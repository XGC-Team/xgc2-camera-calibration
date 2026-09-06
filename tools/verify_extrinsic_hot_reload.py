#!/usr/bin/env python3
"""Isolated real-ROS hot reload acceptance using synthetic extrinsic assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import xmlrpc.client

import numpy as np
from xgc_camera_calibration.solver import (
    ExtrinsicResult, extrinsic_selection_path, save_extrinsic, write_extrinsic_selection,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publisher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=11429)
    parser.add_argument('--static', choices=['true', 'false'], default='true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    env = dict(os.environ, ROS_MASTER_URI='http://127.0.0.1:{}'.format(args.port),
               ROS_IP='127.0.0.1', ROS_LOG_DIR=str(args.output / 'ros-log'))
    env.pop('ROS_HOSTNAME', None)
    os.environ.update(env)
    processes = []
    receipt = {'status': 'failed', 'synthetic_assets': True, 'static': args.static,
               'publisher_sha256': hashlib.sha256(args.publisher.read_bytes()).hexdigest(),
               'clock': 'use_sim_time=true; no clock publisher', 'checks': []}
    def wait(check, description, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if check():
                return
            time.sleep(.025)
        raise AssertionError(description)
    def spawn(command, log):
        with (args.output / log).open('wb') as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        processes.append(process)
        return process
    try:
        master = spawn(['roscore', '-p', str(args.port)], 'master.log')
        proxy = xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        def ready():
            try:
                return proxy.getPid('/acceptance')[0] == 1
            except OSError:
                return False
        wait(ready, 'isolated master failed to start', 15)
        import rospy
        from tf2_msgs.msg import TFMessage
        rospy.init_node('hot_reload_acceptance', anonymous=True, disable_signals=True)
        rospy.set_param('/use_sim_time', True)
        root = args.output / 'synthetic-camera'
        received = []
        lock = threading.Lock()
        def observe(message):
            with lock:
                for transform in message.transforms:
                    if transform.child_frame_id == 'acceptance_link':
                        received.append((time.monotonic(), transform.transform.translation.x))
        topic = '/tf_static' if args.static == 'true' else '/tf'
        subscriber = rospy.Subscriber(topic, TFMessage, observe, queue_size=20)
        publisher = spawn([sys.executable, str(args.publisher),
            '__name:=hot_reload_subject', '_calibration_root:='+str(root),
            '_calibration_mode:=phy', '_camera_name:=usb_cam', '_parent_frame:=world',
            '_runtime_source_parent_frame:=world', '_optical_frame:=acceptance_optical',
            '_runtime_source_optical_frame:=acceptance_optical', '_camera_link_frame:=acceptance_link',
            '_watch_file:=true', '_wait_for_file:=false', '_static:='+args.static,
            '_file_poll_rate:=10'], 'publisher.log')
        def saw(x):
            with lock:
                return any(abs(value-x) < 1e-9 for _, value in received)
        wait(lambda: saw(0), 'default transform not published with paused clock')
        def save(index, x):
            path = root/'phy'/'usb_cam'/('extrinsics-20260907T00000{}.000000Z.yaml'.format(index))
            identity = 'synthetic-{}'.format(index)
            save_extrinsic(path, ExtrinsicResult(
                translation=np.asarray([x, 0., 0.]), quaternion_xyzw=np.asarray([0., 0., 0., 1.]),
                rotation_world_to_camera=np.eye(3), translation_world_to_camera=np.asarray([-x, 0., 0.]),
                reprojection_errors_px=np.asarray([.1]*4), inlier_indices=np.arange(4), warnings=()),
                calibration_mode='phy', camera_name='usb_cam', parent_frame='world',
                child_frame='acceptance_optical', metadata={'candidate_id': identity})
            start = time.monotonic()
            write_extrinsic_selection(str(root), 'phy', 'usb_cam', path, identity)
            wait(lambda: saw(x), 'saved transform {} not published'.format(index))
            receipt['checks'].append({'saved_x': x, 'latency_seconds': time.monotonic()-start})
            return path
        first = save(1, 1.)
        pointer = extrinsic_selection_path(str(root), 'phy', 'usb_cam')
        pointer.write_text('{broken synthetic selection')
        wait(lambda: bool(rospy.get_param('/hot_reload_subject/extrinsic_update_error', '')),
             'invalid selection error not exposed')
        assert publisher.poll() is None, 'publisher exited on bad selection'
        assert rospy.get_param('/hot_reload_subject/active_extrinsic_file') == str(first)
        second = save(2, 2.)
        wait(lambda: rospy.get_param('/hot_reload_subject/active_extrinsic_file', '') == str(second),
             'active provenance did not advance')
        assert rospy.get_param('/hot_reload_subject/extrinsic_update_error') == ''
        transition = rospy.get_param('/hot_reload_subject/active_extrinsic_transition')
        assert transition['previous_file'] == str(first)
        receipt.update(status='passed', transition=transition, publisher_alive=publisher.poll() is None)
        subscriber.unregister()
        rospy.signal_shutdown('acceptance complete')
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        receipt['owned_processes_stopped'] = all(p.poll() is not None for p in processes)
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
