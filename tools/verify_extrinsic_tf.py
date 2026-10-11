#!/usr/bin/env python3
"""Check a real isolated ROS publisher with a Core-supplied frozen camera pose."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import xmlrpc.client

from xgc_camera_calibration.extrinsic_resolver import encode_frozen, resolve_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publisher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--static', choices=('true', 'false'), default='true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roles = {'parentFrame': 'world', 'opticalFrames': {'sim': 'acceptance_optical', 'phy': 'acceptance_optical'}}
    choice = {'mode': 'pose', 'pose': {'convention': 'world_T_camera_optical',
        'translation': [1., 2., 3.], 'quaternionXyzw': [0., 0., 0., 1.],
        'coordinates': {'schemaVersion': 1, 'kind': 'experiment-world', 'frame': 'world', 'savedWorldOffset': [0., 0., 0.]}}}
    frozen = encode_frozen(resolve_selection(str(output), 'acceptance', choice,
        {'frame': 'world', 'worldOffset': [0., 0., 0.]}, roles))
    with socket.socket() as probe:
        while True:
            port = random.randint(20000, 40000)
            try:
                probe.bind(('127.0.0.1', port))
                break
            except OSError:
                continue
    env = dict(os.environ, ROS_MASTER_URI='http://127.0.0.1:{}'.format(port),
        ROS_IP='127.0.0.1', ROS_HOME=str(output / 'ros-home'), ROS_LOG_DIR=str(output / 'ros-log'))
    env.pop('ROS_HOSTNAME', None)
    os.environ.pop('ROS_HOSTNAME', None)
    os.environ.update(env)
    processes = []
    logs = []
    subscriber = None
    def spawn(command, name):
        stream = (output / name).open('wb')
        logs.append(stream)
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process
    try:
        master = spawn(['roscore', '-p', str(port)], 'master.log')
        proxy = xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if master.poll() is not None:
                raise RuntimeError('isolated master exited')
            try:
                if proxy.getPid('/acceptance')[0] == 1:
                    break
            except OSError:
                pass
            time.sleep(.05)
        else:
            raise TimeoutError('isolated master did not start')
        import rospy
        from tf2_msgs.msg import TFMessage
        rospy.init_node('frozen_camera_acceptance', anonymous=True, disable_signals=True)
        received = {}
        condition = threading.Condition()
        def observe(message):
            with condition:
                for transform in message.transforms:
                    received[transform.child_frame_id] = transform
                condition.notify_all()
        subscriber = rospy.Subscriber('/tf_static' if args.static == 'true' else '/tf', TFMessage, observe, queue_size=20)
        publisher = spawn([sys.executable, str(args.publisher), '--resolved-extrinsic-json', frozen,
            '--frame-roles-json', json.dumps(roles), '__name:=frozen_camera_subject',
            '_calibration_mode:=phy', '_camera_name:=acceptance', '_parent_frame:=world',
            '_optical_frame:=acceptance_optical', '_camera_link_frame:=acceptance_link', '_static:=' + args.static], 'publisher.log')
        with condition:
            assert condition.wait_for(lambda: 'acceptance_link' in received, timeout=5), 'camera link was not published'
            transform = received['acceptance_link']
        assert publisher.poll() is None
        assert transform.header.frame_id == 'world'
        translation = transform.transform.translation
        assert [translation.x, translation.y, translation.z] == [1., 2., 3.]
        optical = rospy.wait_for_message('/tf_static', TFMessage, timeout=5)
        assert any(item.child_frame_id == 'acceptance_optical' and item.header.frame_id == 'acceptance_link'
                   for item in optical.transforms)
        assert rospy.get_param('/frozen_camera_subject/extrinsic_ready') is True
        assert not list(output.glob('*.yaml'))
        (output / 'acceptance.json').write_text(json.dumps({'passed': True, 'scope': 'real frozen TF publication',
            'master_port': port, 'static': args.static, 'frozen_context': json.loads(frozen)}, indent=2))
    finally:
        if subscriber is not None:
            subscriber.unregister()
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in logs:
            stream.close()


if __name__ == '__main__':
    main()
