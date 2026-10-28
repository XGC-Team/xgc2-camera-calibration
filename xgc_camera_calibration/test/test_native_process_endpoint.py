"""Real native discovery, held readiness, result reads and node arguments."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
import os
import random
import signal
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from xmlrpc.client import ServerProxy
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from xgc2_xrpc import Client, Endpoint, Fault, ServiceRef, TransportError
from xgc_camera_calibration.media_snapshot import MediaSnapshotClient, MediaSnapshotError
from xgc_camera_calibration.web_service import CalibrationHttpServer


class Domain:
    camera_name = 'test-camera'
    calibration_mode = 'phy'

    def __init__(self):
        self.condition = threading.Condition()
        self.waiting = threading.Event()
        self.ready = False
        self.stopping = False
        self.result = None

    def state(self):
        return {'phase': 'collecting'}

    def solve(self, value):
        return value

    def start_candidate(self):
        return {'accepted': True}

    def read_result(self):
        return copy.deepcopy(self.result)

    def wait_ready(self, timeout):
        with self.condition:
            self.waiting.set()
            self.condition.wait_for(lambda: self.ready or self.stopping, timeout)
            return self.ready and not self.stopping

    def stop_readiness(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()


class NativeProcessEndpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        for name in ('index.html', 'app.js', 'styles.css'):
            (self.root / name).write_text('asset')

    def start(self, kind):
        domain = Domain()
        server = CalibrationHttpServer(('127.0.0.1', 0), domain if kind == 'extrinsic' else None,
            self.root, frame_ancestors="'self'", target_id='local-test',
            rpc_socket=str(self.root / (kind + '.sock')),
            intrinsic_service=domain if kind == 'intrinsic' else None).start()
        self.addCleanup(server.close)
        initial = ServiceRef('local-test', 'xgc2.calibration.v1.Calibration', '1', '', 'http.v1',
                             server.service_ref.endpoint)
        return domain, server, initial

    def test_both_domains_publish_local_readiness_and_result_facts(self):
        for kind in ('intrinsic', 'extrinsic'):
            with self.subTest(kind=kind):
                domain, server, initial = self.start(kind)
                with Client.from_service(initial, runtime=server.runtime, local_target='local-test', discovery=True) as client:
                    started = time.monotonic()
                    description = client.json('/v1/describe?wait_ready_ms=150', method='GET', timeout=1)
                    self.assertLess(time.monotonic() - started, 1)
                    self.assertTrue(description['ready'])
                    self.assertEqual(description['service'], initial.service)
                    self.assertEqual(description['api_version'], '1')
                    self.assertRegex(description['instance_id'], r'^[0-9a-f]{32}$')
                    self.assertEqual(description['facts']['result_path'], '/v1/result')
                    self.assertEqual(description['facts']['result_kind'], kind)
                    self.assertEqual(description['facts']['camera_id'], domain.camera_name)
                    with self.assertRaises(Fault):
                        client.json('/v1/result', method='GET')
                    self.assertFalse(domain.ready)
                    self.assertFalse(domain.waiting.is_set())
                    for query in ('wait_ready_ms=-1', 'wait_ready_ms=30001', 'wait_ready_ms=',
                                  'wait_ready_ms=1&wait_ready_ms=2', 'unknown=1'):
                        with self.subTest(query=query), self.assertRaises(Fault) as error:
                            client.json('/v1/describe?' + query, method='GET')
                        self.assertEqual(error.exception.status, 400)
                server.close()
                self.assertFalse(Path(initial.endpoint.address).exists())
                self.assertTrue(server.runtime.closed)

    def test_result_is_instance_pinned_read_only_and_revision_tracks_changes(self):
        for kind in ('intrinsic', 'extrinsic'):
            with self.subTest(kind=kind):
                domain, server, _ = self.start(kind)
                with Client.from_service(server.service_ref, runtime=server.runtime, local_target='local-test') as client:
                    empty = client.json('/v1/result', method='GET')
                    self.assertIsNone(empty['result'])
                    domain.result = {'candidate_id': 'candidate-1', 'document': {'camera_name': domain.camera_name,
                        'sample_count': 4, 'rms_reprojection_error_px': .1, 'camera_matrix': [1, 2, 3]}}
                    before = copy.deepcopy(domain.result)
                    computed = client.json('/v1/result', method='GET')
                    self.assertEqual(computed['camera_id'], domain.camera_name)
                    self.assertEqual(computed['kind'], kind)
                    self.assertEqual(computed['result'], before)
                    self.assertNotEqual(empty['revision'], computed['revision'])
                    self.assertEqual(client.json('/v1/result', method='GET'), computed)
                    self.assertEqual(domain.result, before)
                    domain.result['document']['rms_reprojection_error_px'] = .2
                    changed = client.json('/v1/result', method='GET')
                    self.assertNotEqual(changed['revision'], computed['revision'])
                    domain.result = None
                    self.assertNotEqual(client.json('/v1/result', method='GET')['revision'], changed['revision'])
                    route = '/api/v1/intrinsic/save' if kind == 'intrinsic' else '/api/v1/save'
                    with self.assertRaises(Fault) as removed:
                        client.json(route, {'candidate_id': 'candidate-1'})
                    self.assertEqual(removed.exception.status, 404)
                    server.private_host.instance_id = 'replaced-instance'
                    with self.assertRaises(TransportError):
                        client.json('/v1/result', method='GET')
                server.close()

    def test_invalid_socket_and_target_fail_before_runtime_allocation(self):
        for path, target in (('relative.sock', 'local-test'), ('', 'local-test'), ('/', 'local-test'),
                             ('//private/socket', 'local-test'), ('/private/\x00.sock', 'local-test'),
                             ('/' + 'a' * 107, 'local-test'), (str(self.root / 'a.sock'), None)):
            with self.subTest(path=path), patch('xgc_camera_calibration.web_service.Runtime') as runtime:
                with self.assertRaises(ValueError):
                    CalibrationHttpServer(('127.0.0.1', 0), object(), self.root,
                        frame_ancestors="'self'", target_id=target, rpc_socket=path)
                runtime.assert_not_called()

    def test_media_users_share_one_held_discovery_without_domain_probes(self):
        from dataclasses import asdict
        from urllib.parse import parse_qs
        from xgc2_xrpc import Host, Runtime
        runtime = Runtime()
        self.addCleanup(runtime.close)
        reference = ServiceRef('local-test', 'media-edge', 'v1', 'media-held', 'http.v1',
                               Endpoint('unix', str(self.root / 'held-media.sock')))
        entered, available = threading.Event(), threading.Event()
        calls = []
        async def describe(context, value):
            calls.append(context.query)
            entered.set()
            wait = int(parse_qs(context.query)['wait_ready_ms'][0])
            await runtime.blocking(host, available.wait, wait / 1000.)
            return {'ready': available.is_set(), 'service_ref': asdict(reference)}
        host = Host(reference.endpoint.address, {('GET', '/v1/describe'): describe}, runtime=runtime,
                    instance_id=reference.instance_id, discovery_routes=('/v1/describe',)).start()
        self.addCleanup(host.close)
        media = MediaSnapshotClient(reference.endpoint.address, 'usb_cam', runtime=runtime, local_target='local-test')
        self.addCleanup(media.close)
        self.addCleanup(available.set)
        self.assertEqual(media.status()['connect_attempts'], 0)
        # Concurrent callers share the outstanding watch. Sequential calls can
        # legitimately outlive its 30-second hold and re-arm discovery.
        with ThreadPoolExecutor(max_workers=8) as executor:
            captures = [executor.submit(media.capture) for _ in range(8)]
            self.assertTrue(entered.wait(1))
            self.assertEqual(calls, ['wait_ready_ms=30000'])
            self.assertFalse(media.status()['ready'])
            available.set()
            for capture in captures:
                with self.assertRaises(MediaSnapshotError):
                    capture.result(timeout=6)
        # These are only one-shot capture users. Their final physical join
        # releases discovery instead of retaining a ready peer indefinitely.
        self.assertFalse(media.status()['ready'])
        self.assertEqual(media.status()['reason'], 'not_requested')
        self.assertIsNone(media.peer._thread)
        self.assertEqual(media.status()['connect_attempts'], 1)

    def test_early_not_ready_describe_replies_use_exponential_backoff(self):
        from dataclasses import asdict
        from xgc2_xrpc import Host, Runtime
        runtime = Runtime()
        self.addCleanup(runtime.close)
        reference = ServiceRef('local-test', 'media-edge', 'v1', 'media-waiting', 'http.v1',
                               Endpoint('unix', str(self.root / 'waiting-media.sock')))
        host = Host(reference.endpoint.address, {('GET', '/v1/describe'):
            lambda context, value: {'ready': False, 'service_ref': asdict(reference)}},
            runtime=runtime, instance_id=reference.instance_id, discovery_routes=('/v1/describe',)).start()
        self.addCleanup(host.close)
        media = MediaSnapshotClient(reference.endpoint.address, 'usb_cam', runtime=runtime, local_target='local-test')
        self.addCleanup(media.close)
        with patch('xgc_camera_calibration.peer_connection.random.uniform', side_effect=lambda start, end: end):
            media.peer.activate()
            time.sleep(1.05)
            self.assertFalse(media.status()['ready'])
            self.assertLessEqual(media.status()['connect_attempts'], 3)
            self.assertGreaterEqual(media.status()['connect_attempts'], 2)
            media.close()

    def test_missing_media_waits_on_change_and_last_detector_stop_wakes_it(self):
        from xgc_camera_calibration.intrinsic_service import IntrinsicCalibrationService
        media = MediaSnapshotClient(str(self.root / 'absent.sock'), 'usb_cam', local_target='local-test')
        service = IntrinsicCalibrationService(board_size=(7, 5), square=.2,
            output_file=str(self.root / 'intrinsics.yaml'), camera_name='camera', media_source='usb_cam')
        service.attach_frame_capture(lambda: None)
        service.capture_peer = media.peer
        release = media.peer.retain()
        waited, entered = [], threading.Event()
        original_wait = media.peer._condition.wait
        def wait(timeout=None):
            if threading.current_thread().name == 'intrinsic-live-detection':
                waited.append(timeout)
                entered.set()
            return original_wait(timeout)
        try:
            deadline = time.monotonic() + 2
            while media.status()['reason'] != 'waiting_socket' and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(media.status()['reason'], 'waiting_socket')
            with patch.object(media.peer._condition, 'wait', side_effect=wait):
                service.retain_live_capture(.05)
                self.assertTrue(entered.wait(1))
                time.sleep(.65)
                self.assertEqual(waited, [None])
                service.release_live_capture()
                self.assertIsNone(service._auto_capture_thread)
            release()
            self.assertIsNone(media.peer._thread)
            print('T15_MISSING_MEDIA_WAIT ' + json.dumps({
                'detector_condition_waits': len(waited), 'timed_waits': sum(value is not None for value in waited),
                'peer_owners_after_join': media.peer._owners, 'peer_thread_after_join': media.peer._thread is not None,
            }, sort_keys=True))
        finally:
            service.stop_auto_capture()
            release()
            media.close()

    def test_node_arguments_forward_socket_and_target(self):
        importlib.import_module('xgc_camera_calibration.ros_lifecycle')
        parameters = {'~target_id': 'local-test', '~socket_path': str(self.root / 'allocated.sock'),
                      '~calibration_mode': 'phy', '~camera_name': 'test-camera',
                      '~parent_frame': 'world', '~child_frame': 'test-camera-optical',
                      '~media_edge_rpc_socket': str(self.root / 'media.sock')}
        for kind in ('intrinsic', 'extrinsic'):
            with self.subTest(kind=kind):
                ros = MagicMock()
                ros.get_param.side_effect = lambda name, *default: parameters[name] if name in parameters else default[0]
                ros.myargv.return_value = ['calibrator', '--resolved-extrinsic-json', '{}', '--frame-roles-json', '{}']
                path = Path(__file__).parents[1] / 'scripts' / (kind + '_calibrator_web.py')
                spec = importlib.util.spec_from_file_location(kind + '_entrypoint_test', path)
                entry = importlib.util.module_from_spec(spec)
                with patch.dict('sys.modules', {'rospy': ros, 'geometry_msgs.msg': MagicMock(), 'sensor_msgs.msg': MagicMock()}):
                    spec.loader.exec_module(entry)
                with patch.object(entry, 'NodeParameters') as params, patch.object(entry, 'RosMasterBinding'), \
                        patch.object(entry, 'Runtime'), patch.object(entry, 'MediaSnapshotClient'), \
                        patch.object(entry.rospkg, 'RosPack'), \
                        patch.object(entry, 'CalibrationHttpServer', side_effect=RuntimeError('stop test startup')) as server:
                    params.return_value.get.side_effect = ros.get_param.side_effect
                    if kind == 'intrinsic':
                        with patch.object(entry, 'IntrinsicCalibrationService'):
                            self.assertEqual(entry.main(), 1)
                    else:
                        with patch.object(entry, 'RosCalibrationSource') as source, patch.object(entry, 'CalibrationService') as service, \
                                patch.object(entry, 'resolve_application'):
                            source.return_value.pose_coordinate_source = 'experiment-world'
                            source.return_value.pose_world_offset = (0, 0, 0)
                            service.return_value.frozen_extrinsic = {'targetCoordinates': {'worldOffset': [0, 0, 0]}}
                            self.assertEqual(entry.main(), 1)
                    self.assertEqual(server.call_args.kwargs['rpc_socket'], parameters['~socket_path'])
                    self.assertEqual(server.call_args.kwargs['target_id'], parameters['~target_id'])

    def test_extrinsic_camera_info_is_status_and_keeps_the_optical_frame_fence(self):
        path = Path(__file__).parents[1] / 'scripts/extrinsic_calibrator_web.py'
        spec = importlib.util.spec_from_file_location('extrinsic_camera_info_test', path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        source = entry.RosCalibrationSource.__new__(entry.RosCalibrationSource)
        source.lock = threading.RLock()
        source.ready_condition = threading.Condition(source.lock)
        source.optical_frame = 'test-camera-optical'
        source.camera_info = None
        source._state_listeners = set()
        source._readiness_stopping = False
        self.assertTrue(source.wait_ready(0))
        info = SimpleNamespace(K=[500, 0, 320, 0, 500, 240, 0, 0, 1], D=[0, 0, 0, 0, 0],
                               width=640, height=480, distortion_model='plumb_bob',
                               header=SimpleNamespace(frame_id='other-camera-optical'))
        source._camera_info_callback(info)
        self.assertIsNone(source.camera_info)
        self.assertTrue(source.wait_ready(0))
        info.header.frame_id = source.optical_frame
        source._camera_info_callback(info)
        self.assertIsNotNone(source.camera_info)
        source.stop_readiness()
        self.assertFalse(source.wait_ready(0))

    def test_extrinsic_state_freshness_notifies_on_real_deadline_and_joins(self):
        path = Path(__file__).parents[1] / 'scripts/extrinsic_calibrator_web.py'
        spec = importlib.util.spec_from_file_location('extrinsic_state_expiry_test', path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        source = entry.RosCalibrationSource('', 'phy', 'test-camera',
            parameters={'~pose_coordinate_source': 'experiment-world',
                        '~child_frame': 'test-camera-optical', '~pose_max_age': 0.15},
            frozen_extrinsic={'targetCoordinates': {'worldOffset': [0, 0, 0]}})
        clock = SimpleNamespace(is_rostime_initialized=lambda: True, is_wallclock=lambda: True)
        fake_ros = SimpleNamespace(rostime=clock, Time=SimpleNamespace(
            now=lambda: SimpleNamespace(to_sec=time.time)))
        notifications = []
        with patch.object(entry, 'rospy', fake_ros):
            release = source.subscribe_state(lambda: notifications.append(time.monotonic()))
            try:
                time.sleep(0.2)
                self.assertEqual(notifications, [])
                stamp = time.time()
                message = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1., y=2., z=3.)),
                    header=SimpleNamespace(frame_id='world', stamp=SimpleNamespace(to_sec=lambda: stamp)))
                source._marker_callback('board')(message)
                self.assertEqual(len(notifications), 1)
                deadline = time.monotonic() + 1
                while len(notifications) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(len(notifications), 2)
                self.assertEqual(source._state_freshness[1], ())
            finally:
                release()
            self.assertIsNone(source._state_thread)
            self.assertEqual(source._state_listeners, set())
            before = len(notifications)
            time.sleep(0.2)
            self.assertEqual(len(notifications), before)

    def test_extrinsic_state_freshness_tracks_sim_clock_edges_without_polling(self):
        path = Path(__file__).parents[1] / 'scripts/extrinsic_calibrator_web.py'
        spec = importlib.util.spec_from_file_location('extrinsic_state_clock_test', path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        source = entry.RosCalibrationSource('', 'phy', 'test-camera',
            parameters={'~pose_coordinate_source': 'experiment-world',
                        '~child_frame': 'test-camera-optical', '~pose_max_age': 3.0},
            frozen_extrinsic={'targetCoordinates': {'worldOffset': [0, 0, 0]}})
        clock_value = [10.]
        subscriptions = []
        class Subscriber:
            def __init__(self, _topic, _kind, callback, **_kwargs):
                self.callback, self.closed = callback, False
                subscriptions.append(self)
            def unregister(self):
                self.closed = True
        fake_ros = SimpleNamespace(rostime=SimpleNamespace(
            is_rostime_initialized=lambda: True, is_wallclock=lambda: False),
            Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: clock_value[0])),
            Subscriber=Subscriber)
        notifications = []
        with patch.object(entry, 'rospy', fake_ros):
            release = source.subscribe_state(lambda: notifications.append(clock_value[0]))
            try:
                stamp = 10.
                message = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1., y=2., z=3.)),
                    header=SimpleNamespace(frame_id='world', stamp=SimpleNamespace(to_sec=lambda: stamp)))
                source._marker_callback('board')(message)
                self.assertEqual(len(subscriptions), 1)
                self.assertEqual(notifications, [10.])
                for value in (10.1, 11., 12.9):
                    clock_value[0] = value
                    subscriptions[0].callback(SimpleNamespace(clock=SimpleNamespace(to_sec=lambda: value)))
                self.assertEqual(notifications, [10.])
                clock_value[0] = 13.1
                subscriptions[0].callback(SimpleNamespace(clock=SimpleNamespace(to_sec=lambda: 13.1)))
                self.assertEqual(notifications, [10., 13.1])
                self.assertEqual(source._state_freshness[1], ())
            finally:
                release()
            self.assertTrue(subscriptions[0].closed)
            self.assertIsNone(source._state_thread)

    def test_extrinsic_observer_admission_failure_joins_expiry_owner(self):
        path = Path(__file__).parents[1] / 'scripts/extrinsic_calibrator_web.py'
        spec = importlib.util.spec_from_file_location('extrinsic_state_admission_test', path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        source = entry.RosCalibrationSource('', 'phy', 'test-camera',
            parameters={'~pose_coordinate_source': 'experiment-world',
                        '~child_frame': 'test-camera-optical'},
            frozen_extrinsic={'targetCoordinates': {'worldOffset': [0, 0, 0]}})
        source.snapshot_client = SimpleNamespace(subscribe_state=MagicMock(side_effect=RuntimeError('admission failed')))
        fake_ros = SimpleNamespace(rostime=SimpleNamespace(is_rostime_initialized=lambda: False))
        with patch.object(entry, 'rospy', fake_ros):
            with self.assertRaisesRegex(RuntimeError, 'admission failed'):
                source.subscribe_state(lambda: None)
        self.assertEqual(source._state_listeners, set())
        self.assertIsNone(source._state_thread)

    def test_last_ros_observer_joins_inflight_callback_before_reveal(self):
        path = Path(__file__).parents[1] / 'scripts/extrinsic_calibrator_web.py'
        spec = importlib.util.spec_from_file_location('extrinsic_source_join_test', path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        source = entry.RosCalibrationSource('', 'phy', 'test-camera',
            parameters={'~pose_coordinate_source': 'experiment-world',
                        '~child_frame': 'test-camera-optical', '~trackers': ['board']},
            frozen_extrinsic={'targetCoordinates': {'worldOffset': [0, 0, 0]}})
        subscribers = []
        class Subscriber:
            def __init__(self, _topic, _kind, callback, **_kwargs):
                self.callback, self.closed = callback, False
                subscribers.append(self)
            def unregister(self):
                self.closed = True
        fake_ros = SimpleNamespace(rostime=SimpleNamespace(is_rostime_initialized=lambda: False),
                                   Subscriber=Subscriber)
        entered, finish = threading.Event(), threading.Event()
        def held(_message):
            entered.set()
            if not finish.wait(3):
                raise RuntimeError('controlled ROS callback was not released')
        with patch.object(entry, 'rospy', fake_ros), patch.object(source, '_camera_info_callback', side_effect=held):
            source.start_ros()
            self.assertEqual(subscribers, [])
            release = source.subscribe_state(lambda: None)
            self.assertEqual(len(subscribers), 3)
            worker = threading.Thread(target=subscribers[0].callback, args=(object(),))
            worker.start()
            self.assertTrue(entered.wait(1))
            retired, revealed, result = threading.Event(), threading.Event(), {}
            retire = threading.Thread(target=lambda: (release(), retired.set()))
            def reveal():
                result['release'] = source.subscribe_state(lambda: None)
                revealed.set()
            successor = threading.Thread(target=reveal)
            try:
                retire.start()
                deadline = time.monotonic() + 1
                while not source._ros_retiring and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertTrue(source._ros_retiring)
                self.assertTrue(all(subscriber.closed for subscriber in subscribers))
                successor.start()
                self.assertFalse(retired.wait(.05))
                self.assertFalse(revealed.is_set())
                self.assertEqual(len(subscribers), 3)
                finish.set()
                worker.join(2)
                retire.join(2)
                successor.join(2)
                self.assertTrue(retired.is_set())
                self.assertTrue(revealed.is_set())
                self.assertEqual(len(subscribers), 6)
                result['release']()
                self.assertTrue(all(subscriber.closed for subscriber in subscribers))
                self.assertEqual((source._ros_users, source._ros_callbacks), (0, 0))
                self.assertIsNone(source._state_thread)
                observer = source.subscribe_state(lambda: None)
                operation = source.retain_source()
                observer()
                self.assertEqual(source._ros_users, 1)
                self.assertTrue(all(not subscriber.closed for subscriber in subscribers[-3:]))
                self.assertIsNone(source._state_thread)
                operation()
                self.assertTrue(all(subscriber.closed for subscriber in subscribers))
                self.assertEqual(source._ros_users, 0)
            finally:
                finish.set()
                worker.join(2)
                retire.join(2)
                if successor.ident is not None:
                    successor.join(2)
                if 'release' in result:
                    result['release']()

    def test_repeated_marker_keeps_one_real_expiry_without_reprojection(self):
        path = Path(__file__).parents[1] / 'scripts/extrinsic_calibrator_web.py'
        spec = importlib.util.spec_from_file_location('extrinsic_state_extension_test', path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        source = entry.RosCalibrationSource('', 'phy', 'test-camera',
            parameters={'~pose_coordinate_source': 'experiment-world',
                        '~child_frame': 'test-camera-optical', '~pose_max_age': 0.5},
            frozen_extrinsic={'targetCoordinates': {'worldOffset': [0, 0, 0]}})
        class Subscriber:
            def __init__(self, *_args, **_kwargs):
                self.closed = False
            def unregister(self):
                self.closed = True
        fake_ros = SimpleNamespace(rostime=SimpleNamespace(
            is_rostime_initialized=lambda: True, is_wallclock=lambda: False),
            Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 10.)),
            Subscriber=Subscriber)
        message = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1., y=2., z=3.)),
            header=SimpleNamespace(frame_id='world', stamp=SimpleNamespace(to_sec=lambda: 10.)))
        notifications, computations = [], []
        original = source._freshness_locked
        def freshness(*args, **kwargs):
            computations.append(threading.current_thread().name)
            return original(*args, **kwargs)
        with patch.object(entry, 'rospy', fake_ros), patch.object(source, '_freshness_locked', side_effect=freshness):
            release = source.subscribe_state(lambda: notifications.append(source._state_freshness))
            try:
                source._marker_callback('board')(message)
                first_receipt = source.marker_receipts['board'][0]
                time.sleep(0.03)
                baseline = len(computations)
                for _ in range(20):
                    source._marker_callback('board')(message)
                    time.sleep(0.005)
                self.assertEqual(len(notifications), 1)
                self.assertEqual(len(computations), baseline)
                time.sleep(max(0., first_receipt + 0.52 - time.monotonic()))
                # The first deadline was superseded by later receipts; it
                # must not make the owner compute an unchanged projection.
                self.assertEqual(len(notifications), 1)
                self.assertEqual(len(computations), baseline)
                deadline = time.monotonic() + 1
                while len(notifications) == 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(len(notifications), 2)
                self.assertEqual(source._state_freshness[1], ())
                self.assertEqual(len(computations), baseline + 1)
            finally:
                release()
        self.assertIsNone(source._state_thread)

    def test_extrinsic_describe_survives_ros_clock_initialization_window(self):
        path = Path(__file__).parents[1] / 'scripts/extrinsic_calibrator_web.py'
        spec = importlib.util.spec_from_file_location('extrinsic_clock_window_test', path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        source = entry.RosCalibrationSource('', 'phy', 'test-camera',
            parameters={'~pose_coordinate_source': 'experiment-world',
                        '~child_frame': 'test-camera-optical'},
            frozen_extrinsic={'targetCoordinates': {'worldOffset': [0, 0, 0]}})
        self.addCleanup(source.close)
        initialized = entry.rospy.core.is_initialized()
        clock_initialized = entry.rospy.rostime.is_rostime_initialized()
        self.addCleanup(entry.rospy.core.set_initialized, initialized)
        self.addCleanup(entry.rospy.rostime.set_rostime_initialized, clock_initialized)
        # Noetic init_node sets the node flag before init_simtime. Local HTTP
        # may run in that window, even with no camera or pose publisher.
        entry.rospy.core.set_initialized(True)
        entry.rospy.rostime.set_rostime_initialized(False)
        with self.assertRaises(entry.rospy.exceptions.ROSInitException):
            entry.rospy.Time.now()
        domain = Domain()
        server = CalibrationHttpServer(('127.0.0.1', 0), domain, self.root,
            frame_ancestors="'self'", target_id='local-test',
            rpc_socket=str(self.root / 'clock-window.sock'),
            peer_status=lambda: {'world': source.status()}).start()
        self.addCleanup(server.close)
        with Client.from_service(server.service_ref, runtime=server.runtime,
                                 local_target='local-test') as client:
            before = client.json('/v1/describe?wait_ready_ms=0', method='GET', timeout=1)
            self.assertTrue(before['ready'])
            self.assertFalse(before['facts']['peers']['world']['world_pose_available'])
            entry.rospy.rostime.set_rostime_initialized(True)
            now = entry.rospy.Time.now().to_sec()
            source.marker_latest['body'] = object()
            source.marker_receipts['body'] = (time.monotonic(), now)
            after = client.json('/v1/describe?wait_ready_ms=0', method='GET', timeout=1)
            self.assertTrue(after['ready'])
            self.assertTrue(after['facts']['peers']['world']['world_pose_available'])
        server.close()
        self.assertTrue(server.runtime.closed)
        self.assertFalse(Path(server.service_ref.endpoint.address).exists())

    def test_stop_during_ros_initialization_does_not_admit_a_bound_lifetime(self):
        from xgc_camera_calibration.ros_lifecycle import RosMasterBinding
        import rospy
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                binding = RosMasterBinding('initialization-stop-test')
                self.addCleanup(binding.close)
                def initialize(*_args, **_kwargs):
                    binding._signal(signal.SIGTERM, None)
                    if interrupted:
                        raise rospy.exceptions.ROSInitException('init_node interrupted')
                with patch.object(binding, '_probe', return_value=(123, 'master-run')), \
                        patch.object(rospy.core, 'is_initialized', return_value=False), \
                        patch.object(rospy, 'init_node', side_effect=initialize), \
                        patch.object(rospy, 'signal_shutdown') as shutdown:
                    self.assertFalse(binding.initialize())
                    shutdown.assert_called_once_with('process stopped')
                self.assertIsNone(binding._thread)
                self.assertFalse(binding.status()['available'])
                binding.close()


class NativeEntrypointTest(unittest.TestCase):
    """Real rospy processes, released SDK hosts, and isolated ROS masters."""

    def setUp(self):
        from xgc2_xrpc import Runtime
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runtime = Runtime()
        self.addCleanup(self.runtime.close)
        self.processes = []
        self.streams = []
        self.peers = []
        self.addCleanup(self.cleanup)
        self.master_port = self.port()
        self.env = {**os.environ, "ROS_MASTER_URI": "http://127.0.0.1:%d" % self.master_port,
                    "ROS_HOME": str(self.root / "ros-home"), "ROS_LOG_DIR": str(self.root / "ros-log"),
                    "ROS_IP": "127.0.0.1"}
        self.env.pop("ROS_HOSTNAME", None)
        package = Path(__file__).parents[1]
        self.env["ROS_PACKAGE_PATH"] = str(package.parent) + ":" + self.env.get("ROS_PACKAGE_PATH", "")
        self.package = package
        self.roles = {"parentFrame": "world", "opticalFrames": {"sim": "usb_cam_optical_frame", "phy": "usb_cam_optical_frame"}}
        from xgc_camera_calibration.extrinsic_resolver import resolve_selection, encode_frozen
        self.frozen = encode_frozen(resolve_selection(str(self.root), "usb_cam", {"mode": "auto"},
            {"frame": "world", "worldOffset": [0, 0, 0]}, self.roles, "process-test"))

    @staticmethod
    def port():
        while True:
            with socket.socket() as peer:
                try:
                    peer.bind(('127.0.0.1', random.randint(20000, 40000)))
                    return peer.getsockname()[1]
                except OSError:
                    pass

    def launch(self, command, name):
        log = self.root / (name + '.log')
        stream = log.open('ab')
        self.streams.append(stream)
        process = subprocess.Popen(['nice', '-n', '19'] + command, env=self.env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)
        return process

    def stop(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                self.fail('process did not stop gracefully: %s' % process.args)
        else:
            process.wait()

    def cleanup(self):
        for peer in reversed(self.peers):
            peer.__exit__()
        self.peers.clear()
        for process in reversed(self.processes):
            self.stop(process)
        for stream in self.streams:
            stream.close()

    def start_master(self):
        return self.launch(['roscore', '-p', str(self.master_port)], 'master')

    def start_helper(self, kind, camera=False):
        port = self.port()
        socket_path = str(self.root / (kind + '.sock'))
        parameters = {'target_id': 'test-target', 'socket_path': socket_path,
            'calibration_mode': 'sim', 'camera_name': 'usb_cam',
            'media_edge_rpc_socket': str(self.root / 'media.sock'), 'media_source_id': 'usb_cam',
            'http_port': port, 'snapshot_timeout': .5, 'pose_coordinate_source': 'experiment-world',
            'parent_frame': 'world', 'child_frame': 'usb_cam_optical_frame', 'optical_frame': 'usb_cam_optical_frame',
            'camera_link_frame': 'camera_link', 'trackers': ['body'], 'camera_control': camera,
            'simulation_endpoint': str(self.root / 'world.sock'), 'camera_model_name': 'camera',
            'camera_control_timeout': .5}
        script = 'extrinsic_tf_publisher.py' if kind == 'tf' else kind + '_calibrator_web.py'
        command = [sys.executable, str(self.package / 'scripts' / script)]
        if kind == 'extrinsic':
            command += ['--extrinsic-asset-json', 'null', '--extrinsic-pin-json', 'null',
                        '--target-offset-json', '{"x":0,"y":0,"z":0}', '--frame-roles-json', json.dumps(self.roles)]
        elif kind == 'tf':
            command += ['--resolved-extrinsic-json', self.frozen, '--frame-roles-json', json.dumps(self.roles)]
        command += ['_' + name + ':=' + json.dumps(value) for name, value in parameters.items()]
        process = self.launch(command, kind)
        service = 'xgc2.calibration.v1.ExtrinsicTf' if kind == 'tf' else 'xgc2.calibration.v1.Calibration'
        initial = ServiceRef('test-target', service, '1', '', 'http.v1', Endpoint('unix', socket_path))
        client = Client.from_service(initial, runtime=self.runtime, local_target='test-target', discovery=True)
        self.addCleanup(client.close)
        self.until(lambda: Path(socket_path).exists(), process=process)
        return process, client, port

    def until(self, predicate, timeout=12, process=None, diagnostic=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                self.fail('entrypoint exited: ' + (self.root / ('tf.log' if 'extrinsic_tf' in str(process.args) else
                    'intrinsic.log' if 'intrinsic_calibrator' in str(process.args) else 'extrinsic.log')).read_text())
            try:
                value = predicate()
                if value:
                    return value
            except (TransportError, OSError):
                pass
            time.sleep(.025)
        self.fail('native condition did not become true: ' + repr(diagnostic() if diagnostic else None))

    def describe(self, client, wait=0):
        return client.json('/v1/describe?wait_ready_ms=%d' % wait, method='GET', timeout=max(2, wait / 1000 + 1))

    def state(self, kind, port):
        path = '/api/v1/intrinsic/state' if kind == 'intrinsic' else '/api/v1/state'
        with urllib.request.urlopen('http://127.0.0.1:%d%s' % (port, path), timeout=2) as response:
            return json.load(response)

    def request(self, port, path, value=None):
        request = urllib.request.Request('http://127.0.0.1:%d%s' % (port, path),
            data=json.dumps(value).encode() if value is not None else None,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.load(response)

    def viewer(self, kind, port):
        path = '/api/v1/intrinsic/events' if kind == 'intrinsic' else '/api/v1/events'
        stream = urllib.request.urlopen('http://127.0.0.1:%d%s' % (port, path), timeout=3)
        self.addCleanup(stream.close)
        return stream

    def start_media(self, instance):
        sys.path.insert(0, str(Path(__file__).parent))
        from test_media_snapshot import FakeMediaEdge
        peer = FakeMediaEdge(str(self.root / 'media.sock'), instance).__enter__()
        self.peers.append(peer)
        return peer

    def stop_peer(self, peer):
        self.peers.remove(peer)
        peer.__exit__()

    def start_world(self, instance):
        sys.path.insert(0, str(Path(__file__).parent))
        from test_camera_control import NativeWorld
        peer = NativeWorld(str(self.root / 'world.sock'), instance).__enter__()
        self.peers.append(peer)
        return peer

    def board_assets(self):
        from xgc_camera_calibration.board_profiles import PROFILES, FIELD_6X6_88MM_30PCT
        root = self.root / 'assets' / 'gazebo_sim_worlds'
        root.mkdir(parents=True)
        (root / 'package.xml').write_text('<package format="2"><name>gazebo_sim_worlds</name><version>0.0.1</version>'
            '<description>Native test asset</description><maintainer email="test@example.com">Test</maintainer>'
            '<license>MIT</license></package>')
        model = root / 'models' / PROFILES[FIELD_6X6_88MM_30PCT].gazebo_model / 'model.sdf'
        model.parent.mkdir(parents=True)
        model.write_text("<sdf version='1.6'><model name='board'/></sdf>")
        self.env['ROS_PACKAGE_PATH'] = str(root.parent) + ':' + self.env['ROS_PACKAGE_PATH']

    def publishers(self):
        return self.launch([sys.executable, '-c', """
import signal, time, rospy
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped
import cv2, numpy as np
rospy.init_node('late_camera_topics', disable_signals=True)
stop = [False]
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda *_: stop.__setitem__(0, True))
info = rospy.Publisher('/usb_cam/camera_info', CameraInfo, queue_size=1)
pose = rospy.Publisher('/vrpn_client_node/body/pose', PoseStamped, queue_size=1)
image = rospy.Publisher('/usb_cam/image_raw/compressed', CompressedImage, queue_size=1)
c = CameraInfo(width=384, height=216, distortion_model='plumb_bob',
    K=[100.,0.,192.,0.,101.,108.,0.,0.,1.], D=[0.,0.,0.,0.,0.])
c.header.frame_id='usb_cam_optical_frame'
p=PoseStamped(); p.header.frame_id='world'; p.pose.position.x=1.; p.pose.orientation.w=1.
i=CompressedImage(format='jpeg', data=cv2.imencode('.jpg', np.zeros((216,384,3), np.uint8))[1].tobytes())
i.header.frame_id=c.header.frame_id
try:
    while not stop[0]:
        c.header.stamp=p.header.stamp=i.header.stamp=rospy.Time.now()
        info.publish(c); pose.publish(p); image.publish(i); time.sleep(.1)
finally:
    rospy.signal_shutdown('test done')
"""], 'publishers')

    def test_helpers_serve_held_unbound_describe_before_master_and_exit_on_replacement(self):
        clients = []
        for kind in ('intrinsic', 'extrinsic', 'tf'):
            process, client, port = self.start_helper(kind)
            clients.append((kind, process, client, port))
            started = time.monotonic()
            self.assertEqual(self.describe(client, 150)['ready'], kind != 'tf')
            if kind == 'tf': self.assertGreaterEqual(time.monotonic() - started, .14)
            self.assertFalse(self.describe(client)['facts']['peers']['ros_master']['available'])
            if kind != 'tf':
                with urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=1) as page:
                    self.assertIn(b'<html', page.read().lower())
        master = self.start_master()
        first_master = None
        bound_identities = {}
        def bound_master(client):
            observed = self.describe(client)['facts']['peers']['ros_master']
            return observed if observed['available'] else None
        for kind, process, client, port in clients:
            # Local HTTP readiness does not prove that this process has bound
            # the first master. Establish that identity before replacing it.
            bound = self.until(lambda: bound_master(client), process=process)
            identity = (bound['pid'], bound['run_id'])
            if first_master is None:
                first_master = identity
            self.assertEqual(identity, first_master)
            bound_identities[kind] = identity
            self.assertTrue(bound['run_id'])
            self.assertIsInstance(bound['pid'], int)
            self.assertTrue(self.describe(client)['ready'])
            if kind != 'tf':
                self.assertFalse(self.describe(client)['facts']['peers']['camera_info']['received'])
        self.stop(master)
        self.start_master()
        replaced_at = time.monotonic()
        exit_seconds = {}
        for kind, process, _client, _port in clients:
            self.until(lambda: process.poll() is not None, timeout=20,
                diagnostic=lambda: {"kind":kind, "bound_master":bound_identities[kind],
                    "log_tail":(self.root / (kind + '.log')).read_text()[-2000:]})
            self.assertEqual(process.wait(), 0)
            exit_seconds[kind] = round(time.monotonic() - replaced_at, 3)
        print(json.dumps({"bound_master":first_master, "replacement_exit_seconds":exit_seconds}))

    def test_calibrators_start_before_media_camera_info_and_pose_and_recover(self):
        self.start_master()
        helpers = []
        for kind in ('intrinsic', 'extrinsic'):
            process, client, port = self.start_helper(kind)
            self.until(lambda: self.describe(client)['ready'], process=process)
            facts = self.describe(client)['facts']['peers']
            self.assertFalse(facts['media']['ready'])
            self.assertEqual(facts['media']['connect_attempts'], 0)
            self.assertFalse(facts['camera_info']['received'])
            self.viewer(kind, port)
            helpers.append((kind, process, client, port))
        media = self.start_media('edge-late')
        for kind, process, client, port in helpers:
            self.until(lambda: self.describe(client)['facts']['peers']['media']['instance_id'] == 'edge-late', process=process)
            self.assertTrue(self.describe(client)['ready'])
            if kind == 'intrinsic':
                self.until(lambda: self.state(kind, port)['image_ready'], process=process, diagnostic=lambda: self.state(kind, port))
        topics = self.publishers()
        extrinsic = helpers[1]
        self.until(lambda: self.describe(extrinsic[2])['facts']['peers']['camera_info']['received'])
        self.until(lambda: self.describe(extrinsic[2])['facts']['peers']['world']['pose_available'])
        self.assertEqual(self.request(extrinsic[3], '/api/v1/freeze', {})['frame']['frame_id'], 'usb_cam_optical_frame')
        self.stop(topics)
        self.until(lambda: not self.describe(extrinsic[2])['facts']['peers']['world']['pose_available'])
        self.publishers()
        self.until(lambda: self.describe(extrinsic[2])['facts']['peers']['world']['pose_available'])
        self.stop_peer(media)
        for kind, process, client, port in helpers:
            self.until(lambda: not self.describe(client)['facts']['peers']['media']['ready'])
            self.assertTrue(self.describe(client)['ready'])
        replacement = self.start_media('edge-restarted')
        for kind, process, client, port in helpers:
            self.until(lambda: self.describe(client)['facts']['peers']['media']['instance_id'] == 'edge-restarted'
                and self.describe(client)['facts']['peers']['media']['ready'], process=process)
        self.until(lambda: len(replacement.requests) >= 1)
        self.assertEqual(self.request(extrinsic[3], '/api/v1/freeze', {})['frame']['frame_id'], 'usb_cam_optical_frame')

    def test_extrinsic_views_and_accepted_freeze_own_real_ros_and_media_until_join(self):
        from xgc2_xrpc import multipart
        self.start_master()
        process, client, port = self.start_helper('extrinsic')
        self.until(lambda: self.describe(client)['ready'], process=process)
        self.publishers()
        media = self.start_media('edge-observed')
        topics = ('/usb_cam/camera_info', '/usb_cam/image_raw/compressed', '/vrpn_client_node/body/pose')
        def subscribers():
            with ServerProxy(self.env['ROS_MASTER_URI']) as master:
                code, _message, state = master.getSystemState('/xgc_t15_observer_test')
            self.assertEqual(code, 1)
            return {name: tuple(nodes) for name, nodes in state[1]}
        def owns_topics():
            current = subscribers()
            return all(any('xgc_camera_extrinsic_calibrator_web' in node for node in current.get(topic, ()))
                       for topic in topics)
        def no_topics():
            current = subscribers()
            return all(not any('xgc_camera_extrinsic_calibrator_web' in node for node in current.get(topic, ()))
                       for topic in topics)
        def owned_topic_count():
            current = subscribers()
            return sum(sum('xgc_camera_extrinsic_calibrator_web' in node for node in current.get(topic, ()))
                       for topic in topics)
        self.until(no_topics)
        first = self.viewer('extrinsic', port)
        second = self.viewer('extrinsic', port)
        self.assertEqual(first.readline(), b'id: 1\n')
        self.assertEqual(second.readline(), b'id: 1\n')
        self.until(owns_topics, process=process)
        self.until(lambda: self.state('extrinsic', port)['source']['camera_info_received'], process=process)
        self.until(lambda: self.state('extrinsic', port)['source']['world_pose_available'], process=process)
        self.until(lambda: self.describe(client)['facts']['peers']['media']['ready'], process=process)
        self.assertEqual(self.describe(client)['facts']['peers']['media']['connect_attempts'], 1)
        active_subscribers = owned_topic_count()
        self.assertEqual(active_subscribers, 3)
        state = self.state('extrinsic', port)
        self.assertNotIn('camera_info_connected', state['source'])
        self.assertEqual(set(state['peers']['camera_info']), {'received'})
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def held_capture(value):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('controlled Media capture was not released')
            raw = media.raw if value['includeRgb'] else b''
            metadata = {**media.metadata, 'snapshotId': media.snapshot_ids[-1],
                        'jpegBytes': len(media.jpeg), 'rgbBytes': len(raw)}
            return multipart(metadata, media.jpeg, raw)
        media.custom_capture = held_capture
        captures_before = len(media.requests)
        result = {}
        def freeze():
            try:
                result['value'] = self.request(port, '/api/v1/freeze', {})
            except Exception as error:
                result['error'] = error
        worker = threading.Thread(target=freeze)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            first.close()
            second.close()
            self.assertTrue(owns_topics(), 'accepted freeze lost its ROS source on park')
            release.set()
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertNotIn('error', result)
            self.assertEqual(result['value']['frame']['frame_id'], 'usb_cam_optical_frame')
            freeze_captures = len(media.requests) - captures_before
            self.assertEqual(freeze_captures, 1)
            self.assertEqual(media.deleted, media.snapshot_ids)
            self.until(no_topics, process=process)
            parked_subscribers = owned_topic_count()
            self.assertEqual(parked_subscribers, 0)
            self.until(lambda: self.describe(client)['facts']['peers']['media']['reason'] == 'not_requested')
            parked = self.describe(client)['facts']['peers']['media']['connect_attempts']
            self.stop_peer(media)
            media = self.start_media('edge-revealed')
            time.sleep(.3)
            hidden_attempts = self.describe(client)['facts']['peers']['media']['connect_attempts']
            self.assertEqual(hidden_attempts, parked)
            self.assertEqual(len(media.requests), 0)
            self.assertTrue(no_topics())
            reveal = self.viewer('extrinsic', port)
            self.assertEqual(reveal.readline(), b'id: 1\n')
            self.until(owns_topics, process=process)
            self.until(lambda: self.describe(client)['facts']['peers']['media']['instance_id'] == 'edge-revealed'
                       and self.describe(client)['facts']['peers']['media']['ready'])
            revealed_attempts = self.describe(client)['facts']['peers']['media']['connect_attempts']
            self.assertEqual(revealed_attempts - parked, 1)
            self.until(lambda: self.state('extrinsic', port)['source']['camera_info_received'], process=process)
            self.until(lambda: self.state('extrinsic', port)['source']['world_pose_available'], process=process)
            reveal.close()
            self.until(no_topics, process=process)
            print('T15_EXTRINSIC_NATIVE_WORK ' + json.dumps({
                'visible_readers': 2, 'ros_subscriptions_active': active_subscribers,
                'ros_subscriptions_parked': parked_subscribers, 'accepted_freeze_captures': freeze_captures,
                'hidden_discovery_attempt_delta': hidden_attempts - parked,
                'reveal_discovery_attempt_delta': revealed_attempts - parked,
                'replacement_media_captures': len(media.requests),
            }, sort_keys=True))
        finally:
            release.set()
            worker.join(5)

    def test_world_created_late_replays_board_and_latest_pose_once_per_instance(self):
        self.board_assets()
        self.start_master()
        process, client, port = self.start_helper('intrinsic', camera=True)
        self.until(lambda: self.describe(client)['ready'], process=process)
        self.assertEqual(self.describe(client)['facts']['peers']['world']['connect_attempts'], 0)
        self.viewer('intrinsic', port)
        world = self.start_world('world-late')
        self.until(lambda: self.describe(client)['facts']['peers']['world']['pose_available'], process=process)
        self.assertEqual([c[0] for c in world.calls].count('create'), 1)
        self.request(port, '/api/v1/intrinsic/goto', {'index': 0})
        desired = copy.deepcopy(world.pose)
        revision = self.describe(client)['facts']['peers']['world']['desired_revision']
        self.stop_peer(world)
        self.until(lambda: not self.describe(client)['facts']['peers']['world']['pose_available'])
        self.assertTrue(self.describe(client)['ready'])
        for index in (1, 2):
            with self.assertRaises(urllib.error.HTTPError) as absent:
                self.request(port, '/api/v1/intrinsic/goto', {'index': index})
            self.assertEqual(absent.exception.code, 503)
        target = self.state('intrinsic', port)['targets'][2]['position']
        revision += 2
        replacement = self.start_world('world-restarted')
        self.until(lambda: self.describe(client)['facts']['peers']['world']['pose_available'], process=process)
        self.assertEqual(replacement.pose['position'], target)
        self.assertEqual([c[0] for c in replacement.calls].count('create'), 1)
        self.assertEqual([c[0] for c in replacement.calls].count('pose'), 1)
        status = self.describe(client)['facts']['peers']['world']
        self.assertEqual(status['applied_revision'], revision)
        self.assertEqual(status['applied_instance'], 'world-restarted')
        before = len(replacement.calls)
        time.sleep(.5)
        self.assertEqual(len(replacement.calls), before)

    @staticmethod
    def resources(process):
        root = Path('/proc') / str(process.pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        cpu = (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
        return cpu, len(list((root / 'task').iterdir())), len(list((root / 'fd').iterdir()))

    @unittest.skipUnless(os.environ.get('XGC_CALIBRATION_LONG_TESTS') == '1', '60-second F11 peer outage')
    def test_media_and_world_down_for_sixty_seconds_stay_idle_and_replay_once(self):
        self.board_assets()
        self.start_master()
        process, client, port = self.start_helper('intrinsic', camera=True)
        self.until(lambda: self.describe(client)['ready'], process=process)
        self.viewer('intrinsic', port)
        media = self.start_media('edge-before-outage')
        world = self.start_world('world-before-outage')
        self.until(lambda: self.describe(client)['facts']['peers']['world']['pose_available'])
        self.until(lambda: self.state('intrinsic', port)['image_ready'])
        self.request(port, '/api/v1/intrinsic/goto', {'index': 0})
        desired = copy.deepcopy(world.pose)
        self.stop_peer(media)
        self.stop_peer(world)
        self.until(lambda: not self.describe(client)['facts']['peers']['media']['ready']
            and not self.describe(client)['facts']['peers']['world']['ready'])
        before = self.resources(process)
        start_facts = self.describe(client)['facts']['peers']
        before_lines = len((self.root / 'intrinsic.log').read_text().splitlines())
        time.sleep(60)
        after = self.resources(process)
        end_facts = self.describe(client)['facts']['peers']
        for name in ('media', 'world'):
            self.assertLessEqual(end_facts[name]['connect_attempts'] - start_facts[name]['connect_attempts'], 15)
        log_lines = len((self.root / 'intrinsic.log').read_text().splitlines()) - before_lines
        self.assertLessEqual(log_lines, 14)
        self.assertLess(after[0] - before[0], 1.2)
        self.assertLessEqual(after[1], before[1])
        self.assertLessEqual(after[2], before[2])
        replacement_media = self.start_media('edge-after-outage')
        replacement_world = self.start_world('world-after-outage')
        self.until(lambda: self.describe(client)['facts']['peers']['world']['ready'])
        self.until(lambda: len(replacement_media.requests) >= 1)
        self.assertEqual(replacement_world.pose, desired)
        self.assertEqual([c[0] for c in replacement_world.calls].count('pose'), 1)
        self.assertEqual([c[0] for c in replacement_world.calls].count('create'), 1)
        print('F11 media/world: cpu=%.3fs threads=%d/%d fds=%d/%d log_lines=%d attempts=%s' %
            (after[0]-before[0], before[1], after[1], before[2], after[2], log_lines,
             {name: end_facts[name]['connect_attempts']-start_facts[name]['connect_attempts'] for name in ('media','world')}))

    @unittest.skipUnless(os.environ.get('XGC_CALIBRATION_LONG_TESTS') == '1', '60-second F11 master outage')
    def test_master_down_for_sixty_seconds_stays_idle_then_becomes_ready(self):
        process, client, port = self.start_helper('intrinsic')
        start = self.describe(client)['facts']['peers']['ros_master']['connect_attempts']
        before = self.resources(process)
        before_lines = len((self.root / 'intrinsic.log').read_text().splitlines())
        time.sleep(60)
        after = self.resources(process)
        facts = self.describe(client)['facts']['peers']['ros_master']
        attempts = facts['connect_attempts'] - start
        lines = len((self.root / 'intrinsic.log').read_text().splitlines()) - before_lines
        self.assertLess(after[0]-before[0], 1.2)
        self.assertLessEqual(attempts, 20)
        self.assertLessEqual(lines, 7)
        self.assertLessEqual(after[1], before[1])
        self.assertLessEqual(after[2], before[2])
        self.assertFalse(self.describe(client)['ready'])
        self.start_master()
        self.until(lambda: self.describe(client)['ready'], timeout=35, process=process)
        print('F11 master: cpu=%.3fs threads=%d/%d fds=%d/%d log_lines=%d attempts=%d' %
              (after[0]-before[0], before[1], after[1], before[2], after[2], lines, attempts))


if __name__ == '__main__':
    unittest.main()
