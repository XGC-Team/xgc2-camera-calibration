"""Real native discovery, held readiness, result reads and node arguments."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from xgc2_xrpc import Client, Endpoint, Fault, ServiceRef, TransportError
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

    def test_both_domains_hold_unbound_readiness_and_publish_result_facts(self):
        for kind in ('intrinsic', 'extrinsic'):
            with self.subTest(kind=kind):
                domain, server, initial = self.start(kind)
                with Client.from_service(initial, runtime=server.runtime, local_target='local-test', discovery=True) as client:
                    started = time.monotonic()
                    description = client.json('/v1/describe?wait_ready_ms=150', method='GET', timeout=1)
                    self.assertGreaterEqual(time.monotonic() - started, .14)
                    self.assertFalse(description['ready'])
                    self.assertEqual(description['service'], initial.service)
                    self.assertEqual(description['api_version'], '1')
                    self.assertRegex(description['instance_id'], r'^[0-9a-f]{32}$')
                    self.assertEqual(description['facts']['result_path'], '/v1/result')
                    self.assertEqual(description['facts']['result_kind'], kind)
                    self.assertEqual(description['facts']['camera_id'], domain.camera_name)
                    with self.assertRaises(Fault):
                        client.json('/v1/result', method='GET')
                    domain.waiting.clear()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        waiting = executor.submit(client.json, '/v1/describe?wait_ready_ms=2000', method='GET', timeout=3)
                        self.assertTrue(domain.waiting.wait(1))
                        self.assertFalse(waiting.done())
                        with domain.condition:
                            domain.ready = True
                            domain.condition.notify_all()
                        self.assertTrue(waiting.result(timeout=1)['ready'])
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

    def test_node_arguments_forward_socket_and_target(self):
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
                with patch.object(entry, 'Runtime'), patch.object(entry, 'MediaSnapshotClient'), \
                        patch.object(entry.rospkg, 'RosPack'), \
                        patch.object(entry, 'CalibrationHttpServer', side_effect=RuntimeError('stop test startup')) as server:
                    if kind == 'intrinsic':
                        with patch.object(entry, 'IntrinsicCalibrationService'):
                            self.assertEqual(entry.main(), 1)
                    else:
                        with patch.object(entry, 'RosCalibrationSource') as source, patch.object(entry, 'CalibrationService') as service, \
                                patch.object(entry, 'parse_frame_roles'), patch.object(entry, 'decode_frozen'):
                            source.return_value.pose_coordinate_source = 'experiment-world'
                            source.return_value.pose_world_offset = (0, 0, 0)
                            service.return_value.frozen_extrinsic = {'targetCoordinates': {'worldOffset': [0, 0, 0]}}
                            self.assertEqual(entry.main(), 1)
                    self.assertEqual(server.call_args.kwargs['rpc_socket'], parameters['~socket_path'])
                    self.assertEqual(server.call_args.kwargs['target_id'], parameters['~target_id'])

    def test_extrinsic_readiness_requires_camera_info_with_the_camera_optical_frame(self):
        path = Path(__file__).parents[1] / 'scripts/extrinsic_calibrator_web.py'
        spec = importlib.util.spec_from_file_location('extrinsic_camera_info_test', path)
        entry = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'rospy': MagicMock(), 'geometry_msgs.msg': MagicMock(),
                                       'sensor_msgs.msg': MagicMock()}):
            spec.loader.exec_module(entry)
        source = entry.RosCalibrationSource.__new__(entry.RosCalibrationSource)
        source.lock = threading.RLock()
        source.ready_condition = threading.Condition(source.lock)
        source.optical_frame = 'test-camera-optical'
        source.camera_info = None
        source.use_camera_info = True
        source.preview_jpeg = b'camera image'
        source.snapshot_available = False
        source._readiness_stopping = False
        self.assertFalse(source.wait_ready(0))
        info = SimpleNamespace(K=[500, 0, 320, 0, 500, 240, 0, 0, 1], D=[0, 0, 0, 0, 0],
                               width=640, height=480, distortion_model='plumb_bob',
                               header=SimpleNamespace(frame_id='other-camera-optical'))
        source._camera_info_callback(info)
        self.assertIsNone(source.camera_info)
        self.assertFalse(source.wait_ready(0))
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(source.wait_ready, 2)
            self.assertFalse(waiting.done())
            info.header.frame_id = source.optical_frame
            source._camera_info_callback(info)
            self.assertTrue(waiting.result(timeout=1))
        source.stop_readiness()
        self.assertFalse(source.wait_ready(0))


if __name__ == '__main__':
    unittest.main()
