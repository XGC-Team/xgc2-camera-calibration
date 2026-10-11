"""Static A8 and native camera wiring checks without ROS or the SDK."""
import ast
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'xgc_camera_calibration'


class ProcessContractTest(unittest.TestCase):
    def test_calibration_code_has_no_storage_credentials_or_source_device_access(self):
        forbidden = re.compile(r'bootstrap|startupInputPath|storage[_-]?(?:ref|grant|token)|xgc2\.storage|token|\bbearer\b', re.I)
        files = list((PACKAGE / 'src').rglob('*.py')) + list((PACKAGE / 'scripts').glob('*.py'))
        files += list((PACKAGE / 'launch').glob('*.launch'))
        files += list((ROOT / 'process-definitions').glob('*.json'))
        files += list((ROOT / 'web-src/src').glob('*'))
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text()
                self.assertIsNone(forbidden.search(text))
                self.assertNotRegex(text, r'VideoCapture\(|/dev/video|v4l2src')
        for path in ('scripts/intrinsic_calibrator_web.py', 'scripts/extrinsic_calibrator_web.py',
                     'scripts/extrinsic_tf_publisher.py', 'src/xgc_camera_calibration/web_service.py',
                     'src/xgc_camera_calibration/intrinsic_service.py'):
            text = (PACKAGE / path).read_text()
            self.assertNotRegex(text, r'\b(?:save_intrinsic|save_extrinsic|SelectionStore|SavedExtrinsicApplication|ExtrinsicApplication)\b')
        self.assertFalse((ROOT / 'contracts/storage-manifest.json').exists())
        self.assertFalse((PACKAGE / 'src/xgc_camera_calibration/preferences.py').exists())

    def test_definitions_bind_only_declared_parameters_and_native_endpoints(self):
        for path in sorted((ROOT / 'process-definitions').glob('*.json')):
            document = json.loads(path.read_text())
            self.assertEqual(document['apiVersion'], 'xgc.execution.process/v1')
            definition, = document['definitions']
            with self.subTest(definition=definition['id']):
                properties = definition['parameters']['properties']
                required = definition['parameters']['required']
                self.assertLessEqual(set(required), set(properties))
                self.assertNotIn('calibrationRoot', properties)
                for group in definition['parameters']['groups']:
                    self.assertLessEqual(set(group['parameters']), set(properties))
                args = definition['command']['args']
                for value in args + list(definition['command']['env'].values()):
                    for name in re.findall(r'\$\{([^}]+)\}', value):
                        self.assertIn(name, properties)
                if 'extrinsic' in definition['id']:
                    self.assertTrue({'parentFrame', 'opticalFrame', 'frameRolesJson'} <= set(required))
                if 'calibrator' in definition['id']:
                    self.assertEqual(definition['services'], [{'service': 'xgc2.calibration.v1.Calibration',
                        'api_version': '1', 'profile': 'http.v1', 'endpointParameter': 'socketPath'}])
                    self.assertTrue({'socketPath', 'targetId'} <= set(required))
                    self.assertIn('_socket_path:=${socketPath}', args)
                    self.assertIn('_target_id:=${targetId}', args)
                    self.assertEqual(definition['readiness']['kind'], 'describe')
                    self.assertIn('GET /v1/result', definition['description'])
                    self.assertIn('_media_edge_rpc_socket:=${mediaEdgeRPCSocket}', args)
                else:
                    self.assertNotIn('filePollRate', properties)
                    self.assertEqual(definition['services'], [{'service': 'xgc2.calibration.v1.ExtrinsicTf',
                        'api_version': '1', 'profile': 'http.v1', 'endpointParameter': 'socketPath'}])
                    self.assertTrue({'socketPath', 'targetId'} <= set(required))
                    self.assertEqual(definition['readiness']['kind'], 'describe')

    def test_ros_inputs_and_frames_match_camera_fragment(self):
        path = ROOT / 'process-definitions/xgc2-camera-extrinsic-calibrator-ros1.json'
        definition = json.loads(path.read_text())['definitions'][0]
        properties = definition['parameters']['properties']
        self.assertEqual(properties['imageTopic']['default'], '/usb_cam/image_raw/compressed')
        self.assertEqual(properties['previewImageTopic']['default'], '/usb_cam/image_raw/compressed')
        self.assertEqual(properties['cameraInfoTopic']['default'], '/usb_cam/camera_info')
        self.assertIn('_camera_info_topic:=${cameraInfoTopic}', definition['command']['args'])
        source = (PACKAGE / 'scripts/extrinsic_calibrator_web.py').read_text()
        self.assertIn('CameraInfo, CompressedImage', source)
        self.assertNotIn('rospy.wait_for_message', source)
        self.assertNotIn('get_published_topics', source)
        self.assertIn('self._latest_image', source)
        self.assertIn('message.header.frame_id != self.optical_frame', source)
        self.assertIn('if frame_id != self.optical_frame:', source)
        for kind in ('intrinsic', 'extrinsic'):
            launch = ET.parse(PACKAGE / 'launch' / (kind + '_calibrator.launch'))
            names = {arg.attrib['name'] for arg in launch.findall('arg')}
            self.assertTrue({'socket_path', 'target_id', 'media_edge_rpc_socket', 'media_source_id'} <= names)
            self.assertNotIn('calibration_root', names)
            node = ast.parse((PACKAGE / 'scripts' / (kind + '_calibrator_web.py')).read_text())
            self.assertIsNotNone(node)
        publisher = (PACKAGE / 'scripts/extrinsic_tf_publisher.py').read_text()
        self.assertIn('decode_frozen(args.resolved_extrinsic_json, camera_name, roles)', publisher)
        self.assertNotIn('application.tick', publisher)


if __name__ == '__main__':
    unittest.main()
