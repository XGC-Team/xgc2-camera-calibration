"""Publication facts and the actual TF publisher seam; no live ROS required."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
FACTS = load('record_facts_under_test', ROOT / 'src/xgc_camera_calibration/record_facts.py')
SOURCE = {'id':'camera:world:optical','instanceId':'epoch','kind':'camera-extrinsic','resolutionId':'frozen-resolution'}
FROZEN = {'status':'resolved','resolvedOpticalPose':{'translation':[1,2,3], 'quaternionXyzw':[0,0,0,1]},
          'targetCoordinates':{'worldOffset':[0,0,0]},'resolutionId':'frozen-resolution'}
TRANSFORMS = [{'parentFrame':'world','childFrame':'optical','translation':[1,2,3],'quaternionXyzw':[0,0,0,1]}]


class PublicationFactsTest(unittest.TestCase):
    def owner(self, emit=None, warn=None):
        self.sent = []
        self.warnings = []
        return FACTS.AppliedTransformFacts(SOURCE, emit or self.sent.append, warn or self.warnings.append,
            lambda ros: {'rosTimeNs':str(ros),'unixTimeNs':'1800000000000000123','monotonicTimeNs':str(ros)})

    def test_immutable_fact_dedup_update_and_exact_clock(self):
        owner = self.owner(); transforms=copy.deepcopy(TRANSFORMS); frozen=copy.deepcopy(FROZEN)
        owner.applied(transforms, frozen, 1800000000000000123)
        owner.applied(transforms, frozen, 1800000000000000124)
        transforms[0]['translation'][0]=9; frozen['resolvedOpticalPose']['translation'][0]=9
        owner.applied(transforms, frozen, 1800000000000000125)
        first,second=map(json.loads,self.sent)
        self.assertEqual(first['values']['transforms'][0]['translation'][0],1)
        self.assertEqual(first['effectiveAt']['rosTimeNs'],'1800000000000000123')
        self.assertEqual(second['sequence'],2)
        self.assertEqual(second['values']['transforms'][0]['translation'][0],9)
        self.assertEqual(first['source'],SOURCE)
        self.assertEqual(first['provenance']['selectionConfirmation'],'owned-by-SelectionStore-not-this-fact')

    def test_failed_transport_retries_same_sequence_timestamp_and_bytes(self):
        attempts=[]; fail=[True]
        def emit(text):
            attempts.append(text)
            if fail[0]: raise OSError('transport')
        owner=self.owner(emit)
        owner.applied(TRANSFORMS,FROZEN,123)
        owner.applied(TRANSFORMS,FROZEN,456)
        fail[0]=False; owner.flush()
        self.assertEqual(len(attempts),3)
        self.assertEqual(len(set(attempts)),1)
        self.assertEqual(json.loads(attempts[-1])['effectiveAt']['rosTimeNs'],'123')

    def test_queue_overflow_preserves_sequence_gap_and_does_not_gate(self):
        def fail(_): raise OSError('transport')
        owner=self.owner(fail); owner.MAX_PENDING=2
        for value in range(5):
            frozen=copy.deepcopy(FROZEN); frozen['candidate']=value
            owner.applied(TRANSFORMS,frozen,value)
        self.assertEqual([json.loads(s)['sequence'] for s in owner.pending],[4,5])
        self.assertTrue(any('overflow' in warning for warning in self.warnings))

    def test_uncalibrated_stop_and_diagnostic_failure(self):
        def fail(_): raise OSError('diagnostic')
        owner=self.owner(warn=fail)
        frozen=copy.deepcopy(FROZEN); frozen['status']='uncalibrated'
        owner.applied(TRANSFORMS,frozen,1); owner.stop(2); owner.stop(3)
        self.assertEqual(len(self.sent),2)
        initial,stop=map(json.loads,self.sent)
        self.assertFalse(initial['values']['calibrated'])
        self.assertEqual(stop['event'],'stopped')
        self.assertEqual(stop['provenance']['boundary'],'producer-shutdown')
        owner=self.owner(emit=fail,warn=fail)
        owner.applied(TRANSFORMS,FROZEN,1); owner.flush(); owner.stop(2)


class PublisherSeamTest(unittest.TestCase):
    def install(self, confirmation_error=False, fail_tf=False, fail_metadata=False):
        sent=[]; order=[]; stamps=[]; state={}; ticked=[False]
        stamp=NS(to_nsec=lambda:1800000000000000123)
        def transform():
            return NS(header=NS(frame_id='',stamp=None),child_frame_id='',
                transform=NS(translation=NS(x=0,y=0,z=0),rotation=NS(x=0,y=0,z=0,w=1)))
        params={'~calibration_root':'/not-read','~calibration_mode':'phy','~camera_name':'camera',
                '~parent_frame':'world','~optical_frame':'optical','~camera_link_frame':'link','~static':True}
        def metadata(*args,**kwargs):
            if fail_metadata: raise OSError('metadata unavailable')
            return NS(publish=lambda message: (order.append('fact'),sent.append(json.loads(message.data))))
        ros=types.ModuleType('rospy')
        ros.myargv=lambda:['publisher']; ros.init_node=lambda *_:None
        ros.get_param=lambda name,*default:params[name] if name in params else default[0]
        ros.set_param=lambda key,value:state.update({key:value})
        ros.Time=NS(now=lambda:stamp); ros.is_shutdown=lambda:ticked[0]
        ros.Publisher=metadata
        ros.logwarn=ros.logwarn_throttle=ros.logfatal=lambda *_:None
        class Application:
            def __init__(self,*args):
                self.frozen=copy.deepcopy(FROZEN)
                self.producer={'instanceEpoch':'epoch','resolutionId':'frozen-resolution'}
                self.ready=False; self.project=args[-1]
            def initial_published(self): self.ready=True; order.append('initial-ready')
            def tick(self,apply):
                ticked[0]=True
                next_value=copy.deepcopy(FROZEN); next_value['resolvedOpticalPose']['translation'][0]=9
                apply(next_value)
                order.append('confirm')
                if confirmation_error: raise RuntimeError('selection CAS conflict')
            def state(self): return {'ready':self.ready}
        app=types.ModuleType('xgc_camera_calibration.extrinsic_application')
        app.ExtrinsicApplication=Application
        app.application_arguments=lambda _:NS(frame_roles_json='roles',resolved_extrinsic_json='frozen')
        app.parse_frame_roles=lambda _:{'parentFrame':'world','opticalFrames':{'phy':'optical','sim':'sim_optical'}}
        transforms=types.ModuleType('xgc_camera_calibration.transforms')
        transforms.split_parent_to_optical_pose=lambda t,q,offset:{'parent_t_link':t,'parent_q_link_xyzw':q,
            'link_t_optical':offset,'link_q_optical_xyzw':[0,0,0,1]}
        def broadcast(chain):
            order.append('tf')
            if fail_tf: raise RuntimeError('TF failed')
            stamps.extend(message.header.stamp for message in chain)
        tf=types.ModuleType('tf2_ros'); tf.StaticTransformBroadcaster=lambda:NS(sendTransform=broadcast)
        tf.TransformBroadcaster=tf.StaticTransformBroadcaster
        geometry=types.ModuleType('geometry_msgs.msg'); geometry.TransformStamped=transform
        messages=types.ModuleType('std_msgs.msg'); messages.String=lambda **kwargs:NS(**kwargs)
        modules={'rospy':ros,'tf2_ros':tf,'geometry_msgs':types.ModuleType('geometry_msgs'),'geometry_msgs.msg':geometry,
            'std_msgs':types.ModuleType('std_msgs'),'std_msgs.msg':messages,
            'xgc_camera_calibration':types.ModuleType('xgc_camera_calibration'),
            'xgc_camera_calibration.extrinsic_application':app,'xgc_camera_calibration.transforms':transforms,
            'xgc_camera_calibration.record_facts':FACTS}
        module_patch=patch.dict(sys.modules,modules)
        module_patch.start(); self.addCleanup(module_patch.stop)
        publisher=load('tf_publisher_under_test',ROOT/'scripts/extrinsic_tf_publisher.py')
        publisher.time=NS(sleep=lambda _:None)
        return publisher,sent,order,stamps,stamp

    def test_actual_publisher_keeps_published_update_after_confirmation_failure(self):
        publisher,sent,order,stamps,stamp=self.install(confirmation_error=True)
        self.assertEqual(publisher.main(),0)
        self.assertEqual([entry['event'] for entry in sent],['applied','applied','stopped'])
        self.assertEqual(sent[1]['values']['transforms'][0]['translation'][0],9)
        self.assertLess(order.index('tf'),order.index('fact'))
        self.assertEqual(order[:3],['tf','fact','initial-ready'])
        self.assertEqual(order[3:6],['tf','fact','confirm'])
        self.assertTrue(all(value is stamp for value in stamps))
        self.assertEqual(sent[0]['source']['instanceId'],'epoch')

    def test_failed_tf_does_not_emit_applied_fact(self):
        publisher,sent,order,_,_=self.install(fail_tf=True)
        self.assertEqual(publisher.main(),2)
        self.assertEqual(sent,[])
        self.assertNotIn('initial-ready',order)

    def test_failed_metadata_does_not_reject_tf_or_selection(self):
        publisher,sent,order,_,_=self.install(fail_metadata=True)
        self.assertEqual(publisher.main(),0)
        self.assertEqual(sent,[])
        self.assertIn('initial-ready',order)
        self.assertIn('confirm',order)


if __name__ == '__main__': unittest.main()
