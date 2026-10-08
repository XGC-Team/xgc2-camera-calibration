"""Real HTTP admission, responsive state and retained evidence during slow solves."""
import json
import threading
import urllib.request
from unittest.mock import patch

import cv2
import pytest

from test_intrinsic_service import (
    make_service, make_diagnostic_result, render_board, CalibrationHttpServer, WEB_ROOT,
    intrinsic_solver, ApiError,
)


def collect(service):
    frame = render_board()
    ok, jpeg = cv2.imencode('.jpg', frame)
    assert ok
    for index in range(3):
        service.process_frame(frame, source_jpeg=jpeg.tobytes(), source_snapshot_id=f'snapshot-{index}')
    assert len(service.samples) == 3
    return frame


def test_http_job_remains_responsive_and_deduplicates_while_solver_is_blocked(tmp_path):
    service = make_service(tmp_path / 'intrinsics.yaml')
    frame = collect(service)
    entered, release = threading.Event(), threading.Event()
    def solve(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return make_diagnostic_result((frame.shape[1], frame.shape[0]), 3)
    server = CalibrationHttpServer(('127.0.0.1', 0), object(), WEB_ROOT,
        frame_ancestors="'self'", intrinsic_service=service)
    server.start()
    base = f'http://127.0.0.1:{server.server_address[1]}/api/v1/intrinsic/'
    def request(path, data=None):
        req = urllib.request.Request(base + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=1) as response:
            return response.status, json.load(response)
    try:
        with patch.object(intrinsic_solver, 'calibrate_intrinsic', side_effect=solve) as compute:
            status, first = request('candidate', b'{}')
            assert status == 202 and entered.wait(1)
            assert request('candidate', b'{}')[1]['job']['id'] == first['job']['id']
            state = request('state')[1]
            assert state['solve_job']['status'] == 'running'
            assert state['candidate_pool']['solve_frozen']
            assert state['evidence']['available']
            before = state['collection_revision']
            service.process_frame(frame, source_snapshot_id='later')
            assert request('state')[1]['collection_revision'] == before
            with pytest.raises(ApiError, match='already running'):
                service.reset()
            release.set()
            service._solve_future.result(timeout=3)
            state = request('state')[1]
            assert state['phase'] == 'candidate_ready'
            assert state['solve_job']['status'] == 'succeeded'
            assert compute.call_count == 1
            candidate_id = state['candidate']['candidate_id']
            saved = request('save', json.dumps({'candidate_id': candidate_id}).encode())[1]
            assert saved['saved'] and saved['candidate_id'] == candidate_id
    finally:
        release.set()
        server.close()
def test_original_capture_survives_reset_and_recovers_with_checkpoint(tmp_path):
    output = tmp_path / 'intrinsics.yaml'
    service = make_service(output)
    collect(service)
    capture = service._evidence_root
    before = (capture / 'source/000.jpg').read_bytes()
    assert (capture / 'observations.npz').is_file()
    restored = make_service(output)
    assert restored._evidence_root == capture
    assert restored.state()['evidence']['available']
    assert len(restored.image_points) == 3
    restored.reset()
    assert (capture / 'source/000.jpg').read_bytes() == before
    assert (capture / 'manifest.json').is_file()
    assert restored.state()['samples'] == 0


def test_failed_job_keeps_evidence_and_exposes_error(tmp_path):
    service = make_service(tmp_path / 'intrinsics.yaml')
    collect(service)
    with patch.object(intrinsic_solver, 'calibrate_intrinsic', side_effect=RuntimeError('numerical worker failed')):
        server = CalibrationHttpServer(('127.0.0.1', 0), object(), WEB_ROOT,
            frame_ancestors="'self'", intrinsic_service=service).start()
        try:
            receipt = service.start_candidate()
            service._solve_future.result(timeout=2)
        finally:
            server.close()
    state = service.state()
    assert state['solve_job']['id'] == receipt['job']['id']
    assert state['solve_job']['status'] == 'failed'
    assert state['solve_job']['error'] == 'numerical worker failed'
    assert state['evidence']['available']
    manifest = json.loads((service._evidence_root / 'manifest.json').read_text())
    assert manifest['solve_job']['status'] == 'failed'
    assert len(manifest['samples']) == 3


def test_result_provenance_is_frozen_at_module_load():
    from pathlib import Path
    from xgc_camera_calibration.intrinsic_service import intrinsic_algorithm_provenance
    before = intrinsic_algorithm_provenance()
    with patch.object(Path, 'read_bytes', side_effect=AssertionError('later checkout edit')):
        assert intrinsic_algorithm_provenance() == before


@pytest.mark.parametrize('workers,auto_run,parallelism', [(2, False, 0), (4, False, 2), (4, True, 1)])
def test_nested_validation_preserves_state_capacity_and_finishes_without_private_pool(tmp_path, workers, auto_run, parallelism):
    from xgc2_xrpc import Runtime
    service = make_service(tmp_path / 'intrinsics.yaml')
    frame = collect(service)
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = [0]
    runtime = Runtime(blocking_workers=workers)
    server = CalibrationHttpServer(('127.0.0.1', 0), None, WEB_ROOT,
        frame_ancestors="'self'", intrinsic_service=service, runtime=runtime, validation_workers=2).start()
    outer = None
    def fit(index):
        with lock:
            active[0] += 1
            if active[0] == max(1, parallelism): entered.set()
        assert release.wait(3)
        return index
    def solve(*args, **kwargs):
        assert kwargs['validation_workers'] == parallelism
        if parallelism:
            futures = [kwargs['validation_submit'](fit, index) for index in range(parallelism)]
            for future in futures: future.result(timeout=3)
        else:
            fit(0)
        return make_diagnostic_result((frame.shape[1], frame.shape[0]), 3)
    try:
        with patch.object(intrinsic_solver, 'calibrate_intrinsic', side_effect=solve):
            if auto_run:
                # The real sweep waits synchronously for its nested solve.
                outer = runtime.submit_blocking(server.host, lambda: service.calibrate(_auto_run=True))
            else:
                service.start_candidate()
            assert entered.wait(1)
            base = 'http://127.0.0.1:{}/api/v1/intrinsic/state'.format(server.server_address[1])
            with urllib.request.urlopen(base, timeout=1) as response:
                assert json.load(response)['solve_job']['status'] == 'running'
            # Outer solve (+ optional sweep) and fits share the actual owner.
            assert len(runtime._jobs) == 1 + int(auto_run) + max(0, parallelism)
            assert server.host._jobs == runtime._jobs
            release.set()
            service._solve_future.result(timeout=3)
            if outer is not None: outer.result(timeout=3)
            assert service.state()['solve_job']['status'] == 'succeeded'
    finally:
        release.set()
        server.close()
        runtime.close()
