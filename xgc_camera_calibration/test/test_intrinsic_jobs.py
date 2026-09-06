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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
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
            service._solve_thread.join(3)
            state = request('state')[1]
            assert state['phase'] == 'candidate_ready'
            assert state['solve_job']['status'] == 'succeeded'
            assert compute.call_count == 1
            candidate_id = state['candidate']['candidate_id']
            saved = request('save', json.dumps({'candidate_id': candidate_id}).encode())[1]
            assert saved['saved'] and saved['candidate_id'] == candidate_id
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


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
        receipt = service.start_candidate()
        service._solve_thread.join(2)
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
