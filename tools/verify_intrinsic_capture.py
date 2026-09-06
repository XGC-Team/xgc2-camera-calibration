#!/usr/bin/env python3
"""Replay a preserved capture through real candidate/status/save HTTP endpoints.

Writes only to a NEW output directory. Never modifies the capture or activates
its result in a robot/experiment. Requires the calibration package dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1] / 'xgc_camera_calibration'
sys.path.insert(0, str(PACKAGE / 'src'))
from xgc_camera_calibration.intrinsic_service import IntrinsicCalibrationService
from xgc_camera_calibration.web_service import CalibrationHttpServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = [p for p in (capture / 'observations.npz', capture / 'intrinsics.yaml.session.npz') if p.is_file()]
    if len(checkpoints) != 1:
        raise ValueError('Capture must contain exactly one supported checkpoint')
    with np.load(checkpoints[0], allow_pickle=False) as archive:
        fingerprint = json.loads(str(archive['fingerprint']))
        arrays = {name: archive[name].copy() for name in archive.files if name != 'capture_id'}
    checkpoint = output / 'intrinsics.yaml.session.npz'
    np.savez_compressed(checkpoint, **arrays)
    manifest = json.loads((capture / 'manifest.json').read_text())
    service = IntrinsicCalibrationService(
        board_size=fingerprint['board_size'], square=fingerprint['square'],
        output_file=str(output / 'intrinsics.yaml'), camera_name=fingerprint['camera_name'],
        calibration_mode='phy', board_type=fingerprint['board_type'],
        media_source=fingerprint['media_source'], tag_spacing=fingerprint['tag_spacing'],
        tag_family=fingerprint['tag_family'], tag_start_id=fingerprint['tag_start_id'],
        board_profile_id='a4_6x6_24mm_30pct_kalibr_v1' if fingerprint['square'] == .024 else 'field_6x6_88mm_30pct',
    )
    entries = manifest['samples']
    assert len(entries) == len(service.samples) > 0, service.state()['recovery']
    for index, entry in enumerate(entries):
        normalized = dict(entry)
        normalized['index'] = index
        normalized.setdefault('image_width', service.image_size[0])
        normalized.setdefault('image_height', service.image_size[1])
        for kind in ('source', 'annotated'):
            relative = f'{kind}/{index:03d}.jpg'
            declared = entry.get(kind, {})
            expected = declared.get('sha256', entry.get(kind + '_sha256'))
            data = (capture / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected
            path = service._evidence_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            normalized[kind + '_path'] = relative
            normalized[kind + '_sha256'] = expected
            normalized[kind + '_bytes'] = len(data)
        service._evidence_samples.append(normalized)
    service._save_checkpoint_locked()
    server = CalibrationHttpServer(('127.0.0.1', 0), object(), PACKAGE / 'web/intrinsic',
        frame_ancestors="'self'", intrinsic_service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}/api/v1/intrinsic/'
    latencies = []
    def request(path, body=None):
        req = urllib.request.Request(base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        began = time.monotonic()
        with urllib.request.urlopen(req, timeout=8) as response:
            result = json.load(response)
            status = response.status
        latencies.append(time.monotonic() - began)
        return status, result
    started = time.monotonic()
    try:
        status, receipt = request('candidate', {})
        assert status == 202
        job_id = receipt['job']['id']
        assert request('candidate', {})[1]['job']['id'] == job_id
        previous = None
        while True:
            _, state = request('state')
            job = state['solve_job']
            assert job['id'] == job_id
            progress = (job['stage'], job['completed'], job['total'])
            if progress != previous:
                print(json.dumps({'elapsed_seconds': time.monotonic() - started, 'job': job}), flush=True)
                previous = progress
            if job['status'] != 'running':
                break
            if time.monotonic() - started > 31 * 60:
                raise TimeoutError('Acceptance deadline exceeded; output evidence retained')
            time.sleep(1)
        (output / 'final-state.json').write_text(json.dumps(state, indent=2))
        assert job['status'] == 'succeeded', job
        candidate = state['candidate']
        assert candidate['quality']['status'] == 'save_ready', candidate['quality']
        _, saved = request('save', {'candidate_id': candidate['candidate_id']})
        assert saved['saved']
        assert request('save', {'candidate_id': candidate['candidate_id']})[1] == saved
        assert request('state')[1]['phase'] == 'saved'
        report = {'passed': True, 'scope': 'real HTTP candidate/status/save; replayed original observations',
                  'sample_count': len(entries), 'selected_count': saved['sample_count'],
                  'rms_px': saved['rms_reprojection_error_px'], 'quality': candidate['quality'],
                  'elapsed_seconds': time.monotonic() - started,
                  'max_http_seconds': max(latencies), 'result': saved}
        (output / 'acceptance.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


if __name__ == '__main__':
    main()
