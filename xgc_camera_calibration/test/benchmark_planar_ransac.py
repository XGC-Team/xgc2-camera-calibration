"""Projected-input microbenchmark; not ROS, browser or camera throughput."""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from xgc_camera_calibration import solver
from test_planar_ransac import scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver-path', type=Path, help='Optional original solver from git show')
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    module = solver
    if args.solver_path is not None:
        source_path = args.solver_path.resolve(strict=True)
        spec = importlib.util.spec_from_file_location('benchmark_original_solver', source_path)
        if spec is None or spec.loader is None:
            parser.error('--solver-path must identify a Python module')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    source = Path(module.__file__).read_bytes()
    source_blob = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
    results = []
    for name, count, outliers, noise in (
        ('clean_16', 16, [], 0.),
        ('one_outlier_16', 16, [0], 0.),
        ('three_outliers_noisy_24', 24, [0, 9, 23], 0.08),
    ):
        world, pixels, intrinsic, distortion, _, position = scene(count)
        if noise:
            pixels += np.random.RandomState(17).normal(0., noise, pixels.shape)
        if outliers:
            pixels[outliers] += [70., -50.] if noise else [400., 0.]
        elapsed = []
        for _ in range(args.repetitions):
            started = time.perf_counter()
            result = module.solve_extrinsic(world, pixels, intrinsic, distortion)
            elapsed.append((time.perf_counter() - started) * 1000.)
        # Count real IPPE calls separately, outside the timed observations.
        with patch.object(cv2, 'solvePnPGeneric', wraps=cv2.solvePnPGeneric) as calls:
            module.solve_extrinsic(world, pixels, intrinsic, distortion)
            ippe_calls = calls.call_count
        results.append({
            'workload': name,
            'median_ms': statistics.median(elapsed),
            'ippe_calls': ippe_calls,
            'inliers': len(result.inlier_indices),
            'position_error_m': float(np.linalg.norm(result.translation - position)),
        })
    print(json.dumps({
        'python': platform.python_version(), 'platform': platform.platform(),
        'opencv': cv2.__version__, 'numpy': np.__version__,
        'solver_git_blob_sha': source_blob, 'repetitions': args.repetitions,
        'scope': 'Projected numerical inputs only; not acquisition, ROS or physical latency.',
        'results': results,
    }, indent=2))


if __name__ == '__main__':
    main()
