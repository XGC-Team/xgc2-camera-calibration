#!/usr/bin/env python3
"""Independent model sanity check, NOT a Gazebo/detector/service acceptance test.

Run: python docs/reviews/intrinsic_math_probe.py
Requires NumPy and OpenCV. Uses the same free-K, five-coefficient flags and
termination criteria as intrinsic_solver._run_extended_calibration. Inputs
are synthetic exact correspondences, so this does not test image localization.
"""
import json
import math
import cv2
import numpy as np


def run():
    size = (3840, 2160)
    k = np.array([[1350., 0., 1920.], [0., 1335., 1080.], [0., 0., 1.]])
    tag, gap = 0.088, 0.0264
    objects = np.array([
        [col * (tag + gap) + x - .33, row * (tag + gap) + y - .33, 0.]
        for row in range(6) for col in range(6)
        for x, y in ((0., 0.), (tag, 0.), (tag, tag), (0., tag))
    ], dtype=np.float32)
    poses = [
        ((.25, -.35, .05), (-.40, -.25, 1.2)),
        ((-.3, .2, -.1), (.4, -.2, 1.1)),
        ((.45, .1, .2), (.1, .25, 1.4)),
        ((-.2, -.45, -.2), (-.2, .2, 1.0)),
        ((.1, .55, .15), (.1, -.1, .9)),
        ((-.5, -.15, -.25), (-.15, .1, 1.5)),
        ((.4, -.25, .05), (.5, .1, 1.3)),
        ((-.25, .4, -.15), (-.4, -.1, 1.4)),
    ]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-12)
    results = []
    for distortion in (np.zeros(5), np.array([-.08, .015, .0005, -.0003, -.001])):
        observations = [cv2.projectPoints(objects, np.array(r), np.array(t), k, distortion)[0]
                        for r, t in poses]
        assert all(np.isfinite(p).all() for p in observations)
        assert all((p[:, 0, 0] >= 0).all() and (p[:, 0, 0] < size[0]).all()
                   and (p[:, 0, 1] >= 0).all() and (p[:, 0, 1] < size[1]).all()
                   for p in observations)
        fit = cv2.calibrateCameraExtended([objects] * len(poses), observations,
                                         size, None, None, flags=0, criteria=criteria)
        rms, estimated_k, estimated_d = fit[:3]
        focal_error = float(np.max(np.abs(np.diag(estimated_k)[:2] / np.diag(k)[:2] - 1)))
        principal_error = float(np.max(np.abs(estimated_k[:2, 2] - k[:2, 2])))
        distortion_error = float(np.max(np.abs(estimated_d.reshape(-1) - distortion)))
        assert focal_error < 1e-4 and principal_error < .02 and distortion_error < 1e-3
        results.append(dict(truth_distortion=distortion.tolist(), rms_px=float(rms),
                            max_relative_focal_error=focal_error,
                            max_principal_point_error_px=principal_error,
                            max_distortion_coefficient_error=distortion_error))
    # Algebraic counterexample to interpreting a relative stability envelope
    # as an absolute accuracy gate. Not an executed service save transaction.
    training = np.full(8, 20.0)
    median = float(np.median(training))
    sigma = 1.4826 * float(np.median(np.abs(training - median)))
    limit = median + 3.0 * math.hypot(.75, sigma)
    assert 20.0 <= limit  # Large consistent error is not rejected by this inequality.
    return dict(opencv=cv2.__version__, numpy=np.__version__,
                scope="independent exact-correspondence model check, not application E2E",
                fits=results,
                relative_gate_counterexample=dict(training_median_px=median,
                                                   held_out_rms_px=20.,
                                                   confidence_limit_px=limit))


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
