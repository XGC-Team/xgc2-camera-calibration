#!/usr/bin/env python3
"""Require the OpenCV contrib detector used by intrinsic calibration."""

try:
    import cv2
except ImportError as error:
    raise SystemExit(f"OpenCV contrib is required for intrinsic calibration: {error}") from error

aruco = getattr(cv2, "aruco", None)
required = ("DetectorParameters", "ArucoDetector", "getPredefinedDictionary", "DICT_APRILTAG_36h11")
missing = [name for name in required if aruco is None or not hasattr(aruco, name)]
if missing:
    raise SystemExit("OpenCV contrib AprilGrid detector is missing: " + ", ".join(missing))

parameters = aruco.DetectorParameters()
parameters.markerBorderBits = 2
aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), parameters)
print(f"OpenCV {cv2.__version__} AprilGrid detector is available")
