"""Explicit frozen camera context arguments shared by the ROS entrypoints."""

import argparse
from .extrinsic_selection import strict_json, validate_roles
from .solver import CalibrationError


def parse_frame_roles(payload):
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > 1024:
        raise CalibrationError("frame roles JSON exceeds its byte limit")
    return validate_roles(strict_json(payload))


def application_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-extrinsic-json", required=True)
    parser.add_argument("--frame-roles-json", required=True)
    return parser.parse_args(argv)
