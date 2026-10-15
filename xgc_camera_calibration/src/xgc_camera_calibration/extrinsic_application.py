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
    parser.add_argument("--resolved-extrinsic-json", default="")
    parser.add_argument("--extrinsic-asset-json", default="null")
    parser.add_argument("--extrinsic-pin-json", default="null")
    parser.add_argument("--target-offset-json", default='{"x":0,"y":0,"z":0}')
    parser.add_argument("--frame-roles-json", required=True)
    args = parser.parse_args(argv)
    return args


def resolve_application(args, camera_name):
    """The product interprets pinned assets before starting its local consumers."""
    from .extrinsic_resolver import decode_frozen, encode_frozen, resolve_pinned_asset
    roles = parse_frame_roles(args.frame_roles_json)
    if args.resolved_extrinsic_json:
        return decode_frozen(args.resolved_extrinsic_json, camera_name, roles)
    offset = strict_json(args.target_offset_json)
    value = resolve_pinned_asset(camera_name, strict_json(args.extrinsic_asset_json),
        strict_json(args.extrinsic_pin_json), {"frame": "world", "worldOffset": [offset[k] for k in ("x", "y", "z")]}, roles)
    args.resolved_extrinsic_json = encode_frozen(value)
    return value
