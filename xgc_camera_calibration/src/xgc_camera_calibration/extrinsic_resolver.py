"""Freeze one explicit extrinsic choice; consumers never re-read global latest."""
import argparse
import math
import sys
import uuid

import numpy as np

from .extrinsic_coordinates import coordinate_provenance
from .extrinsic_selection import (
    CalibrationError, SelectionStore, closed_object, compact_json, finite_vector,
    identifier, pose_from_document, read_version, revision, strict_json,
    validate_result, validate_roles, validate_storage, validate_target,
)

MAX_CHOICE_BYTES = 3072
MAX_FROZEN_BYTES = 3584


def _pose(value):
    closed_object(value, {"translation", "quaternionXyzw"}, "optical pose")
    translation = finite_vector(value["translation"], 3, "optical translation")
    quaternion = finite_vector(value["quaternionXyzw"], 4, "optical quaternion")
    if not math.isclose(sum(v * v for v in quaternion), 1.0, rel_tol=0, abs_tol=1e-6):
        raise CalibrationError("optical quaternion must be unit length")
    return {"translation": translation, "quaternionXyzw": quaternion}


def _document(pose, source):
    metadata = {}
    if source is not None:
        closed_object(source, {"kind", "frame", "worldOffset"}, "source coordinates")
        metadata["pose_coordinates"] = coordinate_provenance(
            source["kind"], source["frame"], finite_vector(source["worldOffset"], 3, "saved offset"))
    return {"parent_frame": "world", "translation_array": np.asarray(pose["translation"]),
            "quaternion_xyzw_array": np.asarray(pose["quaternionXyzw"]), "metadata": metadata}


def _manual_pose(value):
    closed_object(value, {"convention", "translation", "quaternionXyzw", "coordinates"}, "manual pose")
    if value["convention"] != "world_T_camera_optical":
        raise CalibrationError("manual pose convention must be world_T_camera_optical")
    coordinates = closed_object(value["coordinates"], {"schemaVersion", "kind", "frame", "savedWorldOffset"}, "manual coordinates")
    if type(coordinates["schemaVersion"]) is not int or coordinates["schemaVersion"] != 1 or coordinates["kind"] != "experiment-world" or coordinates["frame"] != "world":
        raise CalibrationError("manual pose needs explicit experiment-world coordinate provenance")
    pose = _pose({key: value[key] for key in ("translation", "quaternionXyzw")})
    return _document(pose, {"kind": coordinates["kind"], "frame": coordinates["frame"],
                            "worldOffset": coordinates["savedWorldOffset"]})


def resolve_selection(root, camera_name, choice, target_coordinates, frame_roles, resolution_id=None):
    root = validate_storage(root, camera_name)
    roles = validate_roles(frame_roles)
    target = validate_target(target_coordinates)
    if len(compact_json(choice).encode("utf-8")) > MAX_CHOICE_BYTES:
        raise CalibrationError("extrinsic choice exceeds its byte limit")
    if not isinstance(choice, dict) or choice.get("mode") not in ("auto", "version", "pose"):
        raise CalibrationError("extrinsic choice mode is invalid")
    mode = choice["mode"]
    closed_object(choice, {"mode"} | ({"result"} if mode == "version" else {"pose"} if mode == "pose" else set()), "extrinsic choice")
    # Construction validates the explicit root without creating any state.
    store = SelectionStore(root, camera_name, roles)
    value = {"schemaVersion": 1, "resolutionId": identifier(resolution_id if resolution_id is not None else str(uuid.uuid4()), "resolution id"),
             "cameraName": camera_name, "choiceMode": mode, "targetCoordinates": target}
    if mode == "pose":
        document = _manual_pose(choice["pose"])
        value["status"] = "manual"
    else:
        if mode == "auto":
            selection = store.read()
            applied = selection["applied"]
            value["globalAppliedRevision"] = applied["appliedRevision"] if applied else 0
            if applied is None:
                value["status"] = "uncalibrated"
                encode_frozen(value)
                return value
            result = applied["result"]
        else:
            result = validate_result(choice["result"])
        document = read_version(root, camera_name, result, roles)
        value.update(status="calibrated", result=dict(result), sourceOpticalFrame=document["child_frame"])
    source, original, resolved = pose_from_document(document, target)
    value.update(sourceCoordinates=source, originalOpticalPose=original, resolvedOpticalPose=resolved)
    encode_frozen(value)
    return value


def _validate_frozen(value):
    base = {"schemaVersion", "resolutionId", "cameraName", "choiceMode", "status", "targetCoordinates"}
    if not isinstance(value, dict) or type(value.get("schemaVersion")) is not int or value["schemaVersion"] != 1:
        raise CalibrationError("frozen extrinsic schema is invalid")
    identifier(value.get("resolutionId"), "resolution id")
    validate_storage("/declared-calibration-root", value.get("cameraName"))
    target = validate_target(value.get("targetCoordinates"))
    mode, status = value.get("choiceMode"), value.get("status")
    if mode == "auto":
        base.add("globalAppliedRevision")
        revision(value.get("globalAppliedRevision"))
    if status == "uncalibrated":
        closed_object(value, base, "uncalibrated result")
        if mode != "auto" or value["globalAppliedRevision"] != 0:
            raise CalibrationError("uncalibrated result cannot claim an applied revision")
        return
    if status == "calibrated" and mode in ("auto", "version"):
        base |= {"result", "sourceOpticalFrame"}
        validate_result(value.get("result"))
        frame = value.get("sourceOpticalFrame")
        if not isinstance(frame, str) or not frame or len(frame) > 128:
            raise CalibrationError("frozen source optical frame is invalid")
        if mode == "auto" and value["globalAppliedRevision"] == 0:
            raise CalibrationError("calibrated auto result requires an applied revision")
    elif not (status == "manual" and mode == "pose"):
        raise CalibrationError("frozen extrinsic status does not match choice")
    closed_object(value, base | {"sourceCoordinates", "originalOpticalPose", "resolvedOpticalPose"}, "frozen extrinsic")
    original = _pose(value["originalOpticalPose"])
    resolved = _pose(value["resolvedOpticalPose"])
    source = value["sourceCoordinates"]
    if status == "manual" and (not isinstance(source, dict) or source.get("kind") != "experiment-world"):
        raise CalibrationError("manual frozen pose has no coordinate provenance")
    _, _, expected = pose_from_document(_document(original, source), target)
    if resolved != expected:
        raise CalibrationError("frozen extrinsic must contain exactly one coordinate conversion")


def encode_frozen(value):
    _validate_frozen(value)
    payload = compact_json(value)
    if len(payload.encode("utf-8")) > MAX_FROZEN_BYTES:
        raise CalibrationError("frozen extrinsic exceeds stdout budget")
    return payload


def decode_frozen(payload, camera_name, frame_roles):
    """Validate downstream bytes without consulting files or global selection."""
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_FROZEN_BYTES:
        raise CalibrationError("frozen extrinsic exceeds stdout budget")
    roles = validate_roles(frame_roles)
    value = strict_json(payload)
    _validate_frozen(value)
    if value["cameraName"] != camera_name:
        raise CalibrationError("frozen extrinsic camera does not match")
    if value["status"] == "calibrated" and value["sourceOpticalFrame"] != roles["opticalFrames"][value["result"]["sourceMode"]]:
        raise CalibrationError("frozen extrinsic optical role does not match")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--target-offset-json", required=True)
    parser.add_argument("--frame-roles-json", required=True)
    parser.add_argument("--resolution-id")
    parser.add_argument("--legacy-pose-source", choices=("authored", "file"))
    parser.add_argument("--legacy-file", default="")
    parser.add_argument("--legacy-link-pose-json", default="")
    args = parser.parse_args(argv)
    try:
        if len(args.selection_json.encode("utf-8")) > MAX_CHOICE_BYTES:
            raise CalibrationError("extrinsic choice exceeds its byte limit")
        if len(args.target_offset_json.encode("utf-8")) > 1024 or len(args.frame_roles_json.encode("utf-8")) > 1024:
            raise CalibrationError("resolver context exceeds its byte limit")
        offset = closed_object(strict_json(args.target_offset_json), {"x", "y", "z"}, "target offset")
        target = validate_target({"frame": "world", "worldOffset": [offset[key] for key in ("x", "y", "z")]})
        if args.selection_json:
            choice = strict_json(args.selection_json)
        else:
            from .camera_initial_pose import legacy_selection_choice
            if len(args.legacy_link_pose_json.encode("utf-8")) > 1024:
                raise CalibrationError("legacy camera pose exceeds its byte limit")
            choice = legacy_selection_choice(args.root, args.camera, args.legacy_pose_source,
                args.legacy_file,
                strict_json(args.legacy_link_pose_json) if args.legacy_pose_source == "authored" else None,
                target)
        value = resolve_selection(args.root, args.camera, choice, target,
                                  strict_json(args.frame_roles_json), args.resolution_id)
        payload = encode_frozen(value)
    except (CalibrationError, OSError, ValueError, TypeError, OverflowError, RecursionError) as error:
        print("Extrinsic resolution failed: " + str(error), file=sys.stderr)
        return 1
    sys.stdout.write(payload)
    return 0
