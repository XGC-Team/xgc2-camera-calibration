"""Translate saved optical poses between explicitly identified world origins."""
from collections.abc import Mapping
import numpy as np
from .solver import CalibrationError


def _translation(value, label):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CalibrationError(label + ' must contain three finite coordinates') from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise CalibrationError(label + ' must contain three finite coordinates')
    return result


def coordinate_provenance(kind, frame, world_offset=(0., 0., 0.)):
    if kind not in ('raw-vrpn', 'experiment-world'):
        raise CalibrationError('pose coordinate source must be raw-vrpn or experiment-world')
    if not isinstance(frame, str) or not frame.strip():
        raise CalibrationError('pose source frame must be non-empty')
    offset = _translation(world_offset, 'saved world offset')
    if kind == 'raw-vrpn' and np.any(offset != 0):
        raise CalibrationError('raw VRPN coordinates cannot already contain a world offset')
    return {'schema_version': 1, 'kind': kind, 'frame': frame,
            'world_offset': offset.tolist()}


def optical_translation_in_world(document, target_world_offset):
    """Apply target - saved offset exactly once; never mutate the saved document."""
    target = None if target_world_offset is None else _translation(target_world_offset, 'target world offset')
    translation = _translation(document['translation_array'], 'camera translation')
    metadata = document.get('metadata', {})
    provenance = metadata.get('pose_coordinates') if isinstance(metadata, Mapping) else None
    if provenance is None:
        if target is not None and np.any(target != 0):
            raise CalibrationError('extrinsic coordinate provenance is missing; recalibrate before changing world origin')
        return translation.copy()
    if not isinstance(provenance, Mapping) or provenance.get('schema_version') != 1:
        raise CalibrationError('extrinsic coordinate provenance schema is invalid')
    validated = coordinate_provenance(provenance.get('kind'), provenance.get('frame'),
                                      provenance.get('world_offset'))
    if validated['frame'] != document['parent_frame']:
        raise CalibrationError('extrinsic coordinate provenance frame does not match parent frame')
    if target is None:
        return translation.copy()
    return translation + target - np.asarray(validated['world_offset'])
