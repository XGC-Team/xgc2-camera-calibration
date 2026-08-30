"""Canonical AprilGrid board profiles shared by calibration entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


FIELD_6X6_88MM_30PCT = "field_6x6_88mm_30pct"
A4_6X6_24MM_30PCT = "a4_6x6_24mm_30pct"


@dataclass(frozen=True)
class AprilGridProfile:
    profile_id: str
    columns: int
    rows: int
    tag_size_m: float
    tag_gap_m: float
    tag_family: str = "tag36h11"
    start_id: int = 0
    min_tags: int = 6
    gazebo_model: str = ""


PROFILES: Dict[str, AprilGridProfile] = {
    FIELD_6X6_88MM_30PCT: AprilGridProfile(
        profile_id=FIELD_6X6_88MM_30PCT,
        columns=6,
        rows=6,
        tag_size_m=0.088,
        tag_gap_m=0.0264,
        gazebo_model="aprilgrid_6x6_tag36h11_88mm",
    ),
    A4_6X6_24MM_30PCT: AprilGridProfile(
        profile_id=A4_6X6_24MM_30PCT,
        columns=6,
        rows=6,
        tag_size_m=0.024,
        tag_gap_m=0.0072,
        gazebo_model="aprilgrid_6x6_tag36h11_24mm_a4",
    ),
}


def resolve_aprilgrid_profile(profile_id: str) -> AprilGridProfile:
    key = str(profile_id or "").strip()
    try:
        return PROFILES[key]
    except KeyError as error:
        raise ValueError("Unsupported calibration board profile: {}".format(key)) from error
