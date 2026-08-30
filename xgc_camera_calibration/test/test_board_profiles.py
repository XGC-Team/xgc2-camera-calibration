from xgc_camera_calibration.board_profiles import (
    A4_6X6_24MM_30PCT,
    FIELD_6X6_88MM_30PCT,
    resolve_aprilgrid_profile,
)


def test_supported_profiles_are_complete_six_by_six_contracts():
    field = resolve_aprilgrid_profile(FIELD_6X6_88MM_30PCT)
    a4 = resolve_aprilgrid_profile(A4_6X6_24MM_30PCT)
    assert (field.columns, field.rows, field.tag_size_m, field.tag_gap_m) == (
        6, 6, 0.088, 0.0264
    )
    assert (a4.columns, a4.rows, a4.tag_size_m, a4.tag_gap_m) == (
        6, 6, 0.024, 0.0072
    )
    assert field.tag_family == a4.tag_family == "tag36h11"
    assert field.start_id == a4.start_id == 0
    assert field.min_tags == a4.min_tags == 6
    assert field.gazebo_model != a4.gazebo_model


def test_unknown_profile_fails_closed():
    try:
        resolve_aprilgrid_profile("custom-json")
    except ValueError as error:
        assert "Unsupported calibration board profile" in str(error)
    else:
        raise AssertionError("unknown profile was accepted")
