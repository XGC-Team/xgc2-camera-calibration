"""Freshness admission for stationary camera calibration observations."""
import math


def pose_is_fresh(receipt, wall_now, ros_now, maximum_age):
    """Require recent receipt and source stamp; no motion or alignment model."""
    if receipt is None:
        return False
    received, stamped = receipt
    values = (received, stamped, wall_now, ros_now, maximum_age)
    if not all(math.isfinite(v) for v in values) or maximum_age <= 0:
        return False
    return (stamped > 0 and 0 <= wall_now - received <= maximum_age
            and 0 <= ros_now - stamped <= maximum_age)
