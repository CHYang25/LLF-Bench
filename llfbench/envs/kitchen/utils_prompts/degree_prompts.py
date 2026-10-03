import random
import numpy as np

degree_adverbs = {
    "very_low": [
        "gently",
        "softly",
        "slowly",
    ],
    "low": [
        "steadily",
        "calmly",
        "gradually",
    ],
    "medium": [
        "firmly",
        "quickly",
        "strongly",
    ],
}

#: Metres.  Kitchen free-space reaches run to 0.3-0.5 m, an order of magnitude beyond the
#: maniskill cutoffs (1 cm / 3 cm), so those would report "firmly" on almost every step.
MOVE_LOW_THRESHOLD = 2e-2
MOVE_MEDIUM_THRESHOLD = 8e-2

#: Radians.  ``action[3:6]`` is scaled by 0.5 rad, but the scripted expert caps its wrist
#: command at ``max_rotation_step = 0.4``, so the residual it reports never exceeds 0.2 rad.
#: The old 0.2 medium cutoff therefore coincided with that cap: "medium" only ever meant
#: "saturated" (19% of flagged expert turns, all sitting at exactly 0.200) and two thirds of
#: turn guidance fell in "low". Measured 2026-10-01 on flagged expert turns, 0.04 / 0.12
#: splits them ~17 / 43 / 40%: 0.04 rad is ~2 degrees (a trim), 0.12 rad ~7 degrees.
TURN_LOW_THRESHOLD = 4e-2
TURN_MEDIUM_THRESHOLD = 1.2e-1

#: A magnitude inside each bucket, used when a label is parsed back into a record.  Each
#: value re-buckets to its own key and sits above the merger's deadbands.
MOVE_BUCKET_REPRESENTATIVE = {"very_low": 1e-2, "low": 5e-2, "medium": 1.5e-1}
TURN_BUCKET_REPRESENTATIVE = {"very_low": 3e-2, "low": 8e-2, "medium": 1.6e-1}


def _bucket_key(value: float, low: float, medium: float) -> str:
    value = abs(value)
    if value < low:
        return "very_low"
    if value < medium:
        return "low"
    return "medium"


def move_degree_bucket(value: float) -> str:
    """Bucket key for a Cartesian displacement, in metres."""
    return _bucket_key(value, MOVE_LOW_THRESHOLD, MOVE_MEDIUM_THRESHOLD)


def turn_degree_bucket(value: float) -> str:
    """Bucket key for a wrist rotation, in radians."""
    return _bucket_key(value, TURN_LOW_THRESHOLD, TURN_MEDIUM_THRESHOLD)


def _bucket(value: float, low: float, medium: float) -> str:
    return random.choice(degree_adverbs[_bucket_key(value, low, medium)])


def move_degree_adverb_converter(difference: np.ndarray):
    """Per-axis adverb for a Cartesian displacement, in metres."""
    return [_bucket(v, MOVE_LOW_THRESHOLD, MOVE_MEDIUM_THRESHOLD) for v in difference]


def turn_degree_adverb_converter(difference: np.ndarray):
    """Per-axis adverb for a wrist rotation, in radians."""
    return [_bucket(v, TURN_LOW_THRESHOLD, TURN_MEDIUM_THRESHOLD) for v in difference]
