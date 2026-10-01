import random
import numpy as np

#: Franka Kitchen world frame: +x is to the robot's right, +y points away from the robot
#: into the counter, +z is up.  Every entry is ordered ``[negative, positive]`` so it can
#: be indexed with ``int(value >= 0)``.
move_direction_desc_list = [
    [  # x
        ["to the left", "toward the left"],
        ["to the right", "toward the right"],
    ],
    [  # y
        ["backward", "toward the back"],
        ["forward", "toward the front"],
    ],
    [  # z
        ["downward", "to the bottom"],
        ["upward", "to the top"],
    ],
]


def move_direction_converter(difference: np.ndarray):
    """Per-axis wording for a Cartesian displacement the gripper still has to make."""
    return [
        random.choice(move_direction_desc_list[i][int(difference[i] >= 0)])
        for i in range(3)
    ]


#: Rotations are world-frame axis-angle components, so the wording follows the right-hand
#: rule about each world axis.  About +x (right) the tool's forward direction tips up;
#: about +y (forward) world-up tips toward the robot's right; about +z (up) the wrist
#: turns counterclockwise seen from above.  Ordered ``[negative, positive]`` as above.
#:
#: NOTE the maniskill pool disagrees on the +y sense; kitchen follows the right-hand rule
#: because the expert's ``action[3:6]`` is exactly such a world-frame rotation vector
#: (see ``rotation_action`` in ``scripted_policy.py``).
turn_direction_desc_list = [
    [  # x
        ["pitch to the bottom", "pitch downward"],
        ["pitch to the top", "pitch upward"],
    ],
    [  # y
        ["roll to the left", "roll toward the left"],
        ["roll to the right", "roll toward the right"],
    ],
    [  # z
        ["yaw to the right", "yaw toward the right"],
        ["yaw to the left", "yaw toward the left"],
    ],
]


def turn_direction_converter(difference: np.ndarray):
    """Per-axis wording for a wrist rotation the gripper still has to make."""
    return [
        random.choice(turn_direction_desc_list[i][int(difference[i] >= 0)])
        for i in range(3)
    ]


def move_direction_pool(axis: int, positive: bool):
    """The paraphrase pool for a signed displacement along world axis ``axis``."""
    return move_direction_desc_list[axis][int(bool(positive))]


def turn_direction_pool(axis: int, positive: bool):
    """The paraphrase pool for a signed rotation about world axis ``axis``."""
    return turn_direction_desc_list[axis][int(bool(positive))]
