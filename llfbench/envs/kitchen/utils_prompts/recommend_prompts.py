# The pools above are fixed sentences with no degree slot; the wrapper builds its movement
# guidance from the templates below instead, pairing a direction from
# ``utils_prompts.direction_prompts`` with a degree adverb from ``utils_prompts.degree_prompts``.

move_recommend_templates = [
    "move {direction} {degree}.",
    "please move {direction} {degree}.",
    "you should move {direction} {degree}.",
    "make a move {direction} {degree}.",
    "you need to move {direction} {degree}.",
    "take the gripper {direction} {degree}.",
]

turn_recommend_templates = [
    "you should {direction} {degree}.",
    "you need to {direction} {degree}.",
    "you can {direction} {degree}.",
    "be sure to {direction} {degree}.",
    "make sure to {direction} {degree}.",
]

close_gripper_recommend = [
    "you should close the gripper.",
    "you need to close the gripper.",
    "close the fingers on it.",
    "make sure to close the gripper.",
]

open_gripper_recommend = [
    "you should open the gripper.",
    "you need to open the gripper.",
    "release the fingers.",
    "make sure to open the gripper.",
]