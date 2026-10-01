# Movement-guidance clauses.  ``KitchenMultistepMerger.render`` joins every clause of a
# window into ONE sentence ("Open the gripper, move to the left steadily and pitch upward
# gently."), so the pools below are bare clauses: no leading "you should", no trailing period.
# ``{direction}`` comes from ``utils_prompts.direction_prompts`` and ``{degree}`` from
# ``utils_prompts.degree_prompts``.

move_guidance = (
    "move {direction} {degree}",
)

#: The turn direction words already carry the verb ("pitch upward", "yaw to the left").
turn_guidance = (
    "{direction} {degree}",
)

close_gripper_guidance = (
    "close the gripper",
    "close the fingers",
)

open_gripper_guidance = (
    "open the gripper",
    "open the fingers",
)
