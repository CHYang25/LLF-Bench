"""Stage (task-progress) prompts for FrankaKitchen-v1.

The kitchen goal is a *set* of subtasks that the scripted expert attacks in a randomized
order, so a flat stage enumeration like block pushing's four constants does not apply.
Progress is instead reported on two levels, exactly as ``KitchenWrapper._stage_feedback``
assembles it:

1. which goal subtask is currently being worked on -- ``GOAL_PHRASES`` below; and
2. how far into that subtask the manipulation is -- one template per FSM phase of
   ``llfbench.envs.kitchen.scripted_policy.ScriptedKitchenPolicy``.

Every phase template takes ``{goal}`` (a gerund phrase from ``GOAL_PHRASES``) and, where
it names something to touch, ``{object}`` (a noun phrase from ``OBJECT_PHRASES``).  The
``manipulate`` templates additionally take ``{manipulation}`` from ``MANIPULATION_PHRASES``.
"""

# ---------------------------------------------------------------------------------------
# Per-subtask noun / verb phrases, keyed by the goal names in `OBS_ELEMENT_GOALS`.
#
# Gymnasium-Robotics 1.2.1 spells those keys with spaces and merged the variants 1.2.0
# scored separately: the two hinge doors became one "hinge cabinet" and the four burner
# knobs became "bottom burner" / "top burner". A key that misses here silently degrades
# every sentence to the `unknown_*` fallbacks below, so these have to match exactly.
# ---------------------------------------------------------------------------------------

#: What the whole subtask is, as a gerund phrase: "You are {goal}."
GOAL_PHRASES = {
    "microwave": (
        "opening the microwave door",
        "pulling the microwave door open",
    ),
    "kettle": (
        "moving the kettle onto the top left burner",
        "carrying the kettle over to the top left burner",
    ),
    "light switch": (
        "flipping the light switch on",
        "switching the kitchen light on",
    ),
    "slide cabinet": (
        "sliding the cabinet door open",
        "pushing the sliding cabinet open",
    ),
    "hinge cabinet": (
        "swinging the hinge cabinet door open",
        "opening the hinge cabinet",
    ),
    "bottom burner": (
        "turning the bottom burner knob",
        "twisting the bottom burner knob on",
    ),
    "top burner": (
        "turning the top burner knob",
        "twisting the top burner knob on",
    ),
}

#: The thing the gripper actually reaches for and touches.
OBJECT_PHRASES = {
    "microwave": (
        "the microwave door handle",
        "the handle on the microwave door",
    ),
    "kettle": (
        "the kettle handle",
        "the handle on top of the kettle",
    ),
    "light switch": (
        "the light switch",
        "the light switch on the wall",
    ),
    "slide cabinet": (
        "the sliding cabinet handle",
        "the handle of the sliding cabinet door",
    ),
    "hinge cabinet": (
        "the hinge cabinet handle",
        "the handle of the hinge cabinet door",
    ),
    "bottom burner": ("the bottom burner knob",),
    "top burner": ("the top burner knob",),
}

#: The manipulation itself, as an imperative: "Now {manipulation}."
MANIPULATION_PHRASES = {
    "microwave": (
        "pull the microwave door open",
        "swing the microwave door open",
    ),
    "kettle": (
        "lift the kettle and carry it to the top left burner",
        "pick the kettle up and set it on the top left burner",
    ),
    "light switch": (
        "push the light switch across until it clicks on",
        "slide the light switch on",
    ),
    "slide cabinet": (
        "slide the cabinet door across to open it",
        "push the sliding cabinet door open",
    ),
    "hinge cabinet": (
        "swing the hinge cabinet door open",
        "pull the hinge cabinet door open",
    ),
    "bottom burner": ("twist the bottom burner knob around",),
    "top burner": ("twist the top burner knob around",),
}

#: Fallbacks for a task the pools above do not name.
unknown_goal_phrase = ("working on the next kitchen subtask",)
unknown_object_phrase = ("the target object",)
unknown_manipulation_phrase = ("move the target object into place",)


# ---------------------------------------------------------------------------------------
# Per-phase stage templates.
# ---------------------------------------------------------------------------------------

orient_forward_feedback = (
    "Before reaching for anything, turn the gripper around to face the counter.",
    "Start by rotating the gripper into a front facing pose over the counter.",
    "First square the gripper up with the counter in front of you.",
)

select_subtask_feedback = (
    "Pick the next kitchen subtask to work on.",
    "Choose which kitchen subtask to do next.",
    "Decide on the next subtask in the kitchen goal.",
)

move_to_precontact_feedback = (
    "You are {goal}. Move the gripper toward {object}.",
    "You are {goal}. Bring the gripper over to {object}.",
    "You are {goal}, so get the gripper near {object}.",
)

align_feedback = (
    "You are {goal}. Rotate the wrist so the gripper squares up with {object}.",
    "You are {goal}. Turn the gripper until it lines up with {object}.",
    "You are {goal}, so line the gripper up with {object} before you close in.",
)

approach_feedback = (
    "You are {goal}. Close in on {object}.",
    "You are {goal}. Move the gripper the last stretch onto {object}.",
    "You are {goal}, so bring the fingers right up to {object}.",
)

contact_or_grasp_feedback = (
    "You are {goal}. Take hold of {object}.",
    "You are {goal}. Grasp {object} with the fingers.",
    "You are {goal}, so close the gripper on {object}.",
)

manipulate_feedback = (
    "You are {goal}. Now {manipulation}.",
    "You are {goal}. Go ahead and {manipulation}.",
    "You are {goal}, so {manipulation}.",
)

kettle_transport_feedback = (
    "You are {goal}. Carry the kettle over to the top left burner and set it down.",
    "You have the kettle. Transport it to the top left burner.",
    "The kettle is in the gripper, so move it across to the top left burner.",
)

recede_feedback = (
    "You are {goal}. Let go of {object} and back the gripper away.",
    "You are {goal}. Release {object} and withdraw the gripper.",
    "You are {goal}, so open the fingers on {object} and pull the gripper clear.",
)

verify_feedback = (
    "You are {goal}. Hold steady and check that the subtask is finished.",
    "You are {goal}. Keep still for a moment and confirm the subtask is done.",
    "You are {goal}, so hold the pose and let the subtask settle.",
)

retreat_feedback = (
    "You are done with {object}. Withdraw the gripper and move on to the next subtask.",
    "Pull the gripper back from {object} and go on to the next subtask.",
    "Back the gripper away from {object} so the next subtask can start.",
)

idle_feedback = (
    "Every kitchen subtask in the goal is finished.",
    "All of the goal's kitchen subtasks are complete.",
    "There is nothing left in the kitchen goal to do.",
)
