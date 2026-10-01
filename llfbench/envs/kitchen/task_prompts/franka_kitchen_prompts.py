"""Stage (task-progress) prompts for FrankaKitchen-v1.

The kitchen goal is a *set* of subtasks that the scripted expert attacks in a randomized
order, so progress is reported on two levels:

1. which goal subtask is being worked on -- ``GOAL_PHRASES`` / ``SUBTASK_NOUNS``; and
2. how far into that subtask the manipulation is -- one clause per FSM phase of
   ``llfbench.envs.kitchen.scripted_policy.ScriptedKitchenPolicy`` (``PHASE_GERUND`` for
   what is happening now, ``PHASE_PAST`` for what happened earlier in a multistep window).

Every stage sentence is assembled by ``KitchenMultistepMerger.render`` from one of the
``stage_*`` frames below plus these clauses.  Pools are deliberately small (1-2 variants) so
the merger's ``parse`` can invert the grammar with a finite lookup table.  Clause templates
take ``{object}`` (from ``OBJECT_PHRASES``) or ``{manipulation}`` (from
``MANIPULATION_GERUNDS`` / ``MANIPULATION_PAST``); a phase's gerund and past pools must take
the same slots.
"""

# ---------------------------------------------------------------------------------------
# Per-subtask phrases, keyed by the goal names in `OBS_ELEMENT_GOALS`.
#
# Gymnasium-Robotics 1.2.1 spells those keys with spaces and merged the variants 1.2.0
# scored separately: the two hinge doors became one "hinge cabinet" and the four burner
# knobs became "bottom burner" / "top burner". A key that misses here silently degrades
# every sentence to the `unknown_*` fallbacks below, so these have to match exactly.
# ---------------------------------------------------------------------------------------

#: What the whole subtask is, as a gerund phrase: "You are {goal}, ..."
GOAL_PHRASES = {
    "microwave": (
        "opening the microwave door",
        "opening the microwave",
    ),
    "kettle": (
        "moving the kettle onto the top left burner",
        "moving the kettle to the top left burner",
    ),
    "light switch": (
        "flipping the light switch on",
        "flipping the light switch",
    ),
    "slide cabinet": (
        "sliding the cabinet door open",
        "sliding the cabinet open",
    ),
    "hinge cabinet": (
        "swinging the hinge cabinet door open",
        "swinging the hinge cabinet open",
    ),
    "bottom burner": (
        "turning the bottom burner knob",
        "turning the bottom burner knob on",
    ),
    "top burner": (
        "turning the top burner knob",
        "turning the top burner knob on",
    ),
}

#: The subtask as a short noun phrase, for "You finished {done} and started on {next}".
SUBTASK_NOUNS = {
    "microwave": ("the microwave",),
    "kettle": ("the kettle",),
    "light switch": ("the light switch",),
    "slide cabinet": ("the slide cabinet",),
    "hinge cabinet": ("the hinge cabinet",),
    "bottom burner": ("the bottom burner",),
    "top burner": ("the top burner",),
}

#: The thing the gripper actually reaches for and touches.
OBJECT_PHRASES = {
    "microwave": (
        "the microwave door handle",
        "the microwave handle",
    ),
    "kettle": (
        "the kettle handle",
        "the handle on the kettle",
    ),
    "light switch": (
        "the light switch",
        "the light",
    ),
    "slide cabinet": (
        "the sliding cabinet handle",
        "the sliding cabinet door handle",
    ),
    "hinge cabinet": (
        "the hinge cabinet handle",
        "the hinge cabinet door handle",
    ),
    "bottom burner": ("the bottom burner knob",),
    "top burner": ("the top burner knob",),
}

#: The manipulation itself while it happens, as a gerund clause.  The goal phrase already
#: names the object, so these stay short.
MANIPULATION_GERUNDS = {
    "microwave": (
        "pulling the door open",
        "swinging the door open",
    ),
    "kettle": (
        "lifting the kettle and carrying it to the burner",
        "lifting the kettle and setting it on the burner",
    ),
    "light switch": (
        "pushing the switch across until it clicks on",
        "pushing the switch across to turn it on",
    ),
    "slide cabinet": (
        "sliding the door across to open it",
        "sliding the door open",
    ),
    "hinge cabinet": (
        "swinging the door open",
        "pulling the door open",
    ),
    "bottom burner": ("twisting the knob around",),
    "top burner": ("twisting the knob around",),
}

#: The same manipulation in the past tense, for the two-phase window summary.
MANIPULATION_PAST = {
    "microwave": (
        "pulled the door open",
        "swung the door open",
    ),
    "kettle": (
        "lifted the kettle and carried it to the burner",
        "lifted the kettle and set it on the burner",
    ),
    "light switch": (
        "pushed the switch across until it clicked on",
        "pushed the switch across to turn it on",
    ),
    "slide cabinet": (
        "slid the door across to open it",
        "slid the door open",
    ),
    "hinge cabinet": (
        "swung the door open",
        "pulled the door open",
    ),
    "bottom burner": ("twisted the knob around",),
    "top burner": ("twisted the knob around",),
}

#: Fallbacks for a task the pools above do not name.
unknown_goal_phrase = ("working on the next kitchen subtask",)
unknown_subtask_noun = ("the subtask",)
unknown_object_phrase = ("the target object",)
unknown_manipulation_gerund = ("moving the target object into place",)
unknown_manipulation_past = ("moved the target object into place",)


# ---------------------------------------------------------------------------------------
# Per-phase clauses.  Keys are the phase constants of ``scripted_policy``.
# ---------------------------------------------------------------------------------------

#: What the arm is doing now: "You are {goal}, {clause}."
PHASE_GERUND = {
    "orient_forward": (
        "turning the gripper to face the counter",
        "rotating the gripper to face the counter",
    ),
    "select_subtask": (
        "picking the next subtask",
        "choosing the next subtask",
    ),
    "move_to_precontact": (
        "moving the gripper toward {object}",
        "bringing the gripper over to {object}",
    ),
    "align": (
        "squaring the gripper up with {object}",
        "lining the gripper up with {object}",
    ),
    "approach": (
        "closing in on {object}",
        "moving in close on {object}",
    ),
    "contact_or_grasp": (
        "taking hold of {object}",
        "closing the gripper on {object}",
    ),
    "manipulate": ("{manipulation}",),
    "kettle_transport": (
        "carrying it over to the burner",
        "carrying it to the burner",
    ),
    "recede": (
        "letting go of {object} and backing away",
        "releasing {object} and backing away",
    ),
    "verify": (
        "holding steady while the subtask settles",
        "holding still while the subtask settles",
    ),
    "retreat": (
        "backing the gripper away",
        "pulling the gripper back",
    ),
}

#: What the arm did earlier in the window: "You are {goal}. You {past1}, then {past2}."
PHASE_PAST = {
    "orient_forward": (
        "turned the gripper to face the counter",
        "rotated the gripper to face the counter",
    ),
    "select_subtask": (
        "picked the next subtask",
        "chose the next subtask",
    ),
    "move_to_precontact": (
        "moved the gripper toward {object}",
        "brought the gripper over to {object}",
    ),
    "align": (
        "squared the gripper up with {object}",
        "lined the gripper up with {object}",
    ),
    "approach": (
        "closed in on {object}",
        "moved in close on {object}",
    ),
    "contact_or_grasp": (
        "took hold of {object}",
        "closed the gripper on {object}",
    ),
    "manipulate": ("{manipulation}",),
    "kettle_transport": (
        "carried it over to the burner",
        "carried it to the burner",
    ),
    "recede": (
        "let go of {object} and backed away",
        "released {object} and backed away",
    ),
    "verify": (
        "held steady while the subtask settled",
        "held still while the subtask settled",
    ),
    "retreat": (
        "backed the gripper away",
        "pulled the gripper back",
    ),
}


# ---------------------------------------------------------------------------------------
# Stage sentence frames.
# ---------------------------------------------------------------------------------------

#: Same subtask and phase over the whole window (this is also every single-step label).
stage_single = (
    "You are {goal}, {clause}.",
    "You are {goal} and {clause}.",
)

#: Same subtask, the phase advanced inside the window.
stage_two_phase = (
    "You are {goal}. You {past1}, then {past2}.",
    "You are {goal}. You {past1} and then {past2}.",
)

#: A subtask was completed inside the window and the next one was selected.
stage_switch = (
    "You finished {done} and started on {next}, {clause}.",
    "You finished {done} and moved on to {next}, {clause}.",
)

#: A subtask was completed inside the window and none has been selected yet.
stage_switch_unselected = (
    "You finished {done}. Pick the next subtask.",
    "You finished {done}. Choose the next subtask.",
)

#: The expert switched subtask without completing the first (stall / abandonment).
stage_moved_on = (
    "You moved on from {done} to {next}, {clause}.",
    "You left {done} for {next}, {clause}.",
)

#: The window began with no subtask selected (orienting / choosing) and one is now active.
stage_started = (
    "You started on {next}, {clause}.",
    "You began {next}, {clause}.",
)

#: No subtask selected (orienting between tasks, choosing the next one).
stage_no_subtask = (
    "You are {clause}.",
)

#: Retreating right after a completion, when the expert has already dropped the subtask.
stage_done_retreat = (
    "You are done with {done}, {clause}.",
    "You are finished with {done}, {clause}.",
)

stage_idle = (
    "Every subtask is done.",
    "Every kitchen subtask is finished.",
)
