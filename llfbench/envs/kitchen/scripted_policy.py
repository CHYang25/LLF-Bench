"""Hierarchical scripted expert for the ``FrankaKitchen-v1`` environment.

The policy is *state driven*: every action is recomputed from the live MuJoCo state, so
there is no open-loop action sequence anywhere in this file.  A small planner picks one
unfinished task at a time and hands it to a skill, and each skill runs the same finite
state machine::

    ORIENT_FORWARD -> SELECT_SUBTASK -> MOVE_TO_PRECONTACT -> ALIGN -> APPROACH
                   -> CONTACT_OR_GRASP -> MANIPULATE -> RECEDE -> VERIFY -> RETREAT
                   -> SELECT_SUBTASK

This module deliberately targets ``gymnasium-robotics==1.2.0``.  That release had a
one-off Cartesian action interface which was reverted in 1.2.1; the current 9-D
joint-velocity Franka Kitchen interface is a different controller contract.  The policy
checks the action space when it is constructed and fails loudly rather than silently
feeding Cartesian deltas to a joint-velocity environment.

Action semantics in the supported 1.2.0 environment (verified at runtime):

* ``action[:3]``  end-effector position delta in the **world** frame, scaled by
  ``MAX_CARTESIAN_DISPLACEMENT = 0.2`` m.
* ``action[3:6]`` orientation delta as euler angles, scaled by
  ``MAX_ROTATION_DISPLACEMENT = 0.5`` rad, left-multiplied onto the current EEF quaternion
  (so it is a rotation expressed in the world frame).
* ``action[6]``   gripper. ``+1`` opens (finger joints -> 0.04), ``-1`` closes (-> 0.0).
  This is the *opposite* of the MetaWorld ``grab_effort`` convention and was confirmed by
  stepping the env and reading ``robot:finger_joint1``.

Everything is clipped into ``Box(-1, 1, (7,))``; ``FrankaRobot.step`` clips anyway, so
over-scaling an action buys nothing.

Facts about this particular kitchen model that shape the skills below (all measured, see
``debug_scripted_policy.py`` to re-measure):

* The EEF site sits at the hand flange; the fingertip pads are ``FINGERTIP_OFFSET`` =
  0.108 m further along the gripper's approach (local +z) axis.
* The four ``*_burner`` task joints have range ``[-0.009, 0]`` while their goal is
  ``-0.01``, so ``|achieved - desired| = 0.01 < BONUS_THRESH`` **at reset**: the burner
  tasks are scored as complete before the robot moves.  Equality constraints couple the
  physical ``knob_Joint_N`` joints to those burner joints, but the unusually large reward
  tolerance means no knob motion is needed to score them.
* The slide-cabinet door occupies the space the arm has to swing through to reach the
  oven knobs and the light switch, so ``slide_cabinet`` is first in the default order.
* The stock gripper actuator produces only ~3 N of squeeze at handle diameter.  The policy
  scales its stiffness (without changing its open/closed equilibria) to approximate the
  physical Franka's useful grip range.  The kettle skill grasps the thinner vertical bar
  on the handle's left side and transports it horizontally at its measured height; a
  hand-body push remains available as ``KitchenPolicyConfig.kettle_strategy='push'``.
"""

import mujoco
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from gymnasium_robotics.envs.franka_kitchen.kitchen_env import BONUS_THRESH, OBS_ELEMENT_GOALS
from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, get_site_xmat, get_site_xpos


# ---------------------------------------------------------------------------------------
# Environment constants (mirrored from gymnasium_robotics.envs.franka_kitchen.franka_env)
# ---------------------------------------------------------------------------------------

MAX_CARTESIAN_DISPLACEMENT = 0.2   # metres of EEF travel commanded by action[:3] == 1
MAX_ROTATION_DISPLACEMENT = 0.5    # radians of EEF rotation commanded by action[3:6] == 1
ACTION_DIM = 7

#: Distance from the ``EEF`` site to the fingertip pads, along the gripper approach axis.
#: Measured from ``data.geom_xpos`` of the fingertip pad geoms at reset.
FINGERTIP_OFFSET = 0.108

#: Nominal tool axis for skills that do not request a task-specific grasp orientation.
NOMINAL_APPROACH_AXIS = np.array([0.0, 0.0, -1.0])

#: Room-to-cabinet direction in the fixed Franka Kitchen scene.  Together with world-up
#: (the slide handle) and world-right (the slide joint), this is the third member of the
#: requested mutually orthogonal slide-grasp frame.
FRONT_APPROACH_AXIS = np.array([0.0, 1.0, 0.0])
VERTICAL_HANDLE_AXIS = np.array([0.0, 0.0, 1.0])

#: MuJoCo site each skill reaches for, keyed by task (= goal joint) name.
TASK_SITES = {
    "bottom_right_burner": "knob1_site",
    "bottom_left_burner": "knob2_site",
    "top_right_burner": "knob3_site",
    "top_left_burner": "knob4_site",
    "light_switch": "light_site",
    "slide_cabinet": "slide_site",
    "left_hinge_cabinet": "hinge_site1",
    "right_hinge_cabinet": "hinge_site2",
    "microwave": "microhandle_site",
    "kettle": "kettle_site",
}

#: The joint a skill physically drives, when it differs from the joint the env scores.
#: MuJoCo equality constraints couple ``knob_Joint_N`` to the scored burner slide joint.
#: This mapping is used for physical progress detection.
TASK_MANIPULATED_JOINTS = {
    "bottom_right_burner": "knob_Joint_1",
    "bottom_left_burner": "knob_Joint_2",
    "top_right_burner": "knob_Joint_3",
    "top_left_burner": "knob_Joint_4",
}

#: Default execution order.  ``slide_cabinet`` comes first because its door blocks the
#: arm's path to the oven knobs and the light switch (measured: the light-switch skill
#: fails outright when the slide cabinet is still closed, and succeeds after it is open).
#: ``kettle`` is ahead of ``light_switch`` because the arm's path to the switch can knock
#: the reset kettle away from the handle-grasp corridor.  Once dragged into its goal, the
#: tuned light path leaves it inside the goal threshold.
DEFAULT_TASK_ORDER = [
    "slide_cabinet",
    "kettle",
    "light_switch",
    "microwave",
    "right_hinge_cabinet",
    "left_hinge_cabinet",
    "bottom_right_burner",
    "bottom_left_burner",
    "top_right_burner",
    "top_left_burner",
]


# ---------------------------------------------------------------------------------------
# FSM phases
# ---------------------------------------------------------------------------------------

ORIENT_FORWARD = "orient_forward"
SELECT_SUBTASK = "select_subtask"
MOVE_TO_PRECONTACT = "move_to_precontact"
ALIGN = "align"
APPROACH = "approach"
CONTACT_OR_GRASP = "contact_or_grasp"
MANIPULATE = "manipulate"
KETTLE_TRANSPORT = "kettle_transport"
RECEDE = "recede"
VERIFY = "verify"
RETREAT = "retreat"
IDLE = "idle"

#: The order the phases run in.  ALIGN is skipped by skills that do not need a particular
#: wrist yaw; VERIFY loops back to MOVE_TO_PRECONTACT on a bounded retry.
PHASE_SEQUENCE = (
    MOVE_TO_PRECONTACT,
    ALIGN,
    APPROACH,
    CONTACT_OR_GRASP,
    MANIPULATE,
    RECEDE,
    VERIFY,
    RETREAT,
)


# ---------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------

@dataclass
class KitchenPolicyConfig:
    """Tuning knobs for :class:`ScriptedKitchenPolicy`.

    Distances are metres, angles radians, timeouts are *environment steps*.  Step scales
    are in units of the env's own action space, i.e. a fraction of 0.2 m / 0.5 rad.
    """

    # -- planner -------------------------------------------------------------------------
    task_order: Optional[List[str]] = None
    #: Deterministic by default: a fixed order is far easier to debug, and the default
    #: order encodes a real precedence constraint (see ``DEFAULT_TASK_ORDER``).
    randomize_task_order: bool = True
    #: Re-derive the active task/phase from the live state on every query.  This is
    #: required because ``expert_action`` is also queried while an arbitrary learner
    #: action -- not the previous expert recommendation -- is being executed.
    reactive: bool = True
    #: The four burner tasks are already within ``BONUS_THRESH`` at reset.  Leave this
    #: True so the planner does not waste steps on them; set False to exercise the knob
    #: skill (e.g. when tuning it).
    skip_pre_completed_tasks: bool = True
    #: A task already observed complete is not reselected for tiny excursions around
    #: ``BONUS_THRESH``.  A larger regression still reactivates it, which matters when an
    #: arbitrary learner action disturbs a task that the expert completed earlier.
    reactivation_margin: float = 0.05

    #: Before selecting or translating toward any task, rotate at the reset/home position
    #: into the front-facing horizontal frame used by the slide-cabinet grasp.
    align_forward_at_reset: bool = True

    # -- step scaling --------------------------------------------------------------------
    #: 0.07 m per step in free space.  Measured: at the full 0.2 m the DLS controller
    #: regularly whips the wrist through a singularity in a single env step (the gripper
    #: ends up pointing sideways), which ruins the contact geometry for that skill and
    #: every skill after it.  Smaller free-space steps cost time but keep the posture.
    free_space_step: float = 0.35
    #: Gentle tracking once the tool is close to / touching an object (0.05 m per step);
    #: large steps into contact make the DLS controller fight the constraint.
    contact_step: float = 0.25
    #: Per-skill caps kept separate so the compact demonstration can move fixed joints
    #: quickly without changing the conservative control_steps=1 profile.
    light_approach_step: float = 0.10
    light_manipulation_step: float = 0.02
    microwave_precontact_step: float = 0.35
    microwave_approach_step: float = 0.25
    microwave_manipulation_step: float = 0.15
    microwave_contact_tolerance: float = 0.03
    #: Vertical adjustment to the reported microwave handle position.  The compact
    #: controller has a repeatable upward IK residual at contact; a small negative value
    #: keeps the fingertip pads on the vertical bar instead of pinching its upper end.
    microwave_contact_height_offset: float = 0.0
    #: Microwave-only angular hysteresis for leaving ALIGN.  Keep this below the grasp
    #: orientation tolerance: too small can strand reset-to-microwave transitions in
    #: ALIGN, while too large begins the handle approach before the jaws are square.
    microwave_alignment_exit_tolerance: float = 0.20
    #: After opening the microwave, fully release and withdraw this far along the live
    #: door normal before another randomized skill may begin.
    microwave_recede_distance: float = 0.18
    microwave_recede_tolerance: float = 0.03
    kettle_precontact_step: float = 0.20
    kettle_approach_step: float = 0.10
    #: Optional cap for the kettle's high collision-avoidance transit waypoint. ``None``
    #: uses ``free_space_step``; the final descent keeps ``kettle_precontact_step``.
    kettle_transit_step: Optional[float] = None
    kettle_orientation_tolerance: float = 0.15
    kettle_alignment_exit_tolerance: float = 0.08
    #: Slide-only rightward target lead, measured from the live handle.  This is an offset,
    #: not the total drawer travel: the target advances with the handle until the task's
    #: actual joint-goal predicate is satisfied.
    slide_manipulate_lookahead: float = 0.10
    #: Slide-only normalized Cartesian action cap.  With the environment's 0.2 m action
    #: scale, 0.25 requests at most 0.05 m of rightward EEF motion per policy action.
    slide_manipulation_step: float = 0.20
    #: Once the slide has reached its goal, open at the handle and back this far toward the
    #: room before beginning the normal between-task retreat.
    slide_recede_distance: float = 0.18
    #: After switching the light, release it completely and withdraw along the switch bar
    #: before selecting another subtask.
    light_recede_distance: float = 0.18
    light_recede_tolerance: float = 0.03
    #: Once the kettle is in its goal, fully release its handle and back away this far
    #: along the handle-normal approach line before selecting another subtask.
    kettle_recede_distance: float = 0.4
    #: Kettle-only residual tolerance for the fixed recede waypoint.  The loaded Franka
    #: IK can settle a couple of centimetres laterally from a Cartesian target even after
    #: making the requested backward clearance; using the generic 1.5 cm waypoint
    #: tolerance here can otherwise leave the controller in RECEDE forever.
    kettle_recede_tolerance: float = 0.03
    #: Optional normalized Cartesian cap for kettle retreat. This changes how quickly the
    #: fixed waypoint is reached; it does not shorten the required backward clearance.
    kettle_recede_step: Optional[float] = None
    #: 0.2 rad per step.  The env's IK weights the orientation error by 1/50 relative to
    #: position (``mju_quat2Vel(..., 50)``), so wrist alignment is slow; do not lower this.
    max_rotation_step: float = 0.4
    #: Anti-windup gain for the 1.2.0 controller's joint-actuator target accumulator.
    #: Each upstream DLS correction is augmented by
    #: ``gain * (measured_qpos - old_ctrl)``.  Zero preserves the buggy upstream behavior;
    #: one fully rebases every target and is too weak to maintain useful contact force.
    #: A sweep at ``control_steps=1`` found 0.1 bounded the worst per-joint lag below
    #: 0.8 rad while completing slide -> kettle -> light.
    #: The adapter is installed on the robot controller, so it stabilizes learner actions
    #: as well as expert actions.  Set this to 0 for exact upstream transition dynamics.
    ik_target_leak: float = 0.1
    #: Optional finer controller profile used only while the microwave skill is active.
    #: ``None`` keeps the environment's original control repetition and anti-windup gain.
    microwave_control_steps: Optional[int] = None
    microwave_ik_target_leak: Optional[float] = None
    #: Duration passed to MuJoCo when converting quaternion error to the angular velocity
    #: used by the one-step DLS solver.  Upstream hard-codes 50 seconds, which leaves a
    #: requested horizontal grasp more than a radian away after 100 actions.  A measured
    #: sweep found both 5 and 2 seconds reach the complete side-grasp frame without growing
    #: the actuator-target gap.  Two seconds reached it in 48 actions versus 56 for five,
    #: so the faster setting is the default.
    ik_orientation_duration: float = 2.0
    #: Relative least-squares weight on Cartesian position in the pose solver.  One keeps
    #: the upstream DLS objective; it remains configurable for controller experiments.
    ik_position_weight: float = 1.0
    #: Experimentally drive the gripper back to pointing straight down.  This is disabled
    #: by default because the one-release IK underweights orientation and the resulting
    #: actuator-target accumulation can tilt the wrist farther instead of recovering it.
    #: Contact with doors and handles tilts the wrist, and a tilted wrist both spoils the
    #: contact geometry and
    #: (without ``NOMINAL_APPROACH_AXIS``) would make the tool offset -- and therefore every
    #: target -- swing around.
    keep_gripper_down: bool = False
    #: Above this much tilt (radians) the free-space phases stop chasing their position
    #: target at full speed and let the IK spend its Jacobian on the orientation instead.
    #: The env's IK weights orientation at 1/50 of position, so a wrist that has been
    #: knocked over by contact will never right itself while the arm is also travelling --
    #: and an inverted gripper puts the fingertips *above* the EEF, which silently breaks
    #: every tool offset in this file.
    tilt_recovery_threshold: float = 0.35
    tilt_recovery_position_scale: float = 0.15
    #: Convert a tool point into an EEF target using the *measured* gripper axis
    #: (``data.site_xmat['EEF']``) instead of ``NOMINAL_APPROACH_AXIS``.
    #:
    #: Measured is the physically correct choice -- it is the only way the plan agrees with
    #: where the fingertips actually are -- but it couples the target to a wrist the IK
    #: barely controls, so a tilting wrist makes the target chase itself.  Whichever mode is
    #: selected, ``get_diagnostics()['tool_error']`` always reports the *measured*
    #: fingertip-to-tool-point distance, so the mismatch is never hidden.
    #:
    #: The measured frame is the default: nominal offsets caused the light policy to call
    #: a waypoint reached while its real fingertips were still about a tool length away.
    #: Diagnostics always report measured fingertip error in either mode.
    use_measured_tool_frame: bool = True

    # -- geometry ------------------------------------------------------------------------
    #: Stand-off from the contact point before approaching.  0.15 m clears the handles and
    #: is still inside one free-space step of the contact pose.
    precontact_distance: float = 0.15
    #: Residual stand-off held at the end of APPROACH.  0 = touch the target exactly.
    contact_distance: float = 0.0
    #: How far *past* the contact point the manipulation target is placed, i.e. how hard
    #: the skill leans into the object.  0.08 m keeps the commanded delta below one step.
    manipulate_lookahead: float = 0.08
    #: Radius within which a position target counts as reached.
    position_tolerance: float = 0.015
    #: Downward component of the otherwise switch-parallel light grasp.  This lets the
    #: fingertips reach under the cooker hood while the larger hand flange stays outside.
    light_approach_downward_pitch: float = 0.18
    #: Full tool-frame angular tolerance for ALIGN.
    yaw_tolerance: float = 0.15
    #: Retreat pose = contact point pushed back along the approach direction, plus a lift.
    #: Only used when ``retreat_to_home`` is False.
    retreat_distance: float = 0.18
    retreat_lift: float = 0.10
    #: Steps spent lifting straight up at the start of a retreat, before travelling home.
    retreat_lift_steps: int = 3
    #: Retreat all the way to the pose the arm started the episode in instead.  Skills
    #: leave the arm wherever the object ended up -- fully opening the slide cabinet drags
    #: it 0.35 m to the right with a twisted wrist -- and the next skill then starts from a
    #: configuration it was never tuned for.  Returning to a common home pose decouples the
    #: skills from each other and costs about ten steps.
    retreat_to_home: bool = True
    #: State-derived home radius used between subtasks by the reactive controller.
    reactive_retreat_tolerance: float = 0.12

    # -- gripper -------------------------------------------------------------------------
    #: +1 opens, -1 closes.  Verified by stepping the env and reading the finger joints.
    gripper_open: float = 1.0
    gripper_close: float = -1.0
    #: The gripper joint ranges from 0 (closed) to 0.04 m (open).  Recede motion must wait
    #: until it is within this threshold of fully open so a grasped handle is not dragged.
    release_opening_threshold: float = 0.0395
    #: Multiply actuator8's gain, position bias, and damping together.  Scaling all three
    #: preserves its 0--0.04 m position targets while raising the stock ~3 N handle pinch
    #: to roughly 60 N, within the physical Franka gripper's range.  Set to 1 for the
    #: unmodified Gymnasium-Robotics dynamics.
    gripper_stiffness_scale: float = 20.0

    # -- timing / robustness -------------------------------------------------------------
    #: Steps held in CONTACT_OR_GRASP so the gripper actuator settles before manipulating.
    engage_steps: int = 3
    #: Generic per-phase budget.
    phase_timeout: int = 25
    #: Hard cap on a *reaching* phase (MOVE_TO_PRECONTACT / APPROACH).  Reaches are not
    #: bounded by ``phase_timeout``: how many steps a reach needs depends on how far it has
    #: to travel and on ``control_steps``, which sets how much of a commanded displacement
    #: the IK actually achieves per step.  A reach ends when it arrives or when it stops
    #: making progress -- never merely because a fixed step count elapsed, which used to
    #: send the FSM into its contact sequence while the arm was still 0.7 m from the object.
    reach_timeout: int = 45
    #: A reach is "blocked" when its position error has stopped shrinking by more than this
    #: (metres) over ``progress_window`` steps.  Blocked is a *success* path: these skills
    #: routinely end their reach 10-15 cm short because the arm is pressed against the
    #: cabinetry around the target, jittering rather than cleanly stalling.  Only a reach
    #: that is still closing the gap when ``reach_timeout`` expires counts as failed.
    reach_progress_epsilon: float = 0.005
    #: Residual position error at which a timed-out reach counts as *failed* rather than
    #: as "close enough, carry on".  Several skills legitimately run out of budget still
    #: creeping toward a pre-contact pose that sits inside a collision volume (the light
    #: switch stops ~0.15 m short, its pre-contact being inside the cooker hood) and the
    #: contact phases still work from there.  Half a metre out is a different animal: that
    #: is the arm in mid-air, and running CONTACT_OR_GRASP there does nothing but waste the
    #: episode.  0.25 m sits between the two, about two tool lengths.
    reach_failure_tolerance: float = 0.25
    align_timeout: int = 70
    manipulate_timeout: int = 45
    retreat_timeout: int = 20
    #: Hard bound for release-and-recede.  Completion is normally geometric; this merely
    #: prevents an unreachable Cartesian retreat target from trapping the reactive FSM.
    recede_timeout: int = 30
    #: Bounded retries of the whole approach before the skill is abandoned.
    max_retries: int = 1
    #: Hard budget per subtask, across retries.  Without it a skill that cannot succeed
    #: eats the whole 280 step episode -- and a flailing arm can push
    #: an already-placed object back out of its goal region.
    task_step_budget: int = 190
    #: MANIPULATE keeps driving until the task distance is below
    #: ``completion_margin * BONUS_THRESH`` rather than stopping the instant the env would
    #: score the task.  Stopping exactly at the threshold leaves e.g. the slide cabinet
    #: barely cracked open, which scores but makes a poor demonstration.  Completion itself
    #: is always judged with the env's own threshold.
    completion_margin: float = 0.2
    #: The free kettle needs its own attainable buffer inside the environment's loose
    #: success boundary.  0.6 * BONUS_THRESH = 0.18 makes the side grasp push visibly
    #: farther forward without chasing the much tighter generic 0.06 m threshold.
    kettle_completion_margin: float = 0.6
    #: A phase is "stalled" when the EEF moves less than ``stall_distance`` over
    #: ``stall_window`` steps.  Stalling is *expected* here: this arm is frequently blocked
    #: by cabinetry short of its commanded pose, yet still in useful contact, so a stalled
    #: reaching phase is treated as arrived rather than as a failure.
    stall_window: int = 4
    stall_distance: float = 0.004
    #: MANIPULATE gives up when the manipulated joint has not progressed by this much over
    #: ``progress_window`` steps.
    progress_window: int = 12
    progress_epsilon: float = 5e-3

    # -- skill selection -----------------------------------------------------------------
    #: 'grasp' approaches the left handle and pushes it horizontally toward the goal;
    #: 'push' keeps the older hand-body fallback.
    kettle_strategy: str = "grasp"
    #: Normalized gripper command used to hold the kettle's 4.6 cm left handle bar. -0.25
    #: targets about 0.015 m half-opening; a captured bar physically stops the pads near
    #: 0.021 m, so this provides preload without the impact of a full-close command.
    kettle_grasp_command: float = -0.25
    #: Width-holding target after the microwave handle has been captured by both pads.
    #: This was -1.0, i.e. exactly ``gripper_close``, so the documented "hold at the
    #: measured width" was really a full-force squeeze: with ``gripper_stiffness_scale``
    #: at 20 it drives a smooth 4 cm bar straight back out from between the pads
    #: (measured: a good two-pad capture at 0.0153 m opening collapsed to 0.0000 m with
    #: no remaining contact in a single step).  -0.25 targets about 0.015 m, which a
    #: captured bar stops near 0.020 m, so the hold is preload rather than impact.
    microwave_grasp_command: float = -0.25
    #: How far *behind* the handle bar the leading fingertip should sit, in metres.
    #:
    #: The microwave is the one door here that was pulled by friction alone.  Its handle is
    #: a D-bar standing 0.082 m off the panel (bar radius 0.02, leaving 0.038 m of clear
    #: space behind it), and a smooth bar squeezed between two flat pads slides straight
    #: out as soon as the pull starts -- measured as a good bilateral capture at 0.0151 m
    #: opening followed by zero handle contact on the very next step.  Yawing the grasp
    #: frame about the bar's own axis swings the free-edge pad into that gap, so pulling
    #: loads the pad against the back of the bar instead of relying on grip force.  The
    #: rotation is about the handle axis, so local +x stays parallel to the bar and
    #: ``handle_roll_action``'s invariant is untouched.  Zero restores the square,
    #: friction-only grasp.
    #:
    #: Off by default and enabled in ``fast_demo``: the angle is derived from the *open*
    #: jaw span and was measured against that profile's insertion dynamics
    #: (``control_steps`` 15/8, approach step 0.70).  At the conservative profile's
    #: ``control_steps=5`` and 0.25 approach step the same yaw loses the handle, so the
    #: legacy path keeps its square grasp.
    microwave_hook_depth: float = 0.0
    #: The transport waypoint is rebuilt from the measured fingertip position each step,
    #: this far ahead along the planar direction to the grasped bar's goal point. Its
    #: height is unchanged.  The lead is additionally clamped to the *remaining* distance,
    #: which is what decelerates the loaded push instead of driving at the step cap until
    #: the stop predicate happens to fire.
    kettle_transport_lookahead: float = 0.08
    #: Kettle-only normalized Cartesian cap during loaded horizontal transport. Keeping it
    #: separate from contact_step avoids an abrupt acceleration as soon as the pads close.
    kettle_transport_step: float = 0.10
    #: Smallest rightward (world +x) component the loaded push direction may have, as a
    #: fraction of its forward component.
    #:
    #: The pads hold the bar on the kettle's *left*, so pushing along the raw
    #: kettle-to-goal line walks the body leftward -- toward the microwave, which sits at
    #: x = -0.64 against a kettle goal of x = -0.23.  Measured on seed 45: the body ran
    #: from x = -0.265 to -0.294 while the controller was already commanding +x and
    #: growing, because pushing an off-centre handle rightward mostly just yaws the body.
    #: Holding a minimum rightward lean in the *direction itself* steers the carry away
    #: from the microwave instead of waiting for the position error to win.  Zero merely
    #: forbids aiming left; larger values actively lead to the right.
    kettle_transport_min_rightward: float = 0.20
    #: Normalized world-z rotation cap applied while transporting a captured kettle.  The
    #: side grasp sits 0.092 m off the body's centre line, so dragging it spins the free
    #: body; this turns the wrist back toward the goal yaw.  Keep it small: the pads are
    #: ``FINGERTIP_OFFSET`` ahead of the EEF site the wrist rotates about, so every
    #: radian commanded here also drags the captured bar 0.108 m sideways.
    kettle_transport_yaw_step: float = 0.05
    #: Planar radius within which the kettle body counts as delivered.  Transport stops
    #: there rather than continuing to drive a body that is already at its goal position:
    #: the environment's 7-D distance is dominated by the yaw a side-grasped kettle picks
    #: up, so pushing past this point trades a little position for a lot of orientation.
    kettle_goal_position_tolerance: float = 0.02
    #: Residual kettle yaw (radians) accepted at the end of transport.  A 0.35 rad error
    #: contributes about 0.17 to the environment's 7-D pose distance, leaving room under
    #: ``BONUS_THRESH`` = 0.3 for the remaining position error.
    kettle_goal_yaw_tolerance: float = 0.35
    #: Height of the *EEF site* above the kettle root while pushing.  ``KettlePushSkill``
    #: has ``tool_offset = 0``, so this is the hand flange, and the fingers hang roughly
    #: ``FINGERTIP_OFFSET`` below it and sweep the kettle body (which spans 0.0-0.116 m
    #: above the root).  0.202 = 0.094 + FINGERTIP_OFFSET, i.e. the same contact height the
    #: fingertip-referenced version aimed for, but no longer dependent on the wrist angle.
    kettle_push_height: float = 0.202
    #: Planar stand-off from the kettle root when pushing (body half-width is 0.122 m).
    kettle_push_radius: float = 0.12
    #: How far past the burner-knob rotation axis the knob skill contacts the lever.
    knob_lever_arm: float = 0.04

    def resolved_task_order(self) -> List[str]:
        return list(self.task_order) if self.task_order is not None else list(DEFAULT_TASK_ORDER)

    @classmethod
    def fast_demo(cls, **overrides):
        """Return the measured sub-150-step four-task demonstration profile.

        Geometric success, bilateral-grasp, full-release, and retreat predicates are left
        unchanged. Only controller granularity, task motion caps, and the amount of extra
        goal overdrive are adjusted.
        """
        values = dict(
            free_space_step=0.40,
            ik_target_leak=0.4,
            microwave_control_steps=8,
            microwave_ik_target_leak=0.1,
            completion_margin=0.7,
            kettle_completion_margin=0.85,
            light_approach_step=0.45,
            light_manipulation_step=0.12,
            microwave_precontact_step=0.70,
            microwave_approach_step=0.70,
            microwave_manipulation_step=0.20,
            microwave_contact_tolerance=0.01,
            microwave_contact_height_offset=-0.020,
            microwave_alignment_exit_tolerance=0.23,
            kettle_precontact_step=0.35,
            kettle_approach_step=0.15,
            kettle_orientation_tolerance=0.18,
            kettle_alignment_exit_tolerance=0.10,
            # A full 0.4 loaded step can overshoot the goal after a collision-free grasp
            # (there is no accidental top-handle push providing a head start).
            kettle_transport_step=0.3,
            kettle_recede_step=0.35,
            microwave_hook_depth=0.020,
        )
        values.update(overrides)
        return cls(**values)


# ---------------------------------------------------------------------------------------
# Sim access -- the single place in this package that reaches into MuJoCo
# ---------------------------------------------------------------------------------------

class IKTargetLeakController:
    """Add anti-windup to Gymnasium-Robotics 1.2.0's existing DLS controller.

    The upstream robot applies every returned displacement as ``old_ctrl + dq``.  With one
    IK iteration per environment step the actuator cannot catch up, but the next
    correction is added anyway; after two skills the target can be many radians from the
    measured arm.  This adapter leaves the upstream Jacobian/DLS computation untouched and
    leaks only that accumulated actuator error back out of the next increment.
    """

    def __init__(self, controller, data, gain: float):
        self.controller = controller
        self.data = data
        self.gain = float(gain)

    def compute_qpos_delta(self, target_pos, target_quat):
        dq = np.asarray(
            self.controller.compute_qpos_delta(target_pos, target_quat),
            dtype=np.float64,
        )[:7]
        return dq + self.gain * (self.data.qpos[:7] - self.data.ctrl[:7])


class OrientationAwareIKController:
    """Upstream 1.2.0 DLS IK with a usable quaternion-error duration.

    Gymnasium-Robotics 1.2.0 calls ``mju_quat2Vel(..., 50)``.  That makes Cartesian
    orientation actions far too weak to turn the gripper sideways before contact.  The
    equations below otherwise match the upstream controller exactly; this does not alter
    the public 7-D action contract.
    """

    def __init__(self, model, data, duration: float, position_weight: float,
                 regularization_strength: float = 0.3):
        if duration <= 0.0:
            raise ValueError(f"ik_orientation_duration must be positive, got {duration}.")
        self.model = model
        self.data = data
        self.duration = float(duration)
        if position_weight <= 0.0:
            raise ValueError(f"ik_position_weight must be positive, got {position_weight}.")
        self.position_weight = float(position_weight)
        self.regularization_strength = float(regularization_strength)
        self.eef_id = model.site("EEF").id

    def compute_qpos_delta(self, target_pos, target_quat):
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        error = np.empty(6)
        error_pos, error_rot = error[:3], error[3:]
        eef_quat = np.empty(4)
        inverse_eef_quat = np.empty(4)
        error_quat = np.empty(4)

        error_pos[:] = np.asarray(target_pos) - self.data.site_xpos[self.eef_id]
        mujoco.mju_mat2Quat(eef_quat, self.data.site_xmat[self.eef_id])
        mujoco.mju_negQuat(inverse_eef_quat, eef_quat)
        mujoco.mju_mulQuat(error_quat, target_quat, inverse_eef_quat)
        mujoco.mju_quat2Vel(error_rot, error_quat, self.duration)
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.eef_id)
        weighted_error = error.copy()
        weighted_error[:3] *= self.position_weight
        jacobian = np.concatenate((jac_pos * self.position_weight, jac_rot), axis=0)
        hessian = jacobian.T.dot(jacobian)
        hessian += np.eye(hessian.shape[0]) * self.regularization_strength
        joint_delta = jacobian.T.dot(weighted_error)
        return np.linalg.lstsq(hessian, joint_delta, rcond=-1)[0]


class KitchenSim:
    """Named accessors for the ``KitchenEnv`` MuJoCo state.

    ``env`` may be any wrapper stack whose ``.unwrapped`` is the ``KitchenEnv``.  Keeping
    every ``unwrapped`` / ``model`` / ``data`` access behind this class is deliberate: the
    installed Franka Kitchen differs from the D4RL / relay-policy-learning versions and we
    want exactly one place to fix if it changes again.
    """

    def __init__(self, env):
        self._env = env

    # -- handles -------------------------------------------------------------------------
    @property
    def kitchen(self):
        """The ``gymnasium_robotics`` ``KitchenEnv`` (not the gym wrapper stack)."""
        return self._env.unwrapped

    @property
    def model(self):
        return self.kitchen.model

    @property
    def data(self):
        return self.kitchen.data

    @property
    def model_names(self):
        return self.kitchen.robot_env.model_names

    # -- sites / joints ------------------------------------------------------------------
    def site_xpos(self, name: str) -> np.ndarray:
        return get_site_xpos(self.model, self.data, name).copy()

    def site_xmat(self, name: str) -> np.ndarray:
        return get_site_xmat(self.model, self.data, name).copy()

    def body_xpos(self, name: str) -> np.ndarray:
        """World position of a named MuJoCo body."""
        body_id = self.model_names.body_name2id[name]
        return self.data.xpos[body_id].copy()

    def body_xmat(self, name: str) -> np.ndarray:
        """World rotation matrix of a named MuJoCo body."""
        body_id = self.model_names.body_name2id[name]
        return self.data.xmat[body_id].reshape(3, 3).copy()

    def joint_qpos(self, name: str) -> np.ndarray:
        return get_joint_qpos(self.model, self.data, name).copy()

    def joint_anchor(self, name: str) -> np.ndarray:
        """World-frame anchor point of a joint (``data.xanchor``)."""
        return self.data.xanchor[self.model_names.joint_name2id[name]].copy()

    def joint_axis(self, name: str) -> np.ndarray:
        """World-frame axis of a joint (``data.xaxis``)."""
        return self.data.xaxis[self.model_names.joint_name2id[name]].copy()

    def body_geom_axis(self, name: str) -> np.ndarray:
        """World local-``z`` axis of the first geom attached to a named body.

        MuJoCo capsules extend along local z.  This is used for the unnamed light-switch
        capsule, whose lever axis is different from its vertical hinge axis.
        """
        body_id = self.model_names.body_name2id[name]
        geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
        if not len(geom_ids):
            raise ValueError(f"Body {name!r} has no geom from which to measure an axis.")
        return self.data.geom_xmat[int(geom_ids[0])].reshape(3, 3)[:, 2].copy()

    # -- end effector --------------------------------------------------------------------
    @property
    def eef_pos(self) -> np.ndarray:
        return self.site_xpos("EEF")

    @property
    def eef_mat(self) -> np.ndarray:
        return self.site_xmat("EEF")

    @property
    def eef_approach_axis(self) -> np.ndarray:
        """Unit vector the fingers point along (EEF local +z); ~(0, 0, -1) at reset."""
        return self.eef_mat[:, 2]

    @property
    def eef_finger_axis(self) -> np.ndarray:
        """Unit vector the fingers separate along (EEF local +y)."""
        return self.eef_mat[:, 1]

    @property
    def fingertip_pos(self) -> np.ndarray:
        return self.eef_pos + FINGERTIP_OFFSET * self.eef_approach_axis

    @property
    def finger_yaw(self) -> float:
        """Heading of the finger separation axis in the world xy plane."""
        axis = self.eef_finger_axis
        return float(np.arctan2(axis[1], axis[0]))

    @property
    def finger_opening(self) -> float:
        """Half-opening of the gripper in metres (0 closed, 0.04 open)."""
        return float(self.joint_qpos("robot:finger_joint1")[0])

    def validate_action_contract(self) -> None:
        """Reject Franka Kitchen versions whose actions are not Cartesian deltas.

        Gymnasium-Robotics 1.2.0 is the only release with the 7-D Cartesian/IK contract
        used by this policy.  From 1.2.1 onward Franka Kitchen again uses nine joint
        velocities.  Both are normalized boxes, so checking only bounds would miss a
        catastrophic semantic mismatch.
        """
        robot = self.kitchen.robot_env
        if robot.action_space.shape != (7,) or getattr(robot, "controller", None) is None:
            raise RuntimeError(
                "ScriptedKitchenPolicy requires gymnasium-robotics==1.2.0's 7-D "
                "Cartesian IK action space [dx, dy, dz, drx, dry, drz, gripper]; "
                f"this robot exposes shape {robot.action_space.shape}. Gymnasium-"
                "Robotics >=1.2.1 uses nine joint velocities and needs a separate "
                "joint-space policy adapter.")

    def install_orientation_controller(self, duration: float,
                                       position_weight: float) -> None:
        """Replace only the upstream controller's unusably slow rotation weighting."""
        robot = self.kitchen.robot_env
        controller = robot.controller
        if isinstance(controller, IKTargetLeakController):
            controller = controller.controller
        regularization = float(getattr(controller, "regularization_strength", 0.3))
        robot.controller = OrientationAwareIKController(
            self.model,
            self.data,
            duration,
            position_weight,
            regularization_strength=regularization,
        )

    def install_ik_target_leak(self, gain: float) -> None:
        if not 0.0 <= gain <= 1.0:
            raise ValueError(f"ik_target_leak must be in [0, 1], got {gain}.")
        robot = self.kitchen.robot_env
        if isinstance(robot.controller, IKTargetLeakController):
            robot.controller.gain = float(gain)
        elif gain != 0.0:
            robot.controller = IKTargetLeakController(robot.controller, self.data, gain)

    def scale_gripper_stiffness(self, scale: float) -> None:
        """Strengthen the position servo without changing its open/closed equilibria."""
        if scale <= 0.0:
            raise ValueError(f"gripper_stiffness_scale must be positive, got {scale}.")
        if scale == 1.0:
            return
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if actuator_id < 0:
            raise RuntimeError("Franka gripper actuator 'actuator8' is missing.")
        self.model.actuator_gainprm[actuator_id, 0] *= scale
        self.model.actuator_biasprm[actuator_id, 1:3] *= scale

    # -- tasks ---------------------------------------------------------------------------
    @property
    def goal(self) -> Dict[str, np.ndarray]:
        return self.kitchen.goal

    def task_distance(self, task: str) -> float:
        """``||achieved_goal[task] - desired_goal[task]||`` read from the *sim*, not from
        the (noisy) observation."""
        return float(np.linalg.norm(self.joint_qpos(task) - OBS_ELEMENT_GOALS[task]))

    def task_complete(self, task: str) -> bool:
        return self.task_distance(task) < BONUS_THRESH


# ---------------------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------------------

def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def wrap_to_half_pi(angle: float) -> float:
    """Wrap to (-pi/2, pi/2]; the finger axis is symmetric under a 180 degree flip."""
    return float((angle + np.pi / 2) % np.pi - np.pi / 2)


def limit_to_box(vec: np.ndarray, limit: float) -> np.ndarray:
    """Scale ``vec`` down until every component is within ``limit``.

    Clipping componentwise would keep the vector inside the action box but change its
    direction, which turns a straight reach into a dogleg; scaling keeps the direction.
    """
    vec = np.asarray(vec, dtype=np.float64)
    largest = float(np.max(np.abs(vec))) if vec.size else 0.0
    if largest > limit > 0.0:
        vec = vec * (limit / largest)
    return vec


def position_action(sim: KitchenSim, target_pos: np.ndarray, step_scale: float) -> np.ndarray:
    """Normalized Cartesian delta driving the EEF toward ``target_pos``."""
    delta = np.asarray(target_pos, dtype=np.float64) - sim.eef_pos
    return limit_to_box(delta / MAX_CARTESIAN_DISPLACEMENT, float(step_scale))


def yaw_action(sim: KitchenSim, desired_yaw: float, max_rotation_step: float) -> float:
    """Normalized world-z rotation delta driving the finger axis to ``desired_yaw``."""
    err = wrap_to_half_pi(desired_yaw - sim.finger_yaw)
    return float(np.clip(err / MAX_ROTATION_DISPLACEMENT, -max_rotation_step, max_rotation_step))


def unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(3)
    return vec / norm


def grasp_frame(approach_axis: np.ndarray, handle_axis: np.ndarray) -> np.ndarray:
    """Build a horizontal fingertip-grasp frame in world coordinates.

    EEF local ``+z`` runs from the palm toward the fingertips and is aligned with the
    side approach.  Local ``+x`` follows the handle, while local ``+y`` is consequently
    the jaw-separation direction across it.  Projecting the handle axis removes small
    non-orthogonal components from live MuJoCo geometry before constructing the frame.
    """
    z_axis = unit(approach_axis)
    x_axis = np.asarray(handle_axis, dtype=np.float64)
    x_axis = unit(x_axis - np.dot(x_axis, z_axis) * z_axis)
    if np.linalg.norm(z_axis) < 1e-9 or np.linalg.norm(x_axis) < 1e-9:
        raise ValueError("A grasp frame needs nonzero, nonparallel approach and handle axes.")
    y_axis = unit(np.cross(z_axis, x_axis))
    x_axis = unit(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def rotate_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for ``angle`` radians about a unit ``axis``."""
    axis = unit(axis)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (np.eye(3) + np.sin(angle) * cross
            + (1.0 - np.cos(angle)) * cross.dot(cross))


def orientation_error(sim: KitchenSim, desired: np.ndarray) -> float:
    """Shortest rotation angle from the live EEF frame to ``desired``."""
    relative = np.asarray(desired, dtype=np.float64) @ sim.eef_mat.T
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rotation_action(sim: KitchenSim, desired: np.ndarray,
                    max_rotation_step: float) -> np.ndarray:
    """Normalized world-frame rotation command toward a complete EEF frame.

    FrankaRobot left-multiplies ``euler2quat(action[3:6] * 0.5)`` onto the live
    quaternion.  For the small bounded increments used here, the shortest axis-angle
    vector is the stable local representation of that same world-frame correction.  The
    target is recomputed every step, so the small Euler/rotation-vector difference cannot
    accumulate.
    """
    desired = np.asarray(desired, dtype=np.float64)
    desired_quat = np.empty(4)
    current_quat = np.empty(4)
    inverse_current = np.empty(4)
    error_quat = np.empty(4)
    rotation_vector = np.empty(3)
    mujoco.mju_mat2Quat(desired_quat, desired.reshape(-1))
    mujoco.mju_mat2Quat(current_quat, sim.eef_mat.reshape(-1))
    mujoco.mju_negQuat(inverse_current, current_quat)
    mujoco.mju_mulQuat(error_quat, desired_quat, inverse_current)
    mujoco.mju_quat2Vel(rotation_vector, error_quat, 1.0)
    return limit_to_box(
        rotation_vector / MAX_ROTATION_DISPLACEMENT,
        float(max_rotation_step),
    )


# ---------------------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------------------

class KitchenSkill:
    """Base class for one task's manipulation strategy.

    A skill only has to answer geometric questions -- where to stand off, where to touch,
    which way to drive the object -- and the FSM in :class:`ScriptedKitchenPolicy` turns
    those into actions.  Every quantity is recomputed from the live sim state, so the
    targets follow doors and objects as they move.
    """

    #: Gripper command while reaching / approaching, and while manipulating.
    approach_gripper = "close"
    engage_gripper = "close"
    #: Distance from the EEF site to the point that should touch the object.  Skills that
    #: push with the fingertips use ``FINGERTIP_OFFSET``; skills that push with the hand
    #: body itself use 0.
    tool_offset = FINGERTIP_OFFSET
    #: State-derived phase thresholds.  They measure the real tool point, not merely the
    #: commanded EEF target, so a tilted wrist cannot claim contact while the fingers are
    #: still in free space.
    precontact_tolerance = 0.06
    contact_tolerance = 0.05
    lost_contact_tolerance = 0.18
    #: Maximum stand-off error from which wrist-only alignment is safe.  This is broader
    #: than ``precontact_tolerance`` because rotating a long side-on tool frame moves the
    #: EEF even when the fingertip target is held fixed.
    alignment_tolerance = 0.18
    #: A real grasp stops at the object's radius rather than zero finger qpos.  The largest
    #: handle actually straddled here is the kettle's 0.023 m-radius left bar. The centered
    #: microwave handle grasp settles near 0.019--0.020 m; only the unconstrained 0.040 m
    #: state is still open.
    grasp_ready_opening = 0.039
    #: Optional lower bound for detecting that a thin feature has slipped completely out.
    #: Zero disables reacquisition for skills whose manipulation can continue by contact.
    grasp_contact_min_opening = 0.0
    #: Some collision corridors require wrist rotation before any object-relative reach.
    align_before_reach = False
    #: Align at the current EEF position instead of translating back to the stand-off.
    #: Used where cabinetry makes the latter target unreachable during a large rotation.
    align_in_place = False
    #: Optional per-skill angular tolerance. This prevents one difficult grasp from
    #: weakening the alignment required by every other kitchen skill.
    orientation_tolerance = None
    #: Optional lower threshold for leaving ALIGN, providing angular hysteresis.
    alignment_exit_tolerance = None
    #: Optional per-skill normalized rotation cap.
    rotation_step = None
    #: Optional per-skill caps for the final stand-off reach and contact approach.
    precontact_step = None
    approach_step = None
    transit_tolerance = 0.12
    manipulation_step = None
    recede_after_manipulation = False
    #: Most grasps must open in place before translating so they do not undo the task.
    #: A thin feature can opt into opening while withdrawing when holding it in place
    #: wedges it between the fingertip pads.
    release_while_receding = False
    #: Hold the measured EEF pose while the pads open, instead of tracking the object's
    #: live contact point.  Tracking is right for a hinged door, whose handle can only
    #: move along an arc the skill already models, but for a *free* body the contact
    #: point moves with whatever the closed pads are still doing to it, so chasing it is
    #: a feedback loop that drags the object.
    hold_pose_while_releasing = False
    orient_during_recede = True

    def __init__(self, sim: KitchenSim, task: str, cfg: KitchenPolicyConfig):
        self.sim = sim
        self.task = task
        self.cfg = cfg
        self.site = TASK_SITES[task]
        self.manipulated_joint = TASK_MANIPULATED_JOINTS.get(task, task)

    # -- geometry ------------------------------------------------------------------------
    def touch_point(self) -> np.ndarray:
        """World point the tool should be at when it is engaged with the object."""
        return self.sim.site_xpos(self.site)

    def drive_direction(self) -> np.ndarray:
        """Unit world direction the object has to be driven in, right now."""
        raise NotImplementedError

    def handle_axis(self) -> Optional[np.ndarray]:
        """World axis of the feature to straddle, or None for an unoriented push."""
        return None

    def desired_orientation(self) -> Optional[np.ndarray]:
        """Complete EEF frame for a side grasp, or None to leave orientation alone."""
        handle_axis = self.handle_axis()
        if handle_axis is None:
            return None
        return grasp_frame(self.approach_axis(), handle_axis)

    def precontact_point(self) -> np.ndarray:
        return self.touch_point() - self.drive_direction() * self.cfg.precontact_distance

    def alignment_point(self) -> np.ndarray:
        """Tool waypoint at which the wrist may safely rotate into its grasp frame."""
        return self.precontact_point()

    def transit_point(self) -> Optional[np.ndarray]:
        """Optional collision-avoidance waypoint used before the object-relative reach."""
        return None

    def transit_step_scale(self) -> float:
        return float(self.cfg.free_space_step)

    def transit_reached(self, point: np.ndarray) -> bool:
        """Whether a collision-avoidance waypoint has been reached once."""
        return bool(np.linalg.norm(
            (self.tool_pos() - np.asarray(point, dtype=np.float64))[:2]
        ) <= self.transit_tolerance)

    def contact_point(self) -> np.ndarray:
        return self.touch_point() - self.drive_direction() * self.cfg.contact_distance

    def manipulate_point(self) -> np.ndarray:
        return self.touch_point() + self.drive_direction() * self.cfg.manipulate_lookahead

    def retreat_point(self) -> np.ndarray:
        point = self.touch_point() - self.drive_direction() * self.cfg.retreat_distance
        point[2] += self.cfg.retreat_lift
        return point

    def recede_point(self) -> np.ndarray:
        """Post-release waypoint, opposite the grasp approach direction."""
        return self.touch_point() - self.approach_axis() * self.cfg.slide_recede_distance

    def recede_complete(self) -> bool:
        """Whether the tool has reached its post-release clearance waypoint."""
        return bool(np.linalg.norm(self.tool_pos() - self.recede_point())
                    <= self.cfg.position_tolerance)

    def recede_step_scale(self) -> float:
        """Action-space position cap while moving to the release-clearance waypoint."""
        return float(self.cfg.contact_step)

    # -- tool / EEF conversion -----------------------------------------------------------
    def approach_axis(self) -> np.ndarray:
        """Gripper approach direction (EEF local +z, in world) this skill wants to hold."""
        return NOMINAL_APPROACH_AXIS

    def tool_frame_axis(self) -> np.ndarray:
        """The axis used to place the tool point, measured or nominal per config."""
        if self.cfg.use_measured_tool_frame:
            return self.sim.eef_approach_axis
        return self.approach_axis()

    def tool_pos(self) -> np.ndarray:
        """Where the tool point *actually is*, straight from the sim."""
        return self.sim.eef_pos + self.tool_offset * self.sim.eef_approach_axis

    def manipulation_step_scale(self) -> float:
        """Action-space position cap for object motion, optionally overridden per skill."""
        return (self.cfg.contact_step if self.manipulation_step is None
                else float(self.manipulation_step))

    def orientation_tolerance_value(self) -> float:
        return (self.cfg.yaw_tolerance if self.orientation_tolerance is None
                else float(self.orientation_tolerance))

    def rotation_step_scale(self) -> float:
        return (self.cfg.max_rotation_step if self.rotation_step is None
                else float(self.rotation_step))

    def alignment_exit_tolerance_value(self) -> float:
        return (self.orientation_tolerance_value()
                if self.alignment_exit_tolerance is None
                else float(self.alignment_exit_tolerance))

    def precontact_step_scale(self) -> float:
        return (self.cfg.free_space_step if self.precontact_step is None
                else float(self.precontact_step))

    def approach_step_scale(self) -> float:
        return (self.cfg.contact_step if self.approach_step is None
                else float(self.approach_step))

    def gripper_command(self, mode: str) -> float:
        """Resolve an open/close mode into this skill's actuator command."""
        return (self.cfg.gripper_open if mode == "open" else self.cfg.gripper_close)

    def eef_target(self, tool_point: np.ndarray) -> np.ndarray:
        """Where the EEF site must be so that the tool point lands on ``tool_point``."""
        return np.asarray(tool_point, dtype=np.float64) - self.tool_offset * self.tool_frame_axis()

    # -- progress / completion -----------------------------------------------------------
    def progress(self) -> float:
        """Scalar that must keep changing while MANIPULATE is doing something useful."""
        return float(self.sim.joint_qpos(self.manipulated_joint)[0])

    def complete(self) -> bool:
        """The env's own completion predicate for this task."""
        return self.sim.task_complete(self.task)

    #: Per-skill override of ``KitchenPolicyConfig.completion_margin``; None uses the config.
    margin_override = None

    def manipulation_done(self) -> bool:
        """When MANIPULATE may stop: comfortably inside ``BONUS_THRESH``, not just past it."""
        margin = self.cfg.completion_margin if self.margin_override is None else self.margin_override
        return self.sim.task_distance(self.task) < BONUS_THRESH * margin

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.task})"


class HingeSkill(KitchenSkill):
    """Drive a hinge joint by following the arc traced by its handle site.

    The tangent is recomputed from the *live* anchor / site every step, so the target
    follows the door as it swings.  ``hook_depth`` moves the contact point past the handle
    (into the gap between handle and door) for skills that have to *pull* rather than push.
    """

    hook_depth = 0.0
    #: Extra offset of the contact point away from the site, in the world frame.  Used by
    #: the knob skill to grab the lever instead of the (on-axis) site.
    lever_offset = np.zeros(3)

    def _hinge_joint(self) -> str:
        return self.manipulated_joint

    def _goal_qpos(self) -> float:
        return float(OBS_ELEMENT_GOALS[self.task][0]) if self.task in OBS_ELEMENT_GOALS else 0.0

    def drive_direction(self) -> np.ndarray:
        joint = self._hinge_joint()
        anchor = self.sim.joint_anchor(joint)
        axis = self.sim.joint_axis(joint)
        radius = self.touch_point() - anchor
        tangent = unit(np.cross(axis, radius))
        sign = np.sign(self._goal_qpos() - float(self.sim.joint_qpos(joint)[0]))
        if sign == 0:
            sign = -1.0  # both burner knobs and every door in this scene close at qpos 0
        return tangent * sign

    def touch_point(self) -> np.ndarray:
        return self.sim.site_xpos(self.site) + self.lever_offset

    def contact_point(self) -> np.ndarray:
        # Hooking pulls the tool *past* the handle, i.e. opposite the drive direction.
        return self.touch_point() - self.drive_direction() * (self.cfg.contact_distance - self.hook_depth)


class SlideSkill(KitchenSkill):
    """Drive a prismatic joint (the slide cabinet) along its own axis.

    The vertical handle is approached from the room/front side with a horizontal gripper.
    Its three relevant axes are mutually orthogonal: handle = world +z, slide = world +x,
    and palm-to-fingertips approach = world +y.  The jaws close across the bar along the
    slide axis, then translate it toward the goal (right in the reset scene).
    """

    tool_offset = FINGERTIP_OFFSET
    approach_gripper = "open"
    engage_gripper = "close"
    precontact_tolerance = 0.10
    contact_tolerance = 0.05
    recede_after_manipulation = True

    def drive_direction(self) -> np.ndarray:
        axis = self.sim.joint_axis(self.task)
        goal = float(OBS_ELEMENT_GOALS[self.task][0])
        sign = np.sign(goal - float(self.sim.joint_qpos(self.task)[0]))
        if sign == 0:
            sign = 1.0
        return unit(axis) * sign

    def approach_axis(self) -> np.ndarray:
        # handle x drive is the only axis orthogonal to both.  Keep its sign facing from
        # the room toward the cabinet even if a disturbed drawer has to be driven left.
        normal = unit(np.cross(self.handle_axis(), self.drive_direction()))
        if np.dot(normal, FRONT_APPROACH_AXIS) < 0.0:
            normal = -normal
        return normal

    def handle_axis(self) -> np.ndarray:
        return VERTICAL_HANDLE_AXIS

    def precontact_point(self) -> np.ndarray:
        return self.touch_point() - self.approach_axis() * self.cfg.precontact_distance

    def contact_point(self) -> np.ndarray:
        return self.touch_point() - self.approach_axis() * self.cfg.contact_distance

    def manipulate_point(self) -> np.ndarray:
        return (self.touch_point()
                + self.drive_direction() * self.cfg.slide_manipulate_lookahead)

    def manipulation_step_scale(self) -> float:
        return float(self.cfg.slide_manipulation_step)


class LightSwitchSkill(HingeSkill):
    """Grip parallel to the switch, slide it right-to-left, release, and recede."""

    # This end-on insertion uses the distal pad centre, about 15 cm from the EEF flange.
    # The generic 10.8 cm reference is appropriate for crosswise handle grasps but puts
    # the hand body into the cooker hood before these parallel fingertips reach the bar.
    tool_offset = 0.15
    approach_gripper = "open"
    engage_gripper = "close"
    # The cooker hood makes the generic 1.5 cm waypoint tolerance impractical, but 18 cm
    # (an earlier value) classified free space as contact and skipped the actual poke.
    precontact_tolerance = 0.04
    contact_tolerance = 0.035
    lost_contact_tolerance = 0.14
    align_in_place = True
    alignment_tolerance = 0.30
    orientation_tolerance = 0.20
    def approach_step_scale(self) -> float:
        return float(self.cfg.light_approach_step)

    def manipulation_step_scale(self) -> float:
        return float(self.cfg.light_manipulation_step)
    grasp_contact_min_opening = 0.012
    recede_after_manipulation = True
    release_while_receding = True
    orient_during_recede = False

    def __init__(self, sim, task, cfg):
        super().__init__(sim, task, cfg)
        # Freeze the normal-task grasp frame before contact.  Chasing the lever's rotating
        # live axis while manipulating creates a feedback loop: contact turns the switch,
        # which asks the wrist to rotate again and can pry the capsule out of the jaws.
        self._grasp_axis = self._measure_switch_axis()
        self._capture_confirmed = False
        self._recede_origin = None
        self._recede_backward_axis = None
        self._fixed_recede_point = None
        switch_body_id = self.sim.model_names.body_name2id["lightswitchroot"]
        self._switch_geoms = set(map(int, np.flatnonzero(
            (self.sim.model.geom_bodyid == switch_body_id)
            & ((self.sim.model.geom_contype != 0)
               | (self.sim.model.geom_conaffinity != 0))
        )))
        self._finger_geoms = {}
        for finger_name in ("left_finger", "right_finger"):
            finger_body_id = self.sim.model_names.body_name2id[finger_name]
            for geom_id in np.flatnonzero(
                    (self.sim.model.geom_bodyid == finger_body_id)
                    & (self.sim.model.geom_contype != 0)):
                self._finger_geoms[int(geom_id)] = finger_name
        if not self._switch_geoms or not self._finger_geoms:
            raise ValueError("Could not resolve light-switch and fingertip collision geoms.")

    def _measure_switch_axis(self) -> np.ndarray:
        """Lever axis directed from its exposed tip toward its cabinet-side pivot."""
        toward_pivot = self.sim.joint_anchor(self._hinge_joint()) - self.touch_point()
        axis = unit(toward_pivot)
        if np.linalg.norm(axis) < 1e-9:
            axis = unit(self.sim.body_geom_axis("lightswitchroot"))
        return axis

    def switch_axis(self) -> np.ndarray:
        return self._grasp_axis.copy()

    def contacting_fingers(self) -> set:
        """Fingers currently contacting the light-switch collision geometry."""
        fingers = set()
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            for finger_geom, switch_geom in (
                    (int(contact.geom1), int(contact.geom2)),
                    (int(contact.geom2), int(contact.geom1))):
                if (finger_geom in self._finger_geoms
                        and switch_geom in self._switch_geoms):
                    fingers.add(self._finger_geoms[finger_geom])
        return fingers

    def grasp_retained(self) -> bool:
        """Require one observed bilateral capture; width alone is ambiguous here."""
        fingers = self.contacting_fingers()
        if self.sim.finger_opening >= self.grasp_ready_opening:
            self._capture_confirmed = False
        if len(fingers) == 2:
            self._capture_confirmed = True
        return bool(self._capture_confirmed and fingers)

    def approach_axis(self) -> np.ndarray:
        # Point the horizontal fingertips along the lever, from its exposed end toward
        # the pivot.  The previous -drive tangent was orthogonal to the switch and forced
        # an unnecessary 90-degree wrist roll before contact.  A small downward pitch
        # clears the hood without changing the switch-parallel planar heading.
        pitch = self.cfg.light_approach_downward_pitch
        return unit(self.switch_axis() + np.array([0.0, 0.0, -pitch]))

    def handle_axis(self) -> np.ndarray:
        """Physical long axis of the switch, exposed for geometry diagnostics."""
        return self.sim.body_geom_axis("lightswitchroot")

    def desired_orientation(self) -> np.ndarray:
        # Preserve the same vertical roll used by the slide-cabinet side grasp while the
        # palm-to-fingertip axis follows the horizontal switch bar.
        return grasp_frame(self.approach_axis(), VERTICAL_HANDLE_AXIS)

    def precontact_point(self) -> np.ndarray:
        return self.touch_point() - self.approach_axis() * self.cfg.precontact_distance

    def contact_point(self) -> np.ndarray:
        # Put the capsule inside the pad length rather than pinching it at the distal edge.
        return self.touch_point() + self.approach_axis() * 0.02

    def manipulate_point(self) -> np.ndarray:
        # The hinge tangent points right-to-left toward the light-on joint goal. Track a
        # short live lead without yanking the capsule out of the fingertip pads.
        return self.touch_point() + self.drive_direction() * 0.06

    def recede_point(self) -> np.ndarray:
        """Freeze a post-release waypoint straight back along the switch bar."""
        if self._fixed_recede_point is None:
            self._recede_origin = self.touch_point()
            self._recede_backward_axis = self.approach_axis()
            self._fixed_recede_point = (
                self._recede_origin
                - self._recede_backward_axis * self.cfg.light_recede_distance
            )
        return self._fixed_recede_point.copy()

    def recede_complete(self) -> bool:
        target = self.recede_point()
        tool_pos = self.tool_pos()
        backward_progress = float(np.dot(
            self._recede_origin - tool_pos, self._recede_backward_axis))
        return bool(
            np.linalg.norm(tool_pos - target) <= self.cfg.light_recede_tolerance
            or backward_progress >= (
                self.cfg.light_recede_distance - self.cfg.position_tolerance)
        )


class KnobSkill(HingeSkill):
    """Turn an oven knob.

    The knob's rotation axis is horizontal (world -y) and ``knobN_site`` sits *on* that
    axis, so the site itself does not move when the knob turns.  The skill therefore
    contacts the knob's lever ``knob_lever_arm`` above the axis and sweeps it sideways.

    The env scores ``*_burner``, which is equality-coupled to ``knob_Joint_N`` but is
    already within ``BONUS_THRESH`` of its goal at reset.  The physical skill remains
    useful for forced-policy diagnostics; normal reward rollouts skip it.
    """

    tool_offset = FINGERTIP_OFFSET

    def __init__(self, sim, task, cfg):
        super().__init__(sim, task, cfg)
        self.lever_offset = np.array([0.0, 0.0, cfg.knob_lever_arm])

    def _goal_qpos(self) -> float:
        # Drive the knob to the end of its own range; the burner goal says nothing about it.
        joint_id = self.sim.model_names.joint_name2id[self.manipulated_joint]
        return float(self.sim.model.jnt_range[joint_id][0])

    def complete(self) -> bool:
        # Report the env's own predicate so the planner and the env agree.
        return self.sim.task_complete(self.task)

    def manipulation_done(self) -> bool:
        # The burner task distance is a constant 0.01, so it says nothing about the knob.
        # Judge the manipulation on the knob joint the skill actually drives.
        turned = float(self.sim.joint_qpos(self.manipulated_joint)[0])
        return abs(turned - self._goal_qpos()) < 0.15


class HandlePullSkill(HingeSkill):
    """Open a door that swings *toward* the robot (microwave, both hinge cabinets).

    These are approached from the room side with a horizontal gripper, grasped at the
    fingertips, and pulled along the live hinge tangent.  The complete tool frame follows
    the moving door: fingertips point inward, jaws separate radially, and local x remains
    parallel to the vertical handle.
    """

    tool_offset = FINGERTIP_OFFSET
    # The connector/door collision geometry blocks the nominal site centre.  A small
    # radial bias clears that connector, while insertion places the bar between the pads
    # before they close.
    hook_depth = 0.04
    radial_bias = 0.02
    approach_gripper = "open"
    engage_gripper = "close"
    # The controller can displace the EEF by ~7.5 cm while rotating into the horizontal
    # frame.  Keep that state inside the approach corridor so the reactive policy commits
    # to the handle instead of alternating between stand-off and contact targets.
    precontact_tolerance = 0.10
    contact_tolerance = 0.04
    manipulation_step = 0.05

    def approach_axis(self) -> np.ndarray:
        return -self.drive_direction()

    def handle_axis(self) -> np.ndarray:
        return self.sim.joint_axis(self._hinge_joint())

    def touch_point(self) -> np.ndarray:
        site = self.sim.site_xpos(self.site)
        radial = unit(site - self.sim.joint_anchor(self._hinge_joint()))
        return site + radial * self.radial_bias

    def contact_point(self) -> np.ndarray:
        # Move through the handle far enough that its bar lies along the fingertip pads,
        # then close the jaws across it.
        return self.touch_point() - self.drive_direction() * self.hook_depth

    def manipulate_point(self) -> np.ndarray:
        # A gentle controller step tracks this live tangent lead without outrunning the
        # pinch; the target itself must remain far enough ahead to keep the hinge moving.
        return self.touch_point() + self.drive_direction() * 0.05

    def precontact_point(self) -> np.ndarray:
        # Pulling doors must be approached from the side they open toward.  The generic
        # KitchenSkill formula uses ``touch - drive * standoff`` and is correct for pushes
        # but places this precontact point behind the closed door.
        return self.touch_point() + self.drive_direction() * self.cfg.precontact_distance


class MicrowavePullSkill(HandlePullSkill):
    """Grip the microwave's vertical handle from the room-facing side.

    The gripper points along the door normal, with its jaws still orthogonal to the vertical
    bar.  The fingertip midpoint is driven to the reported handle x/y with a configurable
    height correction, the jaws close around the bar there, and pulling begins only after
    both pads have touched it.
    """

    orientation_tolerance = 0.35
    alignment_exit_tolerance = 0.20
    alignment_tolerance = 0.25
    precontact_tolerance = 0.09
    contact_tolerance = 0.035
    engage_gripper = "close"
    recede_after_manipulation = True
    # Hold the grasp frame while backing straight away from the open door. Reorienting
    # during this narrow exit is what made the hand sweep back through the door in
    # microwave-first randomized orders.
    orient_during_recede = False
    # The 4 cm handle settles the finger joint near 0.019--0.020 m under load. Keep a
    # margin below that physical width so normal compression does not trigger a release.
    grasp_contact_min_opening = 0.015
    manipulation_lookahead = 0.08
    # HandlePullSkill uses 2 cm to clear the bulky cabinet-door connector.  The microwave
    # inherits that value unless it is overridden, but here it shifts the grasp visibly to
    # the right of the much thinner vertical bar.  A contact sweep found 5 mm is the
    # smallest offset that clears the door while allowing both fingers to meet the handle.
    radial_bias = 0.005
    #: Lateral distance from the tool centreline to each pad with the jaws open; the
    #: gripper's finger joint runs 0 (closed) to 0.04 m (open).  Converts a requested hook
    #: depth into the yaw that achieves it.
    JAW_HALF_SPAN = 0.04

    def __init__(self, sim, task, cfg):
        super().__init__(sim, task, cfg)
        self.contact_tolerance = float(cfg.microwave_contact_tolerance)
        self._capture_confirmed = False
        self._capture_offset = None
        self._grasp_reacquire_pending = False
        self._fixed_recede_point = None
        self._recede_origin = None
        self._recede_backward_axis = None
        self._exit_clearance_reached = False
        door_body_id = self.sim.model_names.body_name2id["microdoorroot"]
        self._handle_geoms = set(map(int, np.flatnonzero(
            (self.sim.model.geom_bodyid == door_body_id)
            & (self.sim.model.geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE))
            & ((self.sim.model.geom_contype != 0)
               | (self.sim.model.geom_conaffinity != 0))
        )))
        self._finger_geoms = {}
        for finger_name in ("left_finger", "right_finger"):
            finger_body_id = self.sim.model_names.body_name2id[finger_name]
            for geom_id in np.flatnonzero(
                    (self.sim.model.geom_bodyid == finger_body_id)
                    & (self.sim.model.geom_contype != 0)):
                self._finger_geoms[int(geom_id)] = finger_name
        if not self._handle_geoms or not self._finger_geoms:
            raise ValueError("Could not resolve microwave-handle and fingertip collision geoms.")

    def alignment_exit_tolerance_value(self) -> float:
        """Use the profile's microwave-specific ALIGN hysteresis."""
        return float(self.cfg.microwave_alignment_exit_tolerance)

    def contacting_fingers(self) -> set:
        """Fingers currently contacting one of the microwave handle capsules."""
        fingers = set()
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            for finger_geom, handle_geom in (
                    (int(contact.geom1), int(contact.geom2)),
                    (int(contact.geom2), int(contact.geom1))):
                if (finger_geom in self._finger_geoms
                        and handle_geom in self._handle_geoms):
                    fingers.add(self._finger_geoms[finger_geom])
        return fingers

    def grasp_retained(self) -> bool:
        """Require an observed two-pad capture before allowing the door pull."""
        fingers = self.contacting_fingers()
        handle_sized_opening = (
            self.sim.finger_opening >= self.grasp_contact_min_opening
            and self.sim.finger_opening < self.grasp_ready_opening
        )
        if len(fingers) == 2 and handle_sized_opening:
            if not self._capture_confirmed:
                radial = self.door_radial()
                tangent = self.drive_direction()
                offset = self.tool_pos() - self.touch_point()
                self._capture_offset = np.array([
                    float(np.dot(offset, radial)),
                    float(np.dot(offset, tangent)),
                    float(offset[2]),
                ])
            self._capture_confirmed = True
        if self.sim.finger_opening >= self.grasp_ready_opening:
            self._capture_confirmed = False
            self._capture_offset = None
        return bool(self._capture_confirmed and fingers and handle_sized_opening)

    def contact_depth(self) -> float:
        """Measured fingertip-midpoint depth relative to the handle centre."""
        return float(np.dot(
            self.tool_pos() - self.touch_point(), self.approach_axis()))

    def precontact_step_scale(self) -> float:
        return float(self.cfg.microwave_precontact_step)

    def approach_step_scale(self) -> float:
        return float(self.cfg.microwave_approach_step)

    def manipulation_step_scale(self) -> float:
        return float(self.cfg.microwave_manipulation_step)

    def gripper_command(self, mode: str) -> float:
        """Close firmly to capture, then hold at the handle's measured width."""
        if mode == "close" and self._capture_confirmed:
            return float(self.cfg.microwave_grasp_command)
        return super().gripper_command(mode)

    def hook_yaw(self) -> float:
        """Rotation about the bar that puts the leading pad ``microwave_hook_depth`` behind it.

        With the jaws open each pad sits ``JAW_HALF_SPAN`` off the tool centreline, so
        yawing by ``asin(depth / half_span)`` moves one pad that far along the approach
        while the jaw centre stays on the bar.  The sign is fixed by the frame:
        ``grasp_frame`` makes local +y the door radial (hinge -> handle), and a positive
        rotation about the handle axis carries +y toward the door, so it is the pad on the
        door's free edge that ends up behind the bar.
        """
        depth = float(self.cfg.microwave_hook_depth)
        if depth == 0.0:
            return 0.0
        return float(np.arcsin(np.clip(depth / self.JAW_HALF_SPAN, -1.0, 1.0)))

    def desired_orientation(self) -> np.ndarray:
        """Square door-normal grasp, yawed about the bar so one pad hooks behind it."""
        frame = super().desired_orientation()
        angle = self.hook_yaw()
        if frame is None or angle == 0.0:
            return frame
        return rotate_about_axis(self.handle_axis(), angle) @ frame

    def door_radial(self) -> np.ndarray:
        return unit(self.touch_point() - self.sim.joint_anchor(self._hinge_joint()))

    def approach_axis(self) -> np.ndarray:
        # Point from the room toward the door. This is a 90-degree yaw from the old radial
        # side approach, while remaining orthogonal to the vertical handle.
        return -self.drive_direction()

    def handle_roll_action(self) -> np.ndarray:
        """Keep local +x parallel to the bar without overconstraining approach IK.

        The complete pose correction also tries to recover small pitch/yaw errors.  Near
        the microwave those extra constraints compete with the Cartesian insertion and
        stall the arm short of the handle.  Only roll about the desired forward axis can
        make the jaws cease to be orthogonal to this vertical bar, so correct that one
        component while leaving the remaining wrist freedom to the position solver.
        """
        # Rotate about the *measured* tool axis.  Using the desired forward axis here
        # introduces pitch/yaw whenever the live tool has even a small approach error,
        # which recreates the full-pose IK conflict this correction is meant to avoid.
        rotation_axis = unit(self.sim.eef_approach_axis)
        current_handle_axis = self.sim.eef_mat[:, 0]
        desired_handle_axis = self.handle_axis()
        current_handle_axis = unit(
            current_handle_axis
            - np.dot(current_handle_axis, rotation_axis) * rotation_axis)
        desired_handle_axis = unit(
            desired_handle_axis
            - np.dot(desired_handle_axis, rotation_axis) * rotation_axis)
        if (np.linalg.norm(current_handle_axis) < 1e-9
                or np.linalg.norm(desired_handle_axis) < 1e-9):
            return np.zeros(3, dtype=np.float64)
        angle = float(np.arctan2(
            np.dot(rotation_axis,
                   np.cross(current_handle_axis, desired_handle_axis)),
            np.dot(current_handle_axis, desired_handle_axis),
        ))
        # Reversing local x describes the same parallel jaw/bar relationship.  Taking
        # the nearer of those equivalent frames prevents an unnecessary 180-degree roll.
        angle = wrap_to_half_pi(angle)
        return limit_to_box(
            rotation_axis * angle / MAX_ROTATION_DISPLACEMENT,
            self.rotation_step_scale(),
        )

    def precontact_point(self) -> np.ndarray:
        return self.touch_point() + self.drive_direction() * self.cfg.precontact_distance

    def contact_point(self) -> np.ndarray:
        # Preserve the reported handle's horizontal location.  The fast repeated IK
        # settles about 4.5 cm high; asking for 2 cm lower puts the measured fingertips
        # near the middle of the vertical capsule and prevents it being squeezed out.
        point = self.touch_point().copy()
        point[2] += self.cfg.microwave_contact_height_offset
        return point

    def recede_point(self) -> np.ndarray:
        """Freeze a straight, door-normal exit once the handle has been released."""
        if self._fixed_recede_point is None:
            self._recede_origin = self.touch_point()
            self._recede_backward_axis = unit(self.approach_axis())
            self._fixed_recede_point = (
                self._recede_origin
                - self._recede_backward_axis * self.cfg.microwave_recede_distance
            )
        return self._fixed_recede_point.copy()

    def recede_complete(self) -> bool:
        target = self.recede_point()
        tool_pos = self.tool_pos()
        backward_progress = float(np.dot(
            self._recede_origin - tool_pos, self._recede_backward_axis))
        return bool(
            np.linalg.norm(tool_pos - target) <= self.cfg.microwave_recede_tolerance
            or backward_progress >= (
                self.cfg.microwave_recede_distance - self.cfg.position_tolerance)
        )

    def manipulate_point(self) -> np.ndarray:
        """Follow the handle arc while pulling tangentially from the captured pose.

        A target based on the current tool pose preserves any radial tracking error.  Under
        load that error accumulates until the wrist is pressed into the handle and can no
        longer move tangentially.  Preserve the small capture offset in the moving door
        frame instead, so every action corrects back to the live handle arc.
        """
        if self._capture_offset is None:
            return self.tool_pos() + self.drive_direction() * self.manipulation_lookahead
        radial_offset, tangent_offset, vertical_offset = self._capture_offset
        target = (self.touch_point()
                  + self.door_radial() * radial_offset
                  + self.drive_direction()
                  * (tangent_offset + self.manipulation_lookahead))
        target[2] = self.touch_point()[2] + vertical_offset
        return target


class KettlePushSkill(KitchenSkill):
    """Slide the kettle across the stove toward its goal pose.

    This is the optional hand-body fallback.  It avoids relying on grasp friction and
    moves along the kettle-to-goal line through the body centre so the free joint does not
    spin; the default :class:`KettleGraspSkill` instead uses the requested side grasp.
    """

    #: This opt-in fallback pushes with the hand flange; the default grasp skill below
    #: uses the fingertips and full horizontal tool frame.
    tool_offset = 0.0
    approach_gripper = "close"
    engage_gripper = "close"
    #: The kettle's 7-D distance is dominated by the yaw the body picks up while sliding,
    #: so a tight margin is not reachable; 0.9 * BONUS_THRESH = 0.27 is (measured best
    #: without extra tuning: 0.26).
    margin_override = 0.9
    contact_tolerance = 0.04

    def _root_pos(self) -> np.ndarray:
        return self.sim.joint_qpos("kettle")[:3]

    def _goal_pos(self) -> np.ndarray:
        return np.asarray(OBS_ELEMENT_GOALS["kettle"][:3], dtype=np.float64)

    def drive_direction(self) -> np.ndarray:
        planar = self._goal_pos() - self._root_pos()
        planar[2] = 0.0
        return unit(planar)

    def touch_point(self) -> np.ndarray:
        root = self._root_pos()
        point = root - self.drive_direction() * self.cfg.kettle_push_radius
        point[2] = root[2] + self.cfg.kettle_push_height
        return point

    def progress(self) -> float:
        return -float(np.linalg.norm(self._root_pos() - self._goal_pos()))


class KettleGraspSkill(KettlePushSkill):
    """Grasp the kettle's thinner left handle bar and push it horizontally."""

    # ``kettle_chain.xml`` defines the left handle collision capsule at this pose in the
    # kettleroot frame. The capsule's default local axis is +z; unlike the thick 6.4 cm
    # top capsule used previously, this vertical bar is only 4.6 cm in diameter.
    LEFT_HANDLE_LOCAL_POS = np.array([-0.092, 0.0, 0.18], dtype=np.float64)
    LEFT_HANDLE_LOCAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    #: ``kettle_chain.xml`` puts the top-handle *collision* capsule at local z = 0.259 with
    #: radius 0.032 and half-length 0.1 along local x, so it occupies z = 0.227..0.291 and
    #: reaches 0.132 sideways.  A tool point above ``TOP_HANDLE_GUARD_HEIGHT`` and within
    #: ``TOP_HANDLE_GUARD_RADIUS`` of the root is therefore still *over* that capsule and
    #: cannot descend where it is.
    #:
    #: The height must clear the capsule and the hand's own 0.032 m half-thickness
    #: (0.291 + 0.032), not merely the left bar it is aiming for.  A guard low enough to
    #: overlap the bar approach -- which works between 0.18 and 0.24 -- turns every grasp
    #: retry into a sideways-escape/realign limit cycle instead of preventing a collision.
    TOP_HANDLE_GUARD_HEIGHT = 0.32
    TOP_HANDLE_GUARD_RADIUS = 0.15
    tool_offset = FINGERTIP_OFFSET
    approach_gripper = "open"
    engage_gripper = "close"
    recede_after_manipulation = True
    align_in_place = True
    # The kettle is a free body: its handle goes wherever the still-closed pads push it,
    # so the generic release-while-tracking-the-handle motion chases its own disturbance.
    hold_pose_while_releasing = True
    # control_steps=5 repeats a large rotation long enough to cross the desired frame and
    # bounce back on the next action. A kettle-only cap damps that IK loop while preserving
    # the stricter global contact-alignment threshold.
    rotation_step = 0.10
    # The open jaws provide only about 1.7 cm of lateral clearance around the 4.6 cm bar.
    # This is intentionally separate from the generic waypoint tolerance: the IK cannot
    # exactly attain the full side-grasp stand-off, but it can center the jaws without
    # changing their current forward/backward depth before the final approach.
    grasp_center_tolerance = 0.0025
    lost_contact_tolerance = 0.25
    alignment_tolerance = 0.60
    # Closing 3 cm short leaves the bar beyond the fingertip pads. The centering stage now
    # handles IK residual separately, so require the bar centre to reach the pad depth.
    contact_tolerance = 0.015
    # A correctly captured 4.6 cm-diameter side bar holds a nonzero half-opening.
    # A nearly zero opening means the handle escaped and the reactive controller should
    # reopen and reacquire instead of treating one-finger contact as a grasp.
    grasp_contact_min_opening = 0.015
    #: Planar tolerance for the high cross-kitchen waypoint.
    transit_tolerance = 0.25
    #: Planar tolerance for the sideways escape out of the top-handle column.  This is the
    #: fallback exit only: the escape normally ends as soon as the tool leaves the guarded
    #: volume, which happens well before the stand-off itself is reached.
    descent_column_tolerance = 0.05

    def __init__(self, sim: KitchenSim, task: str, cfg: KitchenPolicyConfig):
        super().__init__(sim, task, cfg)
        self._capture_confirmed = False
        self._capture_approach_axis = None
        self._grasp_reacquire_pending = False
        self._fixed_recede_point = None
        self._recede_origin = None
        self._recede_backward_axis = None
        kettle_body_id = self.sim.model_names.body_name2id["kettleroot"]
        collision_geoms = np.flatnonzero(
            (self.sim.model.geom_bodyid == kettle_body_id)
            & (self.sim.model.geom_contype != 0)
        )
        if collision_geoms.size == 0:
            raise ValueError("Kettle model has no collision geometry for its left handle.")
        local_errors = np.linalg.norm(
            self.sim.model.geom_pos[collision_geoms] - self.LEFT_HANDLE_LOCAL_POS,
            axis=1,
        )
        closest = int(np.argmin(local_errors))
        if local_errors[closest] > 0.01:
            raise ValueError("Could not identify the kettle's left-handle collision capsule.")
        self._left_handle_geom_id = int(collision_geoms[closest])
        self._finger_pad_geoms = {}
        for finger_name in ("left_finger", "right_finger"):
            finger_body_id = self.sim.model_names.body_name2id[finger_name]
            finger_geoms = np.flatnonzero(
                self.sim.model.geom_bodyid == finger_body_id)
            pad_geoms = finger_geoms[
                self.sim.model.geom_type[finger_geoms]
                == int(mujoco.mjtGeom.mjGEOM_BOX)
            ]
            if pad_geoms.size == 0:
                raise ValueError(f"Gripper body {finger_name!r} has no fingertip pad geoms.")
            for geom_id in pad_geoms:
                self._finger_pad_geoms[int(geom_id)] = finger_name

    def touch_point(self) -> np.ndarray:
        body_rotation = self.sim.body_xmat("kettleroot")
        return (self.sim.body_xpos("kettleroot")
                + body_rotation @ self.LEFT_HANDLE_LOCAL_POS)

    def approach_axis(self) -> np.ndarray:
        return self.drive_direction()

    def handle_axis(self) -> np.ndarray:
        # Follow the live vertical capsule as the free kettle body translates and rotates.
        return self.sim.body_xmat("kettleroot") @ self.LEFT_HANDLE_LOCAL_AXIS

    def contact_point(self) -> np.ndarray:
        return self.touch_point()

    def precontact_point(self) -> np.ndarray:
        return self.touch_point() - self.approach_axis() * self.cfg.precontact_distance

    def in_top_handle_corridor(self, point: np.ndarray) -> bool:
        """Whether ``point`` is somewhere the hand can strike the kettle's top handle."""
        root = self.sim.body_xpos("kettleroot")
        point = np.asarray(point, dtype=np.float64)
        planar = float(np.linalg.norm((point - root)[:2]))
        return bool(planar <= self.TOP_HANDLE_GUARD_RADIUS
                    and point[2] >= root[2] + self.TOP_HANDLE_GUARD_HEIGHT)

    def transit_point(self) -> Optional[np.ndarray]:
        """Waypoint that keeps the hand out of the kettle's top-handle volume.

        Two separate obstacles are avoided here.  A cross-kitchen reach to this low handle
        reaches diagonally through the hood/light-switch corridor, so it is flown high over
        the ordinary stand-off first.  Independently -- and whatever the arm did before --
        the hand may never descend to the left bar from over the kettle itself: the top
        handle occupies that column, and coming down onto it presses the kettle into the
        counter and shoves the free body out of the grasp corridor.
        """
        if self.grasp_retained() or self._capture_confirmed:
            # Transport legitimately holds the bar inside the guarded volume, and a loaded
            # kettle that tips raises the pads above the guard height. The corridor is a
            # pre-grasp concern only; re-entering it here would abandon a live grasp.
            return None
        tool = self.tool_pos()
        column = self.precontact_point().copy()
        if self.in_top_handle_corridor(tool):
            # Step sideways to the stand-off column at the height already reached. Climbing
            # back to the high waypoint instead would pay for the whole descent twice.
            column[2] = tool[2]
            return column
        if float(np.linalg.norm((tool - column)[:2])) <= self.transit_tolerance:
            return None
        column[2] = max(column[2] + 0.50, 2.32)
        return column

    def transit_reached(self, point: np.ndarray) -> bool:
        """Leaving the guarded volume ends the escape; the high waypoint keeps its radius.

        The base class measures planar proximity alone, which for this skill would mean
        "roughly over the kettle" rather than "lined up with the approach".
        """
        tool = self.tool_pos()
        planar = float(np.linalg.norm((tool - np.asarray(point, dtype=np.float64))[:2]))
        if self.in_top_handle_corridor(tool):
            return planar <= self.descent_column_tolerance
        return planar <= self.transit_tolerance

    def manipulation_stage(self) -> str:
        return KETTLE_TRANSPORT

    def manipulation_step_scale(self) -> float:
        return float(self.cfg.kettle_transport_step)

    def precontact_step_scale(self) -> float:
        return float(self.cfg.kettle_precontact_step)

    def transit_step_scale(self) -> float:
        return (float(self.cfg.free_space_step)
                if self.cfg.kettle_transit_step is None
                else float(self.cfg.kettle_transit_step))

    def approach_step_scale(self) -> float:
        return float(self.cfg.kettle_approach_step)

    def orientation_tolerance_value(self) -> float:
        return float(self.cfg.kettle_orientation_tolerance)

    def alignment_exit_tolerance_value(self) -> float:
        return float(self.cfg.kettle_alignment_exit_tolerance)

    def recede_step_scale(self) -> float:
        return (float(self.cfg.contact_step)
                if self.cfg.kettle_recede_step is None
                else float(self.cfg.kettle_recede_step))

    def gripper_command(self, mode: str) -> float:
        if mode == "close":
            return float(self.cfg.kettle_grasp_command)
        return super().gripper_command(mode)

    def contacting_fingers(self) -> set:
        """Names of fingers contacting the left-handle collision capsule specifically."""
        contacts = set()
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self._left_handle_geom_id and geom2 in self._finger_pad_geoms:
                contacts.add(self._finger_pad_geoms[geom2])
            elif geom2 == self._left_handle_geom_id and geom1 in self._finger_pad_geoms:
                contacts.add(self._finger_pad_geoms[geom1])
        return contacts

    def grasp_retained(self) -> bool:
        """Require an initial two-pad capture, then tolerate one loaded pushing pad."""
        fingers = self.contacting_fingers()
        handle_sized_opening = (
            self.sim.finger_opening >= self.grasp_contact_min_opening
            and self.sim.finger_opening < self.grasp_ready_opening
        )
        if not handle_sized_opening:
            if self.sim.finger_opening >= self.grasp_ready_opening:
                self._capture_confirmed = False
                self._capture_approach_axis = None
            return False
        if len(fingers) == 2:
            if not self._capture_confirmed:
                # Record the approach while it is still a long, well-conditioned vector;
                # see withdraw_axis().
                self._capture_approach_axis = self.approach_axis()
            self._capture_confirmed = True
        return bool(self._capture_confirmed and fingers)

    def _goal_rotation(self) -> np.ndarray:
        """World rotation the kettle body is asked to end up in."""
        quat = np.asarray(OBS_ELEMENT_GOALS["kettle"][3:], dtype=np.float64)
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, quat / np.linalg.norm(quat))
        return matrix.reshape(3, 3)

    def _goal_yaw(self) -> float:
        rotation = self._goal_rotation()
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _kettle_yaw(self) -> float:
        rotation = self.sim.body_xmat("kettleroot")
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def yaw_error(self) -> float:
        """Signed rotation still needed about world z, in radians."""
        return wrap_to_pi(self._goal_yaw() - self._kettle_yaw())

    def goal_position_error(self) -> float:
        """Planar distance from the live kettle body to its goal position."""
        return float(np.linalg.norm((self._goal_pos() - self._root_pos())[:2]))

    def manipulate_point(self) -> np.ndarray:
        """Short live lead toward the goal, at exactly the current tool height.

        The lead is clamped to the distance the body actually has left, so the commanded
        delta shrinks to zero on arrival.  A fixed lead instead holds the request at
        ``kettle_transport_step`` right up to the step the stop predicate happens to fire
        on, which is what drove the kettle well past its goal in a single loaded push.
        """
        point = self.tool_pos().copy()
        planar = np.asarray(self._goal_pos() - self._root_pos(), dtype=np.float64)[:2]
        remaining = float(np.linalg.norm(planar))
        if remaining > 1e-9:
            lead = min(self.cfg.kettle_transport_lookahead, remaining)
            point[:2] += self.push_direction(planar) * lead
        return point

    def push_direction(self, planar: np.ndarray) -> np.ndarray:
        """Planar push heading, held off the microwave side of the counter.

        See ``kettle_transport_min_rightward``: a left-bar grasp walks the body toward
        the microwave, so the heading carries a minimum rightward lean rather than
        pointing straight at the goal.
        """
        planar = np.asarray(planar, dtype=np.float64)
        forward = abs(float(planar[1]))
        floor = float(self.cfg.kettle_transport_min_rightward) * forward
        planar = np.array([max(float(planar[0]), floor), float(planar[1])])
        return unit(planar)[:2]

    def transport_rotation_action(self) -> np.ndarray:
        """Bounded world-z command turning a captured kettle toward its goal yaw.

        This is not the live handle frame the manipulation branch deliberately stops
        chasing: that target rotates with the body and closes a feedback loop through the
        tool frame.  This is a fixed goal yaw, applied about world z only and capped by
        ``kettle_transport_yaw_step``, so it cannot wind the wrist away from the grasp.

        Only a bilateral hold may be turned.  ``grasp_retained`` deliberately tolerates a
        single loaded pad while pushing, but a wrist torque applied through one pad has
        nothing to react against and simply levers the bar out from between the jaws --
        measured as a lost grasp mid-transport, followed by a long failed reacquisition.
        """
        command = np.zeros(3)
        if len(self.contacting_fingers()) < 2:
            return command
        command[2] = float(np.clip(
            self.yaw_error() / MAX_ROTATION_DISPLACEMENT,
            -self.cfg.kettle_transport_yaw_step,
            self.cfg.kettle_transport_yaw_step,
        ))
        return command

    def manipulation_done(self) -> bool:
        """Stop inside the pose threshold, or once the bar is at its goal point.

        The environment's 7-D distance mixes position with a quaternion difference, and a
        friction grip on one 4.6 cm bar cannot always undo the yaw a dragged kettle picks
        up.  Continuing to push a body that is already at its goal position only spins it
        further, so delivery of the handle ends transport as well -- either with the yaw
        inside tolerance, or once the environment has banked the task and nothing more is
        on offer for keeping the grip.
        """
        if (self.sim.task_distance(self.task)
                < BONUS_THRESH * self.cfg.kettle_completion_margin):
            return True
        if self.goal_position_error() > self.cfg.kettle_goal_position_tolerance:
            return False
        return bool(abs(self.yaw_error()) <= self.cfg.kettle_goal_yaw_tolerance
                    or self.task in self.sim.kitchen.episode_task_completions)

    def withdraw_axis(self) -> np.ndarray:
        """Direction the hand came in from, frozen at capture.

        ``approach_axis`` is the live kettle-to-goal direction, which is exactly what a
        successful transport drives to zero: at the end of the push the remaining vector
        is a couple of centimetres of numerical noise, and its unit vector points
        anywhere.  Backing out along *that* is what dragged a delivered kettle a further
        0.2 m across the counter while the pads were still closed on it.  The direction
        the grasp was made from is well defined and is still the right way out.
        """
        if self._capture_approach_axis is not None:
            return self._capture_approach_axis.copy()
        return self.approach_axis()

    def recede_point(self) -> np.ndarray:
        """Fixed post-release waypoint directly backward from the kettle handle.

        Freeze both the handle position and backward axis when receding begins so the
        target cannot move under the controller. Projecting the withdrawal axis off the
        handle axis makes the retreat exactly orthogonal to the live handle.
        """
        if self._fixed_recede_point is None:
            handle_axis = unit(self.handle_axis())
            backward_axis = self.withdraw_axis()
            backward_axis = unit(
                backward_axis - np.dot(backward_axis, handle_axis) * handle_axis)
            if np.linalg.norm(backward_axis) < 1e-9:
                raise ValueError("Kettle retreat needs a nonparallel handle-normal axis.")
            self._recede_origin = self.touch_point()
            self._recede_backward_axis = backward_axis
            self._fixed_recede_point = (
                self._recede_origin - backward_axis * self.cfg.kettle_recede_distance)
        return self._fixed_recede_point.copy()

    def recede_complete(self) -> bool:
        """Finish at the fixed waypoint or after equivalent backward clearance."""
        target = self.recede_point()  # Lazily freeze the origin and handle-normal axis.
        tool_pos = self.tool_pos()
        backward_progress = float(np.dot(
            self._recede_origin - tool_pos, self._recede_backward_axis))
        tolerance = self.cfg.kettle_recede_tolerance
        return bool(
            np.linalg.norm(tool_pos - target) <= tolerance
            or backward_progress >= (
                self.cfg.kettle_recede_distance - self.cfg.position_tolerance)
        )


def build_skill(sim: KitchenSim, task: str, cfg: KitchenPolicyConfig) -> KitchenSkill:
    """Map a task name onto the skill that knows how to do it."""
    if task == "slide_cabinet":
        return SlideSkill(sim, task, cfg)
    if task == "light_switch":
        return LightSwitchSkill(sim, task, cfg)
    if task in TASK_MANIPULATED_JOINTS:
        return KnobSkill(sim, task, cfg)
    if task == "microwave":
        return MicrowavePullSkill(sim, task, cfg)
    if task in ("left_hinge_cabinet", "right_hinge_cabinet"):
        return HandlePullSkill(sim, task, cfg)
    if task == "kettle":
        if cfg.kettle_strategy == "grasp":
            return KettleGraspSkill(sim, task, cfg)
        return KettlePushSkill(sim, task, cfg)
    raise ValueError(f"No scripted skill for kitchen task {task!r}.")


# ---------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------

class ScriptedKitchenPolicy:
    """Hierarchical scripted expert.

    Usage mirrors the MetaWorld scripted policies::

        policy = ScriptedKitchenPolicy(env)
        policy.reset()
        action = policy.get_action(observation)

    ``observation`` is accepted for API compatibility but deliberately unused: the kitchen
    observation has uniform noise added to it (``robot_noise_ratio`` / ``object_noise_ratio``)
    and is a poor source of truth for controller predicates.  All predicates read the live
    simulator, so "arbitrary observation" here means any learner-visited state of the
    attached environment; a detached/offline observation is not enough to reconstruct EEF
    sites and contact state without a separate forward-kinematics model.
    """

    def __init__(self, env, config: Optional[KitchenPolicyConfig] = None):
        self.cfg = config if config is not None else KitchenPolicyConfig()
        self.sim = KitchenSim(env)
        self.sim.validate_action_contract()
        self._base_control_steps = int(self.sim.kitchen.robot_env.control_steps)
        self._active_control_profile_task = None
        self.sim.scale_gripper_stiffness(self.cfg.gripper_stiffness_scale)
        self.sim.install_orientation_controller(
            self.cfg.ik_orientation_duration,
            self.cfg.ik_position_weight,
        )
        self.sim.install_ik_target_leak(self.cfg.ik_target_leak)
        self._rng = np.random.default_rng()
        self.reset()

    # -- episode state -------------------------------------------------------------------
    def reset(self, observation=None, info=None) -> None:
        """Clear all per-episode bookkeeping.  Call after the env has been reset."""
        self._apply_control_profile(None)
        self._task: Optional[str] = None
        self._skill: Optional[KitchenSkill] = None
        self._phase: str = (ORIENT_FORWARD if self.cfg.align_forward_at_reset
                            else SELECT_SUBTASK)
        self._initial_orientation_complete = not self.cfg.align_forward_at_reset
        self._phase_steps: int = 0
        self._retry_count: int = 0
        self._abandoned: List[str] = []
        self._completed_order: List[str] = []
        self._eef_history: List[np.ndarray] = []
        self._progress_history: List[float] = []
        self._last_target: Optional[np.ndarray] = None
        self._last_position_error: float = 0.0
        self._last_tool_error: float = 0.0
        self._reach_error_history: List[float] = []
        self._retreat_lift_target = None
        self._alignment_hold_target = None
        self._reactive_retreat_pending = False
        self._reactive_recede_pending = False
        self._last_orientation_error: float = 0.0
        self._phase_failures: int = 0
        self._steps_in_task: int = 0
        self._task_step_counts: Dict[str, int] = {}
        self._task_succeeded: bool = False
        self._complete_at_selection: bool = False
        self._reactive_signature = None
        self._order = self._make_order()
        self._initially_complete = {
            task for task in self._order if self.sim.task_complete(task)
        }
        #: The pose every skill retreats to; captured from the arm's reset configuration.
        self._home_pos = self.sim.eef_pos
        #: A horizontal gripper held at the raw reset flange position would put its
        #: 10.8 cm fingertips through the cabinet plane.  Back the flange toward the room
        #: by exactly that tool length while it turns, without moving toward any task.
        self._initial_orientation_target = (
            self._home_pos - FINGERTIP_OFFSET * FRONT_APPROACH_AXIS
        )

    def _make_order(self) -> List[str]:
        """Order the env's *own* task list; never hardcode a four-task set."""
        env_tasks = list(self.sim.goal.keys())
        preferred = self.cfg.resolved_task_order()
        ordered = [t for t in preferred if t in env_tasks]
        ordered += [t for t in env_tasks if t not in ordered]  # anything the order missed
        if self.cfg.randomize_task_order:
            ordered = list(self._rng.permutation(ordered))
        return ordered

    def seed(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)

    def _apply_control_profile(self, task: Optional[str]) -> None:
        """Select controller repetition/gain without changing the public action space."""
        microwave_profile = task == "microwave"
        control_steps = (
            self.cfg.microwave_control_steps
            if microwave_profile and self.cfg.microwave_control_steps is not None
            else self._base_control_steps
        )
        if int(control_steps) <= 0:
            raise ValueError(f"control_steps must be positive, got {control_steps}.")
        leak = (
            self.cfg.microwave_ik_target_leak
            if microwave_profile and self.cfg.microwave_ik_target_leak is not None
            else self.cfg.ik_target_leak
        )
        self.sim.kitchen.robot_env.control_steps = int(control_steps)
        self.sim.install_ik_target_leak(float(leak))
        self._active_control_profile_task = task if microwave_profile else None

    # -- planner -------------------------------------------------------------------------
    def _pending_tasks(self) -> List[str]:
        """Unfinished tasks, recomputed from state predicates.

        ``KitchenEnv.tasks_to_complete`` is not trustworthy across episodes (it is never
        restored on reset), so completion is recomputed from ``BONUS_THRESH`` directly.

        A task the env has already *scored* stays finished even if a later skill knocks the
        object back out of its goal region: the env removed it from ``tasks_to_complete``,
        so redoing it earns nothing and only risks making things worse.
        """
        banked = set(self.sim.kitchen.episode_task_completions)
        pending = []
        for task in self._order:
            if task in self._abandoned or task in self._completed_order:
                continue
            if self.cfg.skip_pre_completed_tasks and (task in banked or self.sim.task_complete(task)):
                self._completed_order.append(task)
                continue
            pending.append(task)
        return pending

    def _select_subtask(self) -> Optional[str]:
        pending = self._pending_tasks()
        if not pending:
            return None
        return pending[0]

    # -- FSM plumbing --------------------------------------------------------------------
    def _enter_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_steps = 0
        self._eef_history = []
        self._progress_history = []
        self._reach_error_history = []
        self._retreat_lift_target = None
        self._alignment_hold_target = (self.sim.eef_pos if phase == ALIGN else None)

    def _stalled(self) -> bool:
        window = self.cfg.stall_window
        if len(self._eef_history) <= window:
            return False
        travelled = np.linalg.norm(self._eef_history[-1] - self._eef_history[-1 - window])
        return bool(travelled < self.cfg.stall_distance)

    def _reach_blocked(self) -> bool:
        """The position error has stopped shrinking: arrived-as-close-as-it-can, or stuck.

        Weaker than :meth:`_stalled`, deliberately.  Pressing the hand into a cabinet makes
        the EEF jitter by more than ``stall_distance`` per step without getting any closer,
        so a pure "has it stopped moving" test never fires for the light switch.
        """
        window = self.cfg.progress_window
        if len(self._reach_error_history) <= window:
            return False
        gained = self._reach_error_history[-1 - window] - self._reach_error_history[-1]
        return bool(gained < self.cfg.reach_progress_epsilon)

    def _no_progress(self) -> bool:
        window = self.cfg.progress_window
        if len(self._progress_history) <= window:
            return False
        change = abs(self._progress_history[-1] - self._progress_history[-1 - window])
        return bool(change < self.cfg.progress_epsilon)

    def _gripper(self, mode: str) -> float:
        return self.cfg.gripper_open if mode == "open" else self.cfg.gripper_close

    def _abandon_or_retry(self) -> None:
        """Bounded retries, then fall back to the safe retreat pose and move on."""
        self._phase_failures += 1
        if self._retry_count < self.cfg.max_retries:
            self._retry_count += 1
            self._enter_phase(MOVE_TO_PRECONTACT)
        else:
            if self._task is not None and self._task not in self._abandoned:
                self._abandoned.append(self._task)
            self._task_succeeded = False
            self._enter_phase(RETREAT)

    def _finish_task(self) -> None:
        """Called once the retreat after a subtask (successful or not) has finished."""
        if self._task is not None:
            self._task_step_counts[self._task] = self._steps_in_task
            if self._task_succeeded and self._task not in self._completed_order:
                self._completed_order.append(self._task)
        self._task = None
        self._skill = None
        self._apply_control_profile(None)
        self._task_succeeded = False
        self._retry_count = 0
        self._steps_in_task = 0
        self._enter_phase(SELECT_SUBTASK)

    # -- action ---------------------------------------------------------------------------
    def get_action(self, observation=None, info=None) -> np.ndarray:
        """One expert action in the env's own 7-D action space.

        Always returns a finite ``float32`` vector inside ``Box(-1, 1, (7,))``.
        """
        action = (self._compute_reactive_action() if self.cfg.reactive
                  else self._compute_action())
        action = np.nan_to_num(np.asarray(action, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _set_reactive_phase(self, phase: str) -> None:
        """Update diagnostics without treating repeated policy queries as time."""
        signature = np.concatenate((self.sim.data.qpos.copy(), self.sim.data.qvel.copy())).tobytes()
        state_changed = signature != self._reactive_signature
        if phase != self._phase:
            self._enter_phase(phase)
        elif state_changed:
            self._phase_steps += 1
        if state_changed and self._task is not None:
            self._steps_in_task += 1
        self._reactive_signature = signature

    def _reactive_task_satisfied(self, task: str) -> bool:
        """Current completion predicate with narrow hysteresis after a real completion."""
        # ``skip_pre_completed_tasks=False`` is a diagnostic mode for physically turning
        # the burner knobs even though the reward predicate starts true.
        force_initial_knob = (
            not self.cfg.skip_pre_completed_tasks
            and task in self._initially_complete
            and task in TASK_MANIPULATED_JOINTS
            and task not in self._completed_order
        )
        if force_initial_knob:
            skill = self._skill if self._task == task else build_skill(self.sim, task, self.cfg)
            return skill.manipulation_done()

        # KitchenEnv permanently banks a scored subtask in this episode and will never
        # reward it again.  In particular, the microwave door can rebound while the open
        # gripper retreats; reselecting it then traps randomized microwave-before-kettle
        # orders in a pointless second approach instead of advancing to the kettle.
        if task in self.sim.kitchen.episode_task_completions:
            return True

        distance = self.sim.task_distance(task)
        if distance < BONUS_THRESH:
            return True
        was_complete = (
            task in self._completed_order
            or task in self.sim.kitchen.episode_task_completions
        )
        return bool(was_complete and distance < BONUS_THRESH + self.cfg.reactivation_margin)

    def _reactive_task(self) -> Optional[KitchenSkill]:
        """Select from current task predicates only; no banked/abandoned-task filtering."""
        active_satisfied = False
        if self._task is not None and self._skill is not None:
            active_satisfied = self._skill.manipulation_done()
            # Keep transporting a captured kettle to the deeper demonstration margin.
            # If it slips only after KitchenEnv has already banked the task, however,
            # reacquisition cannot earn anything and can spend the rest of a randomized
            # episode destabilizing an already-successful kettle. Release and recede.
            if (not active_satisfied
                    and isinstance(self._skill, KettleGraspSkill)
                    and self._task in self.sim.kitchen.episode_task_completions
                    and not self._skill.grasp_retained()):
                active_satisfied = True
        if self._task is not None and active_satisfied:
            if self._skill.recede_after_manipulation:
                self._reactive_recede_pending = True
                return self._skill
            if self._task not in self._completed_order:
                self._completed_order.append(self._task)
            self._task_step_counts[self._task] = self._steps_in_task
            self._task = None
            self._skill = None
            self._apply_control_profile(None)
            self._steps_in_task = 0
            self._reactive_retreat_pending = True
            return None

        pending = []
        for task in self._order:
            satisfied = (self._skill.manipulation_done()
                         if task == self._task and self._skill is not None
                         else self._reactive_task_satisfied(task))
            if not satisfied:
                pending.append(task)
        if not pending:
            self._task = None
            self._skill = None
            self._apply_control_profile(None)
            return None

        selected = pending[0]
        # Always honor the configured precedence.  If an earlier task is disturbed after
        # it was completed, it becomes current again instead of being hidden by history.
        if self._task != selected or self._skill is None:
            self._task = selected
            self._skill = build_skill(self.sim, selected, self.cfg)
            self._apply_control_profile(selected)
            self._steps_in_task = 0
            self._retry_count = 0
            self._set_reactive_phase(MOVE_TO_PRECONTACT)
        return self._skill

    def _finish_reactive_recede(self) -> None:
        """Bank a still-complete task, or re-approach if retreat disturbed it."""
        completion_retained = (
            self._task in self.sim.kitchen.episode_task_completions
            if self._task is not None else True
        )
        if self._skill is not None and not completion_retained:
            # The free kettle can drift substantially during release, so retain its
            # deeper demonstration margin. Fixed joints such as the light switch only
            # need to remain inside the environment's actual completion threshold.
            completion_retained = (
                self._skill.manipulation_done()
                if isinstance(self._skill, KettleGraspSkill)
                else self._skill.complete()
            )
        if (self._task is not None and self._skill is not None
                and not completion_retained):
            task = self._task
            self._skill = build_skill(self.sim, task, self.cfg)
            self._retry_count += 1
            self._reactive_recede_pending = False
            self._enter_phase(MOVE_TO_PRECONTACT)
            return

        if self._task is not None:
            if self._task not in self._completed_order:
                self._completed_order.append(self._task)
            self._task_step_counts[self._task] = self._steps_in_task
        self._task = None
        self._skill = None
        self._apply_control_profile(None)
        self._steps_in_task = 0
        self._retry_count = 0
        self._reactive_recede_pending = False
        # The recede waypoint is already the slide skill's safe exit.  Selecting the next
        # skill from there avoids driving the still-horizontal fingertips forward through
        # the cabinet merely to revisit the reset flange pose.
        self._reactive_retreat_pending = False
        self._enter_phase(SELECT_SUBTASK)

    @staticmethod
    def _tool_distance(skill: KitchenSkill, point: np.ndarray) -> float:
        return float(np.linalg.norm(skill.tool_pos() - np.asarray(point, dtype=np.float64)))

    @staticmethod
    def _segment_coordinates(point: np.ndarray, start: np.ndarray,
                             end: np.ndarray):
        """Return distance along, lateral distance to, and length of a line segment."""
        point = np.asarray(point, dtype=np.float64)
        start = np.asarray(start, dtype=np.float64)
        segment = np.asarray(end, dtype=np.float64) - start
        length = float(np.linalg.norm(segment))
        if length < 1e-9:
            return 0.0, float(np.linalg.norm(point - start)), 0.0
        direction = segment / length
        relative = point - start
        along = float(np.dot(relative, direction))
        lateral = float(np.linalg.norm(relative - along * direction))
        return along, lateral, length

    def _compute_reactive_action(self) -> np.ndarray:
        """Pure state-feedback expert used for arbitrary learner-visited states.

        The old rollout FSM advanced on stalls and timeouts even when its recommendation
        was not executed.  Merely querying the expert while applying zero actions could
        therefore reach MANIPULATE and permanently abandon the untouched task.  Here the
        phase is inferred from live task, tool, yaw, and gripper predicates each time; no
        elapsed-query count can change the returned action.
        """
        if not self._initial_orientation_complete:
            desired = grasp_frame(FRONT_APPROACH_AXIS, VERTICAL_HANDLE_AXIS)
            self._last_orientation_error = orientation_error(self.sim, desired)
            clearance_error = float(np.linalg.norm(
                self._initial_orientation_target - self.sim.eef_pos))
            if (self._last_orientation_error > self.cfg.yaw_tolerance
                    or clearance_error > self.cfg.position_tolerance):
                self._set_reactive_phase(ORIENT_FORWARD)
                return self._track_initial_orientation(desired)
            self._initial_orientation_complete = True
            self._enter_phase(SELECT_SUBTASK)

        if self._reactive_retreat_pending:
            home_error = float(np.linalg.norm(self.sim.eef_pos - self._home_pos))
            if home_error > self.cfg.reactive_retreat_tolerance:
                self._set_reactive_phase(RETREAT)
                return self._track_eef(None, self._home_pos, self.cfg.free_space_step,
                                       "open")
            self._reactive_retreat_pending = False

        skill = self._reactive_task()
        if skill is None:
            if self._reactive_retreat_pending:
                self._set_reactive_phase(RETREAT)
                return self._track_eef(None, self._home_pos, self.cfg.free_space_step,
                                       "open")
            self._set_reactive_phase(IDLE)
            self._last_target = None
            self._last_position_error = 0.0
            self._last_tool_error = 0.0
            return self._hold(self._gripper("open"))

        if self._reactive_recede_pending:
            self._set_reactive_phase(RECEDE)
            # Release completely while holding the live handle point; translating before
            # the pads separate can pull the just-completed object back toward its start.
            released = self.sim.finger_opening >= self.cfg.release_opening_threshold
            if not released and not skill.release_while_receding:
                if skill.hold_pose_while_releasing:
                    return self._track_eef(skill, self.sim.eef_pos,
                                           self.cfg.contact_step, "open", orient=False)
                return self._track_action(skill, skill.contact_point(),
                                          self.cfg.contact_step, "open")
            target = skill.recede_point()
            if isinstance(skill, MicrowavePullSkill):
                # First translate straight backward with a frozen wrist. Combining that
                # clearance with a large rotation makes the DLS controller move toward
                # the open door instead of away from it. Once clear, optionally restore
                # the tracked door-normal frame for an immediately following light skill;
                # doing the rotation here, in free space, prevents its one-pad oscillation.
                if not skill._exit_clearance_reached:
                    if skill.recede_complete():
                        skill._exit_clearance_reached = True
                    elif self._phase_steps < self.cfg.recede_timeout:
                        return self._track_action(
                            skill,
                            target,
                            skill.recede_step_scale(),
                            "open",
                            orient=False,
                        )
                remaining = [
                    task for task in self._order
                    if task != self._task and not self._reactive_task_satisfied(task)
                ]
                if remaining and remaining[0] == "light_switch":
                    desired = skill.desired_orientation()
                    frame_error = orientation_error(self.sim, desired)
                    self._last_orientation_error = frame_error
                    if (frame_error > skill.orientation_tolerance_value()
                            and self._phase_steps < self.cfg.recede_timeout):
                        return self._track_eef(
                            skill,
                            self.sim.eef_pos,
                            skill.recede_step_scale(),
                            "open",
                        )
                self._finish_reactive_recede()
                return self._hold(self._gripper("open"))
            action = self._track_action(
                skill,
                target,
                skill.recede_step_scale(),
                "open",
                orient=skill.orient_during_recede,
            )
            if released and (skill.recede_complete()
                             or self._phase_steps >= self.cfg.recede_timeout):
                self._finish_reactive_recede()
            return action

        transit = skill.transit_point()
        if transit is not None and not skill.transit_reached(transit):
            self._set_reactive_phase(MOVE_TO_PRECONTACT)
            return self._track_action(skill, transit, skill.transit_step_scale(),
                                      skill.approach_gripper, orient=False)

        precontact = skill.precontact_point()
        contact = skill.contact_point()
        manipulate = skill.manipulate_point()
        pre_error = self._tool_distance(skill, precontact)
        contact_error = self._tool_distance(skill, contact)
        tool = skill.tool_pos()
        approach_along, approach_lateral, approach_length = self._segment_coordinates(
            tool, precontact, contact)
        jaw_center_error = abs(float(np.dot(
            contact - tool, self.sim.eef_finger_axis)))
        manipulate_along, manipulate_lateral, manipulate_length = self._segment_coordinates(
            tool, contact, manipulate)
        desired_orientation = skill.desired_orientation()
        frame_error = (orientation_error(self.sim, desired_orientation)
                       if desired_orientation is not None else 0.0)
        self._last_orientation_error = frame_error
        angle_tolerance = skill.orientation_tolerance_value()
        alignment_threshold = (skill.alignment_exit_tolerance_value()
                               if self._phase == ALIGN else angle_tolerance)
        kettle_handle_contacts = (skill.contacting_fingers()
                                  if isinstance(skill, KettleGraspSkill) else set())
        light_switch_contacts = (skill.contacting_fingers()
                                 if isinstance(skill, LightSwitchSkill) else set())
        microwave_handle_contacts = (skill.contacting_fingers()
                                     if isinstance(skill, MicrowavePullSkill) else set())
        if (isinstance(skill, KettleGraspSkill)
                and not skill.grasp_retained()
                and not kettle_handle_contacts
                and not skill._grasp_reacquire_pending
                and self.sim.finger_opening >= self.cfg.release_opening_threshold
                and frame_error <= angle_tolerance
                and approach_along >= -skill.precontact_tolerance
                and approach_along <= approach_length + skill.lost_contact_tolerance
                and jaw_center_error > skill.grasp_center_tolerance):
            # Project onto the precontact->contact line at the tool's current depth. This
            # requests lateral/vertical centering only, so the outside of a finger cannot
            # push the free kettle forward while the bar is still outside the open jaws.
            # Centering translates without commanding rotation.  The DLS solver couples
            # that translation into rotation, so the frame error ramps until ALIGN fires
            # for a step and knocks it back -- a sawtooth that visibly stalls the approach
            # (seed 42: 0.14 -> 0.18 about every eleven steps).  Correcting the wrist here
            # instead is worse, not better: rotating swings the tool on its 0.108 m lever
            # and fights the centering it is interleaved with, measured as seed 42 falling
            # to 2/4 with the centering stage growing from 76 steps to 104.
            approach_direction = unit(contact - precontact)
            centered_depth = float(np.clip(approach_along, 0.0, approach_length))
            centered_point = precontact + approach_direction * centered_depth
            self._set_reactive_phase(MOVE_TO_PRECONTACT)
            return self._track_action(skill, centered_point,
                                      skill.approach_step_scale(),
                                      skill.approach_gripper, orient=False)

        # Once the kettle is captured, keep the loaded grasp and push horizontally. Do not
        # continue chasing the kettle's rotating handle frame: that target moves with the
        # body and its full-pose correction can tip the kettle off-centre and upward. The
        # only orientation command here is the bounded world-z term that turns the body
        # back toward its goal yaw, which the off-centre side grasp otherwise cannot hold.
        if isinstance(skill, KettleGraspSkill) and skill.grasp_retained():
            phase = skill.manipulation_stage()
            self._set_reactive_phase(phase)
            action = self._track_action(
                skill,
                manipulate,
                skill.manipulation_step_scale(),
                skill.engage_gripper,
                orient=False,
            )
            action[3:6] = skill.transport_rotation_action()
            return action

        if isinstance(skill, LightSwitchSkill) and skill.grasp_retained():
            # This thin capsule can drive the finger joint almost to zero even during a
            # valid grasp, so bilateral switch contact is the authoritative signal.
            self._set_reactive_phase(MANIPULATE)
            return self._track_action(
                skill,
                manipulate,
                skill.manipulation_step_scale(),
                skill.engage_gripper,
                orient=False,
            )

        if isinstance(skill, MicrowavePullSkill) and skill.grasp_retained():
            # A loaded handle constrains the wrist naturally. Reissuing live orientation
            # corrections while pulling can lever the vertical bar back out of the pads.
            self._set_reactive_phase(MANIPULATE)
            return self._track_action(
                skill,
                manipulate,
                skill.manipulation_step_scale(),
                skill.engage_gripper,
                orient=False,
            )

        if isinstance(skill, KettleGraspSkill):
            # Never translate toward the transport waypoint until both pads have captured
            # the intended left-handle capsule. The first pad normally touches while the
            # jaws are fully open; holding the measured EEF pose while closing lets the
            # opposite pad come around the bar instead of using that first pad as a pusher.
            if skill._grasp_reacquire_pending:
                self._set_reactive_phase(CONTACT_OR_GRASP)
                if self.sim.finger_opening < self.cfg.release_opening_threshold:
                    return self._track_eef(skill, self.sim.eef_pos,
                                           skill.approach_step_scale(),
                                           skill.approach_gripper, orient=False)
                skill._grasp_reacquire_pending = False
                self._set_reactive_phase(MOVE_TO_PRECONTACT)
                return self._track_action(skill, precontact,
                                          skill.precontact_step_scale(),
                                          skill.approach_gripper, orient=False)

            if kettle_handle_contacts:
                if desired_orientation is not None and frame_error > angle_tolerance:
                    # A misaligned one-pad touch is not a grasp. Open without advancing;
                    # the ordinary approach logic will realign after contact clears.
                    self._set_reactive_phase(ALIGN)
                    return self._track_eef(skill, self.sim.eef_pos,
                                           skill.approach_step_scale(),
                                           skill.approach_gripper)
                self._set_reactive_phase(CONTACT_OR_GRASP)
                if (self.sim.finger_opening < skill.grasp_ready_opening
                        and self._phase_steps > self.cfg.engage_steps):
                    # Closing did not produce bilateral target-handle contact. Fully open
                    # in place, then return to the stand-off before trying again.
                    skill._grasp_reacquire_pending = True
                    return self._track_eef(skill, self.sim.eef_pos,
                                           skill.approach_step_scale(),
                                           skill.approach_gripper, orient=False)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.engage_gripper, orient=False)

            if self.sim.finger_opening < skill.grasp_ready_opening:
                # The gripper closed after losing its only target-handle contact. Do not
                # let geometric proximity reinterpret that empty closure as transport.
                skill._grasp_reacquire_pending = True
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.approach_gripper, orient=False)

        if (isinstance(skill, LightSwitchSkill)
                and self._phase == APPROACH
                and contact_error > skill.contact_tolerance):
            # Once the low stand-off has been reached, commit to the straight insertion.
            # Alternating between the two endpoints makes the repeated IK action bounce
            # above the switch before the jaws ever reach it.
            if desired_orientation is not None and frame_error > angle_tolerance:
                self._set_reactive_phase(ALIGN)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.approach_gripper)
            self._set_reactive_phase(APPROACH)
            return self._track_action(skill, contact, skill.approach_step_scale(),
                                      skill.approach_gripper)

        if isinstance(skill, LightSwitchSkill):
            if light_switch_contacts:
                # Hold and close until both pads surround the capsule. Never promote a
                # one-finger touch into the right-to-left manipulation.
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.engage_gripper, orient=False)
            if self.sim.finger_opening < skill.grasp_ready_opening:
                # Closed without retaining switch contact: reopen in place, then let the
                # state-derived approach center and retry.
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.approach_gripper, orient=False)

        if isinstance(skill, MicrowavePullSkill):
            if skill._grasp_reacquire_pending:
                self._set_reactive_phase(CONTACT_OR_GRASP)
                if self.sim.finger_opening < self.cfg.release_opening_threshold:
                    return self._track_eef(skill, self.sim.eef_pos,
                                           skill.approach_step_scale(),
                                           skill.approach_gripper, orient=False)
                skill._grasp_reacquire_pending = False
                self._set_reactive_phase(MOVE_TO_PRECONTACT)
                return self._track_action(skill, precontact,
                                          skill.precontact_step_scale(),
                                          skill.approach_gripper, orient=False)

            if microwave_handle_contacts:
                # A coarse control repeat can make one pad brush the handle while the bar
                # is still visibly off-centre between the open jaws.  Closing immediately
                # from that state traps the bar on one side and creates a long
                # release/retry pause.  Finish the straight insertion while still open;
                # only the centred contact-point branch below may start closing.  Pulling
                # remains gated by grasp_retained(), which requires bilateral contact.
                if (len(microwave_handle_contacts) < 2
                        and contact_error > skill.contact_tolerance):
                    self._set_reactive_phase(APPROACH)
                    action = self._track_action(
                        skill,
                        contact,
                        skill.approach_step_scale(),
                        skill.approach_gripper,
                        orient=False,
                    )
                    action[3:6] = skill.handle_roll_action()
                    return action
                self._set_reactive_phase(CONTACT_OR_GRASP)
                if (self.sim.finger_opening < skill.grasp_contact_min_opening
                        and not skill.grasp_retained()):
                    skill._grasp_reacquire_pending = True
                    return self._track_eef(skill, self.sim.eef_pos,
                                           skill.approach_step_scale(),
                                           skill.approach_gripper, orient=False)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.engage_gripper, orient=False)

            if self.sim.finger_opening < skill.grasp_ready_opening:
                # The jaws closed without retaining the target handle. Reopen before
                # returning to the stand-off instead of pulling from an empty grasp.
                skill._grasp_reacquire_pending = True
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.approach_gripper, orient=False)

            if contact_error <= skill.contact_tolerance:
                # With the bar correctly centered, fully open pads do not touch it yet.
                # Proximity is therefore the signal to close; subsequent manipulation
                # still requires grasp_retained() to observe bilateral physical contact.
                self._set_reactive_phase(CONTACT_OR_GRASP)
                action = self._track_action(skill, contact,
                                            skill.approach_step_scale(),
                                            skill.engage_gripper, orient=False)
                action[3:6] = skill.handle_roll_action()
                return action

            if self._phase in (APPROACH, CONTACT_OR_GRASP):
                # Continue the straight insertion to microwave_handle_position. A full
                # pose command overconstrains IK here, so correct only roll about the
                # measured approach axis while preserving translation.
                self._set_reactive_phase(APPROACH)
                action = self._track_action(skill, contact,
                                            skill.approach_step_scale(),
                                            skill.approach_gripper, orient=False)
                action[3:6] = skill.handle_roll_action()
                return action
            if (desired_orientation is not None
                    and frame_error > alignment_threshold
                    and pre_error <= skill.alignment_tolerance):
                self._set_reactive_phase(ALIGN)
                return self._track_action(skill, precontact,
                                          skill.approach_step_scale(),
                                          skill.approach_gripper)
            if pre_error <= skill.precontact_tolerance:
                self._set_reactive_phase(APPROACH)
                return self._track_action(skill, contact,
                                          skill.approach_step_scale(),
                                          skill.approach_gripper, orient=False)
            self._set_reactive_phase(MOVE_TO_PRECONTACT)
            orient = (desired_orientation is None
                      or frame_error < angle_tolerance)
            return self._track_action(skill, precontact,
                                      skill.precontact_step_scale(),
                                      skill.approach_gripper, orient=orient)

        if (isinstance(skill, HandlePullSkill)
                and self._phase == APPROACH
                and contact_error > skill.contact_tolerance):
            # The pull-door stand-off and hooked contact points lie on opposite sides of
            # the handle.  One repeated control action can cross the narrow geometric
            # corridor without yet reaching contact; returning to the stand-off on the
            # next query then produces an endless two-point oscillation.  Once aligned at
            # the stand-off, commit to the straight insertion until the handle is between
            # the open pads.
            if (desired_orientation is not None
                    and frame_error > angle_tolerance
                    and not isinstance(skill, MicrowavePullSkill)):
                self._set_reactive_phase(ALIGN)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.approach_gripper)
            self._set_reactive_phase(APPROACH)
            return self._track_action(skill, contact, skill.approach_step_scale(),
                                      skill.approach_gripper, orient=False)

        # Infer engagement from the current tool pose.  The manipulation corridor handles
        # states just past the contact waypoint without consulting the previous phase.
        near_contact = contact_error <= skill.contact_tolerance
        in_manipulation_corridor = (
            manipulate_along >= -skill.contact_tolerance
            and manipulate_along <= manipulate_length + skill.lost_contact_tolerance
            and manipulate_lateral <= skill.lost_contact_tolerance
        )
        # For a pull skill the precontact and post-grasp pull targets can lie on the same
        # side of the handle.  Geometry alone would mistake the open-gripper precontact
        # pose for an already engaged pull pose and close in free space.  A closed gripper
        # is observable state evidence that the CONTACT stage actually occurred.
        if skill.approach_gripper != skill.engage_gripper:
            in_manipulation_corridor = (
                in_manipulation_corridor
                and self.sim.finger_opening >= skill.grasp_contact_min_opening
                and self.sim.finger_opening < skill.grasp_ready_opening)
        if near_contact or in_manipulation_corridor:
            if isinstance(skill, KettleGraspSkill) and not kettle_handle_contacts:
                # The bar is geometrically centered at the fingertip pads. Keep tracking
                # the final few millimetres while closing: some collision-free arm poses
                # settle just outside the nominal contact point, and holding that residual
                # forever closes in empty space. The next state must still show bilateral
                # target-pad contact before the transport branch can translate the kettle.
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_action(skill, contact,
                                          skill.approach_step_scale(),
                                          skill.engage_gripper, orient=False)
            kettle_grasp_retained = (
                isinstance(skill, KettleGraspSkill) and skill.grasp_retained())
            if (desired_orientation is not None
                    and frame_error > angle_tolerance
                    and not kettle_grasp_retained):
                # Contact can compress an open gripper enough to look "closed".  Frame
                # alignment remains mandatory: preserve a real grip if one exists, but do
                # not let finger opening alone start manipulation with a twisted wrist.
                grip_mode = (skill.engage_gripper
                             if self.sim.finger_opening < skill.grasp_ready_opening
                             else skill.approach_gripper)
                self._set_reactive_phase(ALIGN)
                return self._track_action(skill, contact, self.cfg.contact_step,
                                          grip_mode)
            closing = skill.engage_gripper == "close"
            if (isinstance(skill, KettleGraspSkill)
                    and closing
                    and self.sim.finger_opening < skill.grasp_ready_opening
                    and not kettle_grasp_retained):
                # A partial-width command also stops at a handle-sized opening in empty
                # space, so opening alone cannot prove capture. Reopen until the kettle
                # skill has first observed a bilateral capture on the left bar.
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_action(skill, contact, self.cfg.contact_step,
                                          skill.approach_gripper)
            if closing and self.sim.finger_opening < skill.grasp_contact_min_opening:
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_action(skill, contact, self.cfg.contact_step,
                                          skill.approach_gripper)
            gripper_ready = (not closing
                             or self.sim.finger_opening < skill.grasp_ready_opening)
            if not gripper_ready:
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_action(skill, contact, self.cfg.contact_step,
                                          skill.engage_gripper)
            phase = (skill.manipulation_stage()
                     if isinstance(skill, KettleGraspSkill) else MANIPULATE)
            self._set_reactive_phase(phase)
            return self._track_action(skill, manipulate, skill.manipulation_step_scale(),
                                      skill.engage_gripper,
                                      orient=True)

        # Crossing the precontact/contact segment can move the live handle and tool frame
        # enough that a purely geometric reclassification jumps back to the stand-off on
        # the next step. Once a real approach has started, keep advancing to contact (or
        # realign the wrist) instead of alternating equal and opposite translations.
        if isinstance(skill, KettleGraspSkill) and self._phase == APPROACH:
            if jaw_center_error > skill.grasp_center_tolerance:
                # The target bar is not centered between the jaws yet. Back up to the
                # stand-off and finish the lateral correction instead of letting an outer
                # finger surface turn this approach into an accidental push.
                self._set_reactive_phase(MOVE_TO_PRECONTACT)
                orient = (desired_orientation is None
                          or frame_error < angle_tolerance)
                return self._track_action(skill, precontact,
                                          skill.precontact_step_scale(),
                                          skill.approach_gripper, orient=orient)
            if desired_orientation is not None and frame_error > angle_tolerance:
                self._set_reactive_phase(ALIGN)
                if skill.align_in_place:
                    return self._track_eef(skill, self.sim.eef_pos,
                                           self.cfg.contact_step,
                                           skill.approach_gripper)
                return self._track_action(skill, precontact,
                                          skill.precontact_step_scale(),
                                          skill.approach_gripper)
            self._set_reactive_phase(APPROACH)
            return self._track_action(skill, contact, skill.approach_step_scale(),
                                      skill.approach_gripper)

        # A geometric corridor makes waypoint progress state-derived.  The prior version
        # used ``self._phase`` as hysteresis and could return different actions for an
        # identical simulator state depending on what had been queried before it.
        if (desired_orientation is not None
                and frame_error > alignment_threshold
                and pre_error <= skill.alignment_tolerance):
            # Rotation can displace the EEF a few centimetres.  Keep aligning throughout
            # the broader safe stand-off region instead of bouncing back to an unoriented
            # free-space reach whenever that displacement exceeds precontact_tolerance.
            self._set_reactive_phase(ALIGN)
            if skill.align_in_place:
                self._last_tool_error = pre_error
                return self._track_eef(skill, self.sim.eef_pos, self.cfg.contact_step,
                                       skill.approach_gripper)
            return self._track_action(skill, precontact, self.cfg.contact_step,
                                      skill.approach_gripper)

        in_approach_corridor = (
            approach_along >= -skill.precontact_tolerance
            and approach_along <= approach_length + skill.lost_contact_tolerance
            and approach_lateral <= skill.precontact_tolerance
        )
        if pre_error <= skill.precontact_tolerance or in_approach_corridor:
            if (desired_orientation is not None
                    and frame_error > alignment_threshold):
                self._set_reactive_phase(ALIGN)
                if skill.align_in_place:
                    self._last_tool_error = pre_error
                    return self._track_eef(skill, self.sim.eef_pos, self.cfg.contact_step,
                                           skill.approach_gripper)
                return self._track_action(skill, precontact, self.cfg.contact_step,
                                          skill.approach_gripper)
            self._set_reactive_phase(APPROACH)
            return self._track_action(skill, contact, skill.approach_step_scale(),
                                      skill.approach_gripper)

        self._set_reactive_phase(MOVE_TO_PRECONTACT)
        orient = (desired_orientation is None
                  or frame_error < angle_tolerance)
        return self._track_action(skill, precontact, skill.precontact_step_scale(),
                                  skill.approach_gripper, orient=orient)

    def _compute_action(self) -> np.ndarray:
        cfg = self.cfg

        if self._phase == ORIENT_FORWARD:
            self._phase_steps += 1
            desired = grasp_frame(FRONT_APPROACH_AXIS, VERTICAL_HANDLE_AXIS)
            self._last_orientation_error = orientation_error(self.sim, desired)
            action = self._track_initial_orientation(desired)
            if (self._last_orientation_error <= cfg.yaw_tolerance
                    and self._last_position_error <= cfg.position_tolerance):
                self._initial_orientation_complete = True
                self._enter_phase(SELECT_SUBTASK)
            return action

        # Keep the FSM self-consistent if a wrapper/test clears an active skill or if an
        # idle task is externally disturbed.  A phase without a skill cannot make a valid
        # target, so return through selection instead of silently holding forever.
        if self._skill is None and self._phase not in (ORIENT_FORWARD, SELECT_SUBTASK):
            self._enter_phase(SELECT_SUBTASK)

        if self._phase == SELECT_SUBTASK:
            task = self._select_subtask()
            if task is None:
                self._apply_control_profile(None)
                self._enter_phase(IDLE)
            else:
                self._task = task
                self._skill = build_skill(self.sim, task, cfg)
                self._apply_control_profile(task)
                self._retry_count = 0
                self._steps_in_task = 0
                self._task_succeeded = False
                self._complete_at_selection = self._skill.complete()
                first_phase = (ALIGN if self._skill.align_before_reach
                               and self._skill.desired_orientation() is not None
                               else MOVE_TO_PRECONTACT)
                self._enter_phase(first_phase)

        if self._phase == IDLE or self._skill is None:
            self._last_target = None
            self._last_position_error = 0.0
            return self._hold(self._gripper("open"))

        skill = self._skill
        self._phase_steps += 1
        self._steps_in_task += 1
        self._eef_history.append(self.sim.eef_pos)
        self._progress_history.append(skill.progress())

        # A task can complete at any moment (a door can swing past the threshold on its
        # own); check before spending more steps on it.  Tasks that were *already* within
        # BONUS_THRESH when they were selected -- which is how the burner tasks start --
        # do not short-circuit, otherwise their skill could never be exercised.
        if (self._phase in (APPROACH, CONTACT_OR_GRASP, MANIPULATE)
                and skill.manipulation_done() and not self._complete_at_selection):
            self._enter_phase(RECEDE if skill.recede_after_manipulation else VERIFY)

        # Hard per-subtask budget: give up rather than spend the whole episode on a skill
        # that is not converging.
        if (self._phase not in (VERIFY, RETREAT)
                and self._steps_in_task > cfg.task_step_budget):
            if self._task is not None and self._task not in self._abandoned:
                self._abandoned.append(self._task)
            self._phase_failures += 1
            self._task_succeeded = False
            self._enter_phase(RETREAT)

        if self._phase == MOVE_TO_PRECONTACT:
            desired = skill.desired_orientation()
            frame_error = orientation_error(self.sim, desired) if desired is not None else 0.0
            self._last_orientation_error = frame_error
            aligned = (desired is None
                       or frame_error < skill.orientation_tolerance_value())
            after_reach = APPROACH if aligned else ALIGN
            waypoint = skill.precontact_point() if aligned else skill.alignment_point()
            return self._track(skill, waypoint, skill.precontact_step_scale(),
                               skill.approach_gripper, next_phase=after_reach,
                               timeout=cfg.reach_timeout, stall_advances=True,
                               recover_posture=aligned, orient=aligned)

        if self._phase == ALIGN:
            desired = skill.desired_orientation()
            if self._alignment_hold_target is None:
                self._alignment_hold_target = self.sim.eef_pos
            action = self._track_eef(skill, self._alignment_hold_target, cfg.contact_step,
                                     skill.approach_gripper)
            err = orientation_error(self.sim, desired) if desired is not None else 0.0
            self._last_orientation_error = err
            if (err < skill.orientation_tolerance_value()
                    or self._phase_steps >= cfg.align_timeout):
                self._enter_phase(MOVE_TO_PRECONTACT)
            return action

        if self._phase == APPROACH:
            return self._track(skill, skill.contact_point(), skill.approach_step_scale(),
                               skill.approach_gripper, next_phase=CONTACT_OR_GRASP,
                               timeout=cfg.reach_timeout, stall_advances=True)

        if self._phase == CONTACT_OR_GRASP:
            # Hold the pose and (re)set the gripper.  Never re-close an already closed
            # gripper: if the engage command is 'close' and the fingers are already shut,
            # go straight on to manipulation.
            desired = skill.desired_orientation()
            frame_error = orientation_error(self.sim, desired) if desired is not None else 0.0
            aligned = (desired is None
                       or frame_error < skill.orientation_tolerance_value())
            grip_mode = skill.engage_gripper if aligned else skill.approach_gripper
            action = self._track_action(skill, skill.contact_point(), cfg.contact_step,
                                        grip_mode)
            # A grasped handle physically prevents a zero finger joint position.  Treat a
            # handle-sized opening as engaged instead of waiting forever for empty-gripper
            # closure (the kettle side bar has about 0.023 m radius).
            already_closed = (skill.engage_gripper == "close"
                              and self.sim.finger_opening < skill.grasp_ready_opening)
            if aligned and (self._phase_steps >= cfg.engage_steps or already_closed):
                self._enter_phase(MANIPULATE)
            elif self._phase_steps >= cfg.align_timeout:
                self._abandon_or_retry()
            return action

        if self._phase == MANIPULATE:
            action = self._track_action(skill, skill.manipulate_point(),
                                        skill.manipulation_step_scale(),
                                        skill.engage_gripper,
                                        orient=not isinstance(skill, KettleGraspSkill))
            if skill.manipulation_done():
                self._enter_phase(RECEDE if skill.recede_after_manipulation else VERIFY)
            elif self._phase_steps >= cfg.manipulate_timeout:
                self._enter_phase(VERIFY)
            return action

        if self._phase == RECEDE:
            if self.sim.finger_opening < cfg.release_opening_threshold:
                return self._track_action(skill, skill.contact_point(), cfg.contact_step,
                                          "open")
            action = self._track(skill, skill.recede_point(), skill.recede_step_scale(), "open",
                                 next_phase=VERIFY, timeout=cfg.reach_timeout,
                                 stall_advances=True)
            if skill.recede_complete():
                self._enter_phase(VERIFY)
            return action

        if self._phase == VERIFY:
            # Verification is a state predicate on the *task joint*, never on elapsed time
            # or on EEF distance.
            if skill.complete():
                self._task_succeeded = True
                if skill.recede_after_manipulation:
                    self._finish_task()
                else:
                    self._enter_phase(RETREAT)
            else:
                self._abandon_or_retry()
            return self._hold(self._gripper(skill.approach_gripper), skill)

        if self._phase == RETREAT:
            if cfg.retreat_to_home:
                # Lift straight up before travelling home, so the arm does not drag its
                # wrist sideways across the stove (which launches the kettle).
                if self._phase_steps <= cfg.retreat_lift_steps:
                    if self._retreat_lift_target is None:
                        self._retreat_lift_target = self.sim.eef_pos + np.array([0.0, 0.0, cfg.retreat_lift])
                    action = self._track_eef(skill, self._retreat_lift_target,
                                             cfg.free_space_step, "open")
                    return action
                action = self._track_eef(skill, self._home_pos, cfg.free_space_step, "open",
                                         recover_posture=True)
            else:
                action = self._track_action(skill, skill.retreat_point(), cfg.free_space_step, "open")
            orientation_required = (skill.desired_orientation() is not None
                                    or cfg.keep_gripper_down)
            righted = (not orientation_required
                       or self._last_orientation_error <= cfg.tilt_recovery_threshold)
            if ((self._last_position_error < cfg.position_tolerance and righted)
                    or (self._stalled() and righted)
                    or self._phase_steps >= cfg.retreat_timeout):
                self._finish_task()
            return action

        return self._hold(self._gripper("open"))

    def _track_action(self, skill: KitchenSkill, tool_point: np.ndarray, step_scale: float,
                      gripper_mode: str, recover_posture: bool = False,
                      orient: bool = True) -> np.ndarray:
        # Always record how far the tool point is from where the tool *measurably* is, so
        # a plan built on the nominal gripper axis cannot silently disagree with the sim.
        self._last_tool_error = float(np.linalg.norm(np.asarray(tool_point, dtype=np.float64)
                                                     - skill.tool_pos()))
        return self._track_eef(skill, skill.eef_target(tool_point), step_scale, gripper_mode,
                               recover_posture=recover_posture, orient=orient)

    def _track_initial_orientation(self, desired_frame: np.ndarray) -> np.ndarray:
        """Rotate to the front-facing horizontal frame without reaching toward a task."""
        target = self._initial_orientation_target
        self._last_target = target.copy()
        self._last_position_error = float(np.linalg.norm(target - self.sim.eef_pos))
        self._last_tool_error = 0.0
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        # Counter the Cartesian drift induced by a large wrist reorientation and preserve
        # one fingertip-length of clearance from the cabinet throughout the turn.
        action[:3] = position_action(self.sim, target, self.cfg.contact_step)
        action[3:6] = rotation_action(self.sim, desired_frame, self.cfg.max_rotation_step)
        action[6] = self._gripper("open")
        return action

    def _track_eef(self, skill: Optional[KitchenSkill], target: np.ndarray, step_scale: float,
                   gripper_mode: str, recover_posture: bool = False,
                   orient: bool = True) -> np.ndarray:
        self._last_target = np.asarray(target, dtype=np.float64)
        self._last_position_error = float(np.linalg.norm(target - self.sim.eef_pos))
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[3:6] = self._orientation_action(skill) if orient else 0.0
        orientation_required = (skill is not None
                                and (skill.desired_orientation() is not None
                                     or self.cfg.keep_gripper_down))
        if (orientation_required and recover_posture
                and self._last_orientation_error > self.cfg.tilt_recovery_threshold):
            step_scale = step_scale * self.cfg.tilt_recovery_position_scale
        action[:3] = position_action(self.sim, target, step_scale)
        action[6] = (skill.gripper_command(gripper_mode)
                     if skill is not None else self._gripper(gripper_mode))
        return action

    def _orientation_action(self, skill: Optional[KitchenSkill]) -> np.ndarray:
        """Drive a side-grasp frame, or optionally level an unoriented skill.

        Side-grasp orientation is part of the skill geometry and is therefore always
        active.  ``keep_gripper_down`` retains its older meaning only for skills without
        a complete task-specific frame.
        """
        if skill is None:
            return np.zeros(3)
        desired_frame = skill.desired_orientation()
        if desired_frame is not None:
            self._last_orientation_error = orientation_error(self.sim, desired_frame)
            return rotation_action(self.sim, desired_frame, skill.rotation_step_scale())

        desired = skill.approach_axis()
        current = self.sim.eef_approach_axis
        cross = np.cross(current, desired)
        sin_theta = float(np.linalg.norm(cross))
        theta = float(np.arctan2(sin_theta, float(np.dot(current, desired))))
        # Recorded even when levelling is disabled: the tilt is what silently invalidates
        # a tool offset built on the nominal axis, so it must always be observable.
        self._last_orientation_error = theta
        if not self.cfg.keep_gripper_down or sin_theta < 1e-6:
            return np.zeros(3)
        rotvec = cross / sin_theta * theta
        return limit_to_box(rotvec / MAX_ROTATION_DISPLACEMENT, self.cfg.max_rotation_step)

    def _track(self, skill: KitchenSkill, tool_point: np.ndarray, step_scale: float,
               gripper_mode: str, next_phase: str, timeout: int,
               stall_advances: bool, recover_posture: bool = False,
               orient: bool = True) -> np.ndarray:
        action = self._track_action(skill, tool_point, step_scale, gripper_mode,
                                    recover_posture=recover_posture, orient=orient)
        self._reach_error_history.append(self._last_position_error)
        arrived = self._last_position_error < self.cfg.position_tolerance
        # Do not leave a reaching phase while the wrist is still being righted: a stalled
        # EEF during posture recovery is the recovery working, not the arm being blocked.
        orientation_required = (skill.desired_orientation() is not None
                                or self.cfg.keep_gripper_down)
        recovering = (orientation_required and recover_posture
                      and self._last_orientation_error > self.cfg.tilt_recovery_threshold)
        # "Blocked" has to mean the arm physically cannot move any further, not merely that
        # the error stopped shrinking: several skills legitimately end their reach 10-15 cm
        # short because cabinetry stops the arm, and those are still in useful contact.
        blocked = (stall_advances
                   and self._last_position_error <= self.cfg.reach_failure_tolerance
                   and (self._stalled() or self._reach_blocked())
                   and not recovering)
        if (arrived and not recovering) or blocked:
            self._enter_phase(next_phase)
        elif self._phase_steps >= timeout:
            if self._last_position_error > self.cfg.reach_failure_tolerance:
                # Out of budget and still far away: the reach never got there.  Advancing
                # would run the whole contact sequence in mid-air, which is exactly what
                # used to happen at low control_steps, where the arm covers far less ground
                # per step than the phase budget assumed.
                self._abandon_or_retry()
            else:
                self._enter_phase(next_phase)
        return action

    def _hold(self, gripper: float, skill: Optional[KitchenSkill] = None) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[3:6] = self._orientation_action(skill)
        action[6] = gripper
        return action

    # -- introspection -------------------------------------------------------------------
    def get_diagnostics(self) -> Dict[str, Any]:
        joint_target_error = float(np.linalg.norm(
            self.sim.data.ctrl[:7] - self.sim.data.qpos[:7]))
        robot = self.sim.kitchen.robot_env
        active_leak = (float(robot.controller.gain)
                       if isinstance(robot.controller, IKTargetLeakController) else 0.0)
        kettle_skill = (self._skill if isinstance(self._skill, KettleGraspSkill) else None)
        microwave_skill = (self._skill
                           if isinstance(self._skill, MicrowavePullSkill) else None)
        microwave_contacts = (sorted(microwave_skill.contacting_fingers())
                              if microwave_skill is not None else [])
        return {
            "action_contract": "gymnasium-robotics-1.2.0-cartesian-delta-7d",
            "control_steps": int(robot.control_steps),
            "ik_target_leak": active_leak,
            "selected_subtask": self._task,
            "controller_phase": self._phase,
            # Report a manipulation stage only after the controller actually enters it.
            # The kettle skill's planned stage is always transport, which was misleading
            # while the live controller was still in APPROACH.
            "manipulation_stage": (KETTLE_TRANSPORT
                                   if self._phase == KETTLE_TRANSPORT else None),
            "kettle_contacting_fingers": (sorted(kettle_skill.contacting_fingers())
                                           if kettle_skill is not None else []),
            "kettle_grasp_retained": (kettle_skill.grasp_retained()
                                       if kettle_skill is not None else None),
            "microwave_contacting_fingers": microwave_contacts,
            "microwave_contact_depth": (microwave_skill.contact_depth()
                                        if microwave_skill is not None else None),
            "microwave_grasp_captured": (bool(microwave_skill._capture_confirmed)
                                          if microwave_skill is not None else None),
            "microwave_grasp_retained": (
                bool(microwave_skill._capture_confirmed
                     and microwave_contacts
                     and microwave_skill.grasp_contact_min_opening
                     <= self.sim.finger_opening
                     < microwave_skill.grasp_ready_opening)
                if microwave_skill is not None else None),
            "microwave_radial_bias": (float(microwave_skill.radial_bias)
                                       if microwave_skill is not None else None),
            "microwave_handle_position": (microwave_skill.touch_point().tolist()
                                           if microwave_skill is not None else None),
            "microwave_contact_position": (microwave_skill.contact_point().tolist()
                                            if microwave_skill is not None else None),
            "microwave_tool_position": (microwave_skill.tool_pos().tolist()
                                         if microwave_skill is not None else None),
            "target_position": None if self._last_target is None else self._last_target.tolist(),
            "position_error": float(self._last_position_error),
            "tool_error": float(self._last_tool_error),
            "orientation_error": float(self._last_orientation_error),
            "initial_orientation_complete": bool(self._initial_orientation_complete),
            "finger_opening": float(self.sim.finger_opening),
            "ik_joint_target_error": joint_target_error,
            "task_distance": (float(self.sim.task_distance(self._task))
                              if self._task is not None else None),
            "subtask_complete": bool(self._skill.complete()) if self._skill is not None else None,
            "retry_count": int(self._retry_count),
            "phase_steps": int(self._phase_steps),
            "completed_order": list(self._completed_order),
            "abandoned": list(self._abandoned),
            "phase_failures": int(self._phase_failures),
            "steps_per_subtask": dict(self._task_step_counts),
        }
