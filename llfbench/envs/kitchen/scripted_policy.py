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
  stepping the env and reading the first finger joint.

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

from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from gymnasium_robotics.envs.franka_kitchen.kitchen_env import (
    BONUS_THRESH,
    OBS_ELEMENT_GOALS,
    OBS_ELEMENT_INDICES,
)
from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, get_site_xmat, get_site_xpos
from gymnasium_robotics.utils.rotations import euler2quat


# ---------------------------------------------------------------------------------------
# Environment constants (mirrored from gymnasium_robotics.envs.franka_kitchen.franka_env)
# ---------------------------------------------------------------------------------------

#: Scale of the *internal* Cartesian plan.  The policy reasons in end-effector space and
#: only converts to joints as its last act (:meth:`ScriptedKitchenPolicy.get_action`), so
#: these still set how far one policy step may ask the gripper to travel and turn.  They are
#: no longer an environment contract: Gymnasium-Robotics 1.2.1 removed the Cartesian IK
#: action space these once mirrored.
MAX_CARTESIAN_DISPLACEMENT = 0.2   # metres of EEF travel commanded by a unit plan step
MAX_ROTATION_DISPLACEMENT = 0.5    # radians of EEF rotation commanded by a unit plan step

#: Width of the plan the skills produce, ``[dx, dy, dz, drx, dry, drz, gripper]``.
CARTESIAN_ACTION_DIM = 7

#: Width of the env's action space: one normalized velocity per robot joint, seven arm
#: hinges then two finger slides (``FrankaRobot.action_space``).
ACTION_DIM = 9

#: ``act_rng`` in ``FrankaRobot``: a unit action means this many rad/s.  This is the
#: environment's own normalization, not a tuning knob.
ACTION_VELOCITY_RANGE = 2.0

#: Fastest each joint can actually be driven, in rad/s: the position actuator's force limit
#: divided by the joint's damping (``forcerange / dof_damping`` in the Franka assets).  The
#: arm joints damp at 100 N·m·s/rad against an 87 N·m limit and the forearm at 10 against 12,
#: so the action box is far wider than the arm can follow -- commanding more than 0.44
#: (joints 1-4) or 0.60 (joints 5-7) of the normalized range simply saturates.
#:
#: Measured by commanding a unit action on each joint in turn and differencing ``qpos``:
#: 0.85, 0.68, 0.91, 0.91, 1.11, 1.15, 1.13 rad/s (joint 2 is the one fighting gravity).
#: This is what the previously suspected "velocity servo under-delivers by ~0.5" really was.
JOINT_SPEED_LIMIT = np.array([0.87] * 4 + [1.2] * 3 + [0.7] * 2)

#: MuJoCo names 1.2.1 uses for the robot.  1.2.0 called the site ``EEF`` and the joints
#: ``robot:jointN``; both were renamed when the Cartesian IK controller was dropped.
EEF_SITE = "end_effector"
ARM_JOINTS = tuple(f"robot:panda0_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("robot:panda0_finger_joint1", "robot:panda0_finger_joint2")
GRIPPER_ACTUATORS = ("r_gripper_finger_joint", "l_gripper_finger_joint")
FINGER_BODIES = ("panda0_leftfinger", "panda0_rightfinger")

#: Travel of one finger slide, from ``model.jnt_range``.  A gripper plan value of +1 means
#: fully open and -1 fully closed.
FINGER_RANGE = 0.04

#: Distance from the end-effector site to the fingertip pads, along the gripper approach
#: axis.  Measured from ``data.geom_xpos`` of the fingertip pad geoms: they sit 0.0429 ahead
#: of the site with a 0.0091 half-size, so the pad face is 0.052 out.
#:
#: 1.2.0's ``EEF`` site was 0.108 back, at the wrist; 1.2.1's ``end_effector`` sits between
#: the fingers.  The site's axes themselves did not change convention -- +z still points at
#: the fingertips and +y still separates them -- so only this offset needed re-measuring.
#: What changed alongside it is the arm's home pose, which now starts pointing forward into
#: the kitchen rather than straight down.
FINGERTIP_OFFSET = 0.052

#: Nominal tool axis for skills that do not request a task-specific grasp orientation.
NOMINAL_APPROACH_AXIS = np.array([0.0, 0.0, -1.0])

#: Room-to-cabinet direction in the fixed Franka Kitchen scene.  Together with world-up
#: (the slide handle) and world-right (the slide joint), this is the third member of the
#: requested mutually orthogonal slide-grasp frame.
FRONT_APPROACH_AXIS = np.array([0.0, 1.0, 0.0])
VERTICAL_HANDLE_AXIS = np.array([0.0, 0.0, 1.0])

#: Task names are the env's goal keys, which 1.2.1 spells with spaces and which no longer
#: coincide with the MuJoCo joint names (still underscored).  1.2.1 also merged the
#: left/right burner and hinge variants 1.2.0 scored separately, so the goal for one task
#: can span several joints -- ``light switch`` is ``[light_switch, light_joint]``.  Distances
#: are therefore read through ``OBS_ELEMENT_INDICES`` rather than by joint name.

#: MuJoCo site each skill reaches for, keyed by task name.
TASK_SITES = {
    "bottom burner": "knob2_site",
    "top burner": "knob4_site",
    "light switch": "light_site",
    "slide cabinet": "slide_site",
    "hinge cabinet": "hinge_site2",
    "microwave": "microhandle_site",
    "kettle": "kettle_site",
}

#: The MuJoCo joint each skill physically drives.  Task names and joint names were the same
#: string in 1.2.0; 1.2.1 respells the task names and merges the left/right variants, so the
#: two have to be related explicitly.  For the burners the joint also genuinely differs from
#: the one the env scores -- MuJoCo equality constraints couple ``knob_Joint_N`` to the scored
#: burner slide -- which is what this mapping originally existed for.
TASK_MANIPULATED_JOINTS = {
    "bottom burner": "knob_Joint_2",
    "top burner": "knob_Joint_4",
    "light switch": "light_switch",
    "slide cabinet": "slide_cabinet",
    "hinge cabinet": "right_hinge_cabinet",
    "microwave": "microwave",
    "kettle": "kettle",
}

#: Which component of ``OBS_ELEMENT_GOALS[task]`` belongs to the driven joint.  Only the
#: hinge cabinet needs one: 1.2.1 scores ``[left_hinge_cabinet, right_hinge_cabinet]`` under
#: a single task, and the skill drives the right-hand door.
TASK_GOAL_INDEX = {"hinge cabinet": 1}

#: Default execution order.  ``slide cabinet`` comes first because its door blocks the
#: arm's path to the oven knobs and the light switch (measured: the light-switch skill
#: fails outright when the slide cabinet is still closed, and succeeds after it is open).
#: ``kettle`` is ahead of ``light switch`` because the arm's path to the switch can knock
#: the reset kettle away from the handle-grasp corridor.  Once dragged into its goal, the
#: tuned light path leaves it inside the goal threshold.
DEFAULT_TASK_ORDER = [
    "slide cabinet",
    "kettle",
    "light switch",
    "microwave",
    "hinge cabinet",
    "bottom burner",
    "top burner",
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

#: Phases where a folded arm may be reshaped toward the configuration it resets in.  See
#: :meth:`ScriptedKitchenPolicy._solve_arm_delta`.
#:
#: Only the transit home, even though MOVE_TO_PRECONTACT is where most of the folding
#: happens (132 of 164 self-colliding steps) and is free-space travel by the same argument.
#: Including it works, and costs more than it buys: self-collision over the 24-seed harness
#: falls 164 steps to 48, but the harness falls with it, 21/24 to 20/24 and a net +124
#: steps, because a reach that is reshaped mid-flight arrives somewhere its skill did not
#: plan for.  The transit home has no such consumer -- its whole purpose is to arrive at
#: the reset pose, which is exactly what the bias pulls toward -- so there the two agree.
TRANSIT_PHASES = frozenset({ORIENT_FORWARD})
#: Phases in which a joint pinned on its bound gets the escape term in `_solve_arm_delta`.
#: The same set as `TRANSIT_PHASES`, and it must stay that way: the kettle's centring after
#: the slide-cabinet park crawls at 2 mm per step for 20-40 steps with joint 2 on its
#: -1.7628 bound (seeds 264, 389), which looks like exactly the jam the escape exists for,
#: and adding MOVE_TO_PRECONTACT here to free it measured 16 of 16 seeds at the 1000-step
#: limit with 1-2 of 4 tasks -- the kettle reach never converged (220-290 steps) because its
#: stand-off *is* a pose with joint 2 on that bound, and the escape pushes away from the
#: only configuration that reaches it.  A reach that needs a bound is not a jam.
JAM_RESCUE_PHASES = frozenset({ORIENT_FORWARD})
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
    #: How far the hand may travel straight to the next subtask's stand-off before it is
    #: routed via the home pose instead.  0 always parks at home between subtasks, `inf`
    #: never does.  See :meth:`ScriptedKitchenPolicy._direct_transition_safe`.
    #:
    #: Zero, i.e. always park, because every setting that actually skips a park costs
    #: episodes.  Measured over 8 seeds at 0.0 against 0.8:
    #:
    #:   ================  =========  =========
    #:   task order        0.0        0.8
    #:   ================  =========  =========
    #:   fixed             8/8, 452   8/8, 378
    #:   randomized        7/8, 505   3/8, 1000
    #:   microwave first   7/8, 427   0/8, 1000
    #:   ================  =========  =========
    #:
    #: The fixed order alone suggests this is free -- 16% fewer steps at no cost -- and it is
    #: not: that order is simply the one whose hops happen to be clear.  Distance is a poor
    #: proxy for a safe path.  The microwave-to-kettle hop is well under 0.8 m and goes
    #: straight through the door this expert has just opened, and skipping the park there
    #: loses the episode outright.  Doing this properly needs a transit path that clears the
    #: obstacles -- what :meth:`KettleGraspSkill.transit_point` already does for one skill --
    #: rather than a radius.  The knob is kept so that cost can be re-measured against such a
    #: path, and so a caller who only ever runs the fixed order can buy the 16%.
    direct_transition_max_distance: float = 0.0
    #: Absolute height, in metres, of the generic overhead transit waypoint, or None to
    #: disable it.  None: the route measures 5/8 against 8/8 for parking at home, for the
    #: reasons recorded in :meth:`KitchenSkill.transit_point`.
    overhead_transit_clearance: Optional[float] = None
    #: Planar distance from the stand-off beyond which that overhead route is used.  Below
    #: it the reach is already local and is made directly.
    overhead_transit_radius: float = 0.30
    #: Whether to repeat that reorientation *between* subtasks as well.  True sends the hand
    #: back to the home pose and the front-facing frame after every subtask before the next
    #: one starts; False lets the next skill set out from wherever the last one ended.
    reorient_between_tasks: bool = True
    #: Steps ALIGN may spend before the skill stops waiting for a frame it is not going to
    #: reach and lets the ordinary approach carry on.  Nothing else bounds this phase: its
    #: exit test is the frame error alone, so a wrist that converges to just outside the
    #: tolerance holds the task -- and the whole episode after it -- forever.  Measured on
    #: combined seed 2, the microwave lost its grasp with the door 68% open and then sat in
    #: ALIGN from step 487 to the 1000-step limit, orientation pinned at 0.280 against its
    #: 0.23 exit tolerance, 513 steps spent without moving.  The approach tracker corrects
    #: orientation and position in one solve, so handing the frame back to it is a real
    #: attempt, not a surrender; the task budget still bounds the task itself.
    align_budget: int = 80

    #: Ending ALIGN early when it *looks* stalled is the third variation on that idea this
    #: file has measured and the third to lose tasks.  Seed 7's kettle arrives from the light
    #: switch at a pose the wrist cannot reach the grasp frame from: the frame error falls
    #: 0.411 to 0.222 in eleven steps against the 0.100 this skill exits at, and then the
    #: tool holds position to three decimals for seventy more until this budget latches
    #: `_align_exhausted` -- after which the approach succeeds within five steps.  Seventy
    #: idle steps is a fifth of the episode and the saving is real, but taking it is not:
    #: giving up as soon as both the frame error and the tool are provably static (window
    #: 12, epsilon 0.01, 5 mm) hands APPROACH a frame it cannot close on, and the hand goes
    #: back to pressing the kettle's bail bar -- seed 7 from 4 such steps to 61 and its
    #: shoving from 0.113 m to 0.366 m, with the harness unmoved at 20/24.  The arm is idle
    #: because it is stuck, and letting it proceed sooner only makes it stuck against the
    #: kettle.  See also `orient_forward_budget`.
    #: Steps ORIENT_FORWARD may spend before giving up and selecting a task anyway.  A
    #: converging reorientation from the reset pose takes about 15; between subtasks it can
    #: start from a pose it never reaches the target from, and there is nothing else running
    #: to notice.  See the phase itself for the episode this cost.
    #:
    #: The budget looks like the wrong instrument and is not.  The phase has two gates --
    #: yaw and clearance -- corrected by one arm, so turning drags the tool out of one while
    #: reaching turns the wrist out of the other, and they can settle where neither is met:
    #: measured on seed 7's ``light switch -> kettle`` park, clearance is inside at phase
    #: step 24 (0.100 against 0.100) with yaw at 0.219, yaw is inside by step 42 (0.149)
    #: with clearance back out at 0.130, and both then sit still for the last twenty steps.
    #: Ending the phase there is the obvious saving and it measures *worse*: a no-progress
    #: guard on the excess over both tolerances (window 12, epsilon 0.01) takes the kettle
    #: from 23/24 seeds to 21/24 and the harness mean from 3.79 to 3.71.  The steps after
    #: the gates stop closing are still improving the pose the next skill reaches from,
    #: which no exit condition here is measuring.  Do not re-add the guard.
    orient_forward_budget: int = 60
    #: Transitions whose park is driven in *joint space* from its first step: the seven arm
    #: joints are servoed straight to the reset configuration (`_home_qpos`, joint 2 lifted
    #: off its bound by `park_joint_bound_margin`) instead of tracking the Cartesian home
    #: target with the DLS solver.  The exit gates are unchanged.
    #:
    #: What this buys is the *branch* the next reach starts from, and for the light switch
    #: that branch decides whether the lever is held.  Measured over seeds 0-499 with a
    #: test.py-faithful census: the wrist at the light-switch stand-off lands in one of two
    #: configurations for the same tool pose, joint 5 about -0.45 (joint 7 about +0.6) or
    #: joint 5 about +0.2 (joint 7 about +0.15).  Every one of the 11 seeds that needed a
    #: second grip captured in the second one and every one-grip seed in the first.  From
    #: the second, the sweep along the lever's chord is nearly singular for the wrist: the
    #: base joint runs at its 0.87 rad/s ceiling through 1.6 rad while the lever moves
    #: 0.16 rad, the wrist rolls 0.4 rad, the pads walk 3 cm down the lever and it escapes
    #: at -0.20 rad (seeds 106, 428 traced step by step).  Which branch the reach falls
    #: into is decided by where the park leaves the arm: from the reset configuration the
    #: reach lands in the good branch 127 times out of 127, while the Cartesian park after
    #: the slide cabinet exits at its loose yaw tolerance with joint 5 at +0.3..+0.5 and the
    #: reach then keeps that sign 9 times in 108.  A joint-space park makes the start
    #: configuration the reset one by construction.
    #:
    #: Only these two.  ``("slide cabinet", "kettle")`` was tried because the kettle reach
    #: after that park runs 28-36 steps on a few seeds (60, 75, 224, 406) and the joint park
    #: takes it to 3-5 there; over all 124 such episodes it is a wash (kettle reach 5.8 ->
    #: 6.2 steps mean, episode 289.2 -> 288.6) and the joint park itself times out on 6 of
    #: them (0 before), so it is not kept.  The from-start joint park pays where the *next
    #: skill's* branch depends on it, and only there.
    park_joint_transitions: FrozenSet[Tuple[str, str]] = frozenset({
        ("slide cabinet", "light switch"),
        ("kettle", "light switch"),
        ("light switch", "microwave"),
    })
    #: Steps a between-subtask Cartesian park may run with either exit gate still open
    #: before it switches to the same joint-space homing.  The Cartesian tracker corrects
    #: two gates with one arm and can walk 1.4 m to close a 0.3 m error, or settle on the
    #: wrong IK branch (seed 106's park before the microwave: 62 steps, yaw ringing
    #: 0.46-0.53 against 0.25, joint 1 at -1.17 where the grasp needs +1.68); a joint target
    #: cannot wander and picks the branch by construction.  0 disables the fallback.
    #:
    #: OFF, because measured at 15 over seeds 0-499 it is a net loss.  The parks it was
    #: written for no longer time out once the light switch is entered from the reset
    #: branch (0/132 and 0/132 against 4 and 2), so it has nothing to fix there, and on
    #: the long parks that *were* converging it replaces a Cartesian tracker heading
    #: straight for the tool target with a joint interpolation that only satisfies the
    #: gates once every joint is home: slide cabinet -> kettle 30.1 -> 35.3 steps with 8
    #: timeouts, slide cabinet -> microwave 28.1 -> 32.5 with 12, kettle -> microwave
    #: 25.4 -> 29.3 with 9, 31 timeouts in all against 6, and the suite mean 294.1 -> 296.3.
    #: The from-start parks in `park_joint_transitions` are the whole benefit.
    park_joint_fallback_steps: int = 0
    #: Largest per-joint distance from the joint-space home target, in radians, at which a
    #: joint-space park may exit; the Cartesian yaw/clearance gates must pass as well.  The
    #: gates alone are loose for these transitions (0.40 rad, 0.42 m), so the park was
    #: releasing the arm part-way home: the light switch reached from the kettle's joint
    #: park took 31.8 steps of reach against 16.4 from the reset configuration itself.
    #: 0 or less disables the test and leaves the gates in charge.
    #:
    #: Applied to the transitions in `park_joint_exit_transitions`.  Measured on twelve
    #: seeds: after the kettle, going all the way home costs the park 6 steps and saves 12
    #: on the reach and 5-10 on the sweep (seeds 1, 33, 215, 315: -14 to -20 each).
    #:
    #: The slide cabinet -> light switch park is *not* in the set, and it took three suite
    #: runs to place it.  Its gate exit at (0.6, 0.42) releases the arm near the
    #: light-switch stand-off after ~15 steps and is the cheapest option, but it releases
    #: it from a slightly different configuration each time, and on seed 265 that
    #: configuration reached the lever in a marginal wrist branch (joint 5 at +0.53 at
    #: capture, between the good +0.25 and the bad +0.67) and lost it at -0.355 rad -- one
    #: regrip in 500.  Going all the way home instead (this exit) makes the start identical
    #: on every seed but lands in a third branch (joint 5 +0.05..+0.18, frame 0.04 rad at
    #: capture) that is slower and froze seed 153's approach for a stall retry: suite mean
    #: 270.3 -> 272.5, max 327 -> 366.  A looser 0.30 is worse than either (light-switch
    #: tasks of 96-153 steps on seeds 106, 164, 342, 428).  What holds is tightening that
    #: park's *gates* to (0.25, 0.25) -- it exits 5-13 steps later than at (0.6, 0.42), from
    #: a consistent point, 0 regrips on the seeds that mattered; see the table.
    park_joint_exit_tolerance: float = 0.15
    park_joint_exit_transitions: FrozenSet[Tuple[str, str]] = frozenset({
        ("kettle", "light switch"),
        ("light switch", "microwave"),
    })
    #: How far joint 2 is lifted off its -1.7628 bound in the joint-space home target.  The
    #: reset pose is itself a jammed pose (joint 2 at -1.741), and a target on the bound
    #: hands the next reach an arm damped least squares cannot walk off it from.
    park_joint_bound_margin: float = 0.05
    #: How near the home pose ORIENT_FORWARD must get before it releases the arm to the
    #: next subtask.  Separate from `position_tolerance` because it is not a precision
    #: requirement: nothing is being grasped, and the phase exists to put the arm somewhere
    #: the reach can set out from.  It is the *travel* toward home that the reach downstream
    #: depends on -- turning the wrist in place instead measures 1/8 randomized and 0/8 on
    #: the fixed order, exactly as bad as deleting the phase -- so the question is how much
    #: of that travel is needed, not whether it can be skipped.
    orient_forward_position_tolerance: float = 0.015
    #: Steps over which the *opening* park (the one at episode start, with no previous
    #: subtask) must have moved less than `opening_park_rest_distance` for it to count as
    #: finished with the yaw inside tolerance and the clearance inside three times its
    #: tolerance.
    #: 0 disables the test.
    #:
    #: The opening park times out on 17 of seeds 0-499 with the arm motionless: the yaw is
    #: inside at 0.016-0.025, the clearance sits at 0.020-0.027 m against the 0.015 above
    #: and never moves, and the servo action is a constant gravity-droop hold from step 6
    #: to step 61.  There is nothing left for the IK to command -- the last 5-12 mm are the
    #: servo's deadband -- so the 55 remaining steps are pure loss.  The census is bimodal:
    #: 483 seeds finish this park in under 20 steps and none in 20-58, so a test that fires
    #: only on an arm that has provably stopped cannot cut a converging park short.
    #: Between-subtask parks are excluded on purpose: the source records that a no-progress
    #: guard there loses kettles (the pose keeps improving after the gates stop closing).
    opening_park_rest_steps: int = 6
    opening_park_rest_distance: float = 0.001
    #: Let a skill whose ALIGN has just passed its frame test start the approach from
    #: wherever inside `alignment_tolerance` it aligned, instead of first satisfying the
    #: tighter `precontact_tolerance` again.  See the stand-off block in
    #: `_compute_reactive_action` for the one-step ALIGN/MOVE alternation this removes.
    align_exit_to_approach: bool = True
    #: Widest insertion cone the early start may begin from: lateral offset from the
    #: insertion line over the depth still to travel; 0 or less disables the test.
    #:
    #: Off.  The early start's whole value is that the tool may set out from *beside* the
    #: line -- the reach arrives next to the stand-off, not behind it -- so any bound on
    #: the lateral offset, whether a fixed 2x `precontact_tolerance` or this cone at 0.5,
    #: refuses nearly every early start: both measured the light switch at +12 to +14
    #: steps per episode (seeds 0, 15, 58, 480 each +20-30) for the one steep-from-below
    #: start on seed 93 they would have caught.  That start only arises from the later
    #: slide-cabinet -> light-switch park exit that was being tried at the time.
    approach_cone_ratio: float = 0.0
    #: Per-transition overrides for the two conditions ORIENT_FORWARD exits on, keyed by
    #: ``(subtask just finished, subtask about to start)`` and holding
    #: ``(yaw tolerance, clearance tolerance)``.  A pair not in the table, and the park at
    #: episode start where there is no previous subtask, fall back to `yaw_tolerance` and
    #: `orient_forward_position_tolerance` above.
    #:
    #: These apply to the *park only*.  The yaw value here is deliberately not the skill's
    #: grasp tolerance: `yaw_tolerance` doubles as the fallback in
    #: :meth:`KitchenSkill.orientation_tolerance_value`, so routing the transition value
    #: through that would let a loose transition quietly loosen the grasp after it.
    #:
    #: All twelve ordered pairs are listed even though the values currently vary only with
    #: the destination, because the point of the table is to be tuned per pair: how much
    #: park a transition needs depends on where the arm is coming *from* as much as where it
    #: is going, and a pair that turns out to need its own number has somewhere to go.
    #:
    #: Sizing caution: on seed 0 the three between-subtask parks open at clearances of
    #: 0.385, 0.800 and 0.592, so a clearance tolerance of 0.42 releases some parks on their
    #: first step -- that is not a loosened park but a skipped one, which measured 1/8
    #: randomized when applied globally.
    orient_forward_transition_tolerances: Dict[Tuple[str, str], Tuple[float, float]] = field(
        default_factory=lambda: {
            # (from, to): (yaw tolerance, clearance tolerance)
            ("microwave", "kettle"): (0.15, 0.25),
            # 0.10 here was unreachable, and an unreachable park tolerance is not a strict
            # one -- it is a park that always times out.  The two gates are corrected by one
            # arm, so turning drags the tool out of the clearance and reaching turns the
            # wrist out of the yaw, and on this transition they settle at yaw 0.136-0.150
            # against 0.150 and clearance 0.128-0.135 against 0.100: the yaw arrives, the
            # clearance never does, and `orient_forward_budget` ends the phase mid-
            # oscillation with joint 6 pinned at its 2.1127 bound.  The kettle's ALIGN then
            # cannot reach the grasp frame at all -- 82 steps with the frame error flat at
            # 0.222 and no self-collision -- and grasps at 0.13 rad of skew, which works
            # loose after six steps of push.  That is the kettle being nudged rather than
            # carried: on seed 7, 0.089 m of carry ending at a task distance of 0.291
            # against a 0.300 threshold.
            #
            # 0.14 is inside the equilibrium, so the park converges and exits ~20 steps
            # early instead of timing out.  Seed 7 goes 3/4 to 4/4 with 0.372 m of carry and
            # a final distance of 0.173, and the suite median episode falls 282 to 269 with
            # 21/24 four-task episodes and the mean 3.83 unchanged.
            #
            # The wrist jam invites a different fix -- swap to the other jaw-flip
            # representative of the same grasp when a joint is pinned and ALIGN has stopped
            # closing.  It works on its own terms (seed 7 carries 0.415 m) and is the wrong
            # layer: it costs three seeds their microwave, 21/24 down to 19/24, in every
            # variant tried, and applied *with* this park fix it takes seed 7 back to 2/4.
            # Fix the pose the arm arrives in, not the wrist it arrives with.
            ("light switch", "kettle"): (0.25, 0.14),
            ("slide cabinet", "kettle"): (0.15, 0.10),

            ("kettle", "microwave"): (0.15, 0.35),
            # Loosened from (0.25, 0.25), which was unreachable and therefore not a
            # strict park but a park that always timed out.  Measured on seed 106: the park
            # before the microwave runs its whole 60-step budget and gives up
            # mid-oscillation, with the yaw ringing 0.464 <-> 0.527 against 0.25 and the
            # clearance sitting at 0.252-0.265 against 0.25 -- just outside, never inside.
            # The two gates are corrected by one arm, so turning drags the tool out of the
            # clearance and reaching turns the wrist out of the yaw; that equilibrium is
            # where the pair actually lives, and a tolerance below it can only expire.
            #
            # Set above the equilibrium the park does converge and exit early -- 62 steps
            # of timeout become 27 of convergence -- and seed 106's episode shortens:
            #
            #   (0.25, 0.25)  471 steps, park times out at 60
            #   (0.55, 0.25)  464          (0.45, 0.30)  452
            #   (0.55, 0.28)  461          (0.60, 0.35)  452
            #   (0.60, 0.30)  454          (0.70, 0.35)  446
            #
            # KEPT AT (0.25, 0.25) ANYWAY.  Every loosening trades seed 106 for the tail,
            # because a park released before it has converged hands the *next* seed the
            # badly-posed wrist instead.  Over all 500 seeds, all three settings hold
            # 500/500 and the distribution gets steadily worse:
            #
            #   tolerance      mean   sd    max   new slow seeds
            #   (0.25, 0.25)   294.1  28.7  471   --
            #   (0.70, 0.35)   295.6  35.3  533   144, 235, 295, 386, 292
            #   (0.55, 0.28)   296.4  37.2  562   88, 90, 382
            #
            # Matching the tolerance to the measured equilibrium rather than to seed 106
            # (0.55/0.28 rather than 0.70/0.35) does not help; it is slightly worse again.
            # This is the same trap as the light-switch lead angle: optimising the seed in
            # front of you moves the failure rather than removing it.
            #
            # Tightening does nothing at all -- (0.25, 0.10) measures 480 against 481 --
            # because an unreachable tolerance expires on its budget either way.  So no
            # value of this pair pays for itself, and the real fix, if one is wanted, has
            # to change the park's *equilibrium* rather than its threshold.
            ("light switch", "microwave"): (0.25, 0.25),
            ("slide cabinet", "microwave"): (0.15, 0.10),

            ("microwave", "light switch"): (0.40, 0.42),
            # The two parks below are driven in joint space (`park_joint_transitions`).  The
            # kettle one also waits for the joints to arrive (`park_joint_exit_transitions`),
            # so its gates are met at the reset configuration by construction.  The slide
            # cabinet one exits on these gates alone, so they decide *where along the joint
            # interpolation* the arm is released -- and that point decides the wrist branch
            # the light switch is reached in.  At (0.6, 0.42) it released after ~15 steps:
            # 107 of 108 episodes in the good branch and seed 265 in a marginal one that
            # lost the lever (see `park_joint_exit_tolerance`).  (0.40, 0.42) is the same
            # exit -- the clearance gate is the binding one.  (0.25, 0.25) releases 5-13
            # steps later and fixed seed 265 on a twelve-seed check, but over the suite it
            # costs +15 steps on every slide -> light episode (park +8, light switch +7:
            # mean 270.3 -> 272.8) and its own exit point starts seed 93's insertion 17 cm
            # below the line, which froze for a stall retry (406 steps).  Kept at
            # (0.6, 0.42): the cheapest exit, 107 of 108 in the good branch.  Before the
            # joint park, the loose 0.6 yaw here handed the reach a wrist in the wrong
            # branch on 9 of 108 episodes.
            ("kettle", "light switch"): (0.40, 0.42),
            ("slide cabinet", "light switch"): (0.6, 0.42),

            ("microwave", "slide cabinet"): (0.40, 0.42),
            ("kettle", "slide cabinet"): (0.40, 0.3),
            ("light switch", "slide cabinet"): (0.60, 0.42),
        })

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
    microwave_recede_distance: float = 0.01
    microwave_recede_tolerance: float = 0.0
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
    #: How close the fingertips must come to the slide handle before the jaws close.
    #:
    #: The class default this replaces was 0.05, which is a 5 cm ball around the handle
    #: site, and CONTACT_OR_GRASP fires the moment APPROACH is inside it.  Traced on
    #: randomized seed 2: the close command went out with the bar still 3.5 cm ahead of the
    #: pads, and the fingers shut to an opening of 0.0006 m -- on nothing.  The door still
    #: reached its goal, because a closed fist pushes a slide door perfectly well, so the
    #: environment scored the task and the miss was invisible to every success number.
    #:
    #: The opening is what tells the two apart: a grasp settles at the bar's thickness,
    #: a miss collapses to zero.  Over 8 randomized seeds, minimum opening during
    #: MANIPULATE by tolerance -- 0.05: mean 0.0094 with seeds 2 and 3 at 0.0006, i.e. two
    #: of eight closing on air; 0.03: 0.0129, worst seed 0.0059; 0.02: 0.0149, worst seed
    #: 0.0108, every seed holding the bar; 0.015: 0.0139, worst seed 0.0063.  Tightening
    #: past 0.02 starts costing grips again -- the approach has to stop inside a ball it
    #: cannot always reach, so it releases and retries -- which is why this is not simply
    #: as small as possible.
    slide_contact_tolerance: float = 0.02
    slide_manipulate_lookahead: float = 0.10
    #: Slide-only normalized Cartesian action cap.  With the environment's 0.2 m action
    #: scale, 0.25 requests at most 0.05 m of rightward EEF motion per policy action.
    slide_manipulation_step: float = 0.20
    #: Once the slide has reached its goal, open at the handle and back this far toward the
    #: room before beginning the normal between-task retreat.
    slide_recede_distance: float = 0.18
    #: Whether the slide cabinet's jaw-centring stage also tracks the grasp frame.  See the
    #: centring branch in `_compute_reactive_action`.
    slide_centering_orient: bool = True
    #: Rotation step for that frame tracking, in place of `max_rotation_step` (0.4), as the
    #: kettle's centring has (`kettle_centering_rotation_step`).  Not what fixed seed 480:
    #: the frame errors seen while centring are under 0.1 rad, which either step passes in
    #: one go, and the seed measured the same at 0.2 as at 0.4.  See
    #: `slide_centering_orient_lateral`.
    slide_centering_rotation_step: Optional[float] = 0.2
    #: Only track the frame while centring when the tool is within this distance of the
    #: approach line, in metres; farther out the stage translates only, as it always did.
    #:
    #: The frame tracking is for the last centimetres, where each translate-only step let
    #: the frame drift past the ALIGN threshold and cost a step of ALIGN.  Far from the
    #: line it is a hazard: seed 480 reaches the slide cabinet from the light switch 19 cm
    #: below and 6 cm past the stand-off, the centring stage climbs to the handle, and with
    #: the frame tracked over that climb the tool drifts 8 cm across the bar (jaw error
    #: 0.005 -> 0.085 in six steps), one pad lands on the door beside the handle and pushes
    #: against it for 30 steps until the stall retry backs out -- the "misses the handle,
    #: then re-grips" a viewer sees.  Translate-only over the same climb centres cleanly.
    #:
    #: 0.05.  With the reach-posture pull on this skill (since removed, see
    #: `SlideSkill.use_reach_posture`) 0.15 was the value: it fixed seed 480 (327 -> 272)
    #: while leaving every other slide cabinet byte-identical, where 0.10 altered seed 265's
    #: by five steps and the light switch after it regripped.  Without the pull, 0.15 is a
    #: catastrophe -- the reach from home runs 83-97 steps with the frame tracked from 15 cm
    #: out and stalls into a park retry on 23 of 138 episodes (suite mean 269.8 -> 289.9,
    #: max 653) -- because that reach now arrives beside the line rather than ahead of it,
    #: and orienting there swings the jaws the way it did on seed 480.  At 0.05 the same
    #: eight worst seeds run 34-36-step reaches with no retry, at 0.9 rad of roll -- but the
    #: stand-off flicker the tracking exists for is back (5.1 align/centre segments per
    #: episode against 0.9), because it rings with the tool 5-10 cm off the line.  0.10
    #: covers it (2.5 on the twelve seeds checked, 0.08 gives 3.8) with the reach still
    #: 34-36 steps and no retry.
    slide_centering_orient_lateral: float = 0.10
    #: After switching the light, release it completely and withdraw along the switch bar
    #: before selecting another subtask.
    light_recede_distance: float = 0.1
    light_recede_tolerance: float = 0.03
    #: Normalized gripper command used to hold the light-switch lever.  Read the same way as
    #: :attr:`kettle_grasp_command`: ``target = 0.02 * (command + 1)`` against a lever whose
    #: collision capsule is 0.021 m in radius, so -0.3 asks for 0.014 and squeezes the pads
    #: 7 mm into it.  The generic close of -1.0 asks for a *shut* gripper -- 21 mm inside the
    #: lever on each side -- and the servo spends all of it as grip force, which pops the
    #: capsule back out: that is the observed cycle of gripping, losing the switch partway
    #: through the slide, and re-gripping to finish it in two or three goes.
    #:
    #: A gentler hold than this is not better here, and this lever is not the kettle bar.
    #: The switch has to be *shoved* sideways against its own joint, not carried, so it
    #: needs preload the way the kettle needs restraint; -0.1 (0.018, a 3 mm preload)
    #: measured 8/8 but took 242 steps against 158 at -0.3.
    light_grasp_command: float = -0.30
    #: How far the hand backs straight out along the lever after losing its grip mid-sweep,
    #: in metres, before it is allowed to reposition.  0 disables the back-out.
    #:
    #: This is the "switches halfway and goes back" a viewer sees, and it is the *recovery*
    #: route, not the slip.  When the pads lose the lever the hand is at the far end of the
    #: sweep, deep past the switch; the stand-off it must return to is back and to the
    #: other side, and the straight line between them runs through the lever.  So the hand
    #: sweeps the switch back on its way out.  Traced on seed 158, the lever goes -0.227 ->
    #: -0.211 -> -0.170 -> -0.122 -> -0.069 -> -0.013 -> 0.000 over the fourteen steps after
    #: the release: every bit of the throw is undone before the second attempt begins, and
    #: the second attempt therefore starts from scratch.
    #:
    #: Backing out along the lever's own axis first leaves the switch where it was, so the
    #: next grip continues from the angle already won instead of restarting.  The axis
    #: matters: the jaws squeeze across the lever, so sliding along it releases without
    #: pushing, while any lateral move drags.
    #:
    #: OFF, because it works and cannot be afforded.  At 0.06 it does exactly what it says
    #: -- on seed 158 the lever holds at -0.225 through the whole recovery instead of being
    #: swept to 0.000, grips fall 2.33 -> 2.00, and the mean throw reaches -0.699 against a
    #: -0.69 goal -- and it still costs three episodes: seeds 68, 342 and 428 go 4/4 to 3/4,
    #: 497/500, every one of them by abandoning the *microwave* afterwards.  Ablation is
    #: unambiguous; with only this disabled all three return to 4/4 in 438, 448 and 458
    #: steps.
    #:
    #: It is the same coupling that defeats `light_manipulation_lead_angle`: moving where
    #: the light switch leaves the arm breaks the microwave on the seeds with the least
    #: headroom, and those three were already the three slowest episodes in the suite.
    #: Shortening it does not buy the difference -- 0.03 measures 2/6 on the six seeds
    #: involved, and 0.02 is below the arrival threshold below, so it simply never fires.
    #:
    #: Whoever removes the light switch's second grip has to do it without moving the pose
    #: the skill exits in.  Nothing that changes that pose has survived the suite yet.
    light_slip_backout: float = 0.0
    #: Let the wrist follow the lever's own axis while it is held, instead of holding the
    #: frame frozen at the angle the lever had when it was gripped.
    #:
    #: The lever is gripped *end-on*, so the jaws squeeze across it in one direction and it
    #: is free to slide along the other.  With the frame frozen, every radian the lever
    #: turns is a radian of mismatch between it and the pads, and the lever walks out of the
    #: jaws along that free axis.  Traced on seed 106 in the tool frame, the lever site's
    #: offset along the unconstrained axis grows 0.000 -> 0.030 m as the lever reaches
    #: -0.26 rad; the right pad drops at 0.023 m and the grip is gone at 0.030, at which
    #: point the jaws slam shut on nothing.  That is the mid-sweep drop, and it is why the
    #: switch takes two or three bites.
    light_track_lever_axis: bool = False
    #: Hold the wrist's *roll* about the approach axis during the light-switch sweep --
    #: keep the finger axis horizontal -- while leaving the swing that follows the lever
    #: free.  MANIPULATE otherwise re-targets the live orientation every step, so under the
    #: lever's load the roll is a random walk: 0.3-1.2 rad over a sweep on the one-grip
    #: seeds and 2.0-2.2 on seeds 33 and 285, where the wrist leaves the switch a jaw-flip
    #: away from front.  The park after it then accepts that flipped frame as equivalent
    #: (`equivalent_grasp_frame`) and exits in 9 steps, and the microwave's ALIGN spends
    #: 37-57 steps and a stall retry unwinding it.  Feedback on the roll component alone
    #: turns the random walk into a bounded error without fighting the lever's rotation.
    light_hold_roll: bool = True
    #: Lead angle, in radians, of the light switch's arc-following manipulation target.  The
    #: lever swings 0.69 rad in total, so at 0.70 the clamp to the travel remaining always
    #: binds and this is simply "aim at the goal angle".  That is the right answer here and
    #: a short lead is not: the sweep has to stay loaded, and 0.25 measures 228 steps
    #: against 98, 0.15 measures 173.  The cap is kept for levers with more travel than this
    #: one.  See :meth:`LightSwitchSkill.manipulate_point`.
    #:
    #: SHORTENING THIS IS A TRAP, and the trap is that it looks like it works.  The chord of
    #: a 0.69 rad arc leaves the local tangent by half the swept angle -- about 0.35 rad,
    #: pointing inward toward the hinge -- and that inward component really does walk the
    #: pads down the lever until it escapes: traced on seed 106 in the tool frame, the pads
    #: slide from 3.0 cm past the grasp point to 5.5 cm, then the lever leaves the jaws
    #: sideways (0.000 -> 0.030 m along the one axis parallel jaws cannot hold) at -0.26 rad
    #: and the jaws shut on nothing.  Cutting the lead does stop that -- 0.10 takes seeds
    #: 106/158/164 from 2.33 grips to 1.00, and reads 24/24 on seeds 0-23 for one step of
    #: median.
    #:
    #: It fails for two reasons that only a *large* sample and a look at the render show:
    #:
    #: 1. It moves the pose the switch hands to the next subtask, and the suite is only
    #:    marginally stable there.  Over seeds 0-499, lead 0.10 loses seeds 149 and 418
    #:    (kettle ALIGN) and lead 0.12 loses 164 and 343 -- both 498/500 against 500/500 at
    #:    0.70 -- and the failing seeds *reshuffle* between the two values rather than
    #:    degrading.  A 24-seed harness cannot see this at all.
    #: 2. It leaves the switch barely thrown.  Mean throw falls -0.679 -> -0.551 against a
    #:    -0.69 goal.  Grip count is not the objective; a lever pushed most of the way over
    #:    in one bite is worth more than a lever nudged in one bite.
    #:
    #: Making the cut conditional on the measured slide (arm a short lead only once the pads
    #: have moved) was tried and fails the same way on (2): it reaches 6/6 on the six seeds
    #: involved and 5/6 one-grip, but the throw is still only -0.551.  Arming it earlier
    #: inverts it -- at a 0.002 m trigger the sweep is too weak to hold the lever at all and
    #: seed 158 takes six bites.
    #:
    #: Letting the wrist follow the lever's live axis instead (`light_track_lever_axis`) is
    #: worse again: 1.54 grips and 11/24 one-grip seeds at lead 0.10, 1.17 and 20/24 at 0.15.
    #: `LightSwitchSkill.arc_mode = "contact"`, which puts the target on the pads' own arc,
    #: measures the same 2.33 grips and drops a seed.
    #:
    #: The remaining second grip is therefore a real cost that is worth paying, not a bug to
    #: tune out at this layer.  If it is worth removing, remove the *slide* -- the pads walk
    #: because the hand runs ahead of a lever it is pushing through friction -- rather than
    #: the drive that makes the demonstration good.
    light_manipulation_lead_angle: float = 0.7
    # SUPERSEDED, kept because the reasoning is the trap.  This used to read "one bite on the
    # light switch is worth about 0.42 rad, and that is the grasp geometry, not a setting",
    # on the evidence that five separate levers left the -0.42 stopping point alone:
    # `margin_override`, `light_grasp_command`, `light_manipulation_step`,
    # `light_manipulation_lead_angle`, and turning the wrist with the lever.
    #
    # Every one of those measurements is real and the conclusion drawn from them is wrong.
    # The push was ending because `LightSwitchSkill.manipulate_point` rotated the *tool
    # point* about the pivot while the hand's orientation was frozen, which walks the pads
    # along the wrong arc and squeezes the lever sideways out of the jaws -- so nothing done
    # at the object could help, and a sweep of knobs that all leave the geometry error in
    # place reads exactly like a physical limit.  Sweeping five knobs is not the same as
    # checking the target.  See `LightSwitchSkill.arc_mode`; with the arc corrected the same
    # `margin_override` moves the throw -0.411 to -0.538 in a single grip.
    #: How far along the lever, from its exposed tip toward the pivot, the fingertip midpoint
    #: is driven before closing.  Seating the capsule inside the pad length rather than
    #: pinching it at the distal edge is what stops the slide loading the contact straight
    #: off the end.
    #:
    #: The lever's collision capsule is only 0.06 m long, so this cannot be pushed far: 0.02
    #: and 0.028 both measure 8/8, while 0.035 drives the hand past the capsule into the
    #: panel behind it (1/8) and 0.045 misses entirely (0/8).
    light_insertion_depth: float = 0.028
    #: Vertical bias applied to the light switch's contact point, positive upward.
    #:
    #: Where the pads land on the lever decides whether one bite does the job, and the
    #: discriminator is which *side* of it they land on, not how far off centre.  Measured
    #: at the first push, vertical offset of the tool from the lever site: seed 0 +0.0054,
    #: seed 2 +0.0057, seed 3 +0.0237, all of which throw the switch in a single bite --
    #: and seed 7 -0.0158, which is the only one below it, gets 0.02 rad per bite instead
    #: of 0.42, and needs six.  Seed 3 sits further off centre than seed 7 and is fine, so
    #: this is not about tightening a tolerance.
    #:
    #: 0.025 is what puts seed 7 on the right side: its offset goes -0.0158 at 0, -0.0067
    #: at 0.015, and +0.0014 at 0.025, and that crossing is the whole effect -- bites go 6,
    #: 3, then one clean throw.  Over the 24-seed harness it takes the mean bites from 2.04
    #: to 1.50, the worst seed from 10 to 5, the steps spent on the switch from 78 to 63,
    #: and the harness from 21/24 to 22/24.
    #:
    #: A band, not a direction: 0.03 measures 7/8 on the eight-seed sample against 8/8 at
    #: 0.025, because lifting past the lever misses it the other way.  Do not read this as
    #: "higher is better" or tune it without the full harness.
    #:
    #: **Re-swept after `LightSwitchSkill.contact_tolerance` went to 0.018, and 0.025 is no
    #: longer the answer.**  Everything above is still true of the *sign*; what changed is
    #: how much lift the approach needs, because the close now fires from much nearer the
    #: lever.  At 0.025 the jaws shut 2.0 cm above the switch site with no contact at all,
    #: the empty-close guard reopens them, and the skill takes a second bite once the tool
    #: has descended to 1.3 cm -- so the double grip survived the tolerance fix and this is
    #: what actually removes it.  Measured over 24 seeds, mean grips taken before the drive
    #: starts, and how many seeds hold the lever with both pads at the first drive step:
    #:
    #: ========  ===========  ============  ===============  =====================
    #: offset    seeds 0-23   mean grips    both pads at the  seeds 24-47
    #:                                      first drive step
    #: ========  ===========  ============  ===============  =====================
    #: 0.008     24/24 m309   1.00           24/24            --
    #: 0.012     24/24 m314   1.00           24/24            24/24 m307
    #: 0.015     24/24 m309   1.00           23/24            --
    #: 0.018     23/24 m309   1.00           23/24            --
    #: 0.022     24/24 m316   1.08 (worst 3) 22/24            --
    #: 0.025     24/24 m338   1.04 (worst 2) 21/24            24/24 m314
    #: ========  ===========  ============  ===============  =====================
    #:
    #: 0.008 to 0.015 is the band that takes exactly one bite on every seed; 0.012 sits in
    #: the middle of it and is the only value verified 24/24 on the held-out half as well.
    light_contact_height_offset: float = 0.012
    #: Consecutive contact-free steps a closed light-switch grip may survive before the
    #: skill treats the lever as lost.  0 restores the strict live-contact test.
    #:
    #: The lever is spring loaded, so releasing it costs everything done so far -- which
    #: makes this the opposite trade from the kettle, where the same grace measures worse
    #: (see `KettleGraspSkill.grasp_retained`) because a dropped kettle stays dropped and
    #: gets shoved somewhere worse.  Traced on seed 7: the first push reaches -0.170 rad in
    #: seven clean steps, then contact reads 0 for a *single* frame at step 153, the skill
    #: opens the jaws, and over the next eight steps the lever springs back through -0.127,
    #: -0.085, -0.004 to +0.011 -- all the way home.  The skill then starts again from
    #: zero, which is why that seed needs three bites for a throw one bite can do.
    #:
    #: Keyed to `data.time`, not to a call count: `grasp_retained` is queried more than once
    #: per step, so a step-counting grace burns inside a single step.
    light_contact_dropout_grace: int = 2
    #: Consecutive steps both pads must be seen on the lever before the capture counts and
    #: the sweep may start.  One step is not a grasp: seven of the eleven two-grip seeds in
    #: the 0-499 census had both pads on the lever for a single step and one of them gone
    #: within three, because the frame was tilted so one pad led the other.  Two steps is
    #: the cheapest test that survives that; it costs the good seeds one step of sweep.
    light_capture_steps: int = 2
    #: Once the kettle is in its goal, fully release its handle and back away this far
    #: along the handle-normal approach line before selecting another subtask.
    #:
    #: This is a *clearance*, and clearance is all it has to buy -- the kettle body is
    #: 0.122 m half-width, so a hand this far back cannot catch it on the way to the next
    #: task.  It was 0.4, which is not a clearance but a reach, and once transport started
    #: delivering the kettle to the burner rather than stopping short of it that reach ran
    #: out of arm: measured from the delivered pose, the frozen waypoint at
    #: (-0.452, 0.359, 1.838) has an IK residual of 0.47 rad / 0.377 m, i.e. the wrist
    #: cannot hold the grasp frame anywhere near it.  The solver spends the whole recede
    #: timeout grinding at a pose that does not exist and walks joints into their stops
    #: (joint 1 to -2.898, joints 3/4/6 pinned), and from *there* the home pose is no
    #: longer reachable either -- 0.248 rad / 0.077 m residual -- so ORIENT_FORWARD burns
    #: its budget, gives up, and hands the next skill a wound-up arm.  Seed 0 lost the
    #: microwave that way: 80 steps stalled in ALIGN, then 400 frozen in APPROACH.
    #: Backing out with the wrist frozen instead of shortening the reach does not help --
    #: that measures 5/8, because a frozen wrist 0.4 m back is just as unreachable a
    #: *position*.  Shortening it does, and the window is real on both sides: too far
    #: strands the arm as above, too close leaves the hand near enough to disturb the
    #: kettle it just set down.  Over 8 seeds each, scored on the true final poses:
    #:
    #: ===== ================== ============ ==================
    #: value kettle alone       fixed order  randomized order
    #: ===== ================== ============ ==================
    #: 0.40  --                 --           7/8 (460 steps)
    #: 0.12  7/8 (err 0.038)    7/8          8/8 (390)
    #: 0.16  8/8 (err 0.024)    8/8          8/8 (393)
    #: 0.20  8/8 (err 0.028)    8/8          7/8 (385)
    #: ===== ================== ============ ==================
    #: How far the hand backs off the handle after delivering the kettle.
    #:
    #: This was 0.16, and it is the single most expensive number in the skill.  The pads are
    #: fully open by then, but a 4.6 cm post inside an 8 cm jaw gap is only 1.7 cm from
    #: either finger, and 16 cm is a long way to carry that clearance past a body the hand
    #: is standing over: traced, the open right finger stays hooked on the post for the
    #: whole withdrawal, the wrist rises 9 cm, and the kettle is lifted 6.5 cm off the
    #: counter and dragged 9 cm sideways.  Measured over 12 randomized seeds, the push
    #: delivers to 0.031 m of the burner and the retreat then puts it back out to 0.112.
    #:
    #: Swept with the wrist held still (see ``KettleGraspSkill.orient_during_recede``), the
    #: kettle is left at 0.077 m of the burner at 0.10, 0.057 at 0.08, 0.049 at 0.06 and
    #: 0.047 at 0.05, the short end taking the retreat itself from 41 steps to 9.
    #:
    #: But shorter is not better end to end, because this distance is also the clearance the
    #: hand starts its transit to the next subtask with.  Over 24 seeds of the full run,
    #: all-four-true against fixed / randomized task order: 0.05 gives 13 / 21, 0.08 gives
    #: 18 / 20, 0.10 gives 18 / 20 and 0.16 gives 17 / 20.  0.08 is the best of both and the
    #: fastest -- median 379 / 435 steps against 395 / 439 at 0.05.  The fixed order is the
    #: sensitive one: it runs the kettle with three subtasks still to come.
    #:
    #: Re-measured after the radial clearance was removed (see
    #: :meth:`KettleGraspSkill.recede_point`), which used to stretch most withdrawals well
    #: past this figure and so hid it.  On its own the retreat now wants to be a little
    #: longer: 0.12 gives 19 / 22 against 18 / 22 at 0.08, at a median of 375 / 428 steps
    #: against 380 / 438, and it leaves less behind it -- the kettle is disturbed by at
    #: most 0.004 m after its own skill ends against 0.036, and the microwave that follows
    #: re-grips on 2 of 24 seeds against 4.
    kettle_recede_distance: float = 0.12
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
    #: How many 0.08 s env steps one policy action is held for.  The env itself is
    #: untouched: the wrapper simply repeats the action.
    #:
    #: 1 -- re-plan every env step -- is the only setting that works, and the knob is kept
    #: only so coarser episodes stay available for experiments.  1.2.0 expressed the same
    #: granularity as ``control_steps``, the number of IK re-solves *inside* one env step,
    #: so a large value there bought accuracy; here it does the opposite, holding one
    #: open-loop command while the contact geometry moves underneath it.  Measured: at 19
    #: and at 4 the expert never leaves its first phase; at 2 it still stalls.
    action_repeat: int = 1
    #: Optional finer granularity used only while the microwave skill is active, whose
    #: grasp is contact-sensitive.  ``None`` keeps :attr:`action_repeat`.
    microwave_action_repeat: Optional[int] = None
    #: Integral gain of :class:`JointServo`, per step, on the joint position error.  This is
    #: what cancels the position actuators' gravity droop, which ``ctrl``'s per-step
    #: re-anchoring to the measured ``qpos`` would otherwise re-incur forever.  Measured on a
    #: 0.5 rad move: residual 0.0387 rad at ki=0, and 0.0005 rad at ki=0.05, 0.15 or 0.3
    #: alike, so the exact value matters far less than having one at all.
    servo_ki: float = 0.15
    #: Ceiling on the accumulated bias, in radians.  Generous next to the ~0.04 rad droop it
    #: exists to cancel, but it bounds what a phase change can carry over.
    #:
    #: Tightening it to the droop is tempting -- see :attr:`kettle_carry_bias_leak` for what
    #: an over-full integrator does -- and it is wrong.  The integrator earns its bias while
    #: *approaching*, against a target that moves a few millimetres a step, and it is that
    #: same bias that puts the fingertips on the handle: capped at 0.05 the descent onto the
    #: kettle stops 2-3 cm high, which is the droop, back again.  The fix belongs where the
    #: lag is, not on the ceiling that both share.
    servo_integral_clamp: float = 0.15
    #: How much of the integral bias survives a phase change.  A bias earned closing on one
    #: waypoint is not evidence about the next, so it is discarded by default.
    servo_bias_carryover: float = 0.0

    #: Iteration budget for the damped-least-squares solve in
    #: :meth:`ScriptedKitchenPolicy._solve_arm_delta`.  Purely kinematic, so a generous cap
    #: is cheap; it exits early on :attr:`ik_tolerance`.
    ik_iterations: int = 80
    #: Per-joint step below which the solve is considered converged (radians).
    ik_tolerance: float = 1e-4
    #: Duration passed to MuJoCo when converting quaternion error to the angular velocity
    #: used by the one-step DLS solver.  Upstream hard-codes 50 seconds, which leaves a
    #: requested horizontal grasp more than a radian away after 100 actions.  A measured
    #: sweep found both 5 and 2 seconds reach the complete side-grasp frame without growing
    #: the actuator-target gap.  Two seconds reached it in 48 actions versus 56 for five,
    #: so the faster setting is the default.
    ik_orientation_duration: float = 2.0
    #: Strength of the null-space posture bias that keeps the arm off its joint limits.
    #: See :meth:`OrientationAwareIKController._nullspace_escape`; 0 disables it and
    #: restores the plain minimum-norm solve, which stalls against the bounds.
    #:
    #: 0.01 was set from a per-skill sweep of 5 seeds each -- 0 leaves the light switch at
    #: 1/5 (the arm jams against a bound and the solve returns identically zero), 0.01 takes
    #: it to 5/5 -- which established that the term is needed but not how strong it has to
    #: be against a *contact* task.  0.01 is enough to escape a free-space bound and not
    #: enough to win an argument with the task objective: measured over 24 seeds at the full
    #: budget, the kettle's ALIGN spends a mean 19.2 steps and four seeds have a run past 40,
    #: with seed 7 sitting 82 steps in one alignment while joint 6 stays pinned at its 2.1127
    #: bound.  The visible symptom is a gripper that stops beside the kettle and turns itself
    #: for a second and a half.
    #:
    #: ==========  ============  ==========  =============  ====  ==============
    #: gain        mean ALIGN    runs > 40   bar contact    4/4   mean tasks
    #: ==========  ============  ==========  =============  ====  ==============
    #: 0.010               19.2           4             39    21            3.83
    #: 0.015                5.1           1              0    22               -
    #: 0.020                5.2           1              2    23            3.88
    #: **0.025**            5.9           1              1    23            3.96
    #: 0.028                  -           -              -    22            3.92
    #: 0.030               18.8           4              3     -               -
    #: 0.050               27.0           7              8     -               -
    #: ==========  ============  ==========  =============  ====  ==============
    #:
    #: Not monotone -- 0.03 and 0.05 are worse than 0.01 -- so this is a band and has to be
    #: swept, not reasoned about.  At 0.025 the kettle is grasped on 24 of 24 seeds and
    #: carried 8.73 m in total against 8.26 m at 0.01, and the whole suite spends *one* step
    #: with a finger on the kettle's top handle.
    ik_nullspace_gain: float = 0.025
    #: Nullspace pull toward the reset joint configuration, applied during free-space
    #: transits (see `TRANSIT_PHASES`).
    #:
    #: The transit home is the one move in the episode with no object to work around and a
    #: known-good destination *configuration*, and it is also the one that folds the arm
    #: into itself.  Coming off the slide cabinet on seed 4 the solve walks joint 4 to its
    #: -3.01 rad bound while `panda0_link5` closes on `panda0_link2` -- separation 0.557 m
    #: down to 0.222 -- and then grinds in self-contact for 17 steps.  A jammed arm cannot
    #: track, so the wrist is dragged to 1.66 rad from the frame it is trying to hold, and
    #: the phase spends what is left of its budget paying that back.
    #:
    #: `ik_nullspace_gain` cannot see this: its bias is cubic in distance-to-*limit*, worth
    #: 0.009 rad here, and a link pressing on another link is not a joint near a bound.
    #: Preferring the shape the arm resets in is what keeps the elbow out of the shoulder.
    #: Zero restores the plain minimum-norm solve.
    #:
    #: Applied only to rescue a solve that has actually folded -- see
    #: :meth:`ScriptedKitchenPolicy._solve_arm_delta`.  Held on throughout the transit it
    #: taxes the healthy ones instead: measured at every gain from 0.15 to 1.2, seed 0,
    #: which never self-collides, went from 244 steps to 300-308, while seed 4 only ever
    #: got its 26 contact steps down to 6.  Paying a quarter of a healthy episode to half-
    #: fix a jammed one is the wrong trade, and gating on the fold itself avoids it.
    #: Scoped to the transits because that is where the folding is.  Audited over the
    #: 24-seed harness before any of this existed: 164 self-colliding steps on 15 of 24
    #: seeds, essentially one pair (`panda0_link2` on `panda0_link5`, 161 of 164), and by
    #: phase `move_to_precontact` 132, `orient_forward` 31, `align` 1.  So this is not one
    #: seed's quirk, and the long reach out to a stand-off is the common case, not the
    #: transit home.  Both are free-space travel with no object between the arm and its
    #: waypoint, which is what makes the reset shape a legitimate thing to prefer; the
    #: phases that are actually working an object are left alone.
    ik_transit_posture_gain: float = 0.5
    #: Nullspace pull, during MOVE_TO_PRECONTACT, toward the joint configuration the
    #: stand-off pose solves to from where the reach starts (`KitchenSkill._reach_posture`,
    #: solved once per skill in `_commit_grasp_frame`).  0 disables it.
    #:
    #: The long reaches run at the actuators' speed ceiling for their whole length, and
    #: what makes them long is joint travel the tool does not need: measured on seeds 1, 57,
    #: 60 and 106, the reach from home to the slide cabinet moves joint 1 through 1.1-1.6
    #: rad and joint 4 through 1.4-1.8 to end 0.05-0.6 rad from where they began, so the
    #: 34-38 steps are 1.4-1.5 times the per-joint time-optimal bound (25) for the same net
    #: travel; the light switch from the microwave is the same (33-36 against 21-25), and
    #: seed 60's kettle reach is 30 steps against a bound of 7.6.  Parks, by contrast, sit
    #: at 0.93-1.10 of their bound.  The out-and-back is redundancy resolution -- the
    #: minimum-norm step leaves the self-motion direction to drift -- so pulling that one
    #: direction toward the destination configuration shortens the joint path without
    #: touching the tool's Cartesian path, which is what keeps this safe where a joint-space
    #: interpolation would not be.
    reach_posture_gain: float = 0.5
    #: Relative least-squares weight on Cartesian position in the pose solver.  One keeps
    #: the upstream DLS objective; it remains configurable for controller experiments.
    ik_position_weight: float = 1.0
    #: Iteration budget for the two probe solves :meth:`
    #: ScriptedKitchenPolicy._commit_grasp_frame` runs to decide which jaw-flip
    #: representative of a grasp to target.  This is deliberately far larger than
    #: :attr:`ik_iterations`: the probe starts from wherever the arm happens to be when the
    #: skill is selected, which is typically half a metre away, and it is asking whether the
    #: pose is reachable *at all* rather than what to command this step.  At 80 iterations
    #: neither representative has converged and the residuals compare truncation, not
    #: reachability -- measured, the microwave's un-flipped frame reads (0.025 rad, 2.0 cm)
    #: at 80 and (0.0002, 0.0002) at 400.  It runs twice per skill, not per step.
    grasp_frame_probe_iterations: int = 400
    #: Residuals below which a probed grasp pose counts as reachable, and therefore as an
    #: acceptable answer.  Both representatives are usually reachable; see
    #: :meth:`ScriptedKitchenPolicy._commit_grasp_frame` for why the un-flipped one is then
    #: preferred rather than the marginally closer one.
    grasp_frame_reachable_frame_tolerance: float = 0.02
    grasp_frame_reachable_position_tolerance: float = 0.01
    #: Whether a skill may ever target the flipped jaw representative.  False pins every
    #: skill to the un-flipped frame, so the hand never turns upside down.
    #:
    #: The two representatives are the same physical grasp, so this costs nothing in
    #: principle, and the flip was only ever taken because the un-flipped frame measured
    #: unreachable.  That verdict is real but its *cause* is not the pose: probed with 42
    #: random restarts, the un-flipped kettle grasp closes to (0.0000 rad, 0.0000 m) on
    #: every seed that was flipping.  It is unreachable only from the pose the arm arrives
    #: in, because damped least squares cannot walk a joint off a bound -- which is what
    #: :attr:`joint_limit_rescue_margin` fixes.  With that in place the flip is pure cost:
    #:
    #: ==================  ============  =============  ==========  ==============
    #: flip / jam rescue   seeds 0-23    seeds 24-47    combined    seeds flipped
    #: ==================  ============  =============  ==========  ==============
    #: allowed / off       23/24 m285    21/24 m292     44/48       3
    #: forbidden / off     22/24 m275    --             --          0
    #: forbidden / 0.05    24/24 m295    24/24 m297     48/48       0
    #: ==================  ============  =============  ==========  ==============
    #:
    #: Over seeds 0-23 that is 989 steps holding the hand inverted, worst case a full -1.000
    #: of world-up, against 29 steps and a worst case of -0.228 -- the hand tipping slightly
    #: past level on four seeds, not turning over.
    #:
    #: Two of the three seeds that flipped did not need to.  Measured at the kettle commit
    #: on seed 9, the wrist sits 0.31 rad from the un-flipped frame and 3.02 rad from the
    #: flipped one, and the probe still called un-flipped unreachable at (0.116, 0.042);
    #: forcing un-flipped there finishes 4/4 in 293 steps against 348, with the kettle skill
    #: costing 53 steps rather than 108 -- 70 of which were ALIGN spent rolling.  Seed 15
    #: goes 393 to 272 the same way.
    allow_grasp_frame_flip: bool = False
    #: How close to a joint bound (radians) counts as jammed during free-space transit, at
    #: which point the null-space escape is held at full strength
    #: (:attr:`OrientationAwareIKController.escape_floor`) so the solve spends its redundant
    #: freedom unwinding instead of leaving the joint pinned.  0 disables.
    #:
    #: This is the precondition that makes :attr:`allow_grasp_frame_flip` affordable, and it
    #: is the same class of bug as the ALIGN stalls that `ik_nullspace_gain` addresses: the
    #: escape term is scaled by the unmet task error, so it falls silent exactly when the
    #: park has arrived, and a joint parked on a bound therefore stays there.  Measured on
    #: seed 2, the park before the kettle runs its whole 62-step budget with joint 6 at
    #: +2.10 of a +2.1127 bound, and every exact IK solution for the un-flipped kettle grasp
    #: there needs j6 at +0.99 or below.
    #:
    #: Both halves of the rescue are needed, and each alone is a trap.  Measured on the
    #: opening park of seed 9, which the reset pose already satisfies and which used to
    #: cost one step:
    #:
    #: ==================  ============  ============  =============  =========
    #: escape / posture    opening park  seeds 0-23    seeds 24-47    seed 9
    #: ==================  ============  ============  =============  =========
    #: off / off           1 step        22/24 m275    --             --
    #: off / on            60 steps      24/24 m300    24/24 m297     350 steps
    #: on  / off           2 steps       22/24 m283    22/24 m298     --
    #: on  / on            6 steps       24/24 m295    24/24 m297     295 steps
    #: ==================  ============  ============  =============  =========
    #:
    #: Posture alone stalls the park for its whole `orient_forward_budget` and times out --
    #: it is pulling toward a pose whose joint 2 is 0.017 from a bound, so the jam it is
    #: supposed to cure never clears and it never disarms.  The escape alone clears the jam
    #: in about five steps but hands the next skill an arm of no particular shape, and loses
    #: four seeds across the two halves.  Together the escape unpins joint 2, the jam test
    #: goes false, the posture preference disarms with it, and the park exits.
    #:
    #: Choose the margin on held-out seeds, not the tuning set -- the two disagree:
    #:
    #: ========  =================  ======================
    #: margin    seeds 0-23         seeds 24-47 (held out)
    #: ========  =================  ======================
    #: 0.0       22/24 median 275   --
    #: 0.02      24/24 median 293   22/24 median 309
    #: 0.03      24/24 median 298   --
    #: 0.05      24/24 median 300   24/24 median 297
    #: 0.08      24/24 median 313   --
    #: 0.15      24/24 median 309   --
    #: 0.30      24/24 median 314   --
    #: ========  =================  ======================
    #:
    #: 0.02 is the fastest of these on the tuning set and loses two held-out seeds; it fires
    #: only once a joint is already hard against its stop, which is late.  Everything from
    #: 0.05 up is 24/24 and pays for the wider window in transit steps, so take the bottom
    #: of that band.  (That sweep was run with the posture half only; the margin was not
    #: re-swept after the escape half was added, and the shipped 0.05 holds 48/48.)
    joint_limit_rescue_margin: float = 0.05
    #: Strength the escape is held at while a joint is jammed, as a floor under the
    #: ``unmet`` scaling.  1.0 is full strength for as long as the jam lasts.  Below 1 it
    #: measures worse and is kept only to reproduce the table below: 0.15, 0.35 and 0.6 all
    #: read 23/24 on seeds 0-23 against 24/24 at full strength.
    joint_limit_rescue_escape_floor: float = 1.0
    #: Also prefer the reset shape while jammed, at `ik_transit_posture_gain`.  This is the
    #: half that recovers seeds, and it is useless on its own: `_home_qpos` is captured from
    #: the reset pose, which is *itself* jammed at joint 2, so preferring it answers a jam
    #: with a jammed configuration and -- at twenty times the escape's gain -- cancels the
    #: correction that would have fixed it.  The escape unpins, this supplies the shape.
    joint_limit_rescue_posture: bool = True
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
    #: The same trade, applied to MOVE_TO_PRECONTACT instead of to the park.  The reach
    #: solves position and orientation in one IK, but the env weights orientation at 1/50 of
    #: position, so on a long transit the translation drags the wrist out faster than
    #: ``rotation_step_scale`` can pull it back and the arm then has to unwind before it can
    #: align.  Measured on seed 12's microwave -> slide cabinet reach, the frame error grows
    #: from 0.089 to 1.180 rad over 30 steps while joint 4 sits pinned on its -3.0718 bound
    #: for 20 of them, and the tool sails 0.20 m *past* the target before coming back: the
    #: end effector travels 1.83 m for a 0.97 m chord and the wrist turns 4.51 rad to end
    #: 0.20 rad from where it started.  That churn is the "twisting" a viewer sees between
    #: subtasks.
    #:
    #: Above `reach_tilt_threshold` rad of frame error the reach's translation is scaled by
    #: `reach_tilt_position_scale`, which lets the rotation catch up before the arm commits
    #: further.  Set the threshold to 0 to disable.
    #:
    #: On its own it is a real but partial cure -- the reach still executes a distorted
    #: version of what the IK solved, which is the larger half of the problem; see
    #: `arm_command_excess`, with which this was tuned jointly.
    #:
    #:   threshold / scale   seeds 0-23   median   rot path/transition   path/chord
    #:   0 (off)             24/24 m315   315      3.58 rad              2.4x
    #:   0.35 / 0.35         24/24 m295   295      2.93                  2.3x
    #:   0.35 / 0.15         22/24 m297   297      3.18                  2.3x
    #:   0.20 / 0.35         21/24 m309   309      3.50                  2.5x
    #:
    #: Slowing the reach harder (0.15) or arming it earlier (0.20) both lose seeds: the arm
    #: creeps and runs the task budget out.  The gate has to be a correction, not a speed
    #: limit.
    reach_tilt_threshold: float = 0.35
    reach_tilt_position_scale: float = 0.35
    #: Turn, then travel.  On a reach (or park) whose target is farther than this, in
    #: metres, and whose frame error is above `reach_tilt_threshold` (`tilt_recovery_threshold`
    #: for the park), hold position and rotate until the error is under half the threshold,
    #: then translate at the unthrottled step.  0 disables it and leaves the throttled
    #: reach-and-rotate in place.
    #:
    #: The slide cabinet is the case: its stand-off is 0.73 m from home and its frame needs
    #: 0.5-0.95 rad of wrist turn, so the tilt throttle applies for the whole reach and the
    #: tool travels at 1.9 cm per step (33-45 steps, path/net ratio 1.1 -- not wandering,
    #: throttled), and the park back is throttled the same way (28-30 steps for 0.78 m).
    #: Rotating first at `max_rotation_step` costs ~8-10 steps and the translation then
    #: takes ~11 at the free-space step -- on paper.  Measured at 0.30 on twelve seeds it
    #: is worse everywhere but one place: the slide-cabinet reach is unchanged (41 steps
    #: against 38-40), the light-switch reach nearly doubles where the park hands it a
    #: badly turned wrist (58 against 33 on seed 1, 59 against 34 on 315, 48 against 32 on
    #: 358), and the parks it applies to get longer, not shorter (seed 0: 33 and 26 against
    #: 26 and 24).  Holding position while the wrist turns is not free: the turn needs the
    #: shoulder to move, the hold fights it, and the reach-and-rotate the tracker already
    #: does was the faster of the two.  The one gain -- the kettle reach after the slide
    #: cabinet, 30-36 steps down to 5 on seeds 60 and 75 -- is a start-configuration
    #: problem, not a throttle problem; see `park_joint_transitions`.  Off.
    turn_first_distance: float = 0.0
    #: How far the executed joint step may depart from the one the IK solved, as a ratio of
    #: the fastest hinge's command to what that hinge can actually deliver.  0 disables the
    #: scaling entirely and every hinge is clipped to the action box on its own, which is
    #: what makes long reaches crooked; 1.0 executes the solved displacement exactly, at the
    #: rate of the most limited joint.  See :meth:`JointServo.action` for the mechanism.
    #:
    #: The wrist churn falls monotonically as the allowed distortion shrinks, and so does
    #: the completion rate once it goes too far -- below 2.5 the arm travels at the rate of
    #: whichever hinge is most limited and starts running task budgets out.  Every row below
    #: carries `reach_tilt_threshold` 0.35 / 0.35 except where marked, because the two were
    #: tuned together; "roll" is the wrist's rotation about its own approach axis summed
    #: over a transition, which is the twist a viewer actually sees.
    #:
    #:   excess  tilt  seeds 0-23   seeds 24-47  median  path/chord  rot path  roll
    #:   0 (off) off   24/24 m315   22/24 m312   315     2.4x        3.58 rad  2.11 rad
    #:   0 (off) on    24/24 m295   --           295     2.3x        2.93      --
    #:   3.0     off   24/24 m298   --           298     2.0x        2.33      1.44
    #:   3.0     on    24/24 m301   24/24 m295   301     2.0x        2.16      1.25
    #:   2.5     off   23/24 m295   --           295     1.9x        1.95      1.17
    #:   2.5     on    24/24 m292   24/24 m296   292     1.9x        1.83      1.05
    #:   2.0     off   23/24 m298   --           298     1.8x        1.59      0.82
    #:   2.0     on    23/24 m295   --           295     1.8x        1.63      0.84
    #:   1.5     on*   23/24 m388   --           388     1.6x        1.40      0.73
    #:   2.0     on*   21/24 m341   --           341     1.5x        1.36      0.68
    #:
    #: (*) with `servo_feasible_windup`, which buys the last of the churn and costs 90 steps
    #: of median episode; it is off for the same reason 1.5 is not the default.
    #:
    #: 2.5 is the only setting that is better than the baseline on every axis at once, and
    #: it is better on all of them: 24/24 on both halves, the shortest median episode of
    #: anything measured, half the wrist roll, and ALIGN's workload down from 610 steps to
    #: 201 over the tuning seeds -- the wrist now arrives aligned instead of being rescued.
    arm_command_excess: float = 2.5
    #: Judge an arm hinge saturated at what the actuator can deliver rather than at the
    #: edge of the action box, so the integrator stops accumulating on a joint whose error
    #: cannot close.  See :meth:`JointServo.action`.
    servo_feasible_windup: bool = False
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
    retreat_lift_seconds: float = 0.25
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
    engage_seconds: float = 0.25
    #: Whether the generic contact classifier actually waits out `engage_seconds` before
    #: promoting a closing gripper to MANIPULATE.  False restores the original behaviour,
    #: where a single step of jaw travel counted as a grasp.
    engage_settle_required: bool = True
    #: Generic per-phase budget.
    phase_timeout: float = 8.0
    #: Hard cap on a *reaching* phase (MOVE_TO_PRECONTACT / APPROACH).  Reaches are not
    #: bounded by ``phase_timeout``: how many steps a reach needs depends on how far it has
    #: to travel and on ``control_steps``, which sets how much of a commanded displacement
    #: the IK actually achieves per step.  A reach ends when it arrives or when it stops
    #: making progress -- never merely because a fixed step count elapsed, which used to
    #: send the FSM into its contact sequence while the arm was still 0.7 m from the object.
    reach_timeout: float = 14.0
    #: A reach is "blocked" when its position error has stopped shrinking by more than this
    #: (metres) over ``progress_window`` steps.  Blocked is a *success* path: these skills
    #: routinely end their reach 10-15 cm short because the arm is pressed against the
    #: cabinetry around the target, jittering rather than cleanly stalling.  Only a reach
    #: that is still closing the gap when ``reach_timeout`` expires counts as failed.
    reach_progress_epsilon: float = 0.005
    align_timeout: float = 10.0
    manipulate_timeout: float = 14.0
    retreat_timeout: float = 6.0
    #: Hard bound for release-and-recede.  Completion is normally geometric; this merely
    #: prevents an unreachable Cartesian retreat target from trapping the reactive FSM.
    recede_timeout: float = 6.0
    #: Net frame-error improvement (rad) an in-place RECEDE rotation must show across
    #: ``recede_rotation_progress_window`` steps to count as still converging.
    #:
    #: `recede_timeout` alone is far too coarse a backstop for a rotation that is not
    #: converging, because a rotation commands no translation: the arm stands at the object
    #: it has just finished and turns its wrist for the whole budget.  Measured on seed 3,
    #: where the microwave RECEDE was chasing two mutually unsatisfiable frames, it burned
    #: all 75 steps -- 21% of the episode -- and ended *further* from both targets than it
    #: started (1.160 and 1.848 rad against 0.65 and 1.33 at entry).
    #:
    #: Measured over a *window* rather than against the best error seen so far, which is
    #: the same shape as ``reach_progress_epsilon`` and for the same reason.  Best-ever
    #: tracking cannot catch the failure this exists for: an oscillation dips to a new low
    #: every few steps -- seed 3's alternation ran 0.417, 0.510, 0.431, 0.525, 0.421,
    #: 0.303, 0.384, 0.258 -- and each dip resets the reference, so a wrist swinging back
    #: and forth forever reads as progress.  Comparing across the window sees what actually
    #: happened: 0.008 rad closed in eight steps.
    #:
    #: The epsilon is loose on purpose.  A healthy unwind closes 0.06-0.09 rad per step and
    #: is done in four or five, never reaching the window at all; 0.02 rad across eight
    #: steps is 0.0025 rad per step, so even a badly slowed rotation clears it.
    recede_rotation_progress_epsilon: float = 0.02
    recede_rotation_progress_window: int = 8
    #: Hard budget per subtask, across retries.  Without it a skill that cannot succeed
    #: eats the whole 280 step episode -- and a flailing arm can push
    #: an already-placed object back out of its goal region.
    #: The microwave is the binding case: it needs a median 364 steps (29 s) when it
    #: succeeds, so a 34 s budget abandons it on any seed that runs slower than average.
    #: Measured over 8 combined seeds, raising this to 50 s took full 4/4 episodes from
    #: 2/8 to 3/8 and the microwave itself from 2/8 to 3/8, with no other task affected.
    task_step_budget: float = 50.0
    #: How long a selected subtask may make no progress at all before the skill is retried
    #: from its stand-off, in seconds.  0 disables the detector.
    #:
    #: `reach_timeout`, `reach_progress_epsilon`, `stall_window` and `stall_distance` above
    #: describe exactly this failure, and none of them is reachable: they belonged to the
    #: second FSM that was deleted, and `_eef_history` / `_progress_history` /
    #: `_reach_error_history` are still cleared on every phase change but never written to
    #: or read.  So the reactive FSM has had no stall detection whatever, and the only
    #: backstop is `task_step_budget` -- 625 steps, which is most of an episode.
    #:
    #: Both surviving failures over seeds 0-499 are that gap, and neither is a geometry
    #: problem:
    #:
    #: * seed 106 freezes in APPROACH 9 cm short of the microwave handle -- end effector
    #:   pinned at the same millimetre and the position error flat at 0.090 for **604
    #:   steps** -- and the task budget ends it with 10 steps of episode left.
    #: * seeds 158 and 164 grip the light switch, drive the lever from 0.000 to -0.227 rad
    #:   (task distance 0.692 -> 0.464), then slide off the end of it.  ALIGN opens the jaws
    #:   at the far end of the sweep, the return shoves the lever back to 0.000, and the FSM
    #:   settles into a MANIPULATE <-> CONTACT_OR_GRASP cycle -- jaws closed on nothing,
    #:   tool 2 cm from its target, task distance pinned at 0.692 -- for ~570 steps.
    #:
    #: A phase-step timeout cannot catch the second one: the oscillation re-enters a phase
    #: every few steps and `_phase_steps` resets each time, so neither phase ever ages.  The
    #: test has to live at the *task* level and watch two things at once, because either
    #: alone has a legitimate counter-example: a long reach moves a long way while the task
    #: distance sits still, and a manipulation closes the task distance while the tool
    #: barely moves.  Requiring both to stall is what separates them.
    #:
    #: With the detector and its escalating retry, all three recover and seeds 0-499 go
    #: 497/500 to **500/500** at four tasks, with the mean episode 297.7 -> 294.2 steps and
    #: the longest 1000 -> 535:
    #:
    #:   seed 106   3/4 1000 steps -> 4/4 535 (two retries)
    #:   seed 158   3/4 1000 steps -> 4/4 367
    #:   seed 164   3/4 1000 steps -> 4/4 361
    task_stall_seconds: float = 3.2
    #: Net end-effector displacement, in metres, over `task_stall_seconds` that still counts
    #: as going nowhere.  Net rather than path length on purpose -- the light-switch cycle
    #: travels about 3 cm per lap and returns to where it started.
    task_stall_distance: float = 0.02
    #: Task-distance improvement over the same window that still counts as no progress.
    task_stall_progress: float = 0.005
    #: How many times a stalled subtask is retried from its stand-off before it is
    #: abandoned.  The retry rebuilds the skill, so it re-picks the grasp frame from the
    #: pose the arm is actually in.
    task_stall_retries: int = 2
    #: How long ALIGN may fail to close its frame error before it counts as stalled, in
    #: seconds.  0 disables the test.
    #:
    #: The general stall test above cannot see an ALIGN stall.  ALIGN holds position on
    #: purpose -- it is a rotation in place -- so "the end effector has not moved" is true
    #: of every ALIGN, converging or not, and a reach makes no task progress either, so both
    #: halves are trivially satisfied and only the 3.2 s window separates a stall from
    #: normal operation.  Measured on seed 106's microwave, that is 52 steps of a completely
    #: motionless arm -- four seconds, frame error flat at 0.232 rad, position flat at 0.088
    #: -- before anything fires.
    #:
    #: So judge ALIGN on the one quantity it exists to change.  This is the same shape as
    #: `recede_rotation_progress_epsilon`, which was written for the same reason, and it is
    #: safe for the same reason: a converging unwind closes 0.06-0.09 rad per step and is
    #: done in four or five, so even a badly slowed one clears the threshold with room to
    #: spare.  A wrist that cannot reach its frame closes nothing at all.
    #:
    #: Seed 106's microwave ALIGN, and the whole episode with it: 1.6 s cuts the motionless
    #: stretch 52 -> 27 steps and the episode 497 -> 472; 1.0 s gives 464 and 0.8 s gives
    #: 462, so the return flattens once the detection is short relative to the recovery
    #: itself.  What is left is the park, which is not waste -- it is the only thing
    #: measured to cross the IK branch the reach jams in.
    align_stall_seconds: float = 1.0
    #: A reach that is *exactly* motionless -- see the dead-end note above for why the loose
    #: version of this cannot work.  These constants are deliberately far tighter: the jam
    #: this catches pins the end effector to the millimetre for 37 steps with the position
    #: error flat at 0.090, so it clears a 1.5 mm / 5 cm bar by a wide margin, while an arm
    #: settling onto a stand-off does not.
    reach_freeze_seconds: float = 1.6
    reach_freeze_distance: float = 0.0015
    reach_freeze_position_error: float = 0.05
    #: Frame-error improvement, in radians, an ALIGN must show across `align_stall_seconds`
    #: to count as still converging.
    align_stall_progress: float = 0.02
    #: NOT A KNOB -- a recorded dead end.  A fast "the reach has stopped dead" test was
    #: tried here, scoped to MOVE_TO_PRECONTACT/APPROACH, to catch seed 106's microwave jam
    #: before it becomes visible: the end effector pins to the same millimetre with the
    #: position error flat at 0.090 from step 386 to 423, three seconds of a motionless arm,
    #: before `task_stall_seconds` fires.
    #:
    #: It cannot be made safe.  A reach makes no *task* progress by definition -- nothing
    #: has been manipulated yet -- so the task-distance half of the stall test is trivially
    #: satisfied throughout and the test collapses to "the arm barely moved", which is also
    #: true of an arm settling onto its stand-off.  At 1.2 s / 3 mm it costs eight episodes
    #: (492/500, mean 341 steps against 294).  Adding the obvious missing condition -- that
    #: the tool must still be far from its waypoint, 3 cm, since stillness only means stuck
    #: when there is somewhere to go -- recovers six of them and still loses two (498/500,
    #: seeds 342 and 428).
    #:
    #: What does work is the same idea at a far tighter bar -- see `reach_freeze_seconds`
    #: below.  The jam pins the end effector *to the millimetre*, so 1.5 mm over 1.6 s with
    #: a 5 cm error gate separates it cleanly from an arrival: 500/500, and seed 106 481 ->
    #: 471.  The lesson is that the threshold has to be set from the failure's own scale,
    #: not from what looks like a generous margin.
    #: How many times a subtask that was given up on may be picked up again once every
    #: other subtask is finished.  0 disables the revival.  See :meth:`_reactive_task`.
    task_reattempt_limit: int = 1
    #: MANIPULATE keeps driving until the task distance is below
    #: ``completion_margin * BONUS_THRESH`` rather than stopping the instant the env would
    #: score the task.  Stopping exactly at the threshold leaves e.g. the slide cabinet
    #: barely cracked open, which scores but makes a poor demonstration.  Completion itself
    #: is always judged with the env's own threshold.
    completion_margin: float = 0.2
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
    #: Normalized gripper command used to hold the kettle's left handle bar.
    #:
    #: ``cartesian_plan_to_action`` maps this axis onto finger travel as
    #: ``target = 0.02 * (command + 1)``, and :attr:`KitchenSim.finger_opening` is the *half*
    #: opening, so the command has to be read against the bar's radius, not its diameter.
    #: The left-handle collision capsule is 0.023 m in radius, so a hold sits at a
    #: half-opening of about 0.023 and the neutral command for it is 0.0.
    #:
    #: This was -0.25, i.e. a target of 0.015 -- 8 mm inside the bar on each side.  The
    #: servo spends that entire margin as grip force on a smooth cylinder and squirts it
    #: out: measured, every capture broke 6 to 12 steps into the transport (opening 0.0185
    #: at capture, 0.0143 six steps later with no contact left), the skill re-approached,
    #: and the kettle ended barely a third of the way to its goal.  A few millimetres of
    #: preload is all a friction grasp on this bar can use.  Sweeps of the transport at
    #: -0.15, -0.05, 0.0 and +0.05 put the best hold at -0.05: a target of 0.019, i.e. about
    #: 4 mm of preload, which is enough friction to carry without being enough force to
    #: squeeze the cylinder out.
    kettle_grasp_command: float = -0.05
    #: Normalized finger command used for every microwave close (see
    #: ``MicrowavePullSkill.gripper_command``).  Under 1.2.1's joint servo this is a
    #: *position* target: -1.0 asks for a fully shut gripper and spends all the remaining
    #: travel as grip force, which fires the bar out from between the pads.  The 4 cm bar
    #: stops the finger joint at about 0.0123 m, so -0.5 (target 0.010 m) squeezes just
    #: past it and no further.  Measured flat from -0.4 to -0.6 at 5/8 before the insertion
    #: depth was added, so it is a preload setting, not a knife edge.
    #:
    #: Read against the bar's radius, which is 0.02: -0.5 asks for 0.010, i.e. 10 mm inside
    #: the handle on each side.  Solo the insertion depth seats the bar deep enough in the
    #: jaws to survive that, which is why this stayed at -0.5 while the kettle and light
    #: switch were fixed.  Started from another subtask's finishing pose the seating is
    #: worse and the crush wins: traced with the microwave first, the grasp forms at
    #: `finger_opening` 0.0148, the servo drives on to 0.0074 -- well inside a 0.02 m bar --
    #: the wrist is levered from 0.08 to 0.55 rad of frame error, and the bar is gone eight
    #: steps later with the door barely moved.
    #:
    #: -0.35 (a target of 0.013) is the measured optimum and the curve is *not* monotonic:
    #: over 8 seeds with the microwave first, -0.5 measures 5/8, -0.35 measures 7/8 and
    #: -0.25 only 3/8.  Both ends fail for opposite reasons -- too much travel left over
    #: squeezes the bar out, too little preload lets the door's own resistance peel the pads
    #: off it during the pull -- so this wants to sit just inside the bar, not merely
    #: outside the crush.  The kettle that follows improves with it too (position error
    #: 0.172 against 0.237), because it inherits a door that actually opened.
    microwave_grasp_command: float = -0.30
    #: Fixed yaw, in radians, applied to the microwave grasp frame about the vertical
    #: handle bar.  The frame is built from the door normal, so it turns as the door opens:
    #: the wrist is square to the bar at the moment of capture and roughly 0.5 rad off it by
    #: the time the pull has swung the door open, which is what pinches the round bar
    #: obliquely and squeezes it out.  Pre-yawing splits that error either side of zero
    #: instead of spending all of it at the end.  Positive carries the frame's +y (the
    #: hinge-to-handle radial) toward the door.
    #: Negative, i.e. counter-clockwise about the bar seen from above.  Sign and size are
    #: both measured over 8 seeds, counting how often the bar leaves the pads mid-pull (one
    #: re-grip per episode is the floor -- that one is the release at the end):
    #:
    #: ====== ========= ============ ==============
    #: yaw    re-grips  door left at success
    #: ====== ========= ============ ==============
    #:  0.00  2.00      0.169        8/8
    #: -0.15  1.38      0.233        7/8
    #: -0.30  1.25      0.225        8/8
    #: -0.45  1.38      0.198        8/8
    #: +0.30  0.50      0.591        2/8
    #: ====== ========= ============ ==============
    #:
    #: So the bar is dropped on every episode without this and on two seeds in eight with
    #: it, and turning the wrist the other way simply fails to open the door at all.
    #:
    #: Those numbers are the *isolated* skill, and they are not the reason this is on.  Left
    #: applied to every approach the yaw costs a seed in the composed run -- the door that
    #: was pulled to 0.309 in one stroke stops just outside the 0.3 the environment banks
    #: at, the bar comes out there, and a wrist yawed for a shut door cannot re-capture a
    #: half-open one: 178 steps hovering 3.4 cm off the bar, the microwave's whole 626-step
    #: budget, abandoned.  Gating it to the first attempt (see `_capture_ever`) hands the
    #: recovery the square frame it needs, at the price of the retention gain -- 2.12
    #: re-grips, no better than none.  What it keeps is the composed result, and that is
    #: what this is worth having for: over 8 randomized seeds, -0.45 measures 8/8 at a
    #: 396-step median against 7/8 at 457 for no yaw at all.
    #:
    #: Scaling the yaw by the swing the door has left, rather than gating it, is the
    #: physically honest version of the same idea and is much worse where it counts: the
    #: isolated re-grips do fall to 1.25, and the composed run drops to 5/8 at a 688-step
    #: median.
    #: **Superseded by the approach funnel; this is now 0.** Everything above was measured
    #: against a straight-line APPROACH, where the tool routinely arrived at the bar plane
    #: displaced sideways and the yaw was worth having because it changed *which* way it
    #: slid.  With ``microwave_approach_funnel`` putting the tool on the bar's axis before
    #: it commits, that job is gone and the yaw is only a rotation the wrist has to find
    #: mid-insertion.  Measured over 24 randomized seeds, funnel plus this yaw gives 23/24
    #: grasps at a 14 mm mean miss; funnel with no yaw at all gives **24/24 at 7 mm**, and
    #: holds the bar longer (13.4 steps against 11.2).  The isolated microwave harness
    #: still mildly prefers the yaw (8/8 and a door left at 0.153, against 7/8 and 0.172),
    #: which is the same isolated-versus-composed disagreement the height table shows; the
    #: composed run is what ``test.py`` runs, so it decides.
    microwave_grasp_yaw: float = 0.0
    #: How far above the handle site's centre the jaws take the bar, in metres.  See
    #: :meth:`MicrowavePullSkill.touch_point`; the bar's own half-length is 0.13, and its
    #: mounting stubs sit at exactly that, so this cannot approach it.
    #:
    #: More height buys more reachable arc, and the door does end further open for it, but
    #: the isolated skill and the composed run disagree about how much is safe.  Measured
    #: over 8 seeds each:
    #:
    #: ====== ========= ============ ============ =========
    #: height re-grips  door left at randomized   fixed
    #: ====== ========= ============ ============ =========
    #: 0.00   2.12      0.174        8/8 (396)    8/8
    #: 0.02   2.00      0.152        8/8 (372)    8/8
    #: 0.03   2.00      0.115        7/8          8/8
    #: 0.04   2.00      0.113        8/8 (376)    7/8
    #: 0.05   1.88      0.109        7/8          6/8
    #: 0.06   1.88      0.102        7/8          7/8
    #: 0.07   1.50      0.167        --           --
    #: ====== ========= ============ ============ =========
    #:
    #: 0.02 is the only value that holds *both* orders at 8/8, so that is what this is,
    #: even though 0.04 opens the door considerably further.  Read the middle rows as one
    #: seed either way rather than as a trend -- 0.03 and 0.04 each drop a different order,
    #: which is what a single marginal episode looks like at eight seeds, and the honest
    #: reading is that everything from 0.02 to 0.06 opens the door better than 0.00 while
    #: sitting close enough to the edge that one seed decides the score.
    microwave_grasp_height: float = 0.02
    #: How close to the bar, along the approach, the pre-yaw is allowed to start; ``None``
    #: applies it from the stand-off onward.  It has to be withheld at first and it has to
    #: be in by the time the jaws close.  Yawing the frame for the whole reach also yaws the
    #: *approach*, which is what the since-disabled hook did and what made it come in
    #: obliquely -- measured 7/8 at -0.30 with the door left at 0.225, and 0/8 by +0.30.
    #: Withholding it until the bar is already between the pads is equally useless in the
    #: other direction: the pull commands no rotation at all, so a frame that only becomes
    #: yawed at capture is never actually tracked, and -0.15 through +0.30 all measure the
    #: same 2.0-2.25 re-grips as no yaw whatsoever.  Landing it during the final insertion
    #: is the window where the wrist both can and still may turn.
    microwave_yaw_onset_depth: Optional[float] = 0.10
    #: Whether the pre-yaw is dropped once a grasp has been made and lost; see
    #: ``microwave_grasp_yaw`` for why the recovery wants the square frame back.
    microwave_yaw_first_attempt_only: bool = True
    #: How far past the handle bar, along the approach, the fingertip midpoint is driven.
    #: See ``MicrowavePullSkill.contact_point``: this is what keeps the bar in the jaw when
    #: the pull starts.  Zero (aim at the bar) measures 5/8; 0.015-0.025 all measure 8/8;
    #: 0.027 falls to 7/8 and 0.030 to 0/8, where the wrist meets the door panel and the
    #: pull never starts at all (0.5 grasps per episode, door left at 0.638).  Within the
    #: 8/8 band deeper is better -- seating 5 mm further takes the mean door distance from
    #: 0.201 to 0.184 at an unchanged number of re-grips -- so this sits at the top of it.
    microwave_insertion_depth: float = 0.025
    #: Turn applied to the wrist *after* the jaws have closed on the bar, about the bar's
    #: own axis, in radians.  Negative turns the gripper counter-clockwise as seen from the
    #: front -- away from the hinge, the same sense the retired approach-time
    #: ``microwave_grasp_yaw`` used.
    #:
    #: This is the post-grasp version of that idea, and it is a different move.  Yawing the
    #: *approach* rotates the frame the tool is still reaching along, so the insertion comes
    #: in obliquely and the bar is easier to miss -- which is why that knob is now zero.
    #: Yawing after capture costs the approach nothing: the bar is already between the pads
    #: and the wrist turns against it, loading one pad behind the bar before any pull force
    #: is applied.
    #:
    #: It is commanded as a frozen frame snapshotted at the grasp, not as the live grasp
    #: frame, which turns with the door; following *that* through the swing is measured
    #: harmful and is why the pull is otherwise orientation-free (see the MANIPULATE branch).
    #:
    #: The turn has to be unwound before the next subtask.  It is a rotation the arm
    #: carries out of the skill with it, and at +1.0 the wrist leaves MANIPULATE 1.4 rad
    #: from the live grasp frame and reaches SELECT_SUBTASK at 2.6 rad, from where the
    #: light switch is unreachable: 626 steps of MOVE_TO_PRECONTACT with the position error
    #: pinned at 0.57 m and three joints on their stops, then abandoned, on 3 of 24
    #: randomized seeds.  RECEDE undoes it; see :meth:`MicrowavePullSkill.release_frame`.
    #:
    #: Sign, measured.  Negative -- the sense the retired ``microwave_grasp_yaw`` used --
    #: lengthens the hold but does not widen the door, and costs whole-episode success
    #: (24 randomized seeds, before the unwind existed: 21/24 at zero, against 16-19/24
    #: everywhere in -0.15 to -0.45, with the lost seeds being microwave rebounds).  The
    #: positive turn set here is the one that works: it loads a pad behind the bar rather
    #: than levering it toward the free edge, and takes isolated losses from 2.00 to 1.00.
    #: With the unwind and ``MicrowavePullSkill.margin_override``, +1.0 measures 24/24
    #: seeds grasped at a 7 mm miss, the door open at the end on 24/24, and 20/24 whole
    #: episodes.
    #:
    #: What it does not fix is why the pull ends.  The jaws are pried open by the door's
    #: reaction: the finger target holds at 0.014 m while the *measured* opening walks to
    #: 0.040 and the capture predicate gives up.  Gripping harder answers that and buys
    #: nothing, because the binding limit is the arm's reach along the arc, not the grip --
    #: over 16 seeds the hold lengthens monotonically (11.8 steps at -0.30 through 18.1 at
    #: -0.80) while the door ends *less* open (0.150 to 0.186).  That is why
    #: ``microwave_grasp_command`` stays where it is.
    microwave_pull_yaw: float = 0.8
    #: Gain on the approach funnel: how far behind the contact point APPROACH aims, per
    #: metre of lateral offset from the bar's approach axis.  See
    #: :meth:`MicrowavePullSkill.approach_point`.  Zero restores the straight line to the
    #: contact point, which is what let seed 0 arrive 20 cm off the bar, miss it entirely,
    #: and open the door by shoving it with the wrist -- scored as success by the
    #: environment, which only reads the door joint.
    #:
    #: Measured over 8 randomized seeds, by where the tool crossed the bar plane and
    #: whether the jaws ever closed on the bar:
    #:
    #: ====== ============ ============= ==========
    #: gain   grasped      radial miss   held steps
    #: ====== ============ ============= ==========
    #: 0.0    7/8          0.034         11.2
    #: 0.5    8/8          0.036         12.8
    #: 1.0    8/8          0.014         11.6
    #: 1.5    8/8          0.013         12.9
    #: 2.0    8/8          0.012         10.9
    #: 3.0    2/8          --             3.0
    #: 4.0    1/8          --             1.6
    #: ====== ============ ============= ==========
    #:
    #: Above about 2 the funnel stops closing: the lateral error the arm can actually hold
    #: keeps the target further back than the stand-off, so the insertion never starts and
    #: the skill burns its whole budget.  1.0 is the middle of the working band.  Over 24
    #: randomized seeds it takes the strict end-of-episode score from 19/24 to 21/24 and
    #: the banked mean from 3.88 to 4.00.
    microwave_approach_funnel: float = 1.0
    #: How far *behind* the handle bar the leading fingertip should sit, in metres.  Yawing
    #: the grasp frame about the bar swings one pad into the gap behind the D-bar, so the
    #: pull loads that pad against the back of the bar rather than relying on grip force.
    #:
    #: Zero, i.e. a square door-normal approach, and that is now the right answer: the same
    #: retention comes from ``microwave_insertion_depth``, which is a translation rather
    #: than a rotation and therefore costs the wrist nothing.
    #:
    #: This was 0.020 in ``fast_demo`` and was the direct cause of the visible 180-degree
    #: wrist flip.  0.02 m of hook is a 30-degree yaw of the *whole approach frame*, applied
    #: from the stand-off onward, and reaching it drives joint 7 from 2.4 to its +2.897 bound
    #: and pins it there -- after which the arm can only recover by flipping to the other
    #: jaw representative.  It also made the approach oblique, which is what a viewer sees as
    #: the gripper missing the handle to one side.  Measured against the fixed frame and
    #: insertion depth, any nonzero hook is now strictly harmful: 8/8 at zero, 0/8 at 0.020.
    microwave_hook_depth: float = 0.0
    #: The transport waypoint is rebuilt from the measured fingertip position each step,
    #: this far ahead along the planar direction to the grasped bar's goal point. Its
    #: height is unchanged.  The lead is additionally clamped to the *remaining* distance,
    #: which is what decelerates the loaded push instead of driving at the step cap until
    #: the stop predicate happens to fire.
    kettle_transport_lookahead: float = 0.08
    #: How far along the capture-to-burner line the transport aims, in metres.  Zero
    #: restores heading straight for the goal from wherever the body has drifted to; see
    #: :meth:`KettleGraspSkill._transport_remaining` for why the route, and not just the
    #: endpoint, has to be named.
    kettle_transport_corridor_lookahead: float = 0.10
    #: Whether the flat push holds the height the grasp was made at.
    #:
    #: :meth:`KettleGraspSkill.grasp_goal_point` names only a planar displacement, so the
    #: target's height is inherited from wherever the tool currently is and nothing ever
    #: asks it back up.  The position actuators droop, the droop is never an error, and it
    #: accumulates: measured, the tool ends the push 4 cm below the post height it was
    #: given, which is far enough for the finger *bodies* to reach the kettle's own
    #: shoulder at local z = 0.116.  From there the hand pushes on the body box rather than
    #: the handle -- the grasp is levered open in the last few steps of the push, and the
    #: withdrawal drags that finger across the shoulder and rolls the delivered kettle to
    #: 21.5 degrees.
    kettle_transport_hold_height: bool = True
    #: Metres the flat push holds the tool *above* the height the grasp was made at.
    #:
    #: Not the three-legged carry of :attr:`kettle_lift_height`, which needs the overhead
    #: bail-bar grasp and its rise-cross-descend structure; this is the same push, asked to
    #: hold station a little higher.  A side grasp cannot hang a kettle from a smooth
    #: vertical post, so what a small offset buys is unweighting rather than clearance: the
    #: drag moment the push works against is the body's weight on its leading base edge, and
    #: some of that can be carried by the jaws without the grip having to hold the whole
    #: load.
    #:
    #: It works and it does not pay.  The grip carries the kettle 1:1 with what is asked,
    #: to 5 cm, without slipping on any seed -- so the capability is real -- but measured
    #: over the six randomized seeds whose kettle is actually delivered, mean position error
    #: goes 0.059 -> 0.076 and peak body tilt 2.0 -> 3.2 degrees at 0.02, and every one of
    #: the six is worse.  Success is untouched: 19/24 randomized and 24/24 fixed at 0, 0.01
    #: and 0.02 alike, medians within three steps.
    #:
    #: The reason there is nothing to buy is that the moment this would relieve is already
    #: gone.  ``kettle_lift_height`` below is justified by peak transport tilt of 28-37
    #: degrees; with the yawed grasp, the corridor and ``kettle_transport_hold_height`` in
    #: place that is now 2.0 degrees mean and 3.7 worst.  Raising the tool adds tilt back
    #: rather than removing it.
    kettle_push_lift_height: float = 0.02
    #: Kettle-only normalized Cartesian cap during loaded horizontal transport. Keeping it
    #: separate from contact_step avoids an abrupt acceleration as soon as the pads close.
    kettle_transport_step: float = 0.10
    #: Planar radius within which the kettle body counts as delivered.  Transport stops
    #: there rather than continuing to drive a body that is already at its goal position:
    #: the environment's 7-D distance is dominated by the yaw a side-grasped kettle picks
    #: up, so pushing past this point trades a little position for a lot of orientation.
    kettle_goal_position_tolerance: float = 0.02
    #: Metres *short* of the goal, measured along the route, at which the loaded push may
    #: stop; ``None`` keeps stopping on :attr:`kettle_goal_position_tolerance` alone.
    #: Positive stops early, negative demands an overshoot before releasing.
    #:
    #: That radius never fires on the flat push.  Traced on seed 0, the body closes to
    #: 0.056 m of its aim at step 201 -- outside the 0.02 -- and the arm, which has been
    #: leaning on a loaded kettle for sixty steps with a wound-up servo integrator, then
    #: carries it 0.13 m further to ``y = 0.889`` before the bias unwinds and drags it back
    #: to 0.824.  The excursion is the windup, not the plan: the tool's own aim point sits
    #: still at ``(-0.23, 0.79)`` throughout while the tool runs 0.12 m past it.
    #:
    #: Stopping at the plane through the goal, perpendicular to the route the corridor
    #: already defines, ends the push where the body has arrived rather than where a radius
    #: happens to close, and the phase change then discards the wound-up bias outright.
    #: Measured along the route, not as a radius, so a body that has drifted off the line
    #: still has to *arrive* rather than merely pass abeam of the burner.
    kettle_goal_crossing_margin: Optional[float] = 0.04
    #: Metres in world +x -- to the robot's right, away from the microwave -- added to the
    #: kettle goal the *policy* drives to.  Read by :meth:`KettlePushSkill._goal_pos`, so it
    #: moves the push heading, the transport's stop predicate and the tool's goal point
    #: together.
    #:
    #: The environment's own completion predicate is not touched: it measures against the
    #: unshifted `OBS_ELEMENT_GOALS` entry with a 0.3 threshold, and the transport delivers
    #: to about 0.025 m of whatever it aims at, so a few centimetres of offset is spent out
    #: of a very large budget.
    #:
    #: It exists because the last stretch of the carry runs the hand along the microwave.
    #: The kettle goal sits at x = -0.23 and the microwave door occupies x = -0.64 and
    #: leftward, so the approach to the burner brings the gripper in from the microwave's
    #: side; the same -x drift is called out in :meth:`KettleGraspSkill.grasp_goal_point`
    #: as the failure an earlier target was meant to fix.  Aiming a little right of the
    #: burner buys clearance for the hand without moving the kettle out of tolerance.
    #:
    #: Measured over 8 seeds on the fixed order with the microwave opened *first*, so its
    #: door is out in the path -- the collision does not reproduce at all with the door
    #: shut, nor on a kettle-only run, which stops the moment the task banks at the 0.3
    #: threshold and so never performs the carry:
    #:
    #:   ========  ===================  ==================  ==========
    #:   offset    microwave contact    kettle-body touch   goal error
    #:   ========  ===================  ==================  ==========
    #:   0.00      20.6 steps on 6 / 8  35.4 on 8 / 8       0.101
    #:   0.04       0.5 steps on 1 / 8   8.5 on 8 / 8       0.053
    #:   0.08       none                11.8 on 8 / 8       0.086
    #:   0.12       none                14.1 on 8 / 8       0.127
    #:   ========  ===================  ==================  ==========
    #:
    #: All 8 / 8 on the kettle throughout.  0.08 is the default because it clears the door
    #: outright, and it is *more* accurate against the environment's own goal than aiming
    #: straight at it -- the hand scraping the microwave was itself deflecting the carry.
    kettle_goal_lateral_offset: float = 0.08
    #: Steps the loaded transport may make no progress toward its aim before it gives up,
    #: or ``None`` to let it push until a tolerance fires.  Only ever consulted once the
    #: environment has already banked the kettle, so this cannot end a push that is still
    #: earning the task.
    #:
    #: Without it the transport has no exit while the jaws are still holding: the escape in
    #: `_reactive_task_satisfied` requires ``not grasp_retained()``, so a kettle that stops
    #: moving with the grasp intact is pushed forever.  A right-shifted aim is exactly what
    #: provokes that -- at an offset of 0.06 the body overshoots to ``(-0.20, 0.83)``, past
    #: the burner, and the error plateaus at 0.086 against a 0.02 tolerance for the whole
    #: remaining 480 steps of the episode.
    #:
    #: Distinct from the stall test recorded as harmful in
    #: :meth:`KettleGraspSkill.manipulation_done`'s history, which ended the push *before*
    #: the task banked and cost delivery accuracy.  This one can only fire afterwards.
    kettle_transport_stall_steps: Optional[int] = 30
    #: Improvement, in metres, that counts as progress for that stall test.
    kettle_transport_stall_epsilon: float = 0.004
    #: Rotation of the kettle grasp frame about the post the jaws close on, in radians.
    #: That post is a smooth vertical cylinder, so turning the wrist about its axis is a
    #: symmetry of the grasp -- every value grips equally well -- and the angle is therefore
    #: free to be chosen for the *carry* instead.  It is worth choosing: the arm runs out of
    #: wrist past y = 0.5, so a wrist placed square at the handle has to turn to finish the
    #: carry, and a friction grip on a round post passes part of that turn to the kettle.
    kettle_grasp_yaw: float = 0.4
    #: How square on the post the jaws must be before the final insertion, in metres of
    #: lateral offset.  Read by :attr:`KettleGraspSkill.grasp_center_tolerance`.
    #:
    #: This has to be sized against `kettle_grasp_yaw`, and that is not obvious.  The error
    #: it gates, ``|dot(contact - tool, eef_finger_axis)|``, would go to zero for a wrist
    #: exactly on its target frame -- the finger axis is built perpendicular to the approach
    #: -- so what it actually measures at distance ``d`` with a residual frame error ``theta``
    #: is about ``d sin(theta)``.  That is a *floor*, not an offset the centring stage can
    #: drive out, and a yawed grasp frame raises it: the wrist settles at 0.02 rad un-yawed
    #: but rings between 0.10 and 0.19 at a 0.4 yaw, which floors the error near 0.01-0.02.
    #: Below that floor the stage never finishes, and because it drives with ``orient=False``
    #: the frame drifts back up while it tries, ALIGN interrupts, and the skill cycles inside
    #: MOVE_TO_PRECONTACT forever.
    #:
    #: Measured at ``kettle_grasp_yaw = 0.4``, kettle alone over 8 seeds: 0.0025 gives 0/8,
    #: 0.008 gives 1/8, 0.015 gives 6/8, and 0.020 upward gives 8/8.  Over the full four
    #: tasks and 24 randomized seeds, 0.0025 gives 1/24 and 0.025 gives 16/24.  It costs
    #: nothing un-yawed -- kettle alone is 8/8 either way, and a little quicker at 0.025
    #: (median 43 steps against 48) -- so the looser gate is the default rather than
    #: something switched on with the yaw.
    #: How far below its own grasp point the tool may sit and still be allowed to start
    #: the kettle APPROACH.
    #:
    #: The approach runs almost horizontally into a side post, so its height is set by
    #: wherever the reach left the tool, and a reach that arrives low drives the pads into
    #: the kettle's shoulder instead of alongside the post.  Measured over 8 seeds, the
    #: drop below the grasp point at the end of the reach is 0.027-0.047 m on the seeds
    #: that grasp cleanly and 0.069 on seed 4, which comes in from the slide cabinet, and
    #: seed 4 is the one that touches the body for 3 steps and shoves it 6 cm -- after
    #: which the grasp needs three attempts, because the post is no longer where the skill
    #: planned for.
    #:
    #: Blocks *entry* only.  An approach already committed to its corridor is left to
    #: finish: backing a committed approach out is what the corridor test exists to
    #: prevent, and a guard that could fire mid-insertion would reintroduce exactly the
    #: MOVE_TO_PRECONTACT/APPROACH alternation that test was written to stop.
    #: 0.055 rather than the 0.05 the two populations suggest.  Seed 11's reach drops
    #: 0.061 and seed 4's 0.069, so any threshold that catches seed 4 catches seed 11 too;
    #: what matters is how long the guard then holds, because it holds until the tool is
    #: back within it.  At 0.05 seed 11 spends enough extra steps lifting that its light
    #: switch never gets its budget, and the harness pays 21/24 for seed 4's 3 contacts --
    #: a seed traded, not won.  At 0.055 the guard releases sooner, seed 11 is 4/4 again
    #: and seed 4 keeps the better figure of the two (295 steps against 318 at 0.065).
    kettle_approach_height_slack: float = 0.055
    kettle_grasp_center_tolerance: float = 0.025
    #: Whether the kettle's jaw-centring threshold is widened by the offset the *wrist tilt*
    #: alone accounts for, as a multiple of it.  0 disables; 1.0 subtracts it exactly.
    #:
    #: ``jaw_center_error`` is ``|dot(contact - tool, eef_finger_axis)|``, and the finger
    #: axis is built perpendicular to the approach, so for a wrist sitting on its target
    #: frame the quantity is a genuine lateral offset.  For a wrist ``theta`` off that frame
    #: at range ``d`` it also contains ``d sin(theta)``, which centring cannot remove --
    #: translating does not rotate the wrist.  Worse, the centring branch drives with
    #: ``orient=False``, so while it runs the frame decays until ALIGN fires, and that decay
    #: feeds straight back into the error being centred.  Traced at ``kettle_grasp_yaw =
    #: 0.4``, seed 0: the two ring against each other with the frame between 0.11 and 0.19
    #: and the jaw error tracking it, 0.073 at the top of each cycle and 0.042 at the
    #: bottom, creeping down about 0.002 per cycle -- 130 steps of MOVE_TO_PRECONTACT and
    #: ALIGN before the approach finally starts.
    #:
    #: Subtracting the term the tilt explains leaves centring gating on the offset it can
    #: actually correct, and on the kettle alone that is exactly what happens: at
    #: ``kettle_grasp_yaw = 0.4`` over 8 seeds it stays 8/8 and the median falls from 66
    #: steps to **30**, quicker even than the un-yawed grasp's 48.
    #:
    #: **Off anyway, because it does not survive the full episode.**  Over 24 randomized
    #: seeds with the park table disabled: un-yawed it takes 23/24 to **19/24**, and at a 0.4
    #: yaw 20/24 to 19/24.  Letting the approach start with a tilted wrist is what the
    #: centring stage exists to prevent -- the outside of a finger reaching a free body
    #: before the jaws straddle it -- so buying speed by forgiving the tilt gives back the
    #: grasps it was protecting.  A fix that keeps the speed has to stop the frame decaying
    #: during centring rather than excuse the decay, and the obvious form of that (centring
    #: with ``orient=True``) is already recorded as worse in the branch itself.
    #:
    #: Kept, with its measurement, because the ring it targets is real and costs about 130
    #: steps per yawed kettle grasp.
    kettle_centering_tilt_slack: float = 0.0
    #: Per-step rotation cap applied *during* the kettle's jaw-centring stage, or ``None``
    #: to command no rotation at all there.
    #:
    #: Centring translates with the wrist unheld, and that is the whole of the ring: the
    #: frame decays from 0.10 to 0.19 rad while it drives, the decay re-enters the error it
    #: is centring on through the ``d sin(theta)`` term (see `kettle_centering_tilt_slack`),
    #: ALIGN fires to undo it, and the two alternate for about 130 steps.
    #:
    #: Handing the stage the *full* rotation gain was tried and is worse -- the correction
    #: swings the tool on its ``FINGERTIP_OFFSET`` lever and fights the centring it is
    #: interleaved with (seed 42: 4/4 to 2/4, centring 76 steps to 104).  A cap is the
    #: difference: `rotation_action` already points the correction the right way, so a small
    #: bound holds the frame against decay while moving the tool a fraction of what the
    #: uncapped version does.  It is the fix for the ring, and a large one.  Measured at
    #: ``kettle_grasp_yaw = 0.4`` over 24 seeds, against ``None``:
    #:
    #:   ====================  =================  =================
    #:   run                   None               0.20
    #:   ====================  =================  =================
    #:   kettle alone          23 / 24, 71 steps  24 / 24, 38 steps
    #:   four tasks, random    20 / 24, med 432   21 / 24, med 427
    #:   four tasks, fixed     16 / 24, med 477   21 / 24, med 342
    #:   ====================  =================  =================
    #:
    #: and it is free un-yawed: 23 / 24 at a median of 430 either way.  The cap matters at
    #: the top end -- 0.40, i.e. the full `max_rotation_step`, matches 0.20 everywhere
    #: except un-yawed randomized, where it costs a seed (22 / 24), which is the older
    #: uncapped result showing through.  0.10 loses a seed on the kettle alone.
    kettle_centering_rotation_step: Optional[float] = 0.20
    #: Which kettle handle the jaws take: the side post, or the top bail bar.
    #:
    #: The side post is 9.2 cm off the body's centre line, so every newton of push through
    #: it is also a yaw torque, and -- the part that actually decides this -- the wrist
    #: cannot hold that grasp frame at the burner.  Probed directly: the side-grasp pose at
    #: the goal has an IK residual of 0.125 rad / 0.066 m, so the arm *must* turn to finish
    #: the carry, and a friction grip on a round post hands part of that turn to the kettle.
    #: The top bar is on the centre line (no moment arm at all) and its overhead grasp pose
    #: probes at 0.0000 rad / 0.3 mm at the start and 0.0004 rad / 1.0 mm at the goal, i.e.
    #: one wrist orientation serves the whole carry.
    #:
    #: Measured over 8 seeds, with the carry as a *push* along the counter:
    #:
    #: ============================ ======== ===== ==========
    #: grasp                        delivery roll  body tilt
    #: ============================ ======== ===== ==========
    #: side post, wrist free        0.030 m  9.3   0.4
    #: side post, wrist held        0.101 m  16.8  0.0
    #: top bar,   wrist free        0.059 m  61.3  1.3
    #: top bar,   wrist held        0.165 m  2.5   51.5
    #: ============================ ======== ===== ==========
    #:
    #: A horizontal bar in parallel jaws is form-closed in yaw, so where the round post
    #: slips and passes on only part of a wrist rotation, the top bar passes on all of it --
    #: hence 61 degrees with the wrist left free.  Holding the wrist does then deliver the
    #: near-zero roll the geometry promises, and tips the kettle over instead: the grip is
    #: 0.19 m above the centre of mass and the body is only 0.122 m in half-width, so the
    #: hold levers it onto its side (51 degrees) well before the burner.
    #:
    #: Every entry in that table is a *push*, and the last line points at the answer:
    #: carrying a kettle by its bail handle wants a lift.  :attr:`kettle_lift_height` is
    #: that lift, and on the bench it does exactly what the geometry promises -- the body
    #: stays within 0.2 degrees of level from one end of the kitchen to the other, against
    #: the 28-53 degrees of transient tilt the push rocks up to.
    #:
    #: It is still ``side``, because the arm cannot afford it.  A hanging kettle is a
    #: pendulum about the bail bar, and driving it faster than 0.06 of the Cartesian box
    #: turns it right over, which caps the crossing at roughly three times the push's
    #: duration: measured in the randomized suite the kettle went from 54-76 steps to
    #: 236-626, ate the task budget on half the seeds and left the other tasks unreached.
    #: The knobs are all here and the mode works; it needs a faster way across the counter
    #: than a position-servo lead of 0.06, which is a controller change, not a skill one.
    kettle_grasp_target: str = "side"
    #: Height above the kettle root at which the jaws take the side post, in metres.  The
    #: post's collision capsule is centred at 0.18 with a half-length of 0.06, so it can be
    #: held anywhere from about 0.13 to 0.23.
    #:
    #: This is the lever the push tips the kettle with.  Dragging a free body by a handle
    #: ``h`` above the surface it is standing on is a moment about the leading edge of its
    #: own base, resisted by ``m*g*half_width`` -- 1.05 N m for this kettle -- so the drag
    #: force it takes to rock it up is ``1.05 / h``: 5.8 N at 0.18 and 8.1 N at 0.13.  That
    #: is the arithmetic of the roll, and it says to hold the post lower.
    #:
    #: It does not work, and the reason is worth writing down because the measurement looks
    #: like it does.  At 0.12 the kettle finishes upright on every one of 24 seeds with a
    #: peak tilt of 0.7 degrees against the 33 this height gives -- and it is never grasped:
    #: traced, the hand arrives, shoves the body 12 cm with the outside of a finger, and the
    #: environment banks the task on the shove.  The jaws cannot get that low.  The body's
    #: shoulder is at 0.116 and the post is only 0.023 in radius, so a pad reaching for
    #: 0.12 meets the kettle before it meets the handle; 0.135 and 0.14 are the same failure
    #: half the time (5-6/8, and the seeds that do grasp still tip).  Anything that reads as
    #: an improvement here should be checked against ``contacting_fingers`` before it is
    #: believed.
    #:
    #: Raising it is the other direction, and it is a real trade rather than a trap: what
    #: tips the delivered kettle is the finger *bodies* reaching its shoulder at local
    #: z = 0.116, so more clearance is less roll.  Over 24 randomized seeds, 0.19 takes the
    #: worst peak tilt from 44.5 degrees to 15.0 and 0.20 to 5.6 -- but the higher the grip
    #: the less of the post is in the jaws, and the yaw and the delivery pay for it: 0.19
    #: costs mean yaw 16.8 -> 21.5 degrees, pose distance 0.207 -> 0.241 and 18% more steps
    #: at the same 19/24, and 0.20 collapses to 13/24.  0.18 is kept because the tilt it
    #: gives is already below what is visible on all but a handful of seeds -- 17 of those
    #: 24 finish under 4.3 degrees -- and it is the best of the three everywhere else.
    kettle_side_grasp_height: float = 0.18
    #: Finger command for the top bar, which at radius 0.032 is nearly twice the side post's
    #: 0.023 and needs its own set point; see ``KettleGraspSkill.gripper_command``.
    kettle_top_grasp_command: float = 0.45
    #: What the carry does with the wrist.  What the hand does with its wrist is what the
    #: kettle does with its own pose, so this is the whole of the kettle's roll:
    #:
    #: ``"free"``    -- command nothing, which is what this used to do.  The 7-DOF null
    #:                 space is then unconstrained for the thirty steps of the push and the
    #:                 wrist simply drifts into it: measured over 8 seeds, it rolls to 37.4
    #:                 degrees off vertical (44.8 at worst) against a command that never
    #:                 leaves 5.1, and the kettle follows it over to 32.7 (41.7), settling
    #:                 flat again only once the jaws open.
    #: ``"capture"`` -- hold the whole frame the grasp was made in, levelled about vertical
    #:                 (``kettle_upright_carry_frame``).  This is the default.  It needs
    #:                 the other two halves of the fix to pay off -- a lean that makes the
    #:                 frame reachable at the burner (``kettle_approach_lateral_bias``) and
    #:                 a push that holds its height (``kettle_transport_hold_height``) --
    #:                 and with them, over 24 seeds, the kettle's peak tilt is 5.1 degrees
    #:                 against 38.0, and 2.5 during the push itself against 35.6.
    #: ``"level"``   -- command the roll and nothing else, by targeting the *live* wrist
    #:                 levelled about vertical (:func:`upright_grasp_frame`).  The command
    #:                 is rebuilt every step from the measurement, so it says only "stop
    #:                 leaning" and can neither wind up nor fight the yaw the body picks up.
    #:
    #: This used to be off, because with a *side* grasp the arm cannot do both: that frame
    #: has an IK residual of 0.125 rad / 0.066 m at the burner, so holding it stops the
    #: carry short -- 0.101 m out against 0.030 -- and the roll comes out *worse* at 16.8
    #: degrees, the wrist having spent the carry fighting a pose it cannot reach.  The
    #: overhead top-bar frame has no such problem: probed straight down onto the bail bar,
    #: the residual is 0.0001 rad / 0.3 mm at the kettle and 0.0009 rad / 2.0 mm at the
    #: burner, at every bar yaw from -0.6 to +0.6 rad and every lift from 0 to 0.20 m.  One
    #: wrist orientation serves the whole journey, so holding it costs nothing.
    #:
    #: Off, because the default grasp is the side post again; see ``kettle_grasp_target``.
    kettle_carry_orientation: str = "capture"
    #: Whether the frame frozen at capture is levelled about vertical before it is held.
    #:
    #: The frame is the *measured* wrist, so it inherits whatever tilt the approach was
    #: allowed to finish with -- 5.1 degrees on average over 8 seeds, 6.5 at worst.  The
    #: jaws hold the post along the tool's own +x, so that tilt is handed straight to the
    #: kettle and it is being commanded, not drifted into.  See
    #: :func:`upright_grasp_frame`.
    kettle_upright_carry_frame: bool = True
    #: How far the tool lifts the kettle off the counter before carrying it, in metres.
    #: Zero restores the old push, which is the only mode a side grasp has.
    #:
    #: The push is what put the roll in the roll problem.  Dragging a body by a handle
    #: 0.18-0.26 m above a counter it is still resting on is a moment about the leading
    #: edge of its own base, and the kettle answers by rocking up onto that edge: measured
    #: through the full randomized run, the *peak* body tilt during transport is 28-37
    #: degrees on 17 of 24 seeds and 53 on the worst.  It falls back flat on arrival, which
    #: is why an end-of-episode tilt check reads 0.4 degrees and sees none of it.
    #:
    #: Lifting removes the moment rather than balancing it.  With the body off the counter
    #: there is no edge to pivot about and no friction to drag against, the load hangs
    #: under a bail bar that is on the body's centre line, and the form-closed bar takes
    #: its heading straight from the held wrist.
    kettle_lift_height: float = 0.09
    #: How close to :attr:`kettle_lift_height` the tool must come before the carry begins.
    kettle_lift_tolerance: float = 0.015
    #: How close to the height it was picked up from the kettle must come back before the
    #: carry is finished and the jaws may open.
    kettle_place_tolerance: float = 0.012
    #: Cartesian step scale and lead for the vertical leg alone, both deliberately smaller
    #: than the cross-counter ones.
    #:
    #: A bail bar held from directly above is stable only while the pads sit on its widest
    #: line.  The jaws close on a circular cross-section, so pads that ride *up* the
    #: cylinder meet a narrowing profile and, being position servos with 3 mm of
    #: interference commanded, squeeze the bar back down and out: the expulsion component
    #: is ``h / sqrt(r^2 - h^2)`` of the normal force, which passes half at 14 mm of climb
    #: and then runs away.  Nothing about the hold is marginal -- 100 N of grip against a
    #: 8.6 N kettle -- so this is purely a transient: lift faster than the load can follow
    #: and the pads climb before friction has taken up the weight.  Traced on seed 0 at the
    #: transport's own 0.08 lead, the tool accelerated to 9 mm/step, outran the kettle by
    #: 17 mm and shed the bar at 3.6 cm of lift.
    kettle_lift_step: float = 0.05
    kettle_lift_lookahead: float = 0.02
    #: Cartesian step scale and lead for a *lifted* cross-counter carry, as opposed to the
    #: dragged one ``kettle_transport_step`` sizes.
    #:
    #: Half the speed, because the two carries end in different places kinematically.  A
    #: push keeps the wrist low and roughly level with the counter; a lift holds it above
    #: the kettle, and the overhead pose near the burner needs joints 1 and 4 to swing hard
    #: for very little tool travel.  Traced across the last third of the carry at the
    #: transport's own 0.08, both pin at the edge of the action box, the arm stops steering
    #: and the held frame goes with it -- 0.005 rad of orientation error becomes 0.48, the
    #: kettle tips 18 degrees and is flung past the burner.
    #:
    #: With that fixed (see :attr:`kettle_carry_bias_leak`) the ceiling becomes the load
    #: itself.  A kettle hanging from a bail bar can rotate about it -- a cylinder in flat
    #: jaws is form-closed across its axis and free around it -- so the carry is a pendulum
    #: swinging in the direction of travel, and how hard it is driven decides whether it
    #: swings or goes over.  Over 8 seeds, at every lift height from 0.02 to 0.05: 0.06
    #: delivers 8/8 with the body 0.2 degrees off level, 0.08 turns it right over (110
    #: degrees) on 6 of them, and 0.10 on 7.
    kettle_carry_step: float = 0.06
    kettle_carry_lookahead: float = 0.06
    #: Planar distance from the goal at which a lifted carry puts the kettle back down and
    #: finishes on the counter, rather than flying it all the way in.
    #:
    #: The last few centimetres are the hardest part of a carry and the easiest part of a
    #: push.  A hanging kettle is a pendulum whose planar error keeps crossing any tight
    #: tolerance from both sides, so aiming the *airborne* body at
    #: ``kettle_goal_position_tolerance`` never converges -- measured, the transport spent
    #: its whole 626-step task budget hovering 6 cm out and never declared itself done.  A
    #: kettle standing on the counter has no pendulum and plenty of friction, and 5 cm is a
    #: short enough nudge to close precisely without building the tipping moment a
    #: full-length drag does.
    #:
    #: 0.02 -- the arrival tolerance itself, i.e. no handoff -- because the drag turns out to
    #: tip the kettle even over 5 cm: dragging by the bail bar is the worst lever on the
    #: body there is, 0.26 m up, and over 8 seeds the handoff put the mean peak tilt back to
    #: 22.7 degrees from the 0.2 the carry alone gives.  Kept as a knob because the pendulum
    #: it was meant to answer is real; it wants a shorter handoff than 5 cm, or a way to set
    #: the kettle down that does not then push it.
    kettle_setdown_distance: float = 0.02
    #: Ceiling on the servo's integral bias, in radians, while a lifted carry is crossing
    #: the counter.  ``servo_integral_clamp`` applies everywhere else.
    #:
    #: :class:`JointServo` integrates to cancel gravity droop, and it cannot tell droop from
    #: the velocity lag of a joint tracking a target that moves every step.  Most phases
    #: never notice, because they are short and the bias is dropped at each phase change.  A
    #: lifted carry is one phase of about fifty steps with a new waypoint in every one of
    #: them, which is the worst case for that: joints 1 and 4 lag about 0.016 rad a step,
    #: bank it, and reach ``servo_integral_clamp`` two thirds of the way across.  0.15 rad
    #: of bias is 0.94 of the action box by itself, so both joints then pin at 1.0 with
    #: nothing left to steer with -- the held wrist frame sheds 0.48 rad, the kettle tips 18
    #: degrees and is flung past the burner.
    #:
    #: A second, tighter ceiling for that one phase is the whole fix.  It has to be a
    #: ceiling and not a discard: the bias is also what holds a loaded arm up, and shrinking
    #: it every step -- tried at 0.8 -- sags the wrist 4 cm, puts the kettle back on the
    #: counter to be dragged, and stalls it 6 cm short of the burner with the hand pressing
    #: down on it.  0.05 rad still covers the droop (0.018 on the shoulder under load,
    #: measured) and is only 0.31 of the action box, so the arm keeps most of its authority.
    #:
    #: And only that phase.  Applying the same ceiling globally breaks the *approach*, which
    #: legitimately earns a large bias closing the last centimetres onto the handle: capped
    #: at 0.05 throughout, the descent onto the kettle stops 2-3 cm high -- the droop, back
    #: again -- and the grasp is made at the wrong height and slips at once.
    kettle_carry_bias_clamp: float = 0.05
    #: How far past the bail bar's axis the fingertips descend before closing, in metres.
    #:
    #: The Franka pad is not a flat face.  Its contact capsules taper: the inner surface
    #: stands 1.3 mm proud of the finger at the very tip and 2.5 mm at the back of the pad,
    #: so the jaws are a shallow wedge that is narrowest at the fingertips.  A cylinder
    #: clamped in a wedge is driven toward the wide end; here that is *out the front of the
    #: jaws*, which on an overhead grasp means straight down, with the kettle's own weight
    #: pulling the same way.  Stopping the tool at the bar's axis -- the obvious contact
    #: point, and what every other skill here wants -- leaves the bar sitting mid-taper and
    #: free to walk out: measured on seed 0 it slipped 1.4 cm through the pads over a 2.7 cm
    #: rise and was gone by 3.6 cm, at every lift speed tried.
    #:
    #: Going deeper seats the bar at the back of the pad, where the taper works for the
    #: grasp instead of against it -- escaping downward now means wedging into the narrowing
    #: part.  The pad spans 15.5 mm ahead of the tool point and 37 mm behind it, so this has
    #: room up to about 0.03; nothing occupies the column under the bar's centre until the
    #: kettle body itself, 14 cm lower.
    kettle_top_grasp_depth: float = 0.02
    #: How far *past* the left post the side grasp drives before closing, i.e. how deep the
    #: post is seated between the pads.
    #:
    #: Every other grasp in this file has one -- `light_insertion_depth` 0.028, whose own
    #: comment is "put the capsule inside the pad length rather than pinching it at the
    #: distal edge", `microwave_insertion_depth` 0.025 for retention, `kettle_top_grasp_depth`
    #: 0.02 -- and the kettle's side grasp had none.  `contact_point` returned the post
    #: centre bare, and the tool point *is* the fingertip, so the post was held at the very
    #: tip of the jaws.
    #:
    #: That is what breaks the carry.  Under push load a post pinched at the distal edge
    #: chatters, contact reporting drops for a frame, and the skill correctly concludes it
    #: has lost the grasp and goes back for it -- on seed 6 at step 211, with the
    #: half-opening motionless at 0.0185 through the break, costing 14 steps.  Attacking
    #: that from the retention end does not work: bridging the dropout takes the seed 4/4 to
    #: 2/4, gripping harder loses a task at -0.30 and quadruples the breaks at -0.50, and
    #: retaking it in place instead of walking back takes it 4/4 to 2/4.  Seating the post
    #: deeper removes the chatter instead of arguing with its consequences.
    #:
    #: 0.01, and more is not better -- what matters is the depth the post *reaches*, not the
    #: one commanded.  Measured over the 24-seed harness as the mean along-approach offset
    #: of the fingertip point past the post during the carry:
    #:
    #:   ====== ============== ================= ============
    #:   asked  carry breaks   seating reached   full 4/4
    #:   ====== ============== ================= ============
    #:   0.00   2              +0.0300           21/24
    #:   0.01   0              +0.0332           21/24
    #:   0.02   -              -                 19/24
    #:   0.03   2              +0.0209           21/24
    #:   ====== ============== ================= ============
    #:
    #: Asking for 0.03 seats the post *less* deeply than asking for 0.01: the target is far
    #: enough past the bar that the approach drives through the seating pose instead of
    #: stopping in it.  0.02 measures 19/24 while both its neighbours measure 21/24, so it
    #: is a knife edge on particular seeds rather than a trend -- do not read this table as
    #: "deeper is worse" or tune it on a handful of seeds, which is how 0.02 was picked
    #: before the full harness rejected it.
    #: 0.01 was measured on a 400-step harness, which is not the budget this environment
    #: runs at -- `llfbench.envs.kitchen.make` defaults to ``episode_steps=1000`` -- and the
    #: truncation hid the failure it causes.  At 0.01 the post is held at the very tip of
    #: the jaws: on seed 7 the grasp closes cleanly (both pads, the bar 3.5 mm off the jaw
    #: axis), then over eight steps of push the bar walks out to 17 mm, the left pad lets
    #: go, and the jaws reopen.  What follows is not a retry but a *permanent* seven-step
    #: limit cycle -- approach, back off because the bar is 26-33 mm off the jaw axis,
    #: align, approach, close on air, reopen -- which at 400 steps looks like an episode
    #: that merely ran out of time and at 1000 is 650 steps of the gripper pawing at the
    #: kettle until the task is abandoned.  That is what a viewer reports as "it never grips
    #: the kettle properly".
    #:
    #: Re-measured at 1000 steps, seed 7 goes 2/4 to 3/4 and the suite improves: kettles
    #: banked 22/24 to 23/24, four-task episodes 21/24 with the mean 3.79 to 3.83, and total
    #: distance the kettle is *carried* 8.28 m to 8.48 m.  0.03 measures the same success
    #: with slightly less shoving and 8.25 m carried; 0.04 still measures 21/24.  The pad has
    #: room to about 0.03 (see `kettle_top_grasp_depth`), so 0.02 keeps a margin.
    kettle_side_grasp_depth: float = 0.02
    #: Gain on the translation-only kettle yaw correction; see
    #: :meth:`KettleGraspSkill._yaw_correction`.  1.0 asks for exactly the post displacement
    #: that squares the body up, which the friction grip only partly delivers.
    kettle_yaw_correction_gain: float = 1.0
    #: Planar radius within which a kettle that has slipped out of the jaws counts as
    #: delivered anyway, and is left alone rather than reacquired.  Looser than
    #: :attr:`kettle_goal_position_tolerance`, which stops a *held* transport: this one only
    #: has to separate "dropped it on the burner" from "dropped it halfway there".
    kettle_delivered_position_tolerance: float = 0.06
    #: Height of the *EEF site* above the kettle root while pushing.  ``KettlePushSkill``
    #: has ``tool_offset = 0``, so this is the hand flange, and the fingers hang roughly
    #: ``FINGERTIP_OFFSET`` below it and sweep the kettle body (which spans 0.0-0.116 m
    #: above the root).  This is 0.094 above the intended fingertip contact height, plus
    #: FINGERTIP_OFFSET to convert to the site the value is measured from.  It was a bare
    #: 0.202, which is 0.094 + 1.2.0's 0.108 wrist offset; under 1.2.1's 0.052 site that put
    #: the hand 5.6 cm too high.  Only ``kettle_strategy='push'`` uses it.
    kettle_push_height: float = 0.094 + FINGERTIP_OFFSET
    #: How far the kettle side grasp leans away from the push direction, toward the free
    #: side of the handle bar.  0 approaches straight along the push, 1 comes in at 45
    #: degrees.  See :meth:`KettleGraspSkill.approach_axis`.
    #:
    #: This was zero, on the measurement that the lean cost delivery accuracy.  It did, but
    #: only because the carry was not holding the wrist at the time; the lean's real job is
    #: to make a holdable frame, and there is nothing to hold when the wrist is free.
    #:
    #: What it buys is reach.  An upright grasp frame's IK residual at the burner, probed
    #: along the whole push, is 0.122 rad / 0.067 m approaching straight along the push and
    #: 0.0002 / 0.0002 once the approach is turned 10-30 degrees off it -- and 0.067 m is
    #: exactly where a held-frame push stalls, the arm having run out of wrist rather than
    #: out of anything else.  0.40 puts the approach 21 degrees off the push.
    #:
    #: Swept over 24 seeds with the carry holding its frame: 0.30 and 0.40 both reach
    #: 21/24, 0.33 falls to 18/24, and the neighbourhood is noisy enough on 8 seeds
    #: (8/8 at 0.33 and 0.40, 6/8 at 0.36) that nothing here should be read off a short run.
    kettle_approach_lateral_bias: float = 0.40
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
            completion_margin=0.7,
            light_approach_step=0.45,
            light_manipulation_step=0.12,
            microwave_precontact_step=0.70,
            microwave_approach_step=0.70,
            microwave_manipulation_step=0.20,
            # The height offset compensated for the old fraction-of-error controller and is
            # now counterproductive: it cancelled "a repeatable upward IK residual at
            # contact" that the joint servo does not produce.
            #
            # The tolerance was relaxed to 0.03 with it, and that part was wrong.  It is an
            # *arrival* test on `contact_point`, which already sits
            # `microwave_insertion_depth` = 0.025 past the bar, so a 0.03 tolerance lets the
            # jaws close while the tool is still behind it: measured, a commanded seating of
            # +0.0250 reached only +0.0078 at capture on seed 6, after which the pull dragged
            # the bar back out through the fingertips (+0.0078, +0.0024, -0.0035, -0.0144,
            # pads 2 to 1 to 0, jaws collapsing to 0.0081 on air) and the skill re-gripped.
            # This is the slide cabinet's fake grasp again -- closing before going deep
            # enough -- which `slide_contact_tolerance` already fixes for that skill.
            #
            # Deeper is not the lever here: `microwave_insertion_depth` is capped by the
            # door panel at 0.027 (7/8) and 0.030 (0/8).  The depth asked for was right; the
            # arrival test was letting go of it early.  Over 24 seeds, 0.02 takes the pull
            # breaks from 7 to 0, the seating reached from +0.0133 to +0.0208, and the
            # median episode from 275 steps to 264, with 21/24 unchanged.  0.015 seats no
            # better (+0.0214) and measures 20/24.
            microwave_contact_tolerance=0.02,
            microwave_contact_height_offset=0.0,
            microwave_alignment_exit_tolerance=0.23,
            # 0.35 arrives at the bar too fast to stop.  `cartesian_plan_to_action` commands
            # `eef + step * (waypoint - eef)`, so the first step of a long drop is that
            # fraction of the whole drop, and the arm cannot brake inside its own lag: from
            # the park ALIGN leaves on randomized seed 0, a 33 cm descent overshoots 10.3 cm
            # *below* a waypoint commanded at 1.800, bottoming at 1.697 with the bar at
            # 1.799, and the integrator then takes ten steps to haul it back.  The shallower
            # 25 cm drop of the fixed order only overshoots 2.9 cm, which is why this reads
            # as a seed-dependent dip rather than a constant one.
            #
            # 0.30 removes it: seed 0's floor goes 1.697 -> 1.774, which is the ordinary
            # droop every other seed shows, and the worst floor over 8 randomized seeds goes
            # 1.697 -> 1.750.  It costs about ten steps of median episode length and no
            # seeds at all -- 19/24 randomized and 24/24 fixed, the same as before, failing
            # on the same five randomized seeds (7, 15-18).
            #
            # The floor is flat below this and the successes are not, so there is nothing to
            # gain by going lower.  0.25 lands the floor within 3 mm of 0.30 (1.772 / 1.747)
            # but costs randomized seed 20, which finishes at 392 of its 400 steps; 0.20
            # lifts the worst floor only to 1.765 and takes the *fixed* order to 23/24.
            kettle_precontact_step=0.30,
            kettle_approach_step=0.15,
            kettle_orientation_tolerance=0.18,
            kettle_alignment_exit_tolerance=0.10,
            # Loaded transport is the one motion here that is limited by grip rather than
            # by the arm.  The pads straddle the bar along the axis *perpendicular* to the
            # push, so every newton of push is carried by friction alone, and a fast step
            # simply slides the bar out: measured over 8 seeds, 0.30 broke the grasp 6 to 12
            # steps into every transport and left the kettle a third of the way there, 0.15
            # reached 5/8, and 0.08 holds the grasp for the whole carry at 8/8.
            kettle_transport_step=0.08,
            kettle_recede_step=0.35,
        )
        values.update(overrides)
        return cls(**values)


# ---------------------------------------------------------------------------------------
# Sim access -- the single place in this package that reaches into MuJoCo
# ---------------------------------------------------------------------------------------

class JointServo:
    """Turns a joint-space target into the environment's 9-D normalized velocity action.

    ``FrankaRobot.step`` computes ``ctrl = last_measured_qpos + action * 2.0 * dt`` and hands
    ``ctrl`` to position actuators, so asking for ``action = (q_target - q) / (dt * 2)``
    commands exactly ``q_target``.  That is the whole proportional term; the action box then
    caps how far one step may travel, which makes this a saturating P controller.

    The integral term is what makes it accurate.  Because ``ctrl`` re-anchors to the
    *measured* ``qpos`` every step, the position servo's steady-state droop under gravity is
    re-incurred every step rather than being a one-off offset: measured 0.039 rad on the
    shoulder and 0.022 on the elbow, which is 2-4 cm at the fingertips and enough to miss
    every grasp.  Accumulating the residual removes it -- measured 0.039 -> 0.0005 rad, and
    end-effector error under 1 mm.

    The integrator is frozen on any joint whose command is saturated, so a long transit at
    full speed cannot wind up a bias that then overshoots on arrival.
    """

    def __init__(self, sim: "KitchenSim", ki: float, i_clamp: float,
                 proportional: float = 0.0, feasible_windup: bool = False):
        self._sim = sim
        self._ki = float(ki)
        self._i_clamp = float(i_clamp)
        self._bias = np.zeros(ACTION_DIM)
        #: Scale the seven arm commands as one vector instead of clipping them
        #: independently.  See `KitchenPolicyConfig.proportional_arm_command`.
        self._proportional = float(proportional)
        #: Freeze the integrator at the actuator's ceiling rather than the action box's.
        #: See `KitchenPolicyConfig.servo_feasible_windup`.
        self._feasible_windup = bool(feasible_windup)
        #: Largest normalized command each arm hinge can actually execute:
        #: ``ctrl = q + action * 2 * dt`` asks for ``action * 2`` rad/s, and the actuator
        #: tops out at :data:`JOINT_SPEED_LIMIT`, so anything above this ratio is command
        #: the arm silently discards.
        self._feasible = JOINT_SPEED_LIMIT[:7] / ACTION_VELOCITY_RANGE

    def reset(self) -> None:
        self._bias[:] = 0.0

    def relax(self, factor: float = 0.0) -> None:
        """Shrink the accumulated bias, e.g. when the target jumps to a new waypoint.

        A bias earned while closing on one waypoint is not evidence about the next one, and
        applying it unchanged kicks the arm off the new approach line.
        """
        self._bias *= float(factor)

    def limit(self, ceiling: float) -> None:
        """Hold the bias to ``ceiling`` radians for now, without discarding it.

        For a phase that runs long enough to bank velocity lag as if it were droop.  The
        ordinary clamp has to be loose enough for the bias an *approach* earns; a phase that
        needs the integrator only to carry weight can ask for a tighter one while it lasts.
        """
        np.clip(self._bias, -abs(float(ceiling)), abs(float(ceiling)), out=self._bias)

    def action(self, q_target: np.ndarray) -> np.ndarray:
        """``q_target`` is 9 values: seven arm hinges then the two finger slides."""
        q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
        error = q_target - self._sim.data.qpos[:ACTION_DIM]
        dt = self._sim.kitchen.robot_env.dt
        command = (error + self._bias) / (dt * ACTION_VELOCITY_RANGE)

        # Freeze the integrator on a saturated joint.  The box is the wrong saturation
        # test for the arm: `ctrl = q + action * 2 * dt` asks for `action * 2` rad/s and the
        # hinge tops out at `JOINT_SPEED_LIMIT`, so a command of 0.5 is already past what
        # joints 1-4 can deliver and everything from there to 1.0 is discarded by the
        # actuator.  Testing against 1.0 therefore calls a joint unsaturated for more than
        # half its command range and lets the bias wind up on exactly the joints whose error
        # cannot close -- which is the definition of windup.
        ceiling = np.ones(ACTION_DIM)
        if self._feasible_windup:
            ceiling[:7] = self._feasible
        unsaturated = np.abs(command) < ceiling
        self._bias[unsaturated] += self._ki * error[unsaturated]
        np.clip(self._bias, -self._i_clamp, self._i_clamp, out=self._bias)

        command = np.nan_to_num(command, nan=0.0, posinf=1.0, neginf=-1.0)
        # Clipping the arm command per joint is what makes long reaches crooked.  The IK
        # hands back a joint displacement whose *whole vector* realizes the commanded tool
        # pose; executing a per-joint-truncated version of it realizes some other pose,
        # because a Jacobian does not act componentwise.  Scaling the vector instead keeps
        # the executed step parallel to the solved one -- ``J (alpha dq) = alpha (J dq)`` --
        # so the tool travels the commanded direction, rotation included, just slower.
        #
        # The old comment here argued the opposite: that per-joint saturation is asymmetric
        # (0.87 rad/s on the shoulder against 1.2 on the forearm) so scaling would not
        # preserve the achieved direction either.  That is true of scaling against the
        # action box, and it is why `_feasible` normalizes each hinge by its own ceiling
        # first; scaled that way the executed velocities stay proportional to the solved
        # ones.  What the old reasoning missed is that "corrected on the next step" is not
        # free: the correction is a *different* pose error, and the wrist pays it back as
        # rotation.  Measured on seed 12's kettle -> light switch reach, joints 5 and 7 run
        # saturated for nine consecutive steps while the roll error grows from 0.027 to
        # 0.630 rad in the direction opposite the one commanded -- the reach rolls the wrist
        # the wrong way -- and ALIGN then spends its own steps rolling it back, at which
        # point the reach resumes and undoes that.  Over seeds 0-23 the tool frame turns
        # 3.58 rad per transition to end 0.59 rad from where it started.  That six-fold
        # churn is the visible twisting between subtasks.
        #
        # `_proportional` is the *allowed* distortion, not a switch: at 1.0 the executed
        # step is exactly parallel to the solved one, and larger values buy transit speed
        # back by letting the fastest joints outrun the binding one by that ratio.  Strict
        # parallelism is not free -- every joint then travels at the rate of the most
        # limited one -- so the useful setting is a frontier point, not the extreme.
        if self._proportional > 0.0:
            excess = float(np.max(np.abs(command[:7]) / self._feasible))
            if excess > self._proportional:
                command[:7] *= self._proportional / excess
        return np.clip(command, -1.0, 1.0).astype(np.float32)


class OrientationAwareIKController:
    """Damped-least-squares IK from an end-effector pose to a joint displacement.

    Gymnasium-Robotics 1.2.0 shipped this controller and drove it from a Cartesian action
    space; 1.2.1 deleted both and exposes joint velocities instead.  The policy plans in
    end-effector space regardless, so it now owns the controller itself (``KitchenSim``
    holds the instance) and uses it to convert its plan into the joint command the env
    wants.  ``duration`` replaces upstream's ``mju_quat2Vel(..., 50)``, which made
    orientation corrections far too weak to turn the gripper sideways before contact.
    """

    def __init__(self, model, data, duration: float, position_weight: float,
                 regularization_strength: float = 0.3,
                 nullspace_gain: float = 0.0,
                 posture_gain: float = 0.0):
        if duration <= 0.0:
            raise ValueError(f"ik_orientation_duration must be positive, got {duration}.")
        self.model = model
        self.data = data
        self.duration = float(duration)
        if position_weight <= 0.0:
            raise ValueError(f"ik_position_weight must be positive, got {position_weight}.")
        self.position_weight = float(position_weight)
        self.regularization_strength = float(regularization_strength)
        self.nullspace_gain = float(nullspace_gain)
        self.posture_gain = float(posture_gain)
        #: Joint configuration the redundant freedom should prefer, or None to leave the
        #: arm's shape entirely to the task solve.  Set per phase by the policy.
        self.posture_target: Optional[np.ndarray] = None
        #: Floor under the escape term's ``unmet`` scaling, in [0, 1].  The scaling exists
        #: so the posture correction cannot outlive convergence, but that is precisely why
        #: a joint parked on a bound stays there: the arm arrives, the error goes to zero,
        #: and the one term that would unpin it switches off.  The policy raises this while
        #: a hinge is actually against a stop.  See
        #: :meth:`ScriptedKitchenPolicy._arm_joint_jammed`.
        self.escape_floor: float = 0.0
        self.eef_id = model.site(EEF_SITE).id
        #: Mid-range posture the null-space term drifts toward, and the half-widths that
        #: normalize "how close to a limit" into a comparable number across joints.
        self._joint_mid = 0.5 * (model.jnt_range[:7, 0] + model.jnt_range[:7, 1])
        self._joint_half_range = np.maximum(
            0.5 * (model.jnt_range[:7, 1] - model.jnt_range[:7, 0]), 1e-9)

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

        # Solve over the seven arm hinges, the only joints that get commanded. This is
        # numerically identical to solving over all 29 and slicing -- an object degree of
        # freedom cannot move the robot's end effector, so those columns are zero -- but it
        # says so in the types instead of relying on the reader to know it.
        jacobian = jacobian[:, :7]

        hessian = jacobian.T.dot(jacobian)
        hessian += np.eye(hessian.shape[0]) * self.regularization_strength
        joint_delta = jacobian.T.dot(weighted_error)
        step = np.linalg.lstsq(hessian, joint_delta, rcond=-1)[0]

        if self.nullspace_gain > 0.0:
            # Scaled by how much of the task is still unmet.  Without this the posture term
            # survives convergence -- it depends only on `qpos`, not on the error -- so it
            # keeps nudging after the tool has arrived, drags every solve toward the mid
            # range, and prevents the caller's convergence test from ever firing.  Measured:
            # an unscaled gain of 0.05 took all four skills to 0/5.  Tying it to the
            # residual makes it active exactly when the solve is stuck and silent otherwise.
            unmet = min(1.0, float(np.linalg.norm(weighted_error)) / self.NULLSPACE_ERROR_SCALE)
            unmet = max(unmet, float(self.escape_floor))
            step = step + unmet * self._nullspace_escape(jacobian, hessian)
        if self.posture_target is not None and self.posture_gain > 0.0:
            # Same free direction, spent on keeping the arm's *shape* near a configuration
            # known to be collision-free rather than on getting off a bound.  Scaled by the
            # unmet task error for the reason above: it shapes the route, and must fall
            # silent once the tool has arrived so the convergence test can fire.
            unmet = min(1.0, float(np.linalg.norm(weighted_error)) / self.NULLSPACE_ERROR_SCALE)
            bias = self.posture_gain * (
                np.asarray(self.posture_target, dtype=np.float64) - self.data.qpos[:7])
            step = step + unmet * self._project_nullspace(jacobian, hessian, bias)
        return step

    #: Task-error norm at which the null-space term reaches full strength.
    NULLSPACE_ERROR_SCALE = 0.05

    def _nullspace_escape(self, jacobian, hessian):
        """Posture correction that pulls joints off their limits without moving the tool.

        The arm has seven hinges for a six-dimensional task, so one whole direction of joint
        space leaves the end-effector pose untouched.  Plain damped least squares never uses
        it: it returns the minimum-norm solution, which is happy to drive a joint into its
        bound and then stop, because from there the only descent direction it can see is
        blocked.  Measured on the microwave, this is exactly what stalls the skill -- joints
        3 and 7 both pinned at +2.897 rad, a requested tool move of 19 mm, and a solved
        displacement of *identically zero* on every joint, forever.

        Projecting a mid-range bias through ``I - J^+ J`` spends that free direction on
        getting away from the bounds while leaving the tool where the task wants it.  The
        bias is scaled by how far into its range each joint is, so a comfortable joint
        contributes almost nothing and only the pinned ones actually move.
        """
        deviation = (self.data.qpos[:7] - self._joint_mid) / self._joint_half_range
        # Cubic: negligible mid-range, sharply rising as a joint approaches its bound, so
        # this cannot fight the task objective except where it is genuinely needed.
        bias = -self.nullspace_gain * deviation ** 3

        return self._project_nullspace(jacobian, hessian, bias)

    @staticmethod
    def _project_nullspace(jacobian, hessian, bias):
        """Strip from ``bias`` everything that would move the tool."""
        # I - J^+ J with the same damping the task solve uses, so the two stay consistent.
        pseudo_inverse = np.linalg.lstsq(hessian, jacobian.T, rcond=-1)[0]
        projector = np.eye(7) - pseudo_inverse.dot(jacobian)
        return projector.dot(bias)


class KitchenSim:
    """Named accessors for the ``KitchenEnv`` MuJoCo state.

    ``env`` may be any wrapper stack whose ``.unwrapped`` is the ``KitchenEnv``.  Keeping
    every ``unwrapped`` / ``model`` / ``data`` access behind this class is deliberate: the
    installed Franka Kitchen differs from the D4RL / relay-policy-learning versions and we
    want exactly one place to fix if it changes again.
    """

    def __init__(self, env):
        self._env = env
        #: The IK the policy uses to turn its Cartesian plan into a joint command. Owned
        #: here rather than by the robot: 1.2.1 has no controller of its own to borrow.
        self.controller: Optional["OrientationAwareIKController"] = None
        #: Which of the two equivalent grasp representatives the active skill committed to,
        #: or None before any skill has committed.  See :func:`equivalent_grasp_frame`.
        self.grasp_flip: Optional[bool] = None

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
        return self.site_xpos(EEF_SITE)

    @property
    def eef_mat(self) -> np.ndarray:
        return self.site_xmat(EEF_SITE)

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
        return float(self.joint_qpos(FINGER_JOINTS[0])[0])

    def validate_action_contract(self) -> None:
        """Reject Franka Kitchen versions whose actions are not Cartesian deltas.

        Gymnasium-Robotics 1.2.0 is the only release with the 7-D Cartesian/IK contract
        used by this policy.  From 1.2.1 onward Franka Kitchen again uses nine joint
        velocities.  Both are normalized boxes, so checking only bounds would miss a
        catastrophic semantic mismatch.

        A 1.2.0 env additionally carries a `controller` attribute for its Cartesian IK; its
        absence is the second half of the check, so a release that merely happened to expose
        nine dimensions for some other reason cannot pass.
        """
        robot = self.kitchen.robot_env
        if robot.action_space.shape != (ACTION_DIM,):
            raise RuntimeError(
                "ScriptedKitchenPolicy requires gymnasium-robotics>=1.2.1's 9-D joint "
                "velocity action space; this robot exposes shape "
                f"{robot.action_space.shape}. Release 1.2.0 instead exposed a 7-D "
                "Cartesian IK action space and is no longer supported.")
        if getattr(robot, "controller", None) is not None:
            raise RuntimeError(
                "This robot still carries gymnasium-robotics 1.2.0's Cartesian IK "
                "controller; ScriptedKitchenPolicy now owns its own IK and drives joint "
                "velocities directly. Upgrade to gymnasium-robotics>=1.2.1.")

    def install_orientation_controller(self, duration: float, position_weight: float,
                                       nullspace_gain: float = 0.0,
                                       posture_gain: float = 0.0) -> None:
        """Build the IK the policy uses to turn its Cartesian plan into joint commands."""
        self.controller = OrientationAwareIKController(
            self.model,
            self.data,
            duration,
            position_weight,
            nullspace_gain=nullspace_gain,
            posture_gain=posture_gain,
        )

    def scale_gripper_stiffness(self, scale: float) -> None:
        """Strengthen the finger position servos without moving their equilibria.

        1.2.1 drives each finger from its own actuator, where 1.2.0 had a single coupled
        ``actuator8``; both are scaled so the grip stays symmetric.
        """
        if scale <= 0.0:
            raise ValueError(f"gripper_stiffness_scale must be positive, got {scale}.")
        if scale == 1.0:
            return
        for name in GRIPPER_ACTUATORS:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise RuntimeError(f"Franka gripper actuator {name!r} is missing.")
            self.model.actuator_gainprm[actuator_id, 0] *= scale
            self.model.actuator_biasprm[actuator_id, 1:3] *= scale

    # -- tasks ---------------------------------------------------------------------------
    @property
    def goal(self) -> Dict[str, np.ndarray]:
        return self.kitchen.goal

    def task_distance(self, task: str) -> float:
        """``||achieved_goal[task] - desired_goal[task]||`` read from the *sim*, not from
        the (noisy) observation.

        Indexed through ``OBS_ELEMENT_INDICES`` rather than by joint name, which is what
        ``KitchenEnv.compute_reward`` scores. Task names and joint names coincided in 1.2.0
        but no longer do -- ``light switch`` spans the ``light_switch`` and ``light_joint``
        joints -- so a name lookup would read the wrong quantity or raise.
        """
        achieved = self.data.qpos[OBS_ELEMENT_INDICES[task]]
        return float(np.linalg.norm(achieved - OBS_ELEMENT_GOALS[task]))

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


def finger_contact_geoms(sim: KitchenSim) -> Dict[int, str]:
    """Map every *collidable* geom on either finger onto that finger's body name.

    Used to decide whether a finger is physically touching a target handle.  Selecting only
    the terminal box geom -- as this did while it was written against 1.2.0's gripper -- is
    wrong for the 1.2.1 model: each finger's contact surface there is a chain of small
    capsules with a single box at the very tip, and a grasped handle sits against the
    capsules.  Measured on the kettle: both fingers closed on the left-handle capsule with
    four active contacts, all of them on capsule geoms, so a box-only test reported *no*
    contact and the policy reopened and retried forever.

    Geoms that cannot collide (``contype == 0``, i.e. the visual meshes) are skipped: they
    never appear in ``data.contact`` and including them would only invite the same class of
    mistake in reverse.
    """
    pads: Dict[int, str] = {}
    for finger_name in FINGER_BODIES:
        body_id = sim.model_names.body_name2id[finger_name]
        geoms = np.flatnonzero(sim.model.geom_bodyid == body_id)
        collidable = geoms[sim.model.geom_contype[geoms] != 0]
        if collidable.size == 0:
            raise ValueError(f"Gripper body {finger_name!r} has no collidable geoms.")
        for geom_id in collidable:
            pads[int(geom_id)] = finger_name
    return pads


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


def upright_grasp_frame(frame: np.ndarray) -> np.ndarray:
    """Re-level ``frame`` about vertical, keeping its approach heading and its jaw flip.

    Every frame built here has local +x along the handle and local +z along the approach
    (:func:`grasp_frame`).  For a vertical handle -- the kettle's left post -- the handle
    axis *is* the object's own upright axis, so a wrist whose +x is off vertical is a wrist
    that will carry the object over with it.  Levelling puts +x back on world +z, or -z,
    whichever it is already nearer, so the jaw-flip representative the skill committed to
    survives (see :func:`equivalent_grasp_frame`).  The heading is left alone: that is the
    direction the grasp was actually made along, and turning it would drag the pads across
    the handle they are holding.
    """
    frame = np.asarray(frame, dtype=np.float64)
    approach = frame[:, 2].copy()
    approach[2] = 0.0
    if np.linalg.norm(approach) < 1e-9:
        return frame
    return grasp_frame(approach, np.array([0.0, 0.0, 1.0 if frame[2, 0] >= 0.0 else -1.0]))


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


#: Half turn about the tool's own approach axis.  A parallel-jaw gripper is unchanged by it:
#: local +z still points at the fingertips, and the two jaws simply swap roles.
_JAW_FLIP = np.diag([-1.0, -1.0, 1.0])


def equivalent_grasp_frame(sim: KitchenSim, desired: np.ndarray) -> np.ndarray:
    """Resolve ``desired`` to the jaw-flip representative the active skill committed to.

    Every frame in this file comes from :func:`grasp_frame`, where local +x runs along the
    handle, +y separates the jaws and +z is the approach.  Rotating that frame by pi about
    +z maps the gripper onto itself -- the fingers trade places and nothing else moves -- so
    both matrices describe the *same* physical grasp and either is a valid target.

    Treating them as distinct costs the episode.  Measured on the slide cabinet from reset:
    the full-SO(3) error to the single nominal frame is 3.13 rad, so the controller commands
    a half turn of the wrist, which drives joint 7 into its -2.9 rad limit and pins it there.

    Choosing between them *step by step* -- which is what this used to do, by picking
    whichever was nearer the live wrist -- is worse than it looks.  A jammed wrist drifts,
    the drift eventually makes the other representative nearer, and the target frame then
    snaps through 180 degrees under the controller mid-approach.  Measured: exactly one such
    switch per microwave episode, at step 319 on a seed that then succeeded and step 378 on
    one that ran out of time first.  That flip was the skill recovering, not working.

    The commitment is made once, when the skill is selected, by :meth:`
    ScriptedKitchenPolicy._commit_grasp_frame`, and held for the skill's whole lifetime, so
    the target frame cannot change under the controller.  It prefers the un-flipped
    representative -- the one the wrist is already effectively holding -- and departs from it
    only when the IK says that pose cannot be attained; see there.

    ``sim.grasp_flip`` is None only before any skill has committed, which in practice means
    the reset reorientation.  That is a free-space turn with no object to reach, both
    representatives are reachable, and the nearer one is the right answer, so that case
    keeps the proximity rule.
    """
    desired = np.asarray(desired, dtype=np.float64)
    if sim.grasp_flip is not None:
        return desired @ _JAW_FLIP if sim.grasp_flip else desired
    flipped = desired @ _JAW_FLIP
    current = sim.eef_mat.T
    return (flipped if float(np.trace(flipped @ current)) > float(np.trace(desired @ current))
            else desired)


def orientation_error(sim: KitchenSim, desired: np.ndarray) -> float:
    """Shortest rotation angle from the live EEF frame to ``desired``.

    Measured modulo the jaw-flip symmetry (see :func:`equivalent_grasp_frame`), so this is
    the rotation actually needed to seat the gripper rather than the rotation to one
    arbitrarily chosen labelling of it.
    """
    desired = equivalent_grasp_frame(sim, desired)
    relative = desired @ sim.eef_mat.T
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
    # Same jaw-flip symmetry the error metric uses: command the shorter of the two
    # equivalent rotations, never a half turn that only relabels the fingers.
    desired = equivalent_grasp_frame(sim, desired)
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
    #: Lateral offset of the target from the jaw centre line below which a grasp is
    #: considered centred.  Only the side-grasp skills act on it; see
    #: :meth:`centering_entry_tolerance`.
    grasp_center_tolerance = 0.0
    #: Hysteresis on that threshold.  A centring stage that re-arms at the same value it
    #: exits at will chatter forever once the residual settles near it -- measured on the
    #: kettle, whose jaw error parked at 0.0022-0.0034 against a 0.0025 tolerance and
    #: alternated centring with approach for 200 steps without ever closing the gripper.
    #: Re-entering costs this multiple of the exit threshold.
    centering_release_ratio = 3.0
    #: Set while the centring stage is driving; see :meth:`centering_entry_tolerance`.
    _centering_active = False
    #: Maximum stand-off error from which wrist-only alignment is safe.  This is broader
    #: than ``precontact_tolerance`` because rotating a long side-on tool frame moves the
    #: EEF even when the fingertip target is held fixed.
    alignment_tolerance = 0.18
    #: Frame error that interrupts a reach to re-align.  None means "same as
    #: orientation_tolerance"; see :meth:`realign_tolerance_value`.
    realign_tolerance = None
    #: Frame error the wrist must be inside before the insertion may *start* from the
    #: stand-off, or None to leave the ordinary realign gate in charge.  Distinct from
    #: `realign_tolerance`, which judges an insertion already under way: a skill can need a
    #: precise frame to *enter* the object and still tolerate the wrist settling once it is
    #: in.  See :attr:`LightSwitchSkill.approach_entry_tolerance`.
    approach_entry_tolerance = None
    #: How far from the stand-off an ALIGN that has just passed its frame test may start
    #: the approach from, in metres, or None for twice `precontact_tolerance`.  See
    #: `KitchenPolicyConfig.align_exit_to_approach`.
    approach_entry_radius = None
    #: Joint configuration the stand-off pose solves to from the reach's start, or None;
    #: see `KitchenPolicyConfig.reach_posture_gain`.
    _reach_posture = None
    #: Whether this skill's reach takes that pull at all.  Measured per skill on ten seeds:
    #: the slide cabinet's reach from home drops 36 -> 27 steps and the light switch's from
    #: the microwave 36 -> 29, the microwave's is unchanged (+1), and the kettle's goes from
    #: 5 to 57-64 on every seed -- its reach is routed through `transit_point` and the
    #: stand-off solution pulls against that waypoint until the stall detector fires.
    #: The light switch is off too, for the reason in `park_joint_transitions`: from the
    #: microwave (where no joint park runs) the stand-off solution can sit in the wrong
    #: wrist branch, and pulling the reach toward it measured seed 2 at eight grips of the
    #: lever (427 steps against 249) with 20, 57 and 438 slower by 40-50.  Its reach does
    #: shorten where the branch is already fixed by the joint park (315: 34 -> 20, 33: 33
    #: -> 18), but the branch is what decides the grip, so the reach keeps the tracker's
    #: own resolution there.  Only the slide cabinet takes the pull.
    use_reach_posture = True
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
        #: Set once this skill's jaws have been given `engage_seconds` to finish closing,
        #: so a later dip back into CONTACT_OR_GRASP does not pay the settle again.  See
        #: :attr:`KitchenPolicyConfig.engage_settle_required`.
        self._engage_settled = False

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


    def transit_point(self) -> Optional[np.ndarray]:
        """Optional collision-avoidance waypoint used before the object-relative reach.

        The generic route is an overhead one: climb to a clearance height, cross above the
        stand-off, then descend onto it.  It exists so that going straight from one subtask
        to the next has a *path*, not merely a short euclidean distance -- see
        :attr:`KitchenPolicyConfig.direct_transition_max_distance` for why distance alone is
        not enough, and :meth:`KettleGraspSkill.transit_point` for the hand-written version
        this generalizes.

        Disabled unless ``overhead_transit_clearance`` is set, and engaged only for reaches
        that actually cross the kitchen: a stand-off already nearly overhead is approached
        directly, so ordinary within-skill reaching is untouched.

        **It does not pay for itself, and is off by default.**  Measured over 8 seeds on the
        fixed order, against 8/8 in 452 steps for parking at home between subtasks: with the
        park disabled entirely, a clearance of 2.30 measures 5/8 in 584 steps and 2.45
        measures 5/8 in 529.  Two things are wrong with it.  The radius test fires at the
        *start* of every skill, not only between subtasks, so the whole episode is flown
        overhead and pays the detour even where the direct reach was fine; and arriving at a
        stand-off from vertically above is its own hazard, since these stand-offs are
        positioned along an approach axis, not above one.  Making this work needs the route
        to be conditioned on the transition rather than on distance, and to rejoin the
        approach axis rather than drop onto its end.
        """
        clearance = self.cfg.overhead_transit_clearance
        if clearance is None:
            return None
        stand_off = self.precontact_point()
        tool = self.tool_pos()
        if float(np.linalg.norm((tool - stand_off)[:2])) <= self.cfg.overhead_transit_radius:
            return None
        column = np.asarray(stand_off, dtype=np.float64).copy()
        column[2] = max(float(clearance), float(tool[2]), float(stand_off[2]))
        return column

    def transit_step_scale(self) -> float:
        return float(self.cfg.free_space_step)

    def transit_reached(self, point: np.ndarray) -> bool:
        """Whether a collision-avoidance waypoint has been reached once."""
        return bool(np.linalg.norm(
            (self.tool_pos() - np.asarray(point, dtype=np.float64))[:2]
        ) <= self.transit_tolerance)

    def contact_point(self) -> np.ndarray:
        return self.touch_point() - self.drive_direction() * self.cfg.contact_distance

    def approach_point(self) -> np.ndarray:
        """Where APPROACH steers, as opposed to where it is trying to end up.

        The two are the same for every skill that does not override this.  They differ
        wherever arriving *off the approach axis* is worse than arriving late; see
        :meth:`MicrowavePullSkill.approach_point`.  Phase transitions keep measuring
        against :meth:`contact_point`, so this only shapes the path.
        """
        return self.contact_point()

    def manipulate_point(self) -> np.ndarray:
        return self.touch_point() + self.drive_direction() * self.cfg.manipulate_lookahead


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

    def approach_blocked(self) -> bool:
        """True while the tool is somewhere the approach must not start from.

        Skills whose insertion is close to horizontal can be delivered by the reach at the
        wrong *height* and still look correctly placed to the corridor test, which measures
        along and across the stand-off-to-contact line.  Overridden where that matters.
        """
        return False

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

    def realign_tolerance_value(self) -> float:
        """Frame error at which a moving skill breaks off and goes back to ALIGN.

        Defaults to :meth:`orientation_tolerance_value`, which tests the same number in both
        directions and is fine for skills whose reach does not disturb the wrist.  Skills
        that have to reach and rotate at once need this to be the looser of the pair -- see
        :attr:`LightSwitchSkill.realign_tolerance`.
        """
        return (self.orientation_tolerance_value() if self.realign_tolerance is None
                else float(self.realign_tolerance))

    def precontact_step_scale(self) -> float:
        return (self.cfg.free_space_step if self.precontact_step is None
                else float(self.precontact_step))

    def approach_step_scale(self) -> float:
        return (self.cfg.contact_step if self.approach_step is None
                else float(self.approach_step))

    def centering_entry_tolerance(self) -> float:
        """Jaw-centring threshold for the *current* direction of travel.

        Tight while centring is under way, so the stage runs until the jaws really are
        square on the bar; loose once it has finished, so ordinary millimetre-scale drift
        during the insertion cannot restart it.
        """
        if self._centering_active:
            return self.grasp_center_tolerance
        return self.grasp_center_tolerance * self.centering_release_ratio

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
        if self.task not in OBS_ELEMENT_GOALS:
            return 0.0
        return float(OBS_ELEMENT_GOALS[self.task][TASK_GOAL_INDEX.get(self.task, 0)])

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
    #: The reach-posture pull (`KitchenPolicyConfig.reach_posture_gain`) was kept on for this
    #: skill alone, for 36 -> 27 steps on the reach from home.  It is off now: decomposed on
    #: seeds 4, 1 and 25 into net and accumulated rotation, that reach rolls the wrist about
    #: its own axis 2.02 / 1.02 / 1.93 rad *accumulated* for 0.10 / 0.24 / 0.06 rad *net* --
    #: one full radian out and back, on roughly 300 of 500 episodes -- which is exactly the
    #: twisting the per-joint-clipping fix removed, doubled: without the pull the same reaches
    #: measure 0.84 / 0.97 / 0.93 rad for +7 / +3 / -2 steps.  The residual out-and-back is
    #: the damped-least-squares path itself; doubling the IK's orientation weight, halving
    #: its position weight, or doubling the rotation step leave it at 1.9 rad and cost 30 to
    #: 500 steps, so it is not tuned further here.
    use_reach_posture = False
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
    #: Lateral offset along the finger axis the handle may sit at when the approach
    #: advances onto it.  0 disables the centring stage (the base-class default).
    grasp_center_tolerance = 0.008
    #: Overridden from `KitchenPolicyConfig.slide_contact_tolerance` in `__init__`; the
    #: class value is the pre-config default and is not what runs.
    contact_tolerance = 0.05
    recede_after_manipulation = True

    def __init__(self, sim, task, cfg):
        super().__init__(sim, task, cfg)
        self.contact_tolerance = float(cfg.slide_contact_tolerance)

    def drive_direction(self) -> np.ndarray:
        joint = self.manipulated_joint
        axis = self.sim.joint_axis(joint)
        goal = float(OBS_ELEMENT_GOALS[self.task][TASK_GOAL_INDEX.get(self.task, 0)])
        sign = np.sign(goal - float(self.sim.joint_qpos(joint)[0]))
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

    # This end-on insertion reaches past the fingertip pads, so the switch can be poked from
    # further out than a crosswise handle grasp allows -- otherwise the hand body enters the
    # cooker hood before these parallel fingertips reach the bar.
    #
    # The extra reach is 4.2 cm *beyond the pads*, and has to be written that way.  It was a
    # bare 0.15, measured when the site this is relative to was 1.2.0's `EEF` at the wrist,
    # 10.8 cm behind the pads.  1.2.1's `end_effector` site sits between the fingers at
    # 0.052, so the same literal moved the assumed tool point 9.8 cm past the real fingertips
    # -- the skill drove until it believed it was 2 cm from the switch while physically
    # nowhere near it, closed on empty air, reopened and retried for the whole episode.
    tool_offset = FINGERTIP_OFFSET + 0.042
    approach_gripper = "open"
    engage_gripper = "close"
    #: Drive the lever to its true goal rather than stopping just inside the threshold.
    #:
    #: This was 0.85, chosen when a second grip was the thing to avoid: the margin exists to
    #: stop a skill releasing right on ``BONUS_THRESH`` and letting the object rebound out of
    #: it, and this switch settles *further* in after release, so asking for extra depth only
    #: bought re-grips.  0.85 stops the push at a task distance of 0.255, i.e. at -0.435 rad
    #: of the -0.69 goal -- which is where "one bite is worth about 0.42 rad" came from.  It
    #: was the stopping rule being read as a physical limit.
    #:
    #: With :attr:`arc_mode` holding the lever in the jaws the grip count no longer depends
    #: on it at all: across 0.85, 0.70, 0.55, 0.40, 0.35 and 0.05 it is *identical* (1.29 per
    #: episode, one grip on 22 of 24 seeds) and only the throw moves, -0.411, -0.438, -0.462,
    #: -0.484, -0.493, -0.538.  The knob now buys throw and nothing else.
    #:
    #: 0.85 looked mandatory on a 400-step harness, where anything below it cost a whole
    #: episode (20/24 four-task seeds against 19/24, the loss being seed 23's *microwave*
    #: wedging in MOVE_TO_PRECONTACT after the deeper throw left the arm elsewhere).  That
    #: was an artefact of the budget: `llfbench.envs.kitchen.make` defaults to
    #: ``episode_steps=1000``, and re-measured there the two are indistinguishable -- 21/24
    #: and a mean of 3.83 either way, median episode 282 either way, grips per episode 1.33
    #: and one grip on 22 of 24 seeds either way -- while the throw goes -0.412 to **-0.539**
    #: and the worst seed's -0.377 to -0.400.  The deeper throw needs about ten more steps on
    #: the switch and at the real budget those steps are there.
    #:
    #: Measure this in the combined run at the full budget.  ``lstrace.py`` runs the switch
    #: alone and cannot see any of it, and a 400-step harness inverts the answer.
    margin_override = 0.05
    # The cooker hood makes the generic 1.5 cm waypoint tolerance impractical, but 18 cm
    # (an earlier value) classified free space as contact and skipped the actual poke.
    precontact_tolerance = 0.04
    #: How near the contact point the tool must be before the jaws are told to close.  The
    #: lever is 21 mm in radius against a 40 mm jaw half-opening, so closing early shuts the
    #: pads beside it rather than around it: measured on seed 10 at 0.035, the close fires
    #: at t=62, the jaws reach 0.0127 with *no* contact at all, one pad brushes the capsule
    #: two steps later, and the skill has to reopen and take a second bite.  Tightening to
    #: 0.018 takes the mean number of grips before the drive from 1.29 to 1.04 over 24 seeds
    #: and holds 24/24 on both halves; 0.025 is 23/24.  The cost is about 15 steps of
    #: median, since the approach now has to arrive properly before it may close.
    contact_tolerance = 0.018
    lost_contact_tolerance = 0.14
    align_in_place = True
    #: How close to the stand-off the tool has to be before it is worth stopping to align.
    #:
    #: This was 0.30, which is wider than the stand-off itself, so the skill began aligning
    #: from the moment it set out.  Reaching disturbs the wrist and aligning restores it, so
    #: from that distance the two phases simply take turns: measured, an 80-step
    #: MOVE_TO_PRECONTACT/ALIGN limit cycle that crept in about a centimetre per cycle, at a
    #: frame error ringing between 0.12 and 0.21 either side of one 0.20 threshold.  That is
    #: the visible juddering on the way to the switch.  Waiting until the stand-off is
    #: actually near lets the ordinary reach -- which corrects orientation and position in
    #: the same IK solve -- do the approach: 8/8 either way, but 110 steps against 158 and
    #: 53 phase transitions against 88, with grip retention unchanged.  Below about 0.15 it
    #: reverses sharply (0.12 measures 1/8): the skill then arrives unaligned with no room
    #: left to fix it.
    alignment_tolerance = 0.20
    orientation_tolerance = 0.20
    #: Frame error that sends an already-committed insertion back to ALIGN.
    #:
    #: Deliberately looser than :attr:`orientation_tolerance`, which is what gets the wrist
    #: into the hood in the first place.  Once the fingers are in there the frame error
    #: settles right at 0.20, and testing the *same* number in both directions flips
    #: ALIGN/APPROACH every step or two -- measured, 88 phase transitions in a 152-step
    #: episode, which is the visible juddering on the way to the switch.  Tightening the
    #: exit instead (0.12) is the wrong direction and measures 0/8: the wrist cannot reach
    #: 0.12 in there, so the skill aligns forever and never closes.
    #:
    #: Widening it is only worth anything once :meth:`manipulate_point` follows the lever's
    #: arc.  Tried against the old straight-tangent target it did nothing (88.5 phase
    #: transitions against 88.6) and cost contact quality, because that target was pulling
    #: the tool off the arc and manufacturing frame error faster than any threshold could
    #: absorb.  With the arc in place it is decisive: 57 steps against 110, and **10 phase
    #: transitions against 53** -- one clean pass through the phases rather than a limit
    #: cycle -- with the tool held to 0.3 cm off the arc instead of 1.4 cm.
    realign_tolerance = 0.25
    #: Frame error the wrist has to be inside at the stand-off before the insertion starts.
    #:
    #: The grip's success is decided here, before the pads touch anything.  Measured over
    #: seeds 0-499 with a test.py-faithful census of the tool frame at the moment both pads
    #: first hold the lever: every one of the 24 sampled one-grip seeds captured with the
    #: frame within 0.031 rad of the target (typically 0.004-0.02), and every one of the 11
    #: seeds that needed a second grip captured at 0.036-0.067 rad with the jaw axis tilted
    #: below horizontal (z component -0.011 to -0.025), so that one pad led the other along
    #: the lever.  Seven of the eleven lost a pad within three steps of "capture"; the other
    #: four walked 2.5 cm down the lever and lost it at -0.20 to -0.24 rad.  Nine of the
    #: eleven arrived from the slide cabinet, whose park hands the wrist over up to 0.6 rad
    #: from front (`orient_forward_transition_tolerances`), and the ordinary
    #: `realign_tolerance` of 0.25 let the insertion start with that residual still open.
    #:
    #: 0.03 is the measured one-grip band, and gating the entry on it in place does not
    #: work: with ALIGN holding position at the stand-off the wrist cannot get there --
    #: measured on 35 seeds, ALIGN ran 130-300 steps on 24 of them and the switch was never
    #: attempted, and the five seeds that did capture inside the band (68, 93, 106, 342, 428
    #: at 0.026-0.028 rad) still lost the lever at -0.19 rad.  So the frame at capture is
    #: necessary but it is not the whole mechanism, and the in-place turn is the wrong way to
    #: get it.  Left at None (ordinary gate) while that is measured; see the class docstring
    #: history below for what replaces it.
    approach_entry_tolerance = None
    #: Start the insertion from anywhere ALIGN was allowed to run (`alignment_tolerance`),
    #: not only from the stand-off.  The stand-off is a detour for this skill: measured over
    #: seeds 0-499, letting the approach set out from within 0.20 m takes the reach from
    #: 27.4 to 23.5 steps and the sweep from 19.4 to 17.7 per episode, against 20 seeds
    #: whose approach then runs 26 steps instead of 15 -- a net 6 steps per episode.  The
    #: kettle is the opposite case (its `alignment_tolerance` is 0.60 m and an approach
    #: from there re-grasps), which is why this is per skill.
    approach_entry_radius = 0.20
    use_reach_posture = False
    def manipulation_done(self) -> bool:
        """Also stop once the switch is banked and the pads have come off it.

        The margin above is a *demonstration* target, not a success condition, and it is
        only worth driving to while the switch is still in the jaws.  The thin lever loses
        bilateral contact easily, and when it does so a few thousandths short the ordinary
        logic treats it as an unfinished task and goes back for it -- a full
        approach/align/re-approach for the last centimetre of a switch the environment has
        already scored and will never score again.

        Measured on seed 1 of the randomized order: MANIPULATE reached 0.2726 against the
        0.255 this asks for, contact flickered, and the skill spent 41 further steps
        retaking the switch to win 0.017.  That is the second grasp a viewer sees, where
        the gripper goes back, touches the switch and achieves nothing; the episode runs
        381 steps with it and 339 without.  Over 24 randomized seeds it removes the
        re-grip on 3 of them and takes none of the productive ones, which are the seeds
        where the switch genuinely is not over yet.
        """
        if super().manipulation_done():
            return True
        return bool(self.task in self.sim.kitchen.episode_task_completions
                    and not self.grasp_retained())

    def approach_step_scale(self) -> float:
        return float(self.cfg.light_approach_step)

    def manipulation_step_scale(self) -> float:
        return float(self.cfg.light_manipulation_step)
    grasp_contact_min_opening = 0.012
    #: The kettle's jaw-centring stage was tried here, on the theory that the lever is thin
    #: (21 mm radius) next to a 40 mm jaw half-opening so a lateral miss catches one pad
    #: instead of straddling -- which is true, and measured on seed 10 the close fires with
    #: the tool 9 mm off centre.  Running the stage is far worse than the miss: at a 5 mm
    #: tolerance the harness falls from 24/24 to 16/24 and only 14 of 24 light switches
    #: reach their drive at all, at 10 mm 17/24.  The stage translates without rotating and
    #: the spring-loaded lever does not wait.  `grasp_center_tolerance` therefore stays 0
    #: here, which is what disables it.
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
        #: Simulation time both pads were first seen on the lever in the current bilateral
        #: run, or None; see `KitchenPolicyConfig.light_capture_steps`.
        self._bilateral_since = None
        #: Whether this skill has *ever* held the lever.  `_capture_confirmed` is cleared
        #: whenever the jaws reopen, so it cannot tell a first approach apart from a
        #: recovery, and only the recovery may push the switch backwards.
        self._ever_captured = False
        #: Tool point the slip back-out is heading for, or None when not backing out.
        self._slip_backout_target = None
        #: Simulation time a pad was last seen on the lever; see
        #: `light_contact_dropout_grace`.
        self._last_contact_time = None
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
        for finger_name in FINGER_BODIES:
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
        # Frozen before capture, because chasing a live axis during the reach makes the
        # target chase itself.  Once the lever is actually in the jaws the trade reverses:
        # see `KitchenPolicyConfig.light_track_lever_axis`.
        if self.cfg.light_track_lever_axis and self._capture_confirmed:
            return self._measure_switch_axis()
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
        """Require one observed bilateral capture; width alone is ambiguous here.

        A closed grip survives a brief contact dropout, because dropping this lever undoes
        the push -- see ``light_contact_dropout_grace``.
        """
        fingers = self.contacting_fingers()
        if self.sim.finger_opening >= self.grasp_ready_opening:
            self._capture_confirmed = False
            self._last_contact_time = None
        if len(fingers) == 2:
            now = float(self.sim.data.time)
            if self._bilateral_since is None:
                self._bilateral_since = now
            hold = (max(0, int(self.cfg.light_capture_steps) - 1)
                    * self.sim.kitchen.robot_env.dt)
            if now - self._bilateral_since >= hold - 1e-9:
                self._capture_confirmed = True
                self._ever_captured = True
        else:
            self._bilateral_since = None
        if fingers:
            self._last_contact_time = float(self.sim.data.time)
            return bool(self._capture_confirmed)
        grace = int(self.cfg.light_contact_dropout_grace)
        if (not self._capture_confirmed or grace <= 0
                or self._last_contact_time is None):
            return False
        elapsed = float(self.sim.data.time) - self._last_contact_time
        return bool(elapsed <= grace * self.sim.kitchen.robot_env.dt + 1e-9)

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
        point = self.touch_point() + self.approach_axis() * self.cfg.light_insertion_depth
        point[2] += self.cfg.light_contact_height_offset
        return point

    def gripper_command(self, mode: str) -> float:
        """Close onto the lever's own width; see :attr:`~KitchenPolicyConfig.light_grasp_command`."""
        if mode == "close":
            return float(self.cfg.light_grasp_command)
        return super().gripper_command(mode)

    #: How MANIPULATE turns the remaining travel into a target for the tool point.
    #:
    #: ``"chord"`` translates the tool point by the displacement the *lever site* undergoes;
    #: ``"site"`` (the original) and ``"contact"`` rotate the tool point about the pivot, at
    #: the lever site's radius and at the contact point's radius respectively.
    #:
    #: This is what decides whether one grip does the job, and it is a geometry error, not a
    #: tuning knob.  MANIPULATE sets a *position* target and holds the orientation fixed
    #: (`switch_axis` is frozen at grasp time), so the hand translates without rotating --
    #: and a body that translates moves every one of its points by the same vector.  Rotating
    #: the tool point about the pivot therefore moves the *pads* along the tool point's arc
    #: rather than their own, and the two differ because the pads are 2-3 cm further out:
    #: the lever is dragged sideways through the jaws until one pad lets go.  Traced on seed
    #: 7, the lever site's offset across the jaws grows 0.000 -> 0.029 m over the push, the
    #: right pad drops at -0.215 rad, and the grip is gone by -0.364.
    #:
    #: Measured over the 24-seed harness at the true-goal margin, this one line moves grips
    #: per episode 1.62 -> 1.29, one-grip seeds 17/24 -> 22/24, mean throw -0.529 -> -0.538
    #: and worst-seed throw -0.341 -> -0.395; seed 7 goes 3 grips reaching -0.563 to a single
    #: grip reaching -0.605.
    #:
    #: It is also why the insertion depth cannot be tuned: the site-arc error grows with the
    #: depth, so before this the depth sweep read 0.028 -> 20/24, 0.042 -> 20/24, 0.048 ->
    #: 16/24, 0.054 -> 2/24, 0.060 -> 0/24, which looks exactly like a hand jamming into the
    #: panel and is not.  With the chord the same 0.054 measures 19/24.  Depth still buys
    #: nothing (0.042 -> 1.88 grips, 0.054 -> 2.88 against 1.29 at 0.028) -- seating the
    #: lever deeper shortens the moment arm faster than it improves the hold -- but the
    #: reason is now measurable rather than confounded.
    arc_mode = "chord"

    def manipulate_point(self) -> np.ndarray:
        """Lead along the lever's own arc, not along the tangent to it.

        This used to be ``touch_point() + drive_direction() * 0.06``.  The tangent is the
        right *direction*, but a straight lead off a short arc leaves it immediately, and
        this arc is very short: ``light_switch`` is a hinge about world z anchored at the
        switch body, and its site sits only 0.0815 m out from that pivot.  A 6 cm tangent
        lead therefore names a point ``hypot(0.0815, 0.06) = 0.1012`` m from the pivot --
        **2 cm outside the circle the lever can travel on**.  The lever cannot go there, so
        the request resolves as a steady radial pull that drags the pads off the end of it.

        That radial direction is what is seen from the front.  Outward from this pivot is
        ``(0.39, -0.92, 0)``: overwhelmingly -y, straight back toward the viewer.  Hence a
        gripper that visibly backs away while it is supposed to be sweeping left, and loses
        the switch partway through.

        Rotating the contact point about the hinge keeps the target on the arc by
        construction, so the only thing asked of the hand is the sweep itself.  The lead
        angle is clamped to the travel actually remaining, so the request shrinks to zero at
        the goal instead of continuing to drive past it.
        """
        joint = self._hinge_joint()
        anchor = self.sim.joint_anchor(joint)
        axis = self.sim.joint_axis(joint)
        remaining = self._goal_qpos() - float(self.sim.joint_qpos(joint)[0])
        if abs(remaining) < 1e-9:
            return self.touch_point()
        lead = float(np.clip(remaining,
                             -self.cfg.light_manipulation_lead_angle,
                             self.cfg.light_manipulation_lead_angle))
        radius = self.touch_point() - anchor
        swept = rotate_about_axis(axis, lead) @ radius
        if self.arc_mode == "chord":
            # Where the tool should sit on the lever *now*, carried by the displacement the
            # lever site itself is about to undergo.  See :attr:`arc_mode`.
            return self.contact_point() + (swept - radius)
        if self.arc_mode == "contact":
            return anchor + rotate_about_axis(axis, lead) @ (self.contact_point() - anchor)
        return anchor + swept

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
    use_reach_posture = False
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
    # Measured, not assumed: a bilateral capture on the 4 cm bar settles the finger joint
    # at 0.0123 m, not the 0.019--0.020 m this was written for. At 0.015 the retention test
    # rejected every real grasp the instant it closed -- and since `grasp_retained` also
    # requires the opening to be above this bound, a capture that overshot it could never be
    # recovered, so the skill reopened and re-approached forever. Sit clear below the
    # measured stop instead.
    grasp_contact_min_opening = 0.007
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
    #: Stop MANIPULATE at 0.5 * BONUS_THRESH rather than the shared 0.7, so the pull does
    #: not hand back a door that is only just past the environment's threshold.
    #:
    #: This was deliberately absent, and the reasoning was sound at the time: with the old
    #: grip the pull reached 0.105 on almost no seed, so the skill never declared itself
    #: done, and with the microwave *first* in a randomized order there is no banked-task
    #: short circuit to stop it -- it spent its entire 626-step budget and was abandoned
    #: (7/8 at a 653-step median, against 8/8 at 393 with no override).
    #:
    #: What changed is the grip.  With the funnel putting the jaws on the bar and
    #: ``microwave_pull_yaw`` loading a pad behind it, the pull holds long enough to reach
    #: the deeper gate.  Measured over 24 randomized seeds:
    #:
    #: ====== ============= ============= =============
    #: margin door at peak  door at end   all four true
    #: ====== ============= ============= =============
    #: 0.7    0.156         0.160         20/24
    #: 0.5    0.100         0.109         20/24
    #: 0.35   0.095         0.106         19/24
    #: 0.25   0.078         0.114         --
    #: ====== ============= ============= =============
    #:
    #: The door is open at the end on 24/24 seeds at every value, so this buys depth, not
    #: reliability.  0.5 takes almost all of it; below that the peak keeps creeping down
    #: while the *end* stops improving and the episode score starts to slip, because the
    #: extra pull is spent on seeds where the grasp gives out rather than on seeds this
    #: gate was stopping.  0.25 is the clearest case: the deepest peak of the set and a
    #: worse finish than 0.5.
    margin_override = 0.5

    def __init__(self, sim, task, cfg):
        super().__init__(sim, task, cfg)
        self.contact_tolerance = float(cfg.microwave_contact_tolerance)
        self._capture_confirmed = False
        self._capture_ever = False
        self._capture_offset = None
        self._pull_frame = None
        self._release_frame = None
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
        for finger_name in FINGER_BODIES:
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
                # Freeze the pull frame from the pose the jaws actually closed in, turned
                # about the bar by `microwave_pull_yaw`.  Snapshotting rather than
                # recomputing is the point: the live grasp frame turns with the door, and
                # following that is what drags the wrist through the whole swing.
                yaw = float(self.cfg.microwave_pull_yaw)
                frame = np.asarray(self.sim.eef_mat, dtype=np.float64)
                # On the un-flipped representative, for the reason spelled out where the
                # kettle freezes its own capture frame: both of these are handed to
                # `orientation_error` and `rotation_action`, which push a desired frame
                # through `equivalent_grasp_frame`, which re-applies the jaw flip itself.
                # The live wrist is already the flipped one whenever this skill committed
                # to the flip, so storing it raw flips twice and points both the pull hold
                # and the recede unwind a half turn away from the jaws.  Measured over the
                # 24-seed harness, the microwave takes the flip on seed 15.  The world-frame
                # pull yaw below left-multiplies, so it commutes with this and can be
                # applied to either representative.
                if self.sim.grasp_flip:
                    frame = frame @ _JAW_FLIP
                self._pull_frame = (frame if yaw == 0.0
                                    else rotate_about_axis(self.handle_axis(), yaw) @ frame)
                # Kept past the release, unlike `_pull_frame`: RECEDE needs somewhere to
                # put the wrist back to.  See :meth:`release_frame`.
                self._release_frame = frame
            self._capture_confirmed = True
            self._capture_ever = True
        if self.sim.finger_opening >= self.grasp_ready_opening:
            self._capture_confirmed = False
            self._capture_offset = None
            self._pull_frame = None
        return bool(self._capture_confirmed and fingers and handle_sized_opening)

    def release_frame(self) -> Optional[np.ndarray]:
        """The wrist frame the jaws closed in, before ``microwave_pull_yaw`` turned them.

        RECEDE unwinds to this.  A pull yaw is a turn the arm has to carry out of the
        subtask with it, and the next skill inherits it: measured at a yaw of +1.0, the
        wrist leaves MANIPULATE 1.4 rad from the live grasp frame and reaches
        SELECT_SUBTASK at 2.6 rad, from where the light switch is simply unreachable --
        626 steps of MOVE_TO_PRECONTACT with the position error pinned at 0.57 m and three
        joints on their stops, then abandoned, on 3 of 24 randomized seeds.  Unwinding
        costs a few free-space steps and makes the yaw survivable for whatever runs next.
        """
        return None if self._release_frame is None else self._release_frame.copy()

    def pull_frame(self) -> Optional[np.ndarray]:
        """Wrist frame to hold through the pull, or None before the jaws have closed.

        See ``microwave_pull_yaw``.  This is a *fixed* frame captured at the grasp, not the
        live grasp frame, so commanding it turns the wrist once, at the start of the pull,
        and then simply holds it against the load.
        """
        return None if self._pull_frame is None else self._pull_frame.copy()

    def contact_depth(self) -> float:
        """Measured fingertip-midpoint depth relative to the handle centre."""
        return float(np.dot(
            self.tool_pos() - self.touch_point(), self.approach_axis()))

    # The approach reaches the bar's *depth* long before its *side*: measured, depth closes
    # from -0.087 to -0.005 by step 24 while the tool is still 0.11 m off along the handle,
    # so the gripper then grinds sideways along the bar to find it, and that grind is what
    # wedges a finger and jams it shut. Squaring up at a stand-off first is the obvious fix
    # and does not work here: the arm cannot reach a pose square in front of the bar. Driven
    # to a point 0.06 m out along the approach axis it converges and stops dead 0.083 m
    # short, with the wrist correctly oriented (frame error 0.12) and holding that exact
    # pose for the rest of the episode -- 0/8 at every centring tolerance from 0.015 to
    # 0.03, against 4/8 for going straight in. The reachable corridor to this handle appears
    # to run along the door radial, not perpendicular to the door face, so a fix has to work
    # within that corridor rather than stage a square approach.

    #: How far the *bar itself* may be from the point midway between the pads and still be
    #: considered inside the jaws.  Successes measured 0.015-0.022 m here, failures a tight
    #: cluster at 0.037-0.038, so this separates them with margin on both sides.
    capture_radius = 0.028

    #: How near the handle the fingertips must still be for a lost grasp to be re-seated
    #: in place rather than retried from the stand-off.  See the reacquisition branch in
    #: `KitchenScriptedPolicy._compute_reactive_action`.
    reacquire_in_place_radius = 0.16

    def bar_between_pads(self) -> bool:
        """Is the handle bar actually inside the open jaws, ready to be closed on?

        Distinct from the FSM's ``contact_error``, which measures the distance to the
        *hooked contact point* -- a pose offset from the bar by design, so satisfying it
        says nothing about whether the bar is between the pads.  This asks the question
        directly, against the live bar position.
        """
        return bool(np.linalg.norm(self.tool_pos() - self.touch_point())
                    <= self.capture_radius)

    def precontact_step_scale(self) -> float:
        return float(self.cfg.microwave_precontact_step)

    def approach_step_scale(self) -> float:
        return float(self.cfg.microwave_approach_step)

    def manipulation_step_scale(self) -> float:
        return float(self.cfg.microwave_manipulation_step)

    def gripper_command(self, mode: str) -> float:
        """Close onto the bar's own width, not onto zero.

        The finger command is a *position* target under 1.2.1's joint servo, so the generic
        ``gripper_close`` of -1.0 asks for a fully shut gripper.  A 4 cm bar stops the joint
        at about 0.0123 m, and the servo then spends the whole remaining travel as grip
        force -- which fires the bar out from between the pads.  Targeting slightly inside
        the bar instead gives a bounded preload.

        This used to apply only once a capture had been confirmed, which meant every grasp
        began with the full-close command that ejects the bar, and the gentle hold was only
        ever reached on the rebound.
        """
        if mode == "close":
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

    # Retained but disabled; `microwave_hook_depth` is 0. The hook is a property of the
    # *final* grasp, yet applying it rotates the whole approach frame, and that rotation is
    # what broke this skill in two visible ways: it yaws the gripper 30 degrees off the door
    # normal, so it comes in obliquely and one pad leads the jaw centre onto the bar, and it
    # costs joint 7 enough travel to pin it at +2.897 for the rest of the approach. Both are
    # now unnecessary -- `microwave_insertion_depth` buys the same retention by translating
    # deeper instead of rotating -- and measured with the frame commitment in place the hook
    # is strictly harmful: 8/8 without it, 0/8 with it at 0.020.
    #
    # For the record, under the old proximity-picked frame it had a knife-edge optimum:
    # 0.02 gave 4/8 while 0.01 and 0.03 both gave zero door movement on all 8 seeds, because
    # only that one angle threaded the leading pad past the bar. Ramping the yaw in with
    # proximity removed the jam but never bit hard enough to finish the pull.

    def desired_orientation(self) -> np.ndarray:
        """Square door-normal grasp, yawed about the bar so one pad hooks behind it."""
        frame = super().desired_orientation()
        angle = self.hook_yaw()
        onset = self.cfg.microwave_yaw_onset_depth
        gated = self._capture_ever and self.cfg.microwave_yaw_first_attempt_only
        if not gated and (onset is None or self.contact_depth() > -float(onset)):
            angle += float(self.cfg.microwave_grasp_yaw)
        if frame is None or angle == 0.0:
            return frame
        return rotate_about_axis(self.handle_axis(), angle) @ frame

    def touch_point(self) -> np.ndarray:
        """Take the bar higher up than its centre, to buy reach around the swing.

        The pull is limited by reach, not by grip: probed against the door angle, the square
        grasp pose is exact while the door is shut and has a 0.035 rad / 0.068 m residual by
        -0.6 rad, against a goal of -0.75.  That is why the bar leaves the pads at a
        repeatable door distance and the skill has to re-grip to finish.  The bar is 0.26 m
        tall, and taking it higher measurably extends the reachable arc: at +0.10 m the
        residual at -0.6 rad falls to 0.017 / 0.038.

        The offset has to move the whole grasp geometry, which is why it lives here rather
        than in ``contact_point``: ``bar_between_pads`` measures against this point, so
        raising the contact alone just aims the jaws where the predicate says the bar is
        not, and the skill never closes at all (measured: 0 grasps, 0/8, on every seed).
        """
        return (super().touch_point()
                + unit(self.handle_axis()) * self.cfg.microwave_grasp_height)

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
        """Aim past the bar, so it ends up deep in the jaw rather than at the fingertips.

        Aiming the fingertip midpoint *at* the bar leaves the bar sitting right at the tips,
        where the pads are at their thickest and nearly parallel, and a round bar pinched
        there is squeezed straight back out the moment the pull starts -- measured, a clean
        bilateral capture collapsing to zero opening with no remaining contact in a single
        step.  Driving ``microwave_insertion_depth`` further along the approach seats the bar
        against the narrow base of the fingers instead, where escaping toward the tips means
        forcing past the thicker part of both pads.  This is what makes the pull hold: 5/8
        at depth 0, 8/8 anywhere in 0.015--0.025, and 2/8 by 0.030, where the wrist starts
        hitting the door panel.
        """
        point = self.touch_point() - self.drive_direction() * self.cfg.microwave_insertion_depth
        point[2] += self.cfg.microwave_contact_height_offset
        return point

    def approach_point(self) -> np.ndarray:
        """Hold the insertion back until the tool is actually on the bar's approach axis.

        Aiming APPROACH straight at :meth:`contact_point` lets the tool trade depth for
        lateral error: it closes the two together, and whichever the arm can serve more
        cheaply wins.  Near a joint bound that is always depth, so the tool arrives at the
        bar plane still displaced sideways, misses the bar entirely, and then shoves the
        door open with the wrist -- which the environment scores as success.

        Withdrawing the target by a multiple of the lateral error turns the straight line
        into a funnel: off axis, the nearest point of the target is behind the tool, so the
        commanded motion is almost purely lateral; on axis, the funnel collapses and the
        insertion is the ordinary straight one.
        """
        contact = self.contact_point()
        gain = float(self.cfg.microwave_approach_funnel)
        if gain <= 0.0:
            return contact
        drive = unit(self.drive_direction())
        offset = self.tool_pos() - contact
        lateral = float(np.linalg.norm(offset - float(np.dot(offset, drive)) * drive))
        return contact + drive * min(gain * lateral, self.cfg.precontact_distance)

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
    use_reach_posture = False
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
        """The kettle goal this skill aims at: the environment's, shifted right if asked.

        See :attr:`KitchenPolicyConfig.kettle_goal_lateral_offset`.  Everything geometric
        about the push reads this, so the offset stays self-consistent; the environment's
        success test still reads its own unshifted goal.
        """
        goal = np.array(OBS_ELEMENT_GOALS["kettle"][:3], dtype=np.float64)
        goal[0] += float(self.cfg.kettle_goal_lateral_offset)
        return goal

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
    #: The top bail bar: centred over the body, horizontal, along the body's local x.
    TOP_HANDLE_LOCAL_POS = np.array([0.0, 0.0, 0.259], dtype=np.float64)
    TOP_HANDLE_LOCAL_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
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
    #: Planar radius the tool must clear before a sideways escape is called finished.  This
    #: is the bail bar's own reach -- ``kettle_chain.xml`` gives it half-length 0.100 along
    #: the body's local x and radius 0.032 -- rather than `TOP_HANDLE_GUARD_RADIUS`, which
    #: carries padding appropriate to *entering* the guard and costs steps on the way out.
    #:
    #: The guarded volume has an open bottom, and without this the hand leaves through it.
    #: The test above asks "is the tool over the bar and high enough to be above it", which
    #: is the right question for *entering* the escape and the wrong one for ending it: a
    #: hand that descends stops satisfying the height clause while still directly over the
    #: capsule.  See :attr:`TOP_HANDLE_ESCAPE_RADIUS` for what it exits at.  Measured on seed 7, which comes to this handle from the light switch with
    #: the arm over the kettle: the escape holds down to a tool 0.321 above the root, the
    #: guard releases at 0.305, and on that same step both fingers land on the top bail bar
    #: at a planar distance of 0.063 -- a third of the way in.  The hand then presses on the
    #: bar for sixteen steps and slides the whole kettle 5.1 cm across the counter, which is
    #: the kettle being shoved rather than grasped.
    #: Measured over 24 seeds, escaping to 0.132 against the alternatives (the number here
    #: is what the escape exits at; "grasped" counts seeds where the jaws actually held the
    #: bar, which is not what the environment scores):
    #:
    #: ===========  =======  ======  ==================
    #: exit radius  grasped  banked  steps pressing bar
    #: ===========  =======  ======  ==================
    #: no latch          22      23                 613
    #: 0.132             23      22                  30
    #: 0.145             23      20                 105
    #: 0.160             23      22                 104
    #: 0.170             23      21                 171
    #: 0.180             20      20                   0
    #: ===========  =======  ======  ==================
    #:
    #: "no latch" banks the most and is the worst behaviour in the file: on seed 18 the hand
    #: never grasps the kettle at all, presses the bail bar for 556 steps, shoves the free
    #: body 31.5 cm across the counter and the environment scores it.  Read `banked` here
    #: only next to `grasped`.
    TOP_HANDLE_ESCAPE_RADIUS = 0.132
    #: False restores the original semantics, where the escape ends as soon as the height
    #: clause lapses.  Kept as a switch so the table on :attr:`TOP_HANDLE_ESCAPE_RADIUS` can
    #: be reproduced.
    TOP_HANDLE_ESCAPE_LATCH = True
    tool_offset = FINGERTIP_OFFSET
    approach_gripper = "open"
    engage_gripper = "close"
    recede_after_manipulation = True
    align_in_place = True
    # The kettle is a free body: its handle goes wherever the still-closed pads push it,
    # so the generic release-while-tracking-the-handle motion chases its own disturbance.
    hold_pose_while_releasing = True
    # A kettle-only cap on wrist rotation, kept because the side grasp swings the tool on a
    # lever and a full-speed turn overshoots the bar.  It was 0.10, chosen to damp an IK loop
    # that 1.2.0's `control_steps=5` created by repeating one large rotation across several
    # sub-steps; the servo re-plans every 0.08 s and cannot overshoot that way, so the tight
    # cap only made ALIGN slow -- measured, the skill spent 170-270 of its 400 steps there.
    # Sweep over 8 seeds: 0.10 -> 3/8 (median 179 steps), 0.25 -> 6/8 (108), 0.40 -> 5/8.
    rotation_step = 0.25
    # The open jaws provide only about 1.7 cm of lateral clearance around the 4.6 cm bar.
    # This is intentionally separate from the generic waypoint tolerance: the IK cannot
    # exactly attain the full side-grasp stand-off, but it can center the jaws without
    # changing their current forward/backward depth before the final approach.
    #
    # Sized by `KitchenPolicyConfig.kettle_grasp_center_tolerance`, which explains why it
    # cannot simply be made tight: the residual frame error puts a floor under the quantity
    # this gates, and a yawed grasp frame raises that floor.  A property rather than the
    # class constant it replaces, because it is the only skill whose value is configurable
    # and the base class default of 0.0 must keep applying to the others.
    lost_contact_tolerance = 0.25
    alignment_tolerance = 0.60
    # Closing 3 cm short leaves the bar beyond the fingertip pads. The centering stage now
    # handles IK residual separately, so require the bar centre to reach the pad depth.
    contact_tolerance = 0.015
    # A correctly captured 4.6 cm-diameter side bar holds a nonzero half-opening.
    # A nearly zero opening means the handle escaped and the reactive controller should
    # reopen and reacquire instead of treating one-finger contact as a grasp.
    #
    # This must sit *between* an empty close and a loaded one, not on top of either.  It
    # used to be 0.015, which is exactly the width `kettle_grasp_command = -0.25` asks for,
    # so the retention test straddled its own set point: a captured bar settles at
    # 0.016-0.020 and any momentary dip during transport read as a dropped handle and sent
    # the skill back to reacquire.  An empty close still lands at ~0.000, so 0.010
    # discriminates just as well with room to spare.
    grasp_contact_min_opening = 0.010
    #: Do not steer the wrist while backing off a delivered kettle.
    #:
    #: `desired_orientation` falls back to the live handle frame once the jaws reopen, and
    #: that frame is built from `approach_axis`, which is built from the kettle-to-goal
    #: direction -- a vector the push has just driven to about two centimetres, so its unit
    #: vector points wherever the noise says.  The wrist then chases a randomly turning
    #: target while an open finger is still inside the handle.  This is the same argument
    #: `withdraw_axis` makes for the retreat's *position*, which was already frozen; the
    #: orientation was not.  Measured over 12 randomized seeds it is worth 41 retreat steps
    #: down to 34, and with the shorter `kettle_recede_distance` 41 down to 9.
    orient_during_recede = False

    #: Planar tolerance for the high cross-kitchen waypoint.
    transit_tolerance = 0.25
    #: Planar tolerance for the sideways escape out of the top-handle column.  This is the
    #: fallback exit only: the escape normally ends as soon as the tool leaves the guarded
    #: volume, which happens well before the stand-off itself is reached.
    descent_column_tolerance = 0.05

    def __init__(self, sim: KitchenSim, task: str, cfg: KitchenPolicyConfig):
        super().__init__(sim, task, cfg)
        self.top_grasp = (cfg.kettle_grasp_target == "top")
        self.handle_local_pos = (self.TOP_HANDLE_LOCAL_POS if self.top_grasp
                                 else self.LEFT_HANDLE_LOCAL_POS)
        self.handle_local_axis = (self.TOP_HANDLE_LOCAL_AXIS if self.top_grasp
                                  else self.LEFT_HANDLE_LOCAL_AXIS)
        self._capture_confirmed = False
        #: Latched once the tool is found over the top bail bar, cleared only by getting
        #: planar-clear of it.  See :attr:`TOP_HANDLE_ESCAPE_RADIUS`.
        self._top_handle_escape = False
        self._capture_approach_axis = None
        self._capture_frame = None
        #: Tool and body heights at capture; the lift measures against these rather than
        #: against the goal, because the burner and the kettle's rest surface are the same
        #: height to within a millimetre and the *pick-up* height is the one observed.
        self._capture_tool_z = None
        self._capture_root_z = None
        #: Latched once the tool has actually risen, so a carry that dips does not restart
        #: the lift and stall in place.  See :meth:`manipulate_point`.
        self._lift_reached = False
        #: Latched once the carry has arrived and the set-down has begun, so a hanging
        #: kettle that swings back out of tolerance does not climb again.  See
        #: :meth:`_advance_carry`.
        self._descent_started = False
        #: Latched when a lifted carry has finished: set down, at the burner, still held.
        #: Unlike the two above it survives the jaws opening, which is the point of it --
        #: see :meth:`manipulation_done`.
        self._delivered = False
        #: Where the rise happens.  Anchoring it to the tool's position at capture, rather
        #: than to wherever the tool is now, is what makes the vertical leg closed-loop in
        #: the horizontal plane: "current tool, plus some z" never corrects its own drift,
        #: and measured it wandered the hanging kettle 3.1 cm in x and 4.1 cm in y.
        self._capture_tool_xy = None
        self._capture_root_xy = None
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
            self.sim.model.geom_pos[collision_geoms] - self.handle_local_pos,
            axis=1,
        )
        closest = int(np.argmin(local_errors))
        if local_errors[closest] > 0.01:
            raise ValueError("Could not identify the kettle's left-handle collision capsule.")
        self._left_handle_geom_id = int(collision_geoms[closest])
        self._finger_pad_geoms = finger_contact_geoms(self.sim)
        if not self.top_grasp:
            # Where on that capsule to take hold, which is not the same question as which
            # capsule it is: the lookup above has to match the geometry's own centre, and
            # the grasp is free to sit anywhere along the bar.  See
            # `KitchenPolicyConfig.kettle_side_grasp_height`.
            self.handle_local_pos = np.array(
                [self.LEFT_HANDLE_LOCAL_POS[0], self.LEFT_HANDLE_LOCAL_POS[1],
                 float(self.cfg.kettle_side_grasp_height)], dtype=np.float64)

    def touch_point(self) -> np.ndarray:
        body_rotation = self.sim.body_xmat("kettleroot")
        return (self.sim.body_xpos("kettleroot")
                + body_rotation @ self.handle_local_pos)

    def approach_axis(self) -> np.ndarray:
        """Come in diagonally, between the push direction and the bar's free side.

        This used to be :meth:`drive_direction` itself -- straight along the push.  Two
        things are wrong with that, and they are the same thing seen from two ends.

        Mechanically, :func:`grasp_frame` separates the jaws about the axis perpendicular to
        both the handle and the approach, so approaching along the push puts the pad faces
        *parallel* to the direction of travel and every newton of push is carried by
        friction on a smooth cylinder alone.  Kinematically, it asks the wrist to point
        along +y, and the arm runs out of wrist doing that as the kettle goes back: probed
        along the path, the residual of that frame grows from 0.006 rad at the kettle's
        start to 0.097 by y = 0.50 and 0.122 by y = 0.60, which is where the pads are simply
        dragged off the bar.  Measured, the transport lost the grasp at y = 0.586 every
        time and could not re-reach it afterwards.

        Leaning the approach out toward the bar's free side fixes both.  It swings the jaw
        faces round to take part of the push as a normal force on the trailing pad, and it
        is a frame the arm can hold for the whole journey: the same probe reads 0.0003 rad
        from y = 0.40 to y = 0.70, and 0.014 at the burner itself.  It also stays on the
        *un-flipped* grasp representative the whole way, so nothing here provokes a wrist
        flip (see :meth:`ScriptedKitchenPolicy._commit_grasp_frame`).

        The lean is expressed against the live body frame rather than as a fixed world
        heading, so it keeps pointing at the free side if the kettle yaws under the pads.
        """
        if self.top_grasp:
            # Straight down onto a bar that lies across the body: the jaws then close along
            # the push direction, one pad ahead of the bar and one behind it.
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        drive = self.drive_direction()
        outward = self.touch_point() - self._root_pos()
        outward[2] = 0.0
        outward = unit(outward)
        if not np.any(outward):
            return drive
        return unit(drive - float(self.cfg.kettle_approach_lateral_bias) * outward)

    def handle_axis(self) -> np.ndarray:
        # Follow the live vertical capsule as the free kettle body translates and rotates.
        return self.sim.body_xmat("kettleroot") @ self.handle_local_axis

    def desired_orientation(self) -> Optional[np.ndarray]:
        """The live handle frame while reaching; the frame that made the grasp after it.

        The jaws hold a smooth vertical post whose own axis is the kettle's yaw axis, so
        the grip resists yaw by pad friction alone -- which is to say hardly at all.  The
        body's heading is therefore set by whatever the *wrist* does, and until this froze
        the wrist did two things it should not.  While carrying, the base frame is rebuilt
        every step from the live body rotation, so any yaw the kettle picks up turns the
        target, which turns the wrist, which drags the kettle further the same way; and
        while receding it chases that same moving frame with the pads still brushing the
        body it is turning.  Traced on seed 0: the push torque alone yaws the body only
        -5.6 degrees, then the wrist drifts +0.29 rad over the last fourteen steps of the
        carry and takes the kettle from -5.6 to +11.7 degrees, and the retreat adds another
        +14 for a +26 degree finish.

        A frame frozen at capture removes the loop.  It cannot wind up, because it is a
        constant; it cannot chase the body, because it does not depend on it; and it is the
        measured pose the grasp was actually made in -- not the pose it was reaching for --
        so holding it commands no motion at all on the step it is taken, which is the only
        way a held object stays where the jaws found it.  It is released the moment the jaws reopen past
        ``grasp_ready_opening``, so a reacquisition plans against the kettle where it now
        is rather than where it was.
        """
        if self._capture_frame is not None:
            if not self.top_grasp and self.cfg.kettle_carry_orientation == "level":
                # Only the lean is corrected; the heading is whatever the wrist currently
                # has, so this commands no yaw at all.  Returned on the un-flipped
                # representative because that is what `equivalent_grasp_frame` expects to
                # be handed, and the live matrix is already the flipped one when the skill
                # committed to the flip.
                frame = upright_grasp_frame(np.asarray(self.sim.eef_mat, dtype=np.float64))
                return frame @ _JAW_FLIP if self.sim.grasp_flip else frame
            return self._capture_frame
        frame = super().desired_orientation()
        yaw = float(self.cfg.kettle_grasp_yaw)
        if frame is None or yaw == 0.0:
            return frame
        return rotate_about_axis(self.handle_axis(), yaw) @ frame

    def approach_blocked(self) -> bool:
        """Too low to come in: see ``kettle_approach_height_slack``."""
        slack = float(self.cfg.kettle_approach_height_slack)
        if slack <= 0.0 or self.top_grasp:
            return False
        return bool(self.contact_point()[2] - self.tool_pos()[2] > slack)

    def contact_point(self) -> np.ndarray:
        if not self.top_grasp:
            return (self.touch_point()
                    + self.approach_axis() * self.cfg.kettle_side_grasp_depth)
        return (self.touch_point()
                + self.approach_axis() * self.cfg.kettle_top_grasp_depth)

    def precontact_point(self) -> np.ndarray:
        return self.contact_point() - self.approach_axis() * self.cfg.precontact_distance

    def in_top_handle_corridor(self, point: np.ndarray) -> bool:
        """Whether ``point`` is somewhere the hand can strike the kettle's top handle."""
        if self.top_grasp:
            return False  # that handle is what the jaws are here for.
        root = self.sim.body_xpos("kettleroot")
        point = np.asarray(point, dtype=np.float64)
        planar = float(np.linalg.norm((point - root)[:2]))
        return bool(planar <= self.TOP_HANDLE_GUARD_RADIUS
                    and point[2] >= root[2] + self.TOP_HANDLE_GUARD_HEIGHT)

    def clear_of_top_handle(self) -> bool:
        """Whether the tool is far enough out, in plan, to descend past the bail bar."""
        if self.top_grasp:
            return True
        root = self.sim.body_xpos("kettleroot")
        planar = float(np.linalg.norm((self.tool_pos() - root)[:2]))
        return bool(planar > self.TOP_HANDLE_ESCAPE_RADIUS)

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
            self._top_handle_escape = False
            return None
        tool = self.tool_pos()
        column = self.precontact_point().copy()
        if self.in_top_handle_corridor(tool):
            self._top_handle_escape = True
        if self._top_handle_escape:
            if (self.clear_of_top_handle() if self.TOP_HANDLE_ESCAPE_LATCH
                    else not self.in_top_handle_corridor(tool)):
                self._top_handle_escape = False
            else:
                # Step sideways to the stand-off column at the height already reached.
                # Climbing back to the high waypoint instead would pay for the whole descent
                # twice.  Held until the tool is planar-clear rather than until the height
                # clause lapses, because descending out of the volume is descending onto the
                # bar -- see `TOP_HANDLE_ESCAPE_RADIUS`.
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
        if ((self._top_handle_escape and self.TOP_HANDLE_ESCAPE_LATCH)
                or self.in_top_handle_corridor(tool)):
            return planar <= self.descent_column_tolerance
        return planar <= self.transit_tolerance

    #: Best goal error the loaded transport has reached, and how long it has been stuck
    #: there; see :meth:`transport_stalled`.
    _transport_best_error = None
    _transport_stall_steps = 0

    @property
    def grasp_center_tolerance(self) -> float:  # type: ignore[override]
        return float(self.cfg.kettle_grasp_center_tolerance)

    def centering_entry_tolerance(self) -> float:
        """Base threshold plus the part of the error the wrist tilt alone explains.

        See :attr:`KitchenPolicyConfig.kettle_centering_tilt_slack` for why the raw error is
        not a lateral offset and cannot be driven to a fixed small number.
        """
        base = super().centering_entry_tolerance()
        slack = float(self.cfg.kettle_centering_tilt_slack)
        if slack <= 0.0:
            return base
        desired = self.desired_orientation()
        if desired is None:
            return base
        reach = float(np.linalg.norm(self.contact_point() - self.tool_pos()))
        return base + slack * reach * float(np.sin(orientation_error(self.sim, desired)))

    # The kettle deliberately has *no* ALIGN hysteresis: `realign_tolerance` is left None,
    # so it breaks off a reach at the same frame error it stops aligning at.  On the light
    # switch that arrangement is a limit cycle worth removing; here it is the reach working.
    #
    # It certainly looks like the same bug.  Measured on seed 7 arriving from the light
    # switch, ALIGN hands back frame errors of 0.156, 0.120 and 0.101, every
    # MOVE_TO_PRECONTACT step puts them straight back to 0.185-0.218, and the pair
    # alternates ten times before ALIGN burns its 80-step budget -- the visible stall where
    # the gripper hovers beside the kettle turning itself.  But each cycle also creeps the
    # arm in, and taking the cycle away takes the grasp with it: a looser realign threshold
    # measures kettle 23/24 seeds -> 21/24 at 0.24 and 22/24 at 0.30, buying about seven
    # steps of median episode for a grasp.  The same is true of ending the park early
    # (see `orient_forward_budget`).  Both of the obvious cures for the stall are worse than
    # the stall; what it actually needs is a reach that does not fight its own wrist.
    def manipulation_stage(self) -> str:
        return KETTLE_TRANSPORT

    def manipulation_step_scale(self) -> float:
        if not self.lifting():
            return float(self.cfg.kettle_transport_step)
        if not self._lift_reached:
            return float(self.cfg.kettle_lift_step)
        return float(self.cfg.kettle_carry_step)

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
            return float(self.cfg.kettle_top_grasp_command if self.top_grasp
                         else self.cfg.kettle_grasp_command)
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

    def grip_load(self) -> float:
        """Total normal force the pads are pressing into the grasped capsule, in newtons.

        The lift needs this and the push does not, which is the whole asymmetry of the two
        carries.  With the jaws closing along the drive direction, a horizontal push is
        carried by the pad *normals* -- form closure, available the instant the pads touch.
        Straight up there is no form closure at all: a smooth cylinder is held against its
        own weight by friction alone, so the grasp has to be loaded before it can be moved.

        Measuring it is what distinguishes "the pads are touching" from "the pads are
        holding", and only the second is a grasp you can lift with.  See
        :meth:`manipulate_point`.
        """
        total = 0.0
        result = np.zeros(6, dtype=np.float64)
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if ((geom1 == self._left_handle_geom_id and geom2 in self._finger_pad_geoms)
                    or (geom2 == self._left_handle_geom_id
                        and geom1 in self._finger_pad_geoms)):
                mujoco.mj_contactForce(self.sim.model, self.sim.data, index, result)
                total += abs(float(result[0]))
        return total

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
                # `_capture_approach_axis` is deliberately *not* cleared here.  It is read
                # only by `withdraw_axis`, and the one caller of that -- `recede_point` --
                # runs strictly after the release has opened the jaws, so clearing it on
                # the reopen guaranteed the retreat was planned from the live fallback the
                # docstring there exists to avoid.  Measured on the fixed order, seed 1: at
                # the last held step the frozen axis was (0.45, 0.89, 0) and the release
                # flipped it to (-0.21, -0.98, 0) -- very nearly reversed, because a
                # delivered kettle has overshot its goal in y and the live kettle-to-goal
                # vector then points back the way the push came.  The waypoint froze at
                # y = 1.02 against a body at y = 0.80, i.e. 0.22 m past the kettle on its
                # far side, and the hand climbed 13 cm over the body to reach it, dragging
                # the bail bar with an open finger the whole way.  A re-grasp overwrites
                # this on its next bilateral capture, so nothing stale can survive a retry.
                self._capture_frame = None
                self._capture_tool_z = None
                self._capture_root_z = None
                self._capture_tool_xy = None
                self._capture_root_xy = None
                self._lift_reached = False
                self._descent_started = False
            return False
        if len(fingers) == 2:
            if not self._capture_confirmed:
                # Record the approach while it is still a long, well-conditioned vector;
                # see withdraw_axis().
                self._capture_approach_axis = self.approach_axis()
                # And the frame that made the grasp; see desired_orientation().  This is
                # the *measured* wrist, not the geometric target it was reaching for.  The
                # two differ by however much frame error the approach was allowed to close
                # on -- 0.171 rad on seed 0, against a 0.18 tolerance -- and holding the
                # target instead means spending the first steps of the carry turning the
                # wrist through that residual with the object already in the jaws.  On a
                # lift that is the whole failure: the correction swings the hanging kettle
                # 5.7 cm backwards and shakes the bar out of the pads by step 12.
                frame = np.asarray(self.sim.eef_mat, dtype=np.float64)
                # Store it on the *un-flipped* representative, which is the one every
                # consumer of a desired frame is required to be handed: `orientation_error`
                # and `rotation_action` both push it through `equivalent_grasp_frame`, which
                # re-applies the jaw flip itself when the skill has committed to one.  The
                # live wrist already *is* the flipped representative in that case, so
                # storing it raw and handing it over means flipping twice, and the carry is
                # then commanded to a frame a half turn from the one the jaws are in.
                #
                # Invisible on any seed that grasps un-flipped, which is most of them.  On
                # seed 4, where the pose off the slide cabinet makes `_commit_grasp_frame`
                # take the flip, the frame error jumps from 0.038 rad at CONTACT_OR_GRASP to
                # 3.101 the instant KETTLE_TRANSPORT reads this, and the carry spends itself
                # rolling the wrist 180 degrees with the kettle in the jaws -- against 0.030,
                # 0.019 and 0.014 rad on seeds 0, 2 and 5, which grasp un-flipped.
                #
                # The `"level"` branch of `desired_orientation` already converts, with the
                # same reasoning written out; this is the `"capture"` path, which is the
                # default and did not.
                if self.sim.grasp_flip:
                    frame = frame @ _JAW_FLIP
                if self.cfg.kettle_upright_carry_frame and not self.top_grasp:
                    frame = upright_grasp_frame(frame)
                self._capture_frame = frame
                # And the two heights the lift is measured against; see manipulate_point().
                self._capture_tool_z = float(self.tool_pos()[2])
                self._capture_root_z = float(self._root_pos()[2])
                self._capture_tool_xy = self.tool_pos()[:2].copy()
                self._capture_root_xy = self._root_pos()[:2].copy()
                self._lift_reached = False
                self._descent_started = False
            self._capture_confirmed = True
        # Live contact is required on every query, with no grace for single-step dropouts.
        # Smoothing them looks right -- a carried bar breaks and remakes contact constantly,
        # and each break sends the skill back to re-approach a kettle it is already holding
        # -- but it measures strictly worse: tolerating 4 contact-free steps took this from
        # 7/8 (median 104 steps) to 3/8 (median 211), and 10 steps to 4/8. The churn is the
        # skill's recovery working; suppressing it keeps the transport driving a kettle the
        # jaws have actually dropped, which pushes it somewhere worse than where it started.
        #
        # Three sharper attempts on the same break, none of which survives measurement.
        # The case is seed 6, where the carry stops at step 211, releases, walks back and
        # retakes a bar it never dropped -- the half-opening reads 0.0185, 0.0185, 0.0187
        # straight through the break -- costing 14 steps of a 312-step episode.
        #
        #   * Bridge the dropout only while the opening still says the bar is loaded, which
        #     is the direct evidence a blanket grace lacks. Note it must key off
        #     `data.time`, not a call count: this method is queried more than once per step,
        #     so a one-step grace burns inside a single step. Working correctly, seed 6 goes
        #     4/4 to 2/4 -- it bridges the first break, breaks again anyway, and ends with
        #     140 steps of MOVE_TO_PRECONTACT churn and no kettle.
        #   * Grip harder so the bar stops chattering: no fewer breaks at
        #     `kettle_grasp_command` -0.15, a lost task at -0.30, and 1 break becoming 4
        #     at -0.50.
        #   * Retake it in place instead of walking back, as `MicrowavePullSkill` does:
        #     seed 6 goes 4/4 to 2/4 with 94 steps of APPROACH. The walk back to the
        #     stand-off is doing real work here that it does not do for the microwave --
        #     it re-centres the jaws on the bar, and `jaw_center_error` is what the kettle
        #     approach is gated on.
        #
        # So the dropout is real, the recovery is right, and the walk back is the price.
        return bool(self._capture_confirmed and fingers)

    def goal_position_error(self) -> float:
        """Planar distance from the live kettle body to its goal position."""
        return float(np.linalg.norm((self._goal_pos() - self._root_pos())[:2]))

    def grasp_goal_point(self) -> np.ndarray:
        """Where the tool must go to put the kettle *body* at its goal position.

        The jaws hold a point 9.2 cm off the body's centre line, so where the bar has to
        end up depends on how the body is turned.  Composing the offset with the *goal*
        rotation -- what this used to do -- assumes the transport will also have corrected
        the yaw by the time it arrives.  It does not: the pads grip a smooth vertical
        cylinder, whose own axis is the yaw axis, so friction there resists yaw hardly at
        all, and a body dragged from off-centre picks it up faster than the wrist can take
        it out.  Every seed then stopped with the kettle short and skewed -- measured at
        ``root = (-0.35, 0.52)`` against a goal of ``(-0.23, 0.75)``, drifting in -x toward
        the microwave, exactly the transport this target was supposed to fix.

        Referring the offset to the *live* rotation instead removes the assumption.  The
        expression below is that statement with the body frame cancelled out of it: it is
        simply "move the tool by whatever the body still has left to travel", so the
        controller closes the body's own position error whatever the yaw is doing.  Yaw is
        then only a residual in the environment's 7-D distance, and a small one -- the goal
        quaternion is 0.12 rad off the reset yaw, so a kettle that is merely never *made*
        to turn already starts near it.

        The height is deliberately dropped.  Goal and reset z differ by a millimetre, and
        commanding the difference would turn any tip of the free body into a vertical
        request that lifts or presses the kettle mid-transport.
        """
        return (self.tool_pos() + self._transport_remaining()
                + self._yaw_correction())

    def _transport_remaining(self) -> np.ndarray:
        """Planar displacement the body owes, measured against the line it should be on.

        Aiming straight at the goal from wherever the body currently is says nothing about
        the route, and the route matters: the microwave's near side panel is a box reaching
        to ``x = -0.458`` across ``y = 0.590..0.638``, and the kettle's spout sticks 0.207 m
        out of the body's -x side, so a body that wanders past ``x = -0.251`` on its way
        through that band catches the spout on it.  The straight line from where the grasp
        was made to the burner passes at ``x = -0.241`` and is clear; the transport simply
        does not follow it, because the pads slip on the smooth post and the body drifts a
        couple of centimetres to -x, after which "head for the goal" is a diagonal that cuts
        the corner into the panel.  Traced, that is where the flat push stops: the kettle
        rides up the panel, stalls at ``y = 0.634``, and the grasp is levered out of the
        jaws.

        Pure pursuit along that line removes the question -- the target is a point *on* the
        route, ``kettle_transport_corridor_lookahead`` ahead of how far the body has got
        along it, so cross-track drift is an error the controller closes rather than a
        shortcut it takes.  The lead saturates at the goal itself, so the arrival is
        unchanged.
        """
        goal, root = self._goal_pos(), self._root_pos()
        remaining = goal - root
        remaining[2] = 0.0
        start = self._capture_root_xy
        lead = float(self.cfg.kettle_transport_corridor_lookahead)
        if start is None or lead <= 0.0:
            return remaining
        travel = goal[:2] - start
        length = float(np.linalg.norm(travel))
        if length < 1e-6:
            return remaining
        travel = travel / length
        along = float(np.dot(root[:2] - start, travel))
        aim = start + travel * min(max(along, 0.0) + lead, length)
        remaining[:2] = aim - root[:2]
        return remaining

    def reached_goal_plane(self) -> bool:
        """Whether the body has travelled the whole route, measured along the route.

        The stop this backs up is :attr:`KitchenPolicyConfig.kettle_goal_crossing_margin`.
        Progress is projected onto the same start-to-goal line the corridor steers along,
        so it answers "has it arrived" and not "is it abeam": a body pushed wide still owes
        the cross-track error, and the radius test in :meth:`manipulation_done` is what
        closes that.  Frozen from ``_capture_root_xy``, which the grasp clears if it is
        lost, so a dropped kettle cannot report itself delivered.
        """
        margin = self.cfg.kettle_goal_crossing_margin
        start = self._capture_root_xy
        if margin is None or start is None:
            return False
        travel = self._goal_pos()[:2] - start
        length = float(np.linalg.norm(travel))
        if length < 1e-6:
            return False
        along = float(np.dot(self._root_pos()[:2] - start, travel / length))
        return bool(along >= length - float(margin))

    def _yaw_correction(self) -> np.ndarray:
        """Extra planar tool travel that would put the held post back on a square kettle.

        The jaws hold one point, 9.2 cm off the body's centre line, so the body's heading
        and the post's position are the same fact seen twice: a kettle yawed by theta has
        its post displaced by 0.092 * theta from where a square kettle would keep it.
        Asking the tool for that displacement is therefore a yaw command issued entirely in
        translation -- no wrist rotation, which is what made every previous attempt at this
        worse (see the transport branch in the controller).  It is bounded by 0.092 * theta,
        a couple of centimetres at the yaws actually seen, so it cannot fight the carry.

        Yaw only: the goal quaternion also carries a slight tilt, and commanding that
        through a friction grip on a vertical post would just press the kettle into the
        counter.
        """
        error = wrap_to_pi(GOAL_KETTLE_YAW - self._body_yaw())
        if abs(error) < 1e-9:
            return np.zeros(3, dtype=np.float64)
        offset = self.sim.body_xmat("kettleroot") @ self.handle_local_pos
        correction = rotate_about_axis(np.array([0.0, 0.0, 1.0]), error) @ offset - offset
        correction[2] = 0.0
        return correction * float(self.cfg.kettle_yaw_correction_gain)

    def _body_yaw(self) -> float:
        rotation = self.sim.body_xmat("kettleroot")
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def manipulate_point(self) -> np.ndarray:
        """Short live lead toward :meth:`grasp_goal_point`.

        This used to steer by a *heading*: take the planar root-to-goal direction, force a
        minimum rightward lean into it (a since-removed knob), and push along
        it.  A heading says which way to shove but never says where to stop, and the jaws
        hold the bar 9.2 cm off the body's centre line, so shoving forward through it yaws
        the kettle and walks it sideways faster than the lean corrects.  Measured over the
        whole transport: the body ran 0.157 m in -x -- straight at the microwave, which sits
        at x = -0.64 -- while gaining only 0.16 m of the 0.40 m it needed in +y, and the
        episode scored only by scraping the 0.3 threshold.

        Aiming at :meth:`grasp_goal_point` removes the question: the target names where the
        body still has to get to, so lateral drift is an error the controller closes rather
        than a direction it never knew it was wrong about.  The lead is still clamped to the
        distance remaining, so the request shrinks to zero on arrival instead of driving at
        full step until the stop predicate happens to fire.

        On a lifted carry the height comes from :meth:`carry_height` and the horizontal
        request is withheld entirely until the tool is up, so the rise happens in place.
        The planar target is unchanged: the same "move the tool by whatever the body still
        has left to travel", now issued at altitude.
        """
        self._advance_carry()
        point = self.grasp_goal_point()
        lookahead = float(self.cfg.kettle_transport_lookahead)
        tool = self.tool_pos()
        if not self.lifting():
            if self._capture_tool_z is not None and self.cfg.kettle_transport_hold_height:
                point[2] = (self._capture_tool_z
                            + float(self.cfg.kettle_push_lift_height))
            delta = point - tool
            point = tool.copy()
            remaining = float(np.linalg.norm(delta))
            if remaining > 1e-9:
                point += delta / remaining * min(lookahead, remaining)
            return point
        if self._lift_reached:
            lookahead = float(self.cfg.kettle_carry_lookahead)
        else:
            lookahead = float(self.cfg.kettle_lift_lookahead)
            point = np.array([self._capture_tool_xy[0], self._capture_tool_xy[1], 0.0])
        point[2] = self.carry_height()
        delta = point - tool
        point = tool.copy()
        remaining = float(np.linalg.norm(delta))
        if remaining > 1e-9:
            point += delta / remaining * min(lookahead, remaining)
        return point

    def lifting(self) -> bool:
        """Whether this carry is a lift rather than a push along the counter."""
        return bool(self.top_grasp
                    and self.cfg.kettle_lift_height > 0.0
                    and self._capture_tool_z is not None)

    def _advance_carry(self) -> None:
        """Move a lifted carry on to its next leg when the current one is finished.

        Three legs -- rise in place, cross at altitude, set down -- and the height is the
        whole of what distinguishes them, so they need no phase of their own.  What they do
        need is to be one-way, which is what this is: both transitions latch.

        Rising in place matters.  Asking for the lift and the translation at once gives a
        diagonal whose first centimetres are still mostly horizontal, which is exactly the
        counter-dragging moment the lift exists to avoid -- and it is worst at the start,
        with the body still flat on the surface under its full weight.  So the horizontal
        request is withheld (see :meth:`manipulate_point`) until the tool is up, and
        latched so that a dip under load does not send the carry back to the beginning.

        Setting down has to latch for a different reason: a kettle hanging from a bail bar
        is a pendulum, and its planar error keeps crossing any tolerance in both directions.
        Reading it live -- down while inside, up while outside -- makes the descent a limit
        cycle that never lands: over 8 seeds it left the body a mean 0.081 m from the burner
        and as much as 0.12 m past it, having been flown back and forth over it.  So the
        set-down is committed to a deliberate ``kettle_setdown_distance`` out, and the last
        stretch is closed with the kettle standing on the counter.

        Idempotent, so both :meth:`manipulate_point` and :meth:`manipulation_done` can call
        it and neither has to run first.
        """
        if not self.lifting():
            return
        if not self._lift_reached:
            if (self.tool_pos()[2] >= self._capture_tool_z + self.cfg.kettle_lift_height
                    - self.cfg.kettle_lift_tolerance):
                self._lift_reached = True
            return
        if not self._descent_started:
            if self.goal_position_error() <= self.cfg.kettle_setdown_distance:
                self._descent_started = True
            return
        if (self.placed()
                and self.goal_position_error()
                <= self.cfg.kettle_goal_position_tolerance):
            self._delivered = True

    def carry_height(self) -> float:
        """Tool height for the current leg of a lifted carry; see :meth:`_advance_carry`."""
        if self._descent_started:
            return self._capture_tool_z
        return self._capture_tool_z + self.cfg.kettle_lift_height

    def carrying(self) -> bool:
        """Whether the lifted carry is on its long horizontal leg.

        The two vertical legs are short and are working against the load's own weight, so
        they want the full integrator; this one is neither.  See
        :attr:`KitchenPolicyConfig.kettle_carry_bias_leak`.
        """
        return bool(self.lifting() and self._lift_reached and not self._descent_started)

    def placed(self) -> bool:
        """Whether the carried body is back down at the height it was picked up from."""
        if not self.lifting():
            return True
        return bool(self._root_pos()[2]
                    <= self._capture_root_z + self.cfg.kettle_place_tolerance)


    def manipulation_done(self) -> bool:
        """Stop when the kettle body is actually at the goal position, and not before.

        The environment's 7-D distance mixes position with a quaternion difference, and
        nothing here actuates the quaternion: a friction grip on one 4.6 cm bar cannot undo
        the yaw a dragged kettle picks up, and the wrist rotation that used to try was
        removed for making both terms worse (see the transport branch).  Position is
        therefore the whole predicate.  Waiting on a yaw tolerance as well -- which this
        used to do -- cannot improve the yaw, it can only hold a delivered kettle under the
        jaws until the task budget expires.

        A lifted carry adds one condition and one only: the kettle has to be back on the
        surface.  This predicate is what ends MANIPULATE, and what follows it opens the
        jaws, so satisfying it in mid-air would drop the kettle onto the burner from
        ``kettle_lift_height``.  See :meth:`placed`.

        And once it has been set down, it stays delivered.  Everything above is measured
        against a *held* kettle -- ``placed`` reads the height it was picked up from, which
        is only known while the grasp that picked it up is still on -- so when the jaws open
        the whole test would otherwise fall back to the 2 cm the transport steers to, which
        the release itself is enough to break.  It did: the recede's completion check reads
        this, found it false, and retook a kettle that was already on the burner, over and
        over.  Measured in the randomized suite, the task ran 626 steps on 6 seeds of 8 and
        never ended, with the body 0.07-0.13 m from the goal the whole time and nothing to
        gain.  Past the latch the standard for leaving it alone is
        ``kettle_delivered_position_tolerance``, the same one the controller uses to decide
        a slipped kettle is close enough.

        The position test has two halves because a radius alone is the wrong shape for a
        push.  A body being driven along a line arrives *through* the tolerance, and if the
        radius does not happen to close on one of the steps it is inside, the push simply
        keeps driving -- past the goal, and with a wound-up integrator behind it.  So
        arriving at the goal plane ends the push too; see :meth:`reached_goal_plane` and
        ``kettle_goal_crossing_margin``.

        There used to be a prior branch that stopped as soon as the pose distance fell
        inside a fraction of ``BONUS_THRESH``.  That is what left every rollout
        with the kettle stranded mid-counter: the distance starts at 0.40 against a 0.3
        threshold, so 0.255 is reached after about a third of the journey.  Measured on seed
        0, transport was cut at step 36 with the body at ``y = 0.522`` and the burner at
        0.75 -- still holding the handle, still tracking, and told to let go.  The kettle
        has to be carried to the burner, not to the reward threshold.

        Ending the push early on a *stall* -- no gain for N consecutive steps once the
        environment has banked the task -- was tried, on the reasoning that the last stretch
        looked like wasted nudging.  It is not, and it measures worse at every threshold:
        the push closes to a mean 0.031 m of the burner if it is left alone, and to 0.082 if
        it is stopped after ten settled steps, taking the delivered pose distance from 0.238
        to 0.301.  What looked like the push wasting time was the *retreat*; see
        ``kettle_recede_distance`` and ``KettleGraspSkill.orient_during_recede``.
        """
        self._advance_carry()
        if self._delivered:
            return bool(self.goal_position_error()
                        <= self.cfg.kettle_delivered_position_tolerance)
        return bool(self.placed()
                    and (self.goal_position_error()
                         <= self.cfg.kettle_goal_position_tolerance
                         or self.reached_goal_plane()))

    def transport_stalled(self) -> bool:
        """Whether a banked transport has stopped making progress toward its aim.

        See :attr:`KitchenPolicyConfig.kettle_transport_stall_steps`.  The caller is
        responsible for checking that the environment has already credited the task; this
        only answers whether the body is still moving toward where it is being pushed.
        """
        budget = self.cfg.kettle_transport_stall_steps
        if budget is None:
            return False
        error = self.goal_position_error()
        if (self._transport_best_error is None
                or error < self._transport_best_error
                - float(self.cfg.kettle_transport_stall_epsilon)):
            self._transport_best_error = error
            self._transport_stall_steps = 0
            return False
        self._transport_stall_steps += 1
        return self._transport_stall_steps > int(budget)

    def withdraw_axis(self) -> np.ndarray:
        """Direction the hand came in from, frozen at capture.

        ``approach_axis`` is the live kettle-to-goal direction, which is exactly what a
        successful transport drives to zero: at the end of the push the remaining vector
        is a couple of centimetres of numerical noise, and its unit vector points
        anywhere.  Backing out along *that* is what dragged a delivered kettle a further
        0.2 m across the counter while the pads were still closed on it.  The direction
        the grasp was made from is well defined and is still the right way out.

        Leaning that exit out toward the handle's free side, to keep the pads off the body
        on the way, was tried and does nothing: swept at 0.4, 0.8 and 1.2 the delivered tilt
        was 13.5, 13.0 and 13.9 degrees against 13.1 for a straight exit.  What the finger
        catches on is the kettle's shoulder, and the fix for that is to stop the hand
        sinking onto it -- see ``kettle_transport_hold_height``.
        """
        if self._capture_approach_axis is not None:
            return self._capture_approach_axis.copy()
        return self.approach_axis()

    def recede_point(self) -> np.ndarray:
        """Fixed post-release waypoint directly backward from the kettle handle.

        Freeze both the handle position and backward axis when receding begins so the
        target cannot move under the controller. Projecting the withdrawal axis off the
        handle axis makes the retreat exactly orthogonal to the live handle.

        This used to be followed by a radial push, ``_cleared_of_overhead``, that drove the
        waypoint out to a minimum distance from the kettle *root* so the hand could not be
        left parked under the bail bar.  It was fixing the wrong thing.  The reason the
        hand ended up under the bar is that ``withdraw_axis`` was reading a live fallback
        rather than the frozen capture axis -- see the note in ``grasp_retained`` -- so the
        retreat aimed 0.22 m past the kettle on its far side and the arm climbed over the
        body to get there.  With the axis correct the retreat leaves in a straight line and
        the radial push is a 0.25 m sideways detour that costs more than it buys.  Measured
        over 24 randomized seeds, disabling it against holding it at 0.25: all-four-true
        22/24 against 21/24, mean microwave re-grips 3.12 against 4.00, kettle pose error
        0.112 against 0.122, and the kettle is still not disturbed after its own skill ends
        (mean 0.004 m, worst 0.036, nothing lifted on any seed, against a mean of 0.000).
        Seed 7 is the clearest case: the detour left the arm somewhere the microwave that
        followed could not grasp from, and removing it takes that seed from 38 re-grips and
        3/4 to a clean single grasp and 4/4.
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
            point = (self._recede_origin
                     - backward_axis * self.cfg.kettle_recede_distance)
            # A retreat never climbs.  Everything above the delivered kettle -- the bail
            # bar, the posts -- is what an open finger catches on, so a waypoint above the
            # hand asks the arm to leave through exactly the geometry it is trying to get
            # clear of, and the tool covers the vertical part of that first because it is
            # the shortest leg.  The waypoint inherits its height from the handle, which is
            # normally a centimetre or two *below* the hand and so already satisfies this;
            # the clamp is here so that no combination of grasp height and droop can turn
            # the exit into a lift.
            point[2] = min(float(point[2]), float(self.tool_pos()[2]))
            self._fixed_recede_point = point
        return self._fixed_recede_point.copy()

    def recede_complete(self) -> bool:
        """Finish at the fixed waypoint or after equivalent backward clearance."""
        target = self.recede_point()  # Lazily freeze the origin and handle-normal axis.
        tool_pos = self.tool_pos()
        backward_progress = float(np.dot(
            self._recede_origin - tool_pos, self._recede_backward_axis))
        tolerance = self.cfg.kettle_recede_tolerance
        # Against the waypoint actually planned rather than `kettle_recede_distance`: the
        # non-climbing clamp can shorten it, and an escape measured on the nominal figure
        # would then never fire.
        planned = float(np.linalg.norm(target - self._recede_origin))
        return bool(
            np.linalg.norm(tool_pos - target) <= tolerance
            or backward_progress >= planned - self.cfg.position_tolerance
        )


#: The oven knobs, and only those, are served by :class:`KnobSkill`.
#:
#: This used to be spelled ``task in TASK_MANIPULATED_JOINTS``, which was true of the two
#: knobs alone while that table held only them.  Filling the table out for 1.2.1 -- every
#: task needs its manipulated joint, since 1.2.1's task names are no longer the joint names
#: -- silently widened the clause to catch *all seven*, so the microwave, hinge cabinet and
#: kettle were built as knob skills and never reached their own branches below.
#: World yaw the kettle is asked to end up at, taken from the environment's own goal
#: quaternion.  Only the yaw is used: see `KettleGraspSkill._yaw_correction`.
def _kettle_goal_yaw() -> float:
    quat = np.asarray(OBS_ELEMENT_GOALS["kettle"][3:], dtype=np.float64)
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quat / np.linalg.norm(quat))
    matrix = matrix.reshape(3, 3)
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


GOAL_KETTLE_YAW = _kettle_goal_yaw()


KNOB_TASKS = ("bottom burner", "top burner")


def build_skill(sim: KitchenSim, task: str, cfg: KitchenPolicyConfig) -> KitchenSkill:
    """Map a task name onto the skill that knows how to do it."""
    if task == "slide cabinet":
        return SlideSkill(sim, task, cfg)
    if task == "light switch":
        return LightSwitchSkill(sim, task, cfg)
    if task == "microwave":
        return MicrowavePullSkill(sim, task, cfg)
    if task == "hinge cabinet":
        return HandlePullSkill(sim, task, cfg)
    if task == "kettle":
        if cfg.kettle_strategy == "grasp":
            return KettleGraspSkill(sim, task, cfg)
        return KettlePushSkill(sim, task, cfg)
    if task in KNOB_TASKS:
        return KnobSkill(sim, task, cfg)
    raise ValueError(f"No scripted skill for kitchen task {task!r}.")


# ---------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------

def _microwave_recede_diagnostics(
        skill: "Optional[MicrowavePullSkill]") -> Dict[str, Any]:
    """Withdrawal progress for the debug snapshot, or ``None``s before RECEDE begins.

    Read-only by construction: it touches only state `MicrowavePullSkill.recede_point` has
    already frozen, so asking for diagnostics can never be what decides the exit route.
    """
    blank = {
        "microwave_recede_progress": None,
        "microwave_recede_progress_target": None,
        "microwave_recede_error": None,
        "microwave_recede_tolerance": None,
    }
    if skill is None or skill._fixed_recede_point is None:
        return blank
    tool = skill.tool_pos()
    return {
        # Metres withdrawn along the frozen door normal, against the distance that ends
        # the phase.  This is the condition that actually fires: the tolerance below is a
        # radius on the exit point, and the skill leaves on whichever comes first.
        "microwave_recede_progress": float(np.dot(
            skill._recede_origin - tool, skill._recede_backward_axis)),
        "microwave_recede_progress_target": float(
            skill.cfg.microwave_recede_distance - skill.cfg.position_tolerance),
        "microwave_recede_error": float(np.linalg.norm(
            tool - skill._fixed_recede_point)),
        "microwave_recede_tolerance": float(skill.cfg.microwave_recede_tolerance),
    }


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
        self._base_action_repeat = int(self.cfg.action_repeat)
        self._action_repeat = self._base_action_repeat
        self.cfg = self._scale_step_budgets(self.cfg)
        self._active_control_profile_task = None
        self.sim.scale_gripper_stiffness(self.cfg.gripper_stiffness_scale)
        self.sim.install_orientation_controller(
            self.cfg.ik_orientation_duration,
            self.cfg.ik_position_weight,
            nullspace_gain=self.cfg.ik_nullspace_gain,
            posture_gain=self.cfg.ik_transit_posture_gain,
        )
        self.servo = JointServo(self.sim, self.cfg.servo_ki, self.cfg.servo_integral_clamp,
                                proportional=self.cfg.arm_command_excess,
                                feasible_windup=self.cfg.servo_feasible_windup)
        self.reset()

    # -- episode state -------------------------------------------------------------------
    def reset(self, observation=None, info=None) -> None:
        """Clear all per-episode bookkeeping.  Call after the env has been reset."""
        self.servo.reset()
        self._apply_control_profile(None)
        self._task: Optional[str] = None
        self._skill: Optional[KitchenSkill] = None
        self.sim.grasp_flip = None
        #: Set once ALIGN has spent `align_budget` steps failing to reach its frame; see
        #: :attr:`KitchenPolicyConfig.align_budget`.
        self._align_exhausted: bool = False
        self._phase: str = (ORIENT_FORWARD if self.cfg.align_forward_at_reset
                            else SELECT_SUBTASK)
        self._initial_orientation_complete = not self.cfg.align_forward_at_reset
        #: The tolerances the *current* park exits on, resolved from
        #: `orient_forward_transition_tolerances` when its transition is in the table.  The
        #: park at episode start has no previous subtask, so it keeps the globals.
        self._orient_forward_yaw_tolerance = float(self.cfg.yaw_tolerance)
        self._orient_forward_clearance_tolerance = float(
            self.cfg.orient_forward_position_tolerance)
        #: Whether the current park is driven in joint space from its first step, and
        #: whether it is a between-subtask park at all (the opening park never falls back);
        #: see `KitchenPolicyConfig.park_joint_transitions`.
        self._park_joint_from_start: bool = False
        self._park_between_subtasks: bool = False
        self._park_joint_exit_required: bool = False
        #: Nine-value joint target that replaces the IK solve for the step being computed,
        #: or None.  Set by the joint-space park, read by :meth:`cartesian_plan_to_action`.
        self._joint_target_override: Optional[np.ndarray] = None
        #: (phase step, EEF position) samples for `opening_park_rest_steps`.
        self._park_rest_history: List[Tuple[int, np.ndarray]] = []
        #: Whether a turn-first hold is in progress; hysteresis for `turn_first_distance`.
        self._turn_first_active: bool = False
        #: Set by the free-space reach branch for the step being computed, so the reach
        #: posture applies to that reach and not to the centring or transit steps that also
        #: report MOVE_TO_PRECONTACT; see `reach_posture_gain`.
        self._reach_posture_active: bool = False
        self._phase_steps: int = 0
        self._retry_count: int = 0
        self._abandoned: List[str] = []
        #: How many times each subtask has been given up on, so a revived one cannot be
        #: revived for ever.  See `KitchenPolicyConfig.task_reattempt_limit`.
        self._abandon_counts: Dict[str, int] = {}
        self._completed_order: List[str] = []
        self._eef_history: List[np.ndarray] = []
        self._progress_history: List[float] = []
        #: (task step, end-effector position, task distance) samples for the whole selected
        #: subtask, trimmed to `task_stall_seconds`.  Deliberately *not* cleared by
        #: `_enter_phase`: the failure it exists for is an oscillation between two phases,
        #: which a per-phase history can never see.  See :meth:`_task_stalled`.
        self._task_stall_history: List[Tuple[int, np.ndarray, float]] = []
        #: (task step, frame error) samples for the running ALIGN; see `align_stall_seconds`.
        self._align_stall_history: List[Tuple[int, float]] = []
        self._last_target: Optional[np.ndarray] = None
        self._last_position_error: float = 0.0
        self._last_tool_error: float = 0.0
        self._reach_error_history: List[float] = []
        #: (phase step, frame error) samples for the in-place RECEDE rotation, trimmed to
        #: the progress window.  See :meth:`_recede_rotation_stalled`.
        self._recede_rotation_history: List[Tuple[int, float]] = []
        self._retreat_lift_target = None
        self._alignment_hold_target = None
        self._reactive_retreat_pending = False
        self._reactive_recede_pending = False
        self._release_hold_pos = None
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
        #: And the *configuration* it is in there, which is what ORIENT_FORWARD's posture
        #: bias prefers.  The Cartesian home pose does not pin the arm's shape -- seven
        #: hinges for a six-dimensional task -- so the solve is free to reach it folded.
        self._home_qpos = np.asarray(self.sim.data.qpos[:7], dtype=np.float64).copy()
        #: Geoms belonging to the arm, and how often a solve had to be redone because the
        #: first answer folded it into itself.  See :meth:`_arm_self_colliding`.
        self._arm_geoms = None
        self._self_collision_rescues = 0
        #: A horizontal gripper held at the raw reset flange position would put its
        #: fingertips through the cabinet plane.  Back the flange toward the room
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
            # The environment's own generator, so the order replays from the seed passed to
            # `env.reset`. This used to be an unseeded `default_rng()` built in __init__,
            # which made the order differ on every *process launch*: two runs of the same
            # script at the same seed did different tasks in different orders, and since
            # order is worth several episodes in eight (microwave first measures 5/8 against
            # 8/8 last), a failure could not be reproduced or bisected.
            ordered = list(self.sim.kitchen.np_random.permutation(ordered))
        return ordered

    def seed(self, seed: Optional[int]) -> None:
        """Seed the policy's own generator.

        Nothing in the policy draws from it at present: the one random choice, the task
        order, is drawn from the *environment's* generator instead, so that it replays from
        the seed passed to ``env.reset`` even on the resets that pass none.  See
        :meth:`_make_order`.  Kept because the wrapper calls it on every reset.
        """
        self._rng = np.random.default_rng(seed)

    def _apply_control_profile(self, task: Optional[str]) -> None:
        """Select how long one policy step is held for, without changing the action space.

        1.2.0 expressed this as ``robot.control_steps``, the number of IK re-solves inside a
        single env step.  1.2.1 has no such loop -- one env step is a fixed 0.08 s -- so the
        same knob is now the number of env steps the wrapper repeats one action for, and the
        wrapper reads it back through :attr:`action_repeat`.  The contact-sensitive microwave
        grasp still gets the shorter profile.
        """
        microwave_profile = task == "microwave"
        action_repeat = (
            self.cfg.microwave_action_repeat
            if microwave_profile and self.cfg.microwave_action_repeat is not None
            else self._base_action_repeat
        )
        if int(action_repeat) <= 0:
            raise ValueError(f"action_repeat must be positive, got {action_repeat}.")
        self._action_repeat = int(action_repeat)
        self._active_control_profile_task = task if microwave_profile else None

    @property
    def action_repeat(self) -> int:
        """How many env steps the wrapper should hold the current action for."""
        return self._action_repeat

    # -- planner -------------------------------------------------------------------------

    def _enter_phase(self, phase: str) -> None:
        # The waypoint changes with the phase, so the servo's accumulated bias -- earned
        # against the waypoint being left behind -- is stale and would kick the arm off the
        # new approach line.
        #
        # Keeping part of it is tempting and does not work.  The reasoning for keeping it is
        # that the bias has a gravity-droop component which is still true after the phase
        # changes, since the arm has not moved; clipping rather than discarding would keep
        # that and drop only the chase.  Measured, it is not there to keep: the bias at the
        # end of a descent points *down*, because it was earned chasing a target below the
        # arm, and the droop-cancelling bias only accumulates once the arm has stopped.
        # Clipping to 0.02 / 0.03 / 0.05 rad instead of discarding took the kettle grasp
        # from 8/8 to 1/8, 1/8 and 2/8 over 8 seeds and made the sag deeper at every setting
        # (-7.9, -8.5, -9.5 cm against -3.6).  The sag is fixed on the approach side, by not
        # arriving at speed -- see `KitchenPolicyConfig.kettle_precontact_step`.
        self.servo.relax(self.cfg.servo_bias_carryover)
        self._phase = phase
        self._phase_steps = 0
        self._park_rest_history = []
        self._eef_history = []
        self._progress_history = []
        self._reach_error_history = []
        self._recede_rotation_history = []
        self._retreat_lift_target = None
        self._alignment_hold_target = (self.sim.eef_pos if phase == ALIGN else None)



    def _gripper(self, mode: str) -> float:
        return self.cfg.gripper_open if mode == "open" else self.cfg.gripper_close


    def get_cartesian_plan(self, observation=None, info=None) -> np.ndarray:
        """The skills' own output: ``[dx, dy, dz, drx, dry, drz, gripper]`` in ``[-1, 1]``.

        Translation and rotation are normalized by :data:`MAX_CARTESIAN_DISPLACEMENT` and
        :data:`MAX_ROTATION_DISPLACEMENT`; ``gripper`` is +1 open, -1 closed.  This is not
        an action the environment accepts -- see :meth:`get_action` -- but it is what the
        language feedback is worded from, since it names where the gripper should go.
        """
        plan = self._compute_reactive_action()
        plan = np.nan_to_num(np.asarray(plan, dtype=np.float64),
                             nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(plan, -1.0, 1.0)

    def get_action(self, observation=None, info=None) -> np.ndarray:
        """One expert action in the env's action space: ``Box(-1, 1, (9,))``.

        The nine values are normalized joint velocities, seven arm hinges then two finger
        slides.  The skills plan in end-effector space, so this converts:

        1. the Cartesian plan names a target end-effector pose, offset from the current one;
        2. the policy's own DLS IK turns that offset into a joint displacement ``dq``;
        3. holding velocity ``v`` for the ``action_repeat`` env steps the wrapper will apply
           it for moves each joint by ``v * dt`` per step, so ``v = dq / (repeat * dt)``.

        Step 3 inverts ``FrankaRobot._ctrl_velocity_limits`` exactly -- it commands
        ``qpos + v * dt`` every step -- so no simulation is needed to find the action.
        """
        plan = self.get_cartesian_plan(observation, info)
        return self.cartesian_plan_to_action(plan)

    #: Config fields counted in policy steps, so all of them scale together with the step.
    STEP_BUDGET_FIELDS = (
        "retreat_lift_seconds", "engage_seconds", "phase_timeout", "reach_timeout",
        "align_timeout", "manipulate_timeout", "retreat_timeout", "recede_timeout",
        "task_step_budget", "task_stall_seconds", "align_stall_seconds", "reach_freeze_seconds",
    )

    def _scale_step_budgets(self, cfg: KitchenPolicyConfig) -> KitchenPolicyConfig:
        """Convert the budgets, which are written in seconds, into policy steps.

        These were step *counts* tuned when one policy step lasted 1.5 s, and the previous
        conversion rescaled them by the ratio of step durations to preserve those wall-clock
        budgets.  That preserved the wrong thing: 1.2.0 spent its 1.5 s running fifteen IK
        re-solves and covered real ground, so `task_step_budget = 190` meant 285 s of arm
        motion.  Reproducing that at `action_repeat = 1` inflated the budget to 3562 steps,
        far past any episode, so nothing could ever time out -- and a subtask that cannot be
        done then eats the entire episode instead of being abandoned.  Measured: a kettle
        that fails to grasp starved the light switch and the microwave on 4 of 8 seeds.

        Writing them as durations and converting here keeps the meaning explicit and still
        adapts to any `action_repeat`.
        """
        step_seconds = self._base_action_repeat * self.sim.kitchen.robot_env.dt
        scaled = {name: max(1, int(round(getattr(cfg, name) / step_seconds)))
                  for name in self.STEP_BUDGET_FIELDS}
        return replace(cfg, **scaled)

    def _arm_geom_ids(self) -> frozenset:
        """Geoms belonging to the arm itself, cached; used to spot self-collision."""
        if self._arm_geoms is None:
            model, names = self.sim.model, self.sim.model_names
            self._arm_geoms = frozenset(
                g for g in range(model.ngeom)
                if (names.body_id2name[model.geom_bodyid[g]] or "").startswith("panda"))
        return self._arm_geoms

    def _arm_joint_jammed(self) -> bool:
        """Is any arm hinge sitting on a bound in the *measured* pose?

        A joint clipped against its range is the one failure the task solve cannot see its
        way out of: damped least squares returns the minimum-norm step, and from a bound the
        only descent direction it can find is blocked, so it stops (the same mechanism
        :meth:`OrientationAwareIKController._nullspace_escape` exists for).  The escape is
        scaled by the unmet task error, so it goes quiet exactly when the park has arrived
        -- which is where a jam gets *left* rather than fixed.

        Read off ``data.qpos``, not the candidate the solve just produced, because a jam is
        a property of where the arm actually is.
        """
        margin = float(self.cfg.joint_limit_rescue_margin)
        if margin <= 0.0:
            return False
        q = np.asarray(self.sim.data.qpos[:7], dtype=np.float64)
        lower = self.sim.model.jnt_range[:7, 0]
        upper = self.sim.model.jnt_range[:7, 1]
        return bool(np.any(np.minimum(q - lower, upper - q) < margin))

    def _arm_self_colliding(self) -> bool:
        """Is the *currently evaluated* configuration one where the arm touches itself?

        Read straight off `data.ncon`, so it describes whatever pose `mj_forward` was last
        run on -- which inside :meth:`_solve_arm_delta` is the candidate the solve just
        produced, not the pose the robot is in.  That is the point: it rejects a fold before
        it is ever commanded rather than noticing one after the arm is already jammed.

        Any arm-on-arm contact counts.  Normal operation reports none at all -- measured
        over a whole episode on seed 0, zero steps -- so there is no benign case to exclude,
        and pairs adjacent enough to touch by construction are already excluded by the
        model's own contact filtering.
        """
        arm = self._arm_geom_ids()
        data = self.sim.data
        for i in range(data.ncon):
            contact = data.contact[i]
            if int(contact.geom1) in arm and int(contact.geom2) in arm:
                return True
        return False

    def _ik_sweep(self, target_pos, target_quat, start_qpos) -> np.ndarray:
        """One damped-least-squares run to convergence, from the live `data` state."""
        sim = self.sim
        model, data = sim.model, sim.data
        lower, upper = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
        for _ in range(self.cfg.ik_iterations):
            step = np.asarray(
                sim.controller.compute_qpos_delta(target_pos, target_quat),
                dtype=np.float64)[:7]
            # Respect the joint limits the *environment* will enforce on `ctrl`
            # anyway. Left unclipped the solve happily walks a joint past its bound
            # and returns a target the arm can never reach, so the servo spends the
            # episode pushing into a hard stop with a residual it cannot close.
            data.qpos[:7] = np.clip(data.qpos[:7] + step, lower, upper)
            mujoco.mj_forward(model, data)
            # The solve is damped, so a fixed iteration count leaves the target well
            # short and the arm barely moves -- which the 1.2.1 controller punishes
            # doubly, since its position target re-anchors to the measured qpos every
            # step and so sags with the arm instead of holding it up. Run to
            # convergence rather than for a fixed budget.
            if np.abs(step).max() < self.cfg.ik_tolerance:
                break
        # Differenced from the snapshot rather than accumulated: clipping means the
        # steps taken and the displacement achieved are not the same thing.
        return data.qpos[:7] - start_qpos[:7]

    def _solve_arm_delta(self, target_pos, target_quat) -> np.ndarray:
        """Arm joint displacement that puts the end effector at a target pose.

        One damped-least-squares solve is not enough.  ``mj_jacSite`` spans all 29 degrees of
        freedom in the kitchen -- the cabinets and the kettle included -- so the solution
        spreads the correction over joints the arm does not own, and keeping only the arm's
        seven leaves the target well short.  1.2.0 hid this by re-solving once per
        ``control_steps`` iteration while the sim advanced; with one action per wrapper step
        there is no such loop, so the iteration happens here instead.

        It is purely kinematic: ``qpos`` is advanced and ``mj_forward`` re-evaluated, never
        ``mj_step``, and the state is restored before returning.  Nothing about the episode
        moves, so this stays safe to call at any time -- which the wrapper relies on, since it
        queries the expert on every step for feedback.
        """
        sim = self.sim
        model, data = sim.model, sim.data
        qpos_snapshot = data.qpos.copy()
        qvel_snapshot = data.qvel.copy()

        # Minimum-norm least squares is free to fold the arm: seven hinges for a
        # six-dimensional task leaves a whole direction of joint space that does not move
        # the tool, and nothing in the solve prefers one shape over another.  Two ways in,
        # so two tests.
        #
        # Only free-space travel gets a preferred shape to fall back on.  A phase that is
        # working against an object is inside clearances the reset configuration knows
        # nothing about, so there the fold is counted and left alone.
        may_rescue = (self._phase in TRANSIT_PHASES
                      and self.cfg.ik_transit_posture_gain > 0.0
                      and self._home_qpos is not None)
        # The arm is *already* inside itself.  `data` still holds the measured state here,
        # so this is the live pose, and it is the case that actually costs episodes: the
        # servo drives through configurations between one target and the next, so a jam
        # arises with both endpoints clear and no single solve ever predicts it.  Measured
        # on seed 4 with only the candidate test below, 9 rescues fired and not one of them
        # landed on a step the arm was touching itself on.
        live_fold = bool(may_rescue and self._arm_self_colliding())
        # A hinge pinned on its bound is the other way the transit hands the next skill an
        # arm it cannot reach from.  Measured on seed 2: the park before the kettle runs its
        # whole 62-step budget with joint 6 at +2.10 of a +2.1127 bound, and every exact IK
        # solution for the un-flipped kettle grasp there needs j6 at +0.99 or below, so the
        # grasp is unreachable from the pose the park leaves behind -- not because the pose
        # is infeasible, but because damped least squares cannot walk off a bound.
        #
        # This gets the escape term, not `posture_target`, and the difference is not
        # cosmetic.  The reset pose *is* a jammed pose -- joint 2 starts at -1.741 of a
        # -1.7628 bound -- so preferring it answers a jam with the configuration that has
        # one, and at `ik_transit_posture_gain` it outweighs the escape twenty to one and
        # cancels the correction that was unpinning the joint.  Measured that way, the
        # opening park on seed 9 went from 1 step to all 60 of `orient_forward_budget` and
        # timed out.  The escape has no shape preference to impose: it pushes whatever is
        # against a stop and is silent everywhere else.
        may_escape = (self._phase in JAM_RESCUE_PHASES
                      and self.cfg.ik_transit_posture_gain > 0.0
                      and self._home_qpos is not None)
        live_jam = bool(may_escape and self._arm_joint_jammed())

        try:
            sim.controller.escape_floor = (
                float(self.cfg.joint_limit_rescue_escape_floor) if live_jam else 0.0)
            sim.controller.posture_target = (
                self._home_qpos
                if (live_fold or (live_jam and self.cfg.joint_limit_rescue_posture))
                else None)
            reach_posture = (self._skill._reach_posture
                             if self._skill is not None else None)
            if (sim.controller.posture_target is None
                    and reach_posture is not None
                    and self._reach_posture_active
                    and self.cfg.reach_posture_gain > 0.0):
                # Resolve the redundancy toward where the reach is going; see
                # `KitchenPolicyConfig.reach_posture_gain`.
                sim.controller.posture_target = reach_posture
                sim.controller.posture_gain = float(self.cfg.reach_posture_gain)
            if live_fold:
                self._self_collision_rescues += 1
            total = self._ik_sweep(target_pos, target_quat, qpos_snapshot)
            # And the solve's own answer folds.  The candidate is still loaded in `data`,
            # so this tests the pose about to be commanded rather than the one the arm is
            # in -- preventive where the test above is a recovery.
            if may_rescue and not live_fold and self._arm_self_colliding():
                self._self_collision_rescues += 1
                data.qpos[:] = qpos_snapshot
                data.qvel[:] = qvel_snapshot
                mujoco.mj_forward(model, data)
                sim.controller.posture_target = self._home_qpos
                total = self._ik_sweep(target_pos, target_quat, qpos_snapshot)
        finally:
            sim.controller.posture_target = None
            sim.controller.posture_gain = float(self.cfg.ik_transit_posture_gain)
            sim.controller.escape_floor = 0.0
            data.qpos[:] = qpos_snapshot
            data.qvel[:] = qvel_snapshot
            mujoco.mj_forward(model, data)

        return total

    def _ik_pose_residual(self, target_pos, target_quat,
                          iterations: Optional[int] = None) -> Tuple[float, float]:
        """How close the arm can actually get to a pose: (frame error, position error)."""
        _, frame_error, position_error = self._ik_pose_solution(
            target_pos, target_quat, iterations)
        return frame_error, position_error

    def _ik_pose_solution(self, target_pos, target_quat,
                          iterations: Optional[int] = None
                          ) -> Tuple[np.ndarray, float, float]:
        """The arm configuration a pose solves to, with its (frame error, position error).

        Same purely kinematic solve as :meth:`_solve_arm_delta` -- ``qpos`` is advanced and
        ``mj_forward`` re-evaluated, never ``mj_step``, and the state is restored -- but it
        reports the pose the solve *converged to* rather than the displacement.  A pose the
        wrist cannot hold shows up here as a residual the solve cannot close, because the
        iteration clips to ``jnt_range`` every step.
        """
        sim = self.sim
        model, data = sim.model, sim.data
        qpos_snapshot = data.qpos.copy()
        qvel_snapshot = data.qvel.copy()
        lower, upper = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
        # A reachability probe asks what the arm *can* do, so it gets the plain solve: a
        # preferred shape left armed from the last phase would make a pose look unreachable
        # because the bias was pulling away from it.
        sim.controller.posture_target = None
        sim.controller.escape_floor = 0.0
        target_mat = np.empty(9)
        mujoco.mju_quat2Mat(target_mat, np.asarray(target_quat, dtype=np.float64))
        budget = self.cfg.ik_iterations if iterations is None else int(iterations)
        try:
            for _ in range(budget):
                step = np.asarray(
                    sim.controller.compute_qpos_delta(target_pos, target_quat),
                    dtype=np.float64)[:7]
                data.qpos[:7] = np.clip(data.qpos[:7] + step, lower, upper)
                mujoco.mj_forward(model, data)
                if np.abs(step).max() < self.cfg.ik_tolerance:
                    break
            relative = target_mat.reshape(3, 3) @ sim.eef_mat.T
            frame_error = float(np.arccos(
                np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)))
            position_error = float(np.linalg.norm(np.asarray(target_pos) - sim.eef_pos))
            solution = np.asarray(data.qpos[:7], dtype=np.float64).copy()
        finally:
            data.qpos[:] = qpos_snapshot
            data.qvel[:] = qvel_snapshot
            mujoco.mj_forward(model, data)
        return solution, frame_error, position_error

    def _commit_grasp_frame(self, skill: Optional[KitchenSkill]) -> None:
        """Choose, once per skill, which jaw-flip representative of its grasp to target.

        A parallel jaw gripper has two equally valid frames for the same physical grasp
        (:func:`equivalent_grasp_frame`), but they are not always equally reachable: they
        differ by a half turn about the approach axis, and joint 7 has less than a full turn
        of travel.  Picking an unreachable one jams the wrist against its bound for the rest
        of the approach.

        Reachability is decided by running the policy's own IK to the skill's contact pose
        under both frames.  It costs two solves per skill, not per step, and it is asked at
        :attr:`~KitchenPolicyConfig.grasp_frame_probe_iterations` rather than the per-step
        budget, which is far too short to answer this question from half a metre away.

        **Off by default.**  :attr:`~KitchenPolicyConfig.allow_grasp_frame_flip` is False,
        so nothing below runs and every skill targets the un-flipped frame.  The reasoning
        that follows is why the *choice* used to be worth making, and is kept because it is
        also the evidence for why it no longer is: every argument here weighs reachability
        against a half turn of wrist roll, and once the arm stops arriving with a joint on
        its bound (:attr:`~KitchenPolicyConfig.joint_limit_rescue_margin`) the un-flipped
        frame is reachable everywhere measured, so there is nothing left to trade.

        **When both are reachable, the un-flipped frame wins.**  This is not a tie-break for
        its own sake.  The skills all build their frames with :func:`grasp_frame` from an
        approach axis that already points the way the hand travels in, so the un-flipped
        representative is the one the arm is *already* holding when the skill is selected:
        measured at commit time from the reset pose, ``trace(direct @ eef.T)`` is +2.99 out
        of a maximum 3, i.e. the wrist is within a hundredth of a radian of it, while the
        flipped representative sits at -1.00, a dead half turn away.  Choosing the flipped
        one by a residual hair therefore buys nothing and spends a visible 180-degree wrist
        roll on the way to the object -- which is exactly the flip seen in microwave and
        kettle rollouts.  Both of those measure (0.0002, 0.0002) un-flipped once the probe
        is given enough iterations; the earlier (0.143 rad, 8.4 cm) that justified flipping
        was measured against the since-removed 30-degree hook yaw, not this frame.

        An analytic estimate was tried before the IK probe and is not good enough:
        predicting the joint-7 angle from the roll needed about the *current* approach axis
        scores the two frames at +1.31 and +1.35 rad of headroom, i.e. a coin flip, because
        at commit time the arm is still half a metre away and its approach axis is nothing
        like the one it will grasp with.
        """
        self.sim.grasp_flip = None
        if skill is None:
            return
        self._decide_grasp_flip(skill)
        self._solve_reach_posture(skill)

    def _solve_reach_posture(self, skill: KitchenSkill) -> None:
        """Solve the stand-off pose once from here; see `reach_posture_gain`."""
        skill._reach_posture = None
        if self.cfg.reach_posture_gain <= 0.0 or not skill.use_reach_posture:
            return
        desired = skill.desired_orientation()
        if desired is None:
            return
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(
            target_quat, equivalent_grasp_frame(self.sim, desired).reshape(-1))
        solution, frame_error, position_error = self._ik_pose_solution(
            skill.eef_target(skill.precontact_point()), target_quat,
            self.cfg.grasp_frame_probe_iterations)
        if self._pose_reachable((frame_error, position_error)):
            skill._reach_posture = solution

    def _decide_grasp_flip(self, skill: KitchenSkill) -> None:
        if not self.cfg.allow_grasp_frame_flip:
            self.sim.grasp_flip = False
            return
        desired = skill.desired_orientation()
        if desired is None:
            return
        desired = np.asarray(desired, dtype=np.float64)
        target_pos = skill.eef_target(skill.contact_point())
        direct = np.empty(4)
        flipped = np.empty(4)
        mujoco.mju_mat2Quat(direct, desired.reshape(-1))
        mujoco.mju_mat2Quat(flipped, (desired @ _JAW_FLIP).reshape(-1))
        iterations = self.cfg.grasp_frame_probe_iterations
        direct_residual = self._ik_pose_residual(target_pos, direct, iterations)
        flipped_residual = self._ik_pose_residual(target_pos, flipped, iterations)
        direct_reachable = self._pose_reachable(direct_residual)
        flipped_reachable = self._pose_reachable(flipped_residual)
        if direct_reachable and flipped_reachable:
            # Both are the same physical grasp, so take whichever the wrist is already
            # nearer and spend no half turn at all.  The paragraph above holds this
            # constant at "un-flipped", and that is right *from the reset pose* -- where it
            # was measured, and where the route this used to run always committed, because
            # the arm parked there between every pair of subtasks.  Commit from anywhere
            # else and the assumption is simply false: measured over 8 seeds once the
            # curved transit removed the park, 7 of 23 transitions opened with a frame error
            # of 2.7 to 3.14 rad -- a dead half turn, i.e. the flipped representative -- and
            # they cost a mean 124 reach steps against 38 for the rest, with the one outright
            # stall among them.  Under the park not one transition exceeded 1.0 rad.
            self.sim.grasp_flip = bool(
                orientation_error(self.sim, desired @ _JAW_FLIP)
                < orientation_error(self.sim, desired))
            return
        if direct_reachable:
            self.sim.grasp_flip = False
            return
        self.sim.grasp_flip = bool(flipped_reachable
                                   or flipped_residual < direct_residual)

    def _resolve_orient_forward_tolerances(self, previous: Optional[str],
                                           pending: Optional[KitchenSkill]) -> None:
        """Latch the park's exit tolerances for the transition about to be made."""
        entry = None
        table = self.cfg.orient_forward_transition_tolerances
        if table and previous is not None and pending is not None:
            entry = table.get((previous, pending.task))
        yaw, clearance = entry if entry is not None else (
            self.cfg.yaw_tolerance, self.cfg.orient_forward_position_tolerance)
        self._orient_forward_yaw_tolerance = float(yaw)
        self._orient_forward_clearance_tolerance = float(clearance)
        self._park_between_subtasks = previous is not None and pending is not None
        self._park_joint_from_start = bool(
            self._park_between_subtasks
            and (previous, pending.task) in self.cfg.park_joint_transitions)
        self._park_joint_exit_required = bool(
            self._park_joint_from_start
            and (previous, pending.task) in self.cfg.park_joint_exit_transitions)

    def _next_pending_skill(self) -> Optional[KitchenSkill]:
        """The skill that will be selected next, built for inspection only."""
        for task in self._order:
            if task in self._abandoned or task == self._task:
                continue
            if not self._reactive_task_satisfied(task):
                return build_skill(self.sim, task, self.cfg)
        return None

    def _direct_transition_safe(self, skill: Optional[KitchenSkill]) -> bool:
        """Whether the next skill can set out from here instead of parking at home first.

        Parking between subtasks looks unnecessary and is not: removing it measures 8/8 to
        2/8 on the fixed order.  What it actually buys is not a reset wrist -- the IK says
        the next grasp is reachable from the old pose in almost every case, and gating on
        that alone still measures 2/8 -- but a *safe transit waypoint*.  Consecutive
        subtasks can be at opposite ends of the kitchen (the light switch sits at z = 2.28
        inside the cooker hood, the kettle at z = 1.80 half a metre forward), and the
        straight line between them runs through the counter and the hood.  The home pose is
        a known clear point that every stand-off can be reached from.

        So the test is distance, not reachability: a short hop goes direct, a long one is
        routed via home.  Both conditions are required -- a near stand-off the wrist cannot
        attain is not a safe direct transition either.
        """
        if skill is None:
            return True
        if not np.isfinite(self.cfg.direct_transition_max_distance):
            return True
        hop = float(np.linalg.norm(skill.eef_target(skill.precontact_point())
                                   - self.sim.eef_pos))
        return bool(hop <= self.cfg.direct_transition_max_distance
                    and self._grasp_pose_reachable(skill))

    def _grasp_pose_reachable(self, skill: KitchenSkill) -> bool:
        """Whether the arm could still attain ``skill``'s contact pose from where it is.

        Asked only when a grasp has been lost, to tell "it slipped, go and get it again"
        apart from "it slipped somewhere the arm cannot follow". Those need opposite
        answers and look identical from the FSM's point of view: both are a skill with an
        open gripper and a waypoint it has not reached.

        Without the distinction the kettle transport is the worst case. Its grasp frame
        asks the wrist to point along the push, and the arm runs out of wrist doing that
        partway down the counter (see :meth:`KettleGraspSkill.approach_axis`), so the bar is
        lost at a pose whose IK residual is 0.12 rad and 4 cm -- unattainable -- and the
        skill then re-approaches it forever. Measured over 8 combined episodes, the kettle
        spent *exactly* its 626-step task budget on every one of them, cycling
        MOVE_TO_PRECONTACT/ALIGN without the body moving a millimetre, and the microwave
        after it was starved of budget on three seeds. It costs one probe solve, and only on
        the step a grasp is found to be gone.
        """
        desired = skill.desired_orientation()
        if desired is None:
            return True
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(
            target_quat,
            equivalent_grasp_frame(self.sim, desired).reshape(-1))
        return self._pose_reachable(self._ik_pose_residual(
            skill.eef_target(skill.contact_point()), target_quat,
            self.cfg.grasp_frame_probe_iterations))

    def _pose_reachable(self, residual: Tuple[float, float]) -> bool:
        """Whether an :meth:`_ik_pose_residual` result counts as the arm attaining the pose."""
        frame_error, position_error = residual
        return bool(frame_error <= self.cfg.grasp_frame_reachable_frame_tolerance
                    and position_error <= self.cfg.grasp_frame_reachable_position_tolerance)

    def _turn_first(self, distance: float, frame_error: float, threshold: float) -> bool:
        """Whether to hold position and only rotate this step; see `turn_first_distance`.

        Engages when the target is far and the frame is badly off, releases once the frame
        error is under half the threshold, so the reach cannot alternate between the two
        modes every step as the travel disturbs the wrist again.
        """
        limit = float(self.cfg.turn_first_distance)
        if limit <= 0.0 or threshold <= 0.0:
            self._turn_first_active = False
            return False
        if self._turn_first_active:
            self._turn_first_active = (frame_error > 0.5 * threshold
                                       and distance > limit)
        else:
            self._turn_first_active = (frame_error > threshold and distance > limit)
        return self._turn_first_active

    def _opening_park_at_rest(self, clearance_error: float) -> bool:
        """Whether the opening park has stopped moving inside a relaxed clearance band.

        See :attr:`KitchenPolicyConfig.opening_park_rest_steps`.  Samples are keyed on the
        phase step so repeated policy queries within one env step do not count as time.
        """
        rest = int(self.cfg.opening_park_rest_steps)
        if rest <= 0 or self._park_between_subtasks:
            return False
        # Three times the tolerance, not two: seed 291 rests at 0.031-0.032 m against
        # 0.015 from step 18 to the budget, just outside twice.  The band only says where
        # the deadband may be; the motion test below is what keeps a moving park running.
        if (self._last_orientation_error > self._orient_forward_yaw_tolerance
                or clearance_error > 3.0 * self._orient_forward_clearance_tolerance):
            return False
        step = int(self._phase_steps)
        history = self._park_rest_history
        if not history or history[-1][0] != step:
            history.append((step, np.asarray(self.sim.eef_pos, dtype=np.float64).copy()))
            del history[:-(rest + 1)]
        if len(history) <= rest:
            return False
        moved = float(np.linalg.norm(history[-1][1] - history[0][1]))
        return moved < float(self.cfg.opening_park_rest_distance)

    def _park_joints_home(self) -> bool:
        """Whether a from-start joint-space park has actually arrived; see
        `park_joint_exit_tolerance`.  True for every other park."""
        tolerance = float(self.cfg.park_joint_exit_tolerance)
        if not self._park_joint_exit_required or tolerance <= 0.0:
            return True
        target = self._park_joint_target(self._gripper("open"))[:7]
        error = np.abs(np.asarray(self.sim.data.qpos[:7], dtype=np.float64) - target)
        return bool(float(np.max(error)) <= tolerance)

    def _park_joint_target(self, gripper: float) -> np.ndarray:
        """Nine-value servo target for the joint-space park: home arm, commanded fingers."""
        target = np.empty(ACTION_DIM)
        home = np.asarray(self._home_qpos, dtype=np.float64).copy()
        lower = self.sim.model.jnt_range[:7, 0]
        upper = self.sim.model.jnt_range[:7, 1]
        margin = float(self.cfg.park_joint_bound_margin)
        target[:7] = np.clip(home, lower + margin, upper - margin)
        target[7:] = FINGER_RANGE * (float(gripper) + 1.0) / 2.0
        return target

    def cartesian_plan_to_action(self, plan) -> np.ndarray:
        """Convert a 7-D Cartesian plan into the env's 9-D joint-velocity action.

        The plan names a pose one step ahead of the current one -- the skills already cap how
        far that is via their step scales -- so this resolves it to a joint-space target and
        hands it to :class:`JointServo`, which drives the arm *to* that target rather than
        issuing a fraction of the way there.

        The previous conversion divided the joint displacement by ``action_repeat * dt`` and
        multiplied by a ``velocity_gain`` meant to compensate a servo lag.  There is no such
        lag: the shortfall it was tuned against is the actuators' hard speed ceiling
        (:data:`JOINT_SPEED_LIMIT`), which no gain can lift.  What the old path really lacked
        was integral action, which is now where the accuracy comes from.
        """
        sim = self.sim
        plan = np.asarray(plan, dtype=np.float64).reshape(-1)

        target_pos = sim.eef_pos + plan[:3] * MAX_CARTESIAN_DISPLACEMENT
        eef_quat = np.empty(4)
        mujoco.mju_mat2Quat(eef_quat, sim.data.site_xmat[sim.controller.eef_id])
        spin = euler2quat(plan[3:6] * MAX_ROTATION_DISPLACEMENT)
        target_quat = np.empty(4)
        mujoco.mju_mulQuat(target_quat, spin, eef_quat)

        if self._joint_target_override is not None:
            # A joint-space park: the plan was computed for the feedback text, but the
            # servo is handed the home configuration directly.  See `_park_joint_target`.
            return self.servo.action(self._joint_target_override)
        q_target = np.empty(ACTION_DIM)
        q_target[:7] = sim.data.qpos[:7] + self._solve_arm_delta(target_pos, target_quat)
        # The gripper axis is an absolute command: map [-1, 1] onto the finger travel.
        q_target[7:] = FINGER_RANGE * (float(plan[6]) + 1.0) / 2.0

        return self.servo.action(q_target)

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

    def _task_stalled(self) -> bool:
        """Has the selected subtask stopped getting anywhere at all?

        True when, across `task_stall_seconds`, the end effector's *net* displacement is
        under `task_stall_distance` **and** the task distance has closed by less than
        `task_stall_progress`.  See that field for why both halves are needed and what this
        exists to catch.

        Samples are keyed on `_steps_in_task` so that repeated policy queries against an
        unadvanced simulator -- the wrapper asks for the plan and the action separately --
        cannot age the window.  Phases that are deliberately not making task progress are
        excluded: the park and the retreat have no task to advance, and RECEDE and VERIFY
        run after the manipulation is already done.
        """
        window = int(self.cfg.task_stall_seconds)
        if window <= 0 or self._task is None or self._skill is None:
            return False
        if self._phase in (ORIENT_FORWARD, SELECT_SUBTASK, RECEDE, VERIFY, RETREAT, IDLE):
            return False
        step = int(self._steps_in_task)
        if self._task_stall_history and self._task_stall_history[-1][0] == step:
            return False
        position = self.sim.eef_pos
        distance = float(self.sim.task_distance(self._task))
        self._task_stall_history.append((step, position.copy(), distance))
        # ALIGN is judged on its frame error instead; see `align_stall_seconds`.
        align_window = int(self.cfg.align_stall_seconds)
        if self._phase == ALIGN and align_window > 0:
            self._align_stall_history.append((step, float(self._last_orientation_error)))
            del self._align_stall_history[:-(align_window + 1)]
            if (len(self._align_stall_history) > align_window
                    and (self._align_stall_history[0][1] - self._last_orientation_error)
                    < self.cfg.align_stall_progress):
                return True
        else:
            self._align_stall_history = []
        freeze = int(self.cfg.reach_freeze_seconds)
        if (freeze > 0 and self._phase in (MOVE_TO_PRECONTACT, APPROACH)
                and len(self._task_stall_history) > freeze
                and self._last_position_error > self.cfg.reach_freeze_position_error):
            _, frozen_position, _ = self._task_stall_history[-(freeze + 1)]
            if (float(np.linalg.norm(position - frozen_position))
                    < self.cfg.reach_freeze_distance):
                return True
        if len(self._task_stall_history) <= window:
            return False
        del self._task_stall_history[:-(window + 1)]
        _, start_position, start_distance = self._task_stall_history[0]
        moved = float(np.linalg.norm(position - start_position))
        closed = start_distance - distance
        return bool(moved < self.cfg.task_stall_distance
                    and closed < self.cfg.task_stall_progress)

    def _reactive_task(self) -> Optional[KitchenSkill]:
        """Select the next unfinished, unabandoned task from live state predicates."""
        # Give up on a subtask that is not converging, so the rest of the episode is not
        # spent on it. This was enforced only in a second, unreachable FSM (since deleted),
        # so
        # in practice there was no per-task bound at all: measured, a kettle that fails to
        # grasp holds the selection for the whole episode and the light switch and
        # microwave after it are never even attempted.
        # A stall is retried before it is abandoned: the arm is usually stuck *against*
        # something, and the stand-off is 15 cm back, so simply re-running the reach from
        # there is enough to free it.  Rebuilding the skill also re-picks the grasp frame
        # from the pose the arm is actually in now.
        stalled = (self._task is not None and self._skill is not None
                   and not self._reactive_recede_pending
                   and self._task_stalled())
        if stalled and self._retry_count < int(self.cfg.task_stall_retries):
            # Escalate, because re-reaching cannot fix a stall whose cause is the arm's
            # *configuration* rather than its position.  Seed 106's microwave approach is
            # that case: it freezes with joint 6 pinned to the millirad on its +2.1127
            # bound, and a 60-restart global IK says the contact pose is reachable to
            # 0.0001 rad -- but only with joint 1 at +1.676 where the arm is sitting at
            # -1.171.  The arm is on the wrong side of the microwave, and no amount of
            # damped least squares crosses that: every approach from the same shape jams at
            # the same place.
            #
            # So the second retry re-parks.  Position alone is not enough -- retreating to
            # `_home_pos` leaves seed 106 at 3/4, because the arm arrives home in the same
            # branch it left in -- but clearing `_initial_orientation_complete` runs the
            # *whole* park, whose orientation gate turns the wrist through configurations
            # the reach never visits and which has the joint-limit rescue the reaching
            # phases do not (see `TRANSIT_PHASES`).  With both, seed 106 completes in 535
            # steps.  Enabling the jam rescue in the reaching phases directly was tried
            # instead and is worse on both counts: it does not free seed 106, and it costs
            # steps everywhere else (seed 158, 367 -> 435), because `_arm_joint_jammed` is
            # true for a third of all MOVE_TO_PRECONTACT steps -- joint 2 starts the episode
            # 0.022 rad from its bound and never really leaves.
            # A stall in a *reaching* phase is a configuration problem by construction --
            # the arm is not where it wants to be and cannot get there from the shape it is
            # in -- so re-running the reach from the stand-off cannot fix it, and the only
            # thing that can is the park.  Skip straight to it rather than spending a whole
            # stall window proving the point again: on seed 106 that third of three
            # identical frozen approaches costs 35 steps and ends exactly where the first
            # two did.  A manipulation stall still gets the cheap retry first, because there
            # the arm *is* in contact and re-seating is often all it needs.
            if (self._retry_count >= 1
                    or self._phase in (MOVE_TO_PRECONTACT, APPROACH, ALIGN)):
                self._reactive_retreat_pending = True
                self._initial_orientation_complete = False
            self._skill = build_skill(self.sim, self._task, self.cfg)
            self._align_exhausted = False
            self._commit_grasp_frame(self._skill)
            self._retry_count += 1
            self._task_stall_history = []
            self._align_stall_history = []
            self._release_hold_pos = None
            self._enter_phase(MOVE_TO_PRECONTACT)
            return self._skill

        if (self._task is not None
                and (stalled or self._steps_in_task > self.cfg.task_step_budget)
                and not self._reactive_recede_pending):
            if self._task not in self._abandoned:
                self._abandoned.append(self._task)
            self._abandon_counts[self._task] = self._abandon_counts.get(self._task, 0) + 1
            self._task_step_counts[self._task] = self._steps_in_task
            self._phase_failures += 1
            self._task = None
            self._skill = None
            self._task_stall_history = []
            self.sim.grasp_flip = None
            self._apply_control_profile(None)
            self._steps_in_task = 0
            self._reactive_retreat_pending = True
            return None

        active_satisfied = False
        if self._task is not None and self._skill is not None:
            active_satisfied = self._skill.manipulation_done()
            # Keep transporting a captured kettle to the deeper demonstration margin.
            # If it slips after KitchenEnv has already banked the task *and* the body is
            # essentially where it belongs, reacquisition cannot earn anything and can
            # spend the rest of a randomized episode destabilizing a delivered kettle.
            # Release and recede.
            #
            # The delivery test is the point of this clause and used to be missing.
            # `episode_task_completions` is not evidence that the kettle arrived: its 7-D
            # pose distance starts at 0.39 against a 0.3 threshold, so the first shove banks
            # the task while the body is still a third of a metre from the burner. Without
            # the test, the first slipped grasp -- which the crushing grip made near
            # certain -- abandoned the transport there, and the withdrawing hand then
            # dragged the kettle back most of what it had gained: measured on seed 7, root
            # y 0.537 at release and 0.403 nine steps later, worse than where it started.
            if (not active_satisfied
                    and isinstance(self._skill, KettleGraspSkill)
                    and self._task in self.sim.kitchen.episode_task_completions
                    and not self._skill.grasp_retained()
                    and (self._skill.goal_position_error()
                         <= self.cfg.kettle_delivered_position_tolerance
                         or not self._grasp_pose_reachable(self._skill))):
                active_satisfied = True
            # The same question for a kettle that is *still held*.  The test above cannot
            # ask it -- it requires the grasp to be gone -- so a body that jams with the
            # jaws closed is pushed for the rest of the episode.
            if (not active_satisfied
                    and isinstance(self._skill, KettleGraspSkill)
                    and self._task in self.sim.kitchen.episode_task_completions
                    and self._skill.transport_stalled()):
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
            self.sim.grasp_flip = None
            self._apply_control_profile(None)
            self._steps_in_task = 0
            self._reactive_retreat_pending = True
            return None

        pending = []
        for task in self._order:
            if task in self._abandoned:
                continue
            satisfied = (self._skill.manipulation_done()
                         if task == self._task and self._skill is not None
                         else self._reactive_task_satisfied(task))
            if not satisfied:
                pending.append(task)
        if not pending:
            # Nothing selectable -- but "nothing selectable" and "nothing left to do" are
            # different, and the difference is worth an entire subtask.  An abandoned task
            # is never reconsidered, so an episode that gives one up simply stops: measured
            # over seeds 0-499, every single failure at every light-switch lead angle ends
            # the same way, with the arm holding still for 318 to 327 steps -- a third of
            # the episode -- after the give-up.  The budget to try again was always there.
            #
            # Re-attempt from a clean slate rather than resuming: the arm has since retreated
            # home and re-parked, which is exactly the configuration change a stalled reach
            # needed and could not get in place (see the stall retry above).  That is why
            # this is worth anything at all -- it is a third attempt from a *different* pose,
            # not a repeat of the second.
            revivable = [task for task in self._order
                         if task in self._abandoned
                         and self._abandon_counts.get(task, 0) <= self.cfg.task_reattempt_limit
                         and not self._reactive_task_satisfied(task)]
            if revivable:
                task = revivable[0]
                self._abandoned.remove(task)
                self._steps_in_task = 0
                self._task_stall_history = []
                return self._reactive_task()
            self._task = None
            self._skill = None
            self.sim.grasp_flip = None
            self._apply_control_profile(None)
            return None

        selected = pending[0]
        # Always honor the configured precedence.  If an earlier task is disturbed after
        # it was completed, it becomes current again instead of being hidden by history.
        if self._task != selected or self._skill is None:
            self._task = selected
            self._skill = build_skill(self.sim, selected, self.cfg)
            self._align_exhausted = False
            self._commit_grasp_frame(self._skill)
            self._apply_control_profile(selected)
            self._steps_in_task = 0
            self._retry_count = 0
            self._task_stall_history = []
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
            self._align_exhausted = False
            self._commit_grasp_frame(self._skill)
            self._retry_count += 1
            self._task_stall_history = []
            self._reactive_recede_pending = False
            self._release_hold_pos = None
            self._enter_phase(MOVE_TO_PRECONTACT)
            return

        if self._task is not None:
            if self._task not in self._completed_order:
                self._completed_order.append(self._task)
            self._task_step_counts[self._task] = self._steps_in_task
        # Route the hand via the home pose before the next subtask. What this buys is not a
        # reset wrist -- the IK says the next grasp is reachable from the old pose in almost
        # every case -- but a *safe transit waypoint*: consecutive subtasks sit at opposite
        # ends of the kitchen (the light switch is at z = 2.28 inside the cooker hood, the
        # kettle at z = 1.80 half a metre forward), and the straight line between them runs
        # through the counter. Skipping it measures 8/8 -> 2/8 on the fixed order.
        #
        # `_direct_transition_safe` decides, and is asked here rather than at task selection
        # because ORIENT_FORWARD expects to run with no task selected and no phase set.
        pending = self._next_pending_skill()
        self._resolve_orient_forward_tolerances(self._task, pending)
        self._initial_orientation_complete = not (
            self.cfg.align_forward_at_reset and self.cfg.reorient_between_tasks
            and not self._direct_transition_safe(pending))
        self._task = None
        self._skill = None
        self.sim.grasp_flip = None
        self._apply_control_profile(None)
        self._steps_in_task = 0
        self._retry_count = 0
        self._task_stall_history = []
        self._reactive_recede_pending = False
        self._release_hold_pos = None
        # The recede waypoint is already the slide skill's safe exit.  Selecting the next
        # skill from there avoids driving the still-horizontal fingertips forward through
        # the cabinet merely to revisit the reset flange pose.
        self._reactive_retreat_pending = False
        self._enter_phase(SELECT_SUBTASK)

    def _alignment_hold_point(self) -> np.ndarray:
        """The pose ALIGN froze on entry, which is what `align_in_place` must track.

        These branches used to pass `self.sim.eef_pos`, the *live* measurement.  A target
        read from the live measurement carries no error, so the servo commands nothing and
        the position actuators droop under gravity for as long as the rotation lasts -- the
        same defect the kettle release hit, and dealt with there by `_release_hold_pos`.

        In place is where it does the most damage, because ALIGN is the one phase whose
        whole job is to turn the wrist without going anywhere.  Measured on seed 4, coming
        off the slide cabinet: six steps of ALIGN sank the hand 13 cm (z 1.967 to 1.837)
        while the frame error moved 0.489 to 0.446.  Sinking is what ended the phase -- the
        drop carried the tool out of `alignment_tolerance`, MOVE_TO_PRECONTACT drove back
        up, the wrist came out of `realign_tolerance` again -- and the pair alternated for
        77 steps, which is what starved the last subtask of its budget.

        `_enter_phase` already snapshots this on every ALIGN entry, and `site_xpos` copies,
        so the frozen pose needs nothing further to stay frozen.  Re-entering ALIGN
        re-freezes it, which is what should happen: the pose to hold is wherever the arm
        has just been asked to align from.
        """
        if self._alignment_hold_target is None:
            self._alignment_hold_target = self.sim.eef_pos
        return self._alignment_hold_target

    def _recede_rotation_stalled(self, error: float) -> bool:
        """True once an in-place RECEDE rotation has stopped closing its frame error.

        A rotation that commands no translation has no other way to fail: it holds the
        measured position, turns the wrist, and either converges or stands there turning
        until `recede_timeout`.  That backstop is 75 steps, which is a fifth of an episode
        spent at the object the arm has already finished with.

        Query-count dependence is the cost, and it is the same cost `recede_timeout` in the
        same branch already pays -- both read `_phase_steps`.  It is bounded the same way:
        this can only *end* a rotation early, never start one or change which target is
        commanded, so an expert queried without its actions being applied still returns the
        rotation for as long as the rotation is making progress.
        """
        window = int(self.cfg.recede_rotation_progress_window)
        if window <= 0:
            return False
        history = self._recede_rotation_history
        # Keyed by phase step, so querying the expert twice in one step samples once and a
        # rotation cannot be judged stalled by being asked about more often.
        if history and history[-1][0] == self._phase_steps:
            history[-1] = (self._phase_steps, float(error))
        else:
            history.append((self._phase_steps, float(error)))
        oldest = self._phase_steps - window
        while len(history) > 1 and history[0][0] < oldest:
            history.pop(0)
        if history[0][0] > oldest:
            return False
        return bool(history[0][1] - history[-1][1]
                    <= float(self.cfg.recede_rotation_progress_epsilon))

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
        self._joint_target_override = None
        self._reach_posture_active = False
        if not self._initial_orientation_complete:
            desired = grasp_frame(FRONT_APPROACH_AXIS, VERTICAL_HANDLE_AXIS)
            self._last_orientation_error = orientation_error(self.sim, desired)
            clearance_error = float(np.linalg.norm(
                self._initial_orientation_target - self.sim.eef_pos))
            # Bounded, because this phase has nothing to fall back on. It is a convenience
            # -- hand the next skill a neutral wrist rather than whatever the last one left
            # -- and it holds the whole episode hostage while it runs: no task is selected,
            # so nothing else can make progress and no task budget is counting. Between
            # subtasks the arm does not always start somewhere this can be reached from, and
            # a stall here is silent and total. Measured on combined seed 1: from step 235
            # to the 1000-step limit, orientation error pinned at 0.163 against a 0.10
            # tolerance and position at 0.158, with the light switch and microwave never
            # attempted. Giving up and selecting the next task is strictly better than not
            # selecting one at all.
            if (self._phase == ORIENT_FORWARD
                    and self._phase_steps > self.cfg.orient_forward_budget):
                self._initial_orientation_complete = True
                self._enter_phase(SELECT_SUBTASK)
            elif (((self._last_orientation_error > self._orient_forward_yaw_tolerance
                        or clearance_error
                        > self._orient_forward_clearance_tolerance)
                     and not self._opening_park_at_rest(clearance_error))
                    or not self._park_joints_home()):
                self._set_reactive_phase(ORIENT_FORWARD)
                plan = self._track_initial_orientation(desired)
                # Joint-space homing: either from the first step for the transitions that
                # need the reset branch, or as the fallback once the Cartesian tracker has
                # had its chance.  The plan above is still what the feedback is worded
                # from; only the joint target the servo receives is replaced.
                fallback = int(self.cfg.park_joint_fallback_steps)
                if (self._park_joint_from_start
                        or (self._park_between_subtasks and fallback > 0
                            and self._phase_steps >= fallback)):
                    self._joint_target_override = self._park_joint_target(plan[6])
                return plan
            else:
                self._initial_orientation_complete = True
                # The representative committed to before the turn was chosen from the old
                # wrist; re-ask now that the arm is where it will actually reach from.
                if self._skill is not None:
                    self._commit_grasp_frame(self._skill)
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
            # Nothing left to do, so go and stand somewhere harmless first.  IDLE holds the
            # measured pose, and the pose the last subtask leaves behind is usually resting
            # against the thing it just finished: `_finish_reactive_recede` clears the
            # retreat flag on purpose, because a *next* skill is better started from the
            # recede waypoint than from the reset flange, but when there is no next skill
            # that hands the episode to a hold with the hand still touching.  Measured on
            # the kettle alone, the fingertip settles on the spout at step 98 of a 400-step
            # episode and leans on it for the remaining 300, walking the delivered body
            # from a pose distance of 0.209 to 0.272 -- on 10 of 12 seeds.
            home_error = float(np.linalg.norm(self.sim.eef_pos - self._home_pos))
            if (self._reactive_retreat_pending
                    or home_error > self.cfg.reactive_retreat_tolerance):
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
                    # Freeze the pose on the first step of the release rather than
                    # re-reading it.  Tracking the *live* measurement is a target with no
                    # error in it, so the servo commands nothing and the position
                    # actuators simply droop: measured on the kettle, the hand sank 1.3 cm
                    # over the six steps the pads take to open.  On its own that changed
                    # nothing measurable -- the drop that actually put a finger on the
                    # kettle's shoulder happens during the push, not the release, and is
                    # dealt with by `kettle_transport_hold_height`.  Kept because a target
                    # read from the live measurement carries no error and so commands
                    # nothing, which is not what "hold this pose" is supposed to mean.
                    if self._release_hold_pos is None:
                        self._release_hold_pos = np.asarray(self.sim.eef_pos,
                                                            dtype=np.float64).copy()
                    return self._track_eef(skill, self._release_hold_pos,
                                           self.cfg.contact_step, "open", orient=False)
                return self._track_action(skill, skill.contact_point(),
                                          self.cfg.contact_step, "open")
            target = skill.recede_point()
            if isinstance(skill, MicrowavePullSkill):
                # First translate straight backward with a frozen wrist. Combining that
                # clearance with a large rotation makes the DLS controller move toward
                # the open door instead of away from it. Once clear, unwind the pull yaw --
                # and nothing else. Posing the wrist for whatever runs next is
                # ORIENT_FORWARD's job, and it runs immediately after this phase.
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
                release_frame = skill.release_frame()
                if release_frame is not None and self.cfg.microwave_pull_yaw != 0.0:
                    # Undo the pull yaw now, in free space, with the door already cleared.
                    # Leaving it wound is what strands the next skill; see
                    # `MicrowavePullSkill.release_frame`.
                    unwind_error = orientation_error(self.sim, release_frame)
                    self._last_orientation_error = unwind_error
                    if (unwind_error > skill.orientation_tolerance_value()
                            and not self._recede_rotation_stalled(unwind_error)
                            and self._phase_steps < self.cfg.recede_timeout):
                        action = self._track_eef(skill, self.sim.eef_pos,
                                                 skill.recede_step_scale(), "open",
                                                 orient=False)
                        action[3:6] = rotation_action(self.sim, release_frame,
                                                      skill.rotation_step_scale())
                        return action
                # Nothing else turns the wrist here.  RECEDE used to pre-rotate to the
                # *light switch's* approach frame whenever that task came next, on the
                # theory that doing it in free space spared the light skill its one-pad
                # oscillation.  It cannot work from inside this phase: that frame and the
                # unwind target above are 0.65-0.75 rad apart, each gated at 0.35, so
                # satisfying both at once needs the pair to be under 0.70 and they usually
                # are not.  The two tests run in sequence every step, so each one's
                # rotation pushes the other back out of tolerance and the wrist ping-pongs
                # in place.  Measured over the four seeds that open the microwave first:
                #
                #   ====== ===================== =============== ==============
                #   seed   task after microwave  recede steps    frame gap
                #   ====== ===================== =============== ==============
                #   4      slide cabinet         5               block skipped
                #   5      kettle                5               block skipped
                #   2      light switch          23              0.649 rad
                #   3      light switch          75 (timed out)  0.714 rad
                #   ====== ===================== =============== ==============
                #
                # Seed 2 escaped only by threading both errors under the bar in the same
                # step (0.335 and 0.336); seed 3's gap exceeds 0.70, so no wrist pose could
                # have satisfied both and it ran the whole budget out.  The recede's job is
                # to get clear of the microwave.  ORIENT_FORWARD, which runs next, is what
                # hands the following skill a workable wrist.
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
        # Hysteresis, and it has to be this way round.  Reaching disturbs the wrist and
        # aligning restores it, so on a skill that does both at once the frame error rings
        # between the two -- measured on the light switch, a MOVE step ends at 0.21 and the
        # ALIGN step after it at 0.12, either side of a single 0.20 threshold, giving an
        # 80-step MOVE_TO_PRECONTACT/ALIGN limit cycle that creeps toward the stand-off a
        # centimetre per cycle.  The *entry* threshold is therefore the loose one: break off
        # a reach only for a genuinely bad frame, and leave the small ringing to the
        # tracker, which corrects orientation and position in the same IK solve.
        alignment_threshold = (skill.alignment_exit_tolerance_value()
                               if self._phase == ALIGN else skill.realign_tolerance_value())
        # A frame the wrist cannot reach must not be able to hold the episode. Latched
        # rather than tested per step so the FSM does not simply re-enter ALIGN on the next
        # one and cycle with the budget as its period; it is cleared whenever a skill is
        # selected. See `align_budget`.
        if self._phase == ALIGN and self._phase_steps > self.cfg.align_budget:
            self._align_exhausted = True
        realign_threshold = skill.realign_tolerance_value()
        if self._align_exhausted:
            alignment_threshold = float('inf')
            realign_threshold = float('inf')
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
                and jaw_center_error > skill.centering_entry_tolerance()):
            # Project onto the precontact->contact line at the tool's current depth. This
            # requests lateral/vertical centering only, so the outside of a finger cannot
            # push the free kettle forward while the bar is still outside the open jaws.
            # Centering translates without commanding rotation.  The DLS solver couples
            # that translation into rotation, so the frame error ramps until ALIGN fires
            # for a step and knocks it back -- a sawtooth that visibly stalls the approach
            # (seed 42: 0.14 -> 0.18 about every eleven steps).  Correcting the wrist here
            # instead is worse, not better: rotating swings the tool on its FINGERTIP_OFFSET lever
            # and fights the centering it is interleaved with, measured as seed 42 falling
            # to 2/4 with the centering stage growing from 76 steps to 104.
            skill._centering_active = True
            approach_direction = unit(contact - precontact)
            centered_depth = float(np.clip(approach_along, 0.0, approach_length))
            centered_point = precontact + approach_direction * centered_depth
            self._set_reactive_phase(MOVE_TO_PRECONTACT)
            centering_rotation = self.cfg.kettle_centering_rotation_step
            return self._track_action(skill, centered_point,
                                      skill.approach_step_scale(),
                                      skill.approach_gripper,
                                      orient=centering_rotation is not None,
                                      rotation_step=centering_rotation)
        if jaw_center_error <= skill.grasp_center_tolerance:
            skill._centering_active = False

        # Once the kettle is captured, keep the loaded grasp, push horizontally, and hold
        # the wrist at the frame the grasp was made in (see
        # `KettleGraspSkill.desired_orientation`).  Two other rotations were tried here and
        # both are wrong.  Chasing the kettle's *live* handle frame closes a feedback loop
        # through a target that moves with the body, and its full-pose correction tips the
        # kettle off-centre and upward.  Turning the wrist about world z toward the kettle's
        # goal yaw -- a fixed target, bounded per step -- looks safer and is worse: the pads are
        # FINGERTIP_OFFSET ahead of the site the wrist turns about, so the correction drags
        # the captured bar sideways, which yaws the free body *further* the way it was
        # already going.  Traced on seed 0: the command started at a 0.12 rad error, the
        # body turned the wrong way to 0.26 rad, the wrist answered by winding 0.8 rad
        # clockwise over twenty steps, and the grasp tore out at y = 0.586 with the burner
        # at 0.75.  Over 8 seeds the loop's gain simply picks how badly it ends -- 0.05 ->
        # 4/8, 0.15 -> 8/8 but every kettle 0.15 m short.  Pushing straight, with the wrist
        # left where the grasp put it, delivers to a mean 0.025 m of the burner at 8/8.
        # Residual yaw is then just a term in the environment's 7-D distance, and a small
        # one next to the position error it costs to chase.
        if isinstance(skill, KettleGraspSkill) and skill.grasp_retained():
            phase = skill.manipulation_stage()
            self._set_reactive_phase(phase)
            if skill.carrying():
                # Ordinarily the servo's bias is dropped at phase changes, which is often
                # enough because phases are short.  This one is not; see
                # `KitchenPolicyConfig.kettle_carry_bias_clamp`.
                self.servo.limit(self.cfg.kettle_carry_bias_clamp)
            return self._track_action(
                skill,
                manipulate,
                skill.manipulation_step_scale(),
                skill.engage_gripper,
                orient=self.cfg.kettle_carry_orientation != "free",
            )

        if isinstance(skill, LightSwitchSkill) and skill.grasp_retained():
            # This thin capsule can drive the finger joint almost to zero even during a
            # valid grasp, so bilateral switch contact is the authoritative signal.
            self._set_reactive_phase(MANIPULATE)
            action = self._track_action(
                skill,
                manipulate,
                skill.manipulation_step_scale(),
                skill.engage_gripper,
                orient=False,
            )
            if self.cfg.light_hold_roll and desired_orientation is not None:
                # Roll only: the component of the frame correction along the tool's own
                # approach axis.  See `KitchenPolicyConfig.light_hold_roll`.
                correction = rotation_action(self.sim, desired_orientation,
                                             skill.rotation_step_scale())
                axis = unit(np.asarray(self.sim.eef_approach_axis, dtype=np.float64))
                action[3:6] = axis * float(np.dot(correction, axis))
            return action

        if isinstance(skill, MicrowavePullSkill) and skill.grasp_retained():
            # Do not reissue the live grasp frame while pulling.  That frame turns with the
            # door -- its approach axis is the door normal -- so following it means turning
            # the wrist through the whole 60-degree swing under load.  Measured over 8
            # seeds, that halves the number of losses (1.25 against 2.00 re-grips) and still
            # ends worse: the first pull runs slightly longer, but the recovery after the
            # one remaining loss then fails on every seed, leaving the door at 0.276 mean
            # distance against 0.184 for letting the loaded handle constrain the wrist and
            # simply re-gripping.
            self._set_reactive_phase(MANIPULATE)
            action = self._track_action(
                skill,
                manipulate,
                skill.manipulation_step_scale(),
                skill.engage_gripper,
                orient=False,
            )
            pull_frame = skill.pull_frame()
            if pull_frame is not None and self.cfg.microwave_pull_yaw != 0.0:
                # Not `orient=True`: that would reissue the live, door-following frame.
                # This drives the one frozen frame taken at the grasp, so the wrist turns
                # once into the pull and then holds.
                action[3:6] = rotation_action(self.sim, pull_frame,
                                              skill.rotation_step_scale())
                self._last_orientation_error = orientation_error(self.sim, pull_frame)
            return action

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
                        and self._phase_steps > self.cfg.engage_seconds):
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
            #
            # Abandoning the insertion is judged against `realign_tolerance`, not the
            # tolerance that started it: see there for the chatter that testing one number
            # in both directions produces.
            if desired_orientation is not None and frame_error > realign_threshold:
                self._set_reactive_phase(ALIGN)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.approach_gripper)
            self._set_reactive_phase(APPROACH)
            return self._track_action(skill, contact, skill.approach_step_scale(),
                                      skill.approach_gripper)

        if (isinstance(skill, SlideSkill)
                and skill.grasp_center_tolerance > 0.0
                and self.sim.finger_opening >= self.cfg.release_opening_threshold
                and frame_error <= angle_tolerance
                and approach_along >= -skill.precontact_tolerance
                and approach_along <= approach_length + skill.lost_contact_tolerance
                and jaw_center_error > skill.centering_entry_tolerance()):
            # Square the open jaws on the handle bar before advancing onto it.  The tool
            # arrives from one side and sweeps about 1.7 cm across the bar during APPROACH,
            # which drags the door along its own slide axis: measured, seeds 10, 16 and 19
            # open the cabinet the whole way like this and never enter MANIPULATE at all,
            # and the environment banks it, so a push scores exactly like a grasp.
            skill._centering_active = True
            approach_direction = unit(contact - precontact)
            centered_depth = float(np.clip(approach_along, 0.0, approach_length))
            centered_point = precontact + approach_direction * centered_depth
            self._set_reactive_phase(MOVE_TO_PRECONTACT)
            # Keep the frame while centring, once the tool is near the line.  Without it
            # each centring step lets the frame drift past the ALIGN entry threshold and
            # ALIGN takes one step back in place: align/centre/align/centre, 5-8 steps on
            # every slide cabinet.  Far from the line it swings the jaws instead; see
            # `slide_centering_orient_lateral`.
            orient = (self.cfg.slide_centering_orient
                      and approach_lateral <= self.cfg.slide_centering_orient_lateral)
            return self._track_action(skill, centered_point,
                                      skill.approach_step_scale(),
                                      skill.approach_gripper,
                                      orient=orient,
                                      rotation_step=self.cfg.slide_centering_rotation_step)

        if isinstance(skill, LightSwitchSkill):
            # Lost the lever mid-sweep: leave along the lever before repositioning, so the
            # trip back to the stand-off does not sweep the switch shut again.  See
            # `KitchenPolicyConfig.light_slip_backout`.
            if (self.cfg.light_slip_backout > 0.0
                    and skill._ever_captured
                    and not skill.grasp_retained()
                    and not skill.manipulation_done()):
                if skill._slip_backout_target is None:
                    skill._slip_backout_target = (
                        skill.tool_pos()
                        - skill.approach_axis() * self.cfg.light_slip_backout)
                if float(np.linalg.norm(
                        skill.tool_pos() - skill._slip_backout_target)) > 0.02:
                    self._set_reactive_phase(MOVE_TO_PRECONTACT)
                    return self._track_action(skill, skill._slip_backout_target,
                                              skill.approach_step_scale(),
                                              skill.approach_gripper, orient=False)
                skill._slip_backout_target = None
                skill._ever_captured = False
            if light_switch_contacts:
                # Hold and close until both pads surround the capsule. Never promote a
                # one-finger touch into the right-to-left manipulation.
                self._set_reactive_phase(CONTACT_OR_GRASP)
                return self._track_eef(skill, self.sim.eef_pos,
                                       skill.approach_step_scale(),
                                       skill.engage_gripper, orient=False)
            if (self.sim.finger_opening < skill.grasp_ready_opening
                    and self._phase_steps > self.cfg.engage_seconds):
                # Closed without retaining switch contact: reopen in place, then let the
                # state-derived approach center and retry.
                #
                # The budget is what makes this a test of the *finished* close rather than
                # of a close in progress.  `grasp_ready_opening` is 0.039 against a 0.040
                # open jaw, so without it the branch fires on the first step the fingers
                # move at all: measured on seed 10, the jaws reach 0.0127 one step after the
                # close is commanded, the pads have not touched the lever yet, and this
                # reopens them all the way to 0.0411 before closing again and catching it.
                # That is the visible double grip -- close, release, close -- and it is a
                # race against the tool's last few millimetres of approach, not a missed
                # grasp.  The kettle's equivalent branch has always carried the budget.
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
                if (np.linalg.norm(skill.tool_pos() - skill.touch_point())
                        <= skill.reacquire_in_place_radius):
                    # Re-seat from where the hand is instead of walking back to the
                    # stand-off.  The stand-off sits `precontact_distance` *back along the
                    # arc*, so once the door has swung it is nowhere near the hand: traced
                    # on seed 16, a loss at step 342 with the pads 2 cm off the bar sent
                    # the tool out to y = 0.168 against a handle at y = 0.40 and took 30
                    # steps to come back, and the loss at 382 took 97 -- 289 microwave
                    # steps on that episode, which is what starved the light switch that
                    # followed it.  Nothing about a bar that slipped out of the jaws
                    # invalidates the approach that put them there, so as long as the hand
                    # is still in front of the handle, go straight back in.
                    self._set_reactive_phase(APPROACH)
                    return self._track_action(skill, skill.approach_point(),
                                              skill.approach_step_scale(),
                                              skill.approach_gripper, orient=True)
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
                # Roll about the measured approach axis only, preserving translation: this
                # is the hooking motion that swings the free-edge pad into the gap behind
                # the D-bar, and it is what makes the pull work at all -- replacing it with
                # a full-frame correction measured 0/8, with the door never moving on any
                # seed. The cost is that the wrist is otherwise unmanaged here, so a single
                # pad brushing the bar can sit for hundreds of steps while the frame drifts
                # (measured: 0.18 -> 2.06 rad over 300 steps). That is the approach phase
                # eating 306-382 of every 400-step episode, and it is still open.
                if (len(microwave_handle_contacts) < 2
                        and contact_error > skill.contact_tolerance):
                    # Deliberately unbounded. These branches call `_track_action` directly
                    # rather than a bounded tracker, so no phase budget applies, and adding one made
                    # things strictly worse: forcing a stand-off retry after 100 (or 50)
                    # steps measured 0/8 against 4/8 for simply persisting, because the
                    # re-approach lands in the same one-pad state it backed away from.
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
                if not skill.bar_between_pads():
                    # Touching is not the same as holding. A pad can brush the outside of
                    # the bar while the bar is still well clear of the jaws, and closing
                    # from there traps nothing and wedges the finger. Keep inserting.
                    action = self._track_action(skill, contact,
                                                skill.approach_step_scale(),
                                                skill.approach_gripper, orient=False)
                    action[3:6] = skill.handle_roll_action()
                    return action
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

            if (contact_error <= skill.contact_tolerance
                    and skill.bar_between_pads()):
                # With the bar correctly centered, fully open pads do not touch it yet.
                # Proximity is therefore the signal to close; subsequent manipulation
                # still requires grasp_retained() to observe bilateral physical contact.
                #
                # `contact_error` alone is not enough in principle: it measures the distance
                # to the hooked *contact point*, which sits offset from the bar by design,
                # so the tool can satisfy it while the bar is still outside the jaws.
                # Guarding it measured neutral (4/8 either way) -- on the failing seeds the
                # policy never reaches this branch at all -- but a closing command issued
                # with the bar outside the pads is not recoverable, so the guard stays.
                self._set_reactive_phase(CONTACT_OR_GRASP)
                action = self._track_action(skill, contact,
                                            skill.approach_step_scale(),
                                            skill.engage_gripper, orient=False)
                action[3:6] = skill.handle_roll_action()
                return action

            if self._phase in (APPROACH, CONTACT_OR_GRASP):
                # Continue the straight insertion to microwave_handle_position, correcting
                # the full grasp frame as it goes.  This used to command roll about the
                # measured approach axis only, on the grounds that a full pose overconstrains
                # the IK; that was true of the old fraction-of-error controller, but the
                # joint servo solves position and orientation in one IK, and withholding the
                # rest of the frame here left ALIGN as the only phase that could correct it
                # -- which bounced APPROACH/ALIGN forever with the frame error stuck around
                # 0.3-1.1 rad, far outside the 0.23 needed to close on the handle.
                self._set_reactive_phase(APPROACH)
                return self._track_action(skill, skill.approach_point(),
                                          skill.approach_step_scale(),
                                          skill.approach_gripper, orient=True)
            if (desired_orientation is not None
                    and frame_error > alignment_threshold
                    and pre_error <= skill.alignment_tolerance):
                self._set_reactive_phase(ALIGN)
                return self._track_action(skill, precontact,
                                          skill.approach_step_scale(),
                                          skill.approach_gripper)
            if pre_error <= skill.precontact_tolerance:
                self._set_reactive_phase(APPROACH)
                return self._track_action(skill, skill.approach_point(),
                                          skill.approach_step_scale(),
                                          skill.approach_gripper, orient=False)
            self._set_reactive_phase(MOVE_TO_PRECONTACT)
            orient = True  # see the generic reach below
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
            # `grasp_ready_opening` only says the jaws have left the open stop, which is
            # true one step after the close is commanded and says nothing about whether
            # they have reached the handle.  Measured on the slide cabinet, seed 10: the
            # close is issued at t=296, CONTACT_OR_GRASP lasts exactly one step, and
            # MANIPULATE starts at t=297 with the jaws still travelling through 0.0213 --
            # so the drive begins before the pads have settled on the bar and the cabinet
            # is shoved by whichever finger arrives first rather than carried by a grip.
            # `engage_seconds` was written for precisely this ("steps held in
            # CONTACT_OR_GRASP so the gripper actuator settles before manipulating") but
            # nothing on this path enforced it.
            #
            # Latched per skill, not tested per step.  The classifier is geometric and
            # re-runs every step, so a drive that dips back into CONTACT_OR_GRASP resets
            # `_phase_steps` and would pay the settle again: measured unlatched, the slide
            # entered MANIPULATE 697 times across 24 seeds instead of 20, and the median
            # episode went from 295 steps to 388.  One settled close per skill is what the
            # budget was for.
            settled = (not self.cfg.engage_settle_required
                       or skill._engage_settled
                       or self.sim.finger_opening < skill.grasp_contact_min_opening)
            if (not settled
                    and self._phase == CONTACT_OR_GRASP
                    and self._phase_steps >= self.cfg.engage_seconds):
                # Only steps spent *in the grasp phase* count.  `_phase_steps` belongs to
                # whatever phase is running, and this line is reached from APPROACH too, so
                # without the phase test the latch is earned during the reach and the gate
                # has already lapsed by the time the jaws are told to close.
                skill._engage_settled = True
                settled = True
            gripper_ready = (not closing
                             or (self.sim.finger_opening < skill.grasp_ready_opening
                                 and settled))
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
            if jaw_center_error > skill.centering_entry_tolerance():
                # The target bar is not centered between the jaws yet. Back up to the
                # stand-off and finish the lateral correction instead of letting an outer
                # finger surface turn this approach into an accidental push.
                self._set_reactive_phase(MOVE_TO_PRECONTACT)
                orient = True  # see the generic reach below
                return self._track_action(skill, precontact,
                                          skill.precontact_step_scale(),
                                          skill.approach_gripper, orient=orient)
            if desired_orientation is not None and frame_error > angle_tolerance:
                self._set_reactive_phase(ALIGN)
                if skill.align_in_place:
                    return self._track_eef(skill, self._alignment_hold_point(),
                                           self.cfg.contact_step,
                                           skill.approach_gripper)
                return self._track_action(skill, precontact,
                                          skill.precontact_step_scale(),
                                          skill.approach_gripper)
            self._set_reactive_phase(APPROACH)
            return self._track_action(skill, contact, skill.approach_step_scale(),
                                      skill.approach_gripper)

        in_approach_corridor = (
            approach_along >= -skill.precontact_tolerance
            and approach_along <= approach_length + skill.lost_contact_tolerance
            and approach_lateral <= skill.precontact_tolerance
        )
        # Once the approach has started, committing to it is what keeps it monotone.
        # `pre_error` is the distance back to the *stand-off*, so it necessarily grows as
        # the tool advances toward contact; judging the phase on it alone drops the skill
        # back into MOVE_TO_PRECONTACT, which drives the tool backwards, which shrinks
        # `pre_error` again.  Measured on the kettle: MOVE_TO_PRECONTACT/APPROACH alternated
        # on every single step from t=111 to t=125.  Staying in APPROACH while the tool is
        # still on the stand-off -> contact segment turns that into one clean insertion.
        committed_to_approach = (
            self._phase in (APPROACH, CONTACT_OR_GRASP)
            and approach_along >= -skill.precontact_tolerance
            and approach_lateral <= skill.precontact_tolerance + skill.lost_contact_tolerance
        )
        # The frame an insertion may *start* from can be stricter than the one it may
        # continue at; see `KitchenSkill.approach_entry_tolerance`.  Only before the commit:
        # once the tool is on its way in, `realign_tolerance` alone decides.
        entry_threshold = alignment_threshold
        if (skill.approach_entry_tolerance is not None
                and not committed_to_approach
                and not self._align_exhausted):
            entry_threshold = min(entry_threshold, float(skill.approach_entry_tolerance))

        # A geometric corridor makes waypoint progress state-derived.  The prior version
        # used ``self._phase`` as hysteresis and could return different actions for an
        # identical simulator state depending on what had been queried before it.
        if (desired_orientation is not None
                and frame_error > entry_threshold
                and pre_error <= skill.alignment_tolerance):
            # Rotation can displace the EEF a few centimetres.  Keep aligning throughout
            # the broader safe stand-off region instead of bouncing back to an unoriented
            # free-space reach whenever that displacement exceeds precontact_tolerance.
            self._set_reactive_phase(ALIGN)
            if skill.align_in_place:
                self._last_tool_error = pre_error
                return self._track_eef(skill, self._alignment_hold_point(),
                                       self.cfg.contact_step, skill.approach_gripper)
            return self._track_action(skill, precontact, self.cfg.contact_step,
                                      skill.approach_gripper)

        # An aligned wrist inside the stand-off region goes straight in.  ALIGN is only
        # entered within `alignment_tolerance` of the stand-off, so once its frame test
        # passes the tool is where the reach was taking it; sending it back to
        # MOVE_TO_PRECONTACT for the last few millimetres disturbs the wrist again and the
        # two phases alternate one step at a time -- measured on every episode at the slide
        # cabinet's stand-off, 6-8 steps of align/move/align/move before APPROACH starts.
        # Within twice `precontact_tolerance`, not `alignment_tolerance`: the latter is
        # 0.20 m for the light switch and 0.60 m for the kettle, and starting the insertion
        # from that far out measured 18 light-switch approaches of 26 steps and 0.46 m of
        # path for a 0.15 m insertion, plus two kettle re-grasps, on seeds 0-499.
        entry_radius = (2.0 * skill.precontact_tolerance
                        if skill.approach_entry_radius is None
                        else float(skill.approach_entry_radius))
        # ... and only from inside a cone about the insertion line.  The radius alone let
        # seed 93 start the light-switch insertion 0.20 m out but 17 cm below the line; the
        # diagonal approach met the lever 7 cm off-centre across the jaws, one pad landed on
        # its side, the jaws closed on nothing and the arm sat frozen for 30 steps (4 mm of
        # motion, under the reach-freeze test's error floor) until the task stall fired.
        # Requiring the tool to be *on* the line instead (lateral within 2x
        # `precontact_tolerance`) throws the early start away for nearly every seed and
        # measured the light switch at +12 steps per episode over the suite; the cone keeps
        # the shallow diagonals and refuses the steep ones.
        remaining_depth = max(float(approach_length) - float(approach_along), 1e-6)
        aligned_at_standoff = (
            self.cfg.align_exit_to_approach
            and self._phase == ALIGN
            and pre_error <= entry_radius
            and (self.cfg.approach_cone_ratio <= 0.0
                 or approach_lateral <= self.cfg.approach_cone_ratio * remaining_depth)
        )
        if (pre_error <= skill.precontact_tolerance or in_approach_corridor
                or committed_to_approach or aligned_at_standoff):
            if (desired_orientation is not None
                    and frame_error > entry_threshold):
                self._set_reactive_phase(ALIGN)
                if skill.align_in_place:
                    self._last_tool_error = pre_error
                    return self._track_eef(skill, self._alignment_hold_point(),
                                           self.cfg.contact_step, skill.approach_gripper)
                return self._track_action(skill, precontact, self.cfg.contact_step,
                                          skill.approach_gripper)
            if not committed_to_approach and skill.approach_blocked():
                # Laterally in the corridor but at the wrong height, which the corridor
                # cannot see.  Finish the reach first; its waypoint is the stand-off, so
                # staying here is what lifts the tool back to the insertion line.
                self._set_reactive_phase(MOVE_TO_PRECONTACT)
                return self._track_action(skill, precontact,
                                          skill.precontact_step_scale(),
                                          skill.approach_gripper)
            self._set_reactive_phase(APPROACH)
            return self._track_action(skill, contact, skill.approach_step_scale(),
                                      skill.approach_gripper)

        self._set_reactive_phase(MOVE_TO_PRECONTACT)
        # Reach and rotate together.  Withholding the rotation until the wrist happened
        # to be aligned meant only ALIGN could ever turn it, and turning displaces the
        # end effector -- which pushed the reach predicate back out of tolerance and
        # bounced the FSM between the two phases indefinitely (measured: the slide
        # cabinet cycled MOVE_TO_PRECONTACT/ALIGN every few steps for 600 steps).  The
        # joint servo drives a full pose target, so both are solved in one IK.
        orient = True
        # ... but the two are not solved at the same rate: see `reach_tilt_threshold`.
        step_scale = skill.precontact_step_scale()
        if (skill.desired_orientation() is not None
                and self._turn_first(pre_error, self._last_orientation_error,
                                     self.cfg.reach_tilt_threshold)):
            # Far away and badly turned: rotate in place first, then travel unthrottled.
            return self._track_eef(skill, self.sim.eef_pos, step_scale,
                                   skill.approach_gripper, orient=True)
        if (self.cfg.reach_tilt_threshold > 0.0
                and skill.desired_orientation() is not None
                and self._last_orientation_error > self.cfg.reach_tilt_threshold):
            step_scale *= self.cfg.reach_tilt_position_scale
        # Only this reach takes the destination posture.  Applying it to every step that
        # reports MOVE_TO_PRECONTACT -- the slide's jaw-centring stage does -- measured
        # seed 5's slide cabinet at 268 steps: the pull perturbs the centimetre-scale
        # centring translation, the approach and the centring alternate, and the stall
        # detector retries twice.
        self._reach_posture_active = True
        return self._track_action(skill, precontact, step_scale,
                                  skill.approach_gripper, orient=orient)

    def _track_action(self, skill: KitchenSkill, tool_point: np.ndarray, step_scale: float,
                      gripper_mode: str, recover_posture: bool = False,
                      orient: bool = True,
                      rotation_step: Optional[float] = None) -> np.ndarray:
        # Always record how far the tool point is from where the tool *measurably* is, so
        # a plan built on the nominal gripper axis cannot silently disagree with the sim.
        self._last_tool_error = float(np.linalg.norm(np.asarray(tool_point, dtype=np.float64)
                                                     - skill.tool_pos()))
        return self._track_eef(skill, skill.eef_target(tool_point), step_scale, gripper_mode,
                               recover_posture=recover_posture, orient=orient,
                               rotation_step=rotation_step)

    def _track_initial_orientation(self, desired_frame: np.ndarray) -> np.ndarray:
        """Rotate to the front-facing horizontal frame without reaching toward a task."""
        target = self._initial_orientation_target
        self._last_target = target.copy()
        self._last_position_error = float(np.linalg.norm(target - self.sim.eef_pos))
        self._last_tool_error = 0.0
        action = np.zeros(CARTESIAN_ACTION_DIM, dtype=np.float64)
        # Counter the Cartesian drift induced by a large wrist reorientation and preserve
        # one fingertip-length of clearance from the cabinet throughout the turn.
        #
        # Give way to the rotation once the wrist is badly out.  The travel home from the
        # slide cabinet is 0.8 m, and the joint motion that covers it drags the wrist far
        # faster than `max_rotation_step` can correct: measured across that transit the
        # front-frame error *grows* from 0.006 to 0.742 rad on seed 0 and from 0.019 to
        # 1.657 on seed 4, while the correction is already saturated.  The arm then reaches
        # the home position with most of the phase gone and has to spend what is left
        # unwinding -- seed 0 just makes it in 31 steps, seed 4 runs `orient_forward_budget`
        # out at 62 and hands the kettle a wrist 0.32 rad from front, which is what its
        # ALIGN then cannot recover.  Slowing the translation is the only lever: the
        # rotation is already at its ceiling.
        #
        # `tilt_recovery_threshold` and `tilt_recovery_position_scale` are exactly this
        # trade and were written for it, but nothing passed `recover_posture=True`, so they
        # had never run.  Applying them here is what they were for.
        step_scale = self.cfg.contact_step
        if self._turn_first(self._last_position_error, self._last_orientation_error,
                            self.cfg.tilt_recovery_threshold):
            # Turn first; see `turn_first_distance`.
            action[:3] = 0.0
        else:
            if self._last_orientation_error > self.cfg.tilt_recovery_threshold:
                step_scale *= self.cfg.tilt_recovery_position_scale
            action[:3] = position_action(self.sim, target, step_scale)
        action[3:6] = rotation_action(self.sim, desired_frame, self.cfg.max_rotation_step)
        action[6] = self._gripper("open")
        return action

    def _track_eef(self, skill: Optional[KitchenSkill], target: np.ndarray, step_scale: float,
                   gripper_mode: str, recover_posture: bool = False,
                   orient: bool = True,
                   rotation_step: Optional[float] = None) -> np.ndarray:
        self._last_target = np.asarray(target, dtype=np.float64)
        self._last_position_error = float(np.linalg.norm(target - self.sim.eef_pos))
        action = np.zeros(CARTESIAN_ACTION_DIM, dtype=np.float64)
        action[3:6] = (self._orientation_action(skill, rotation_step)
                       if orient else 0.0)
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

    def _orientation_action(self, skill: Optional[KitchenSkill],
                            rotation_step: Optional[float] = None) -> np.ndarray:
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
            return rotation_action(self.sim, desired_frame,
                                   skill.rotation_step_scale() if rotation_step is None
                                   else float(rotation_step))

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


    def _hold(self, gripper: float, skill: Optional[KitchenSkill] = None) -> np.ndarray:
        action = np.zeros(CARTESIAN_ACTION_DIM, dtype=np.float64)
        action[3:6] = self._orientation_action(skill)
        action[6] = gripper
        return action

    # -- introspection -------------------------------------------------------------------
    def get_diagnostics(self) -> Dict[str, Any]:
        # In 1.2.1 `data.ctrl` is a joint *position* target recomputed from the measured
        # qpos every step, so this gap is the servo's tracking lag rather than the unbounded
        # accumulator 1.2.0 had. It stays useful as a "the arm is being held back" signal.
        joint_target_error = float(np.linalg.norm(
            self.sim.data.ctrl[:7] - self.sim.data.qpos[:7]))
        kettle_skill = (self._skill if isinstance(self._skill, KettleGraspSkill) else None)
        microwave_skill = (self._skill
                           if isinstance(self._skill, MicrowavePullSkill) else None)
        microwave_contacts = (sorted(microwave_skill.contacting_fingers())
                              if microwave_skill is not None else [])
        return {
            "action_contract": "gymnasium-robotics-1.2.1-joint-velocity-9d",
            "action_repeat": int(self._action_repeat),
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
            # How far the withdrawal has actually got, reported straight off the frozen
            # exit line rather than by asking the skill.  `recede_point` *freezes* that
            # line the first time it is called, and it is only correct once the handle has
            # been released, so a diagnostic that called it would decide the exit route
            # early and from the wrong pose -- the same defect that once had the kettle
            # climbing over its own body.  Nothing here is reported until the skill has
            # frozen the line itself.
            **_microwave_recede_diagnostics(microwave_skill),
            # The end effector's world translation, reported unconditionally so a trace can
            # be read as a path in the scene frame rather than only as error magnitudes.
            "eef_position": self.sim.eef_pos.tolist(),
            # The kettle body's world translation, likewise unconditional: it is a free
            # body, so it can be moved by any skill's arm, not only by its own.
            "kettle_position": self.sim.joint_qpos("kettle")[:3].tolist(),
            "target_position": None if self._last_target is None else self._last_target.tolist(),
            "position_error": float(self._last_position_error),
            "tool_error": float(self._last_tool_error),
            "orientation_error": float(self._last_orientation_error),
            "initial_orientation_complete": bool(self._initial_orientation_complete),
            # ORIENT_FORWARD's two exit conditions, reported live alongside the thresholds
            # they are tested against, so the phase can be tuned by watching it rather than
            # by sweeping.  The clearance is how far the tool still is from the home pose,
            # which is what `orient_forward_position_tolerance` gates; the orientation half
            # is `orientation_error` above against `yaw_tolerance`.  Both must fall inside
            # their tolerance for the phase to release the arm to the next subtask.
            "orient_forward_clearance_error": float(np.linalg.norm(
                self._initial_orientation_target - self.sim.eef_pos)),
            "orient_forward_position_tolerance": float(
                self._orient_forward_clearance_tolerance),
            "yaw_tolerance": float(self._orient_forward_yaw_tolerance),
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
