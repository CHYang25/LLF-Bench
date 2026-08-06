"""Tests for the LLF-Bench Franka Kitchen wrapper and its scripted expert.

Run everything::

    python tests/kitchen_test.py

Run one group::

    python tests/kitchen_test.py --only rollout
"""

if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)

import numpy as np
import gymnasium as gym
import mujoco

import llfbench
from llfbench.envs.llf_env import LLFWrapper
from llfbench.envs.kitchen import DEFAULT_TASKS_TO_COMPLETE
from llfbench.envs.kitchen.scripted_policy import (
    BONUS_THRESH,
    FINGERTIP_OFFSET,
    FRONT_APPROACH_AXIS,
    HandlePullSkill,
    VERTICAL_HANDLE_AXIS,
    IKTargetLeakController,
    KitchenPolicyConfig,
    KitchenSkill,
    KitchenSim,
    KettleGraspSkill,
    MicrowavePullSkill,
    OrientationAwareIKController,
    build_skill,
    limit_to_box,
    position_action,
)

ENV_ID = "llf-kitchen-FrankaKitchen-v1"


def make_env(**kwargs):
    return llfbench.make(ENV_ID, **kwargs)


# ---------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------

def test_registration():
    """`llf-kitchen-FrankaKitchen-v1` is registered by `import llfbench` and needs no kwargs."""
    assert ENV_ID in gym.envs.registry, f"{ENV_ID} is not registered by importing llfbench"
    env = gym.make(ENV_ID)          # bug #4: this used to raise TypeError: set(None)
    assert sorted(env.kitchen_env.goal.keys()) == sorted(DEFAULT_TASKS_TO_COMPLETE)
    env.close()
    print("  registration + no-kwarg gym.make .......... ok")


def test_wrapper_stack():
    env = make_env()
    inner = env
    while not isinstance(inner, LLFWrapper) and hasattr(inner, "env"):
        inner = inner.env
    assert isinstance(inner, LLFWrapper), "the kitchen env is not wrapped in an LLFWrapper"
    assert len(env.reward_range) == 2, "tests/test_envs.py asserts len(env.reward_range) == 2"
    assert env.reward_range == (0.0, float(len(env.kitchen_env.goal)))
    env.close()
    print("  LLFWrapper stack + reward_range ........... ok")


def test_reset_contract():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict) and set(obs) == {"instruction", "observation", "feedback"}
    assert isinstance(obs["instruction"], str) and obs["instruction"]
    assert isinstance(obs["observation"], str) and obs["observation"]
    # Like MetaworldWrapper._reset, the kitchen wrapper offers an `fp` suggestion at reset,
    # so the feedback is already a verbalized (non-empty) string here.
    assert isinstance(obs["feedback"], str) and obs["feedback"]
    assert info["success"] is False
    assert isinstance(info["tasks_to_complete"], list)      # bug #2: dict at reset, set at step
    assert all(isinstance(t, str) for t in info["tasks_to_complete"])
    env.close()
    print("  reset() contract .......................... ok")


def test_step_contract():
    env = make_env()
    env.reset(seed=0)
    action = env.action_space.sample()
    out = env.step(action)
    assert isinstance(out, tuple) and len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, dict) and set(obs) == {"instruction", "observation", "feedback"}
    assert obs["instruction"] is None
    assert isinstance(obs["observation"], str) and obs["observation"]
    assert isinstance(obs["feedback"], str) and obs["feedback"], \
        "feedback must verbalize to a non-empty string"
    assert type(reward) is float
    assert type(terminated) is bool and type(truncated) is bool
    assert isinstance(info, dict) and isinstance(info["success"], bool)
    assert info["video"] is None                       # render_video(False) by default
    assert isinstance(info["tasks_to_complete"], list)
    assert env.reward_range[0] <= reward <= env.reward_range[1]
    env.close()
    print("  step() contract ........................... ok")


def test_expert_action_is_valid():
    env = make_env()
    env.reset(seed=0)
    for _ in range(20):
        expert = env.expert_action
        assert expert.shape == env.action_space.shape, (expert.shape, env.action_space.shape)
        assert np.all(np.isfinite(expert)), expert
        assert env.action_space.contains(expert.astype(np.float32)), expert
        env.step(expert)
    env.close()
    print("  expert_action shape / bounds / finite ..... ok")


def test_expert_action_is_memoized_per_step():
    """Reading `expert_action` twice in a step must not advance the expert's FSM twice."""
    env = make_env()
    env.reset(seed=0)
    first = env.expert_action
    second = env.expert_action
    assert np.array_equal(first, second), (first, second)
    phase_a = env.kt_policy.get_diagnostics()["phase_steps"]
    _ = env.expert_action
    phase_b = env.kt_policy.get_diagnostics()["phase_steps"]
    assert phase_a == phase_b, "querying expert_action advanced the FSM"
    env.close()
    print("  expert_action memoized per step ........... ok")


def test_repeated_policy_query_is_state_feedback():
    """Direct queries at one simulator state cannot advance a timer-driven FSM."""
    env = make_env(control_steps=1)
    env.reset(seed=0)
    policy = env.kt_policy
    first = policy.get_action()
    first_diag = policy.get_diagnostics()
    for _ in range(50):
        action = policy.get_action()
        diag = policy.get_diagnostics()
        assert np.array_equal(action, first), (first, action)
        assert diag["selected_subtask"] == first_diag["selected_subtask"]
        assert diag["controller_phase"] == first_diag["controller_phase"]
        assert diag["phase_steps"] == first_diag["phase_steps"]
    env.close()
    print("  repeated query leaves state/action fixed .. ok")


def test_expert_survives_unexecuted_recommendations():
    """Learner actions may differ from the recommendation without corrupting the expert."""
    env = make_env(control_steps=1)
    env.reset(seed=0)
    policy = env.kt_policy
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(40):
        env.step(zero)
    diag = env.kt_policy.get_diagnostics()
    # The requested reset behavior is strict: if the learner does not execute the expert's
    # rotation, task selection and task-directed translation must not begin behind its back.
    assert diag["selected_subtask"] is None, diag
    assert diag["controller_phase"] == "orient_forward", diag
    assert diag["completed_order"] == [] and diag["abandoned"] == [], diag
    expert = env.expert_action
    assert np.linalg.norm(expert[3:6]) > 0.0, expert
    # Zero learner actions still let the actuator state settle away from the captured reset
    # pose.  Translation is allowed only to cancel that drift, never toward a task waypoint.
    sim = KitchenSim(env)
    clearance_delta = policy._initial_orientation_target - sim.eef_pos
    assert np.allclose(diag["target_position"], policy._initial_orientation_target), diag
    assert np.dot(expert[:3], clearance_delta) > 0.0, (expert, clearance_delta)
    env.close()
    print("  arbitrary learner actions do not age FSM . ok")


# ---------------------------------------------------------------------------------------
# Policy / skills
# ---------------------------------------------------------------------------------------

def test_zero_error_gives_zero_translation():
    env = make_env()
    env.reset(seed=0)
    sim = KitchenSim(env)
    action = position_action(sim, sim.eef_pos, 1.0)
    assert np.allclose(action, 0.0, atol=1e-9), action
    env.close()
    print("  zero position error -> zero translation ... ok")


def test_action_points_at_the_target():
    env = make_env()
    env.reset(seed=0)
    sim = KitchenSim(env)
    for delta in (np.array([0.05, 0.0, 0.0]),
                  np.array([0.0, -0.07, 0.0]),
                  np.array([0.0, 0.0, 0.03]),
                  np.array([0.4, -0.3, 0.9])):     # far enough that the step is limited
        action = position_action(sim, sim.eef_pos + delta, 1.0)
        assert float(np.dot(action, delta)) > 0.0, (action, delta)
        cos = float(np.dot(action, delta) / (np.linalg.norm(action) * np.linalg.norm(delta)))
        assert cos > 1 - 1e-6, f"limiting changed the direction: cos={cos}"
    env.close()
    print("  action points toward a perturbed target ... ok")


def test_limit_to_box_preserves_direction():
    for vec in (np.array([3.0, -1.0, 0.5]), np.array([0.1, 0.2, -0.05]), np.zeros(3)):
        out = limit_to_box(vec, 1.0)
        assert np.all(np.abs(out) <= 1.0 + 1e-12), out
        if np.linalg.norm(vec) > 0:
            cos = float(np.dot(out, vec) / (np.linalg.norm(out) * np.linalg.norm(vec)))
            assert cos > 1 - 1e-9, cos
    print("  limit_to_box keeps direction .............. ok")


def test_gripper_convention():
    """Index 6 is the gripper, +1 opens and -1 closes (opposite to MetaWorld)."""
    cfg = KitchenPolicyConfig()
    assert cfg.gripper_open > 0 > cfg.gripper_close

    env = make_env()
    env.reset(seed=0)
    sim = KitchenSim(env)
    action = np.zeros(7, dtype=np.float32)
    action[6] = cfg.gripper_close
    for _ in range(6):
        env.step(action)
    closed = sim.finger_opening
    env.reset(seed=0)
    action[6] = cfg.gripper_open
    for _ in range(6):
        env.step(action)
    opened = sim.finger_opening
    assert closed < 1e-2 < opened, f"gripper convention wrong: closed={closed}, open={opened}"
    env.close()
    print(f"  gripper: +1 -> {opened:.3f} m, -1 -> {closed:.3f} m ..... ok")


def test_ik_target_anti_windup_is_configurable():
    env = make_env(control_steps=1)
    env.reset(seed=0)
    controller = KitchenSim(env).kitchen.robot_env.controller
    assert isinstance(controller, IKTargetLeakController)
    assert np.isclose(controller.gain, KitchenPolicyConfig().ik_target_leak)
    assert isinstance(controller.controller, OrientationAwareIKController)
    assert np.isclose(controller.controller.duration,
                      KitchenPolicyConfig().ik_orientation_duration)
    env.close()

    cfg = KitchenPolicyConfig(ik_target_leak=0.0)
    env = make_env(control_steps=1, policy_config=cfg)
    env.reset(seed=0)
    controller = KitchenSim(env).kitchen.robot_env.controller
    assert not isinstance(controller, IKTargetLeakController)
    env.close()
    print("  IK target anti-windup can be disabled ..... ok")


def test_side_grasp_frames_follow_live_geometry():
    """The four reference tasks use fingertips and complete object-relative frames."""
    env = make_env()
    env.reset(seed=0)
    sim = KitchenSim(env)
    cfg = KitchenPolicyConfig()

    for task in ("slide_cabinet", "microwave", "kettle", "light_switch"):
        skill = build_skill(sim, task, cfg)
        frame = skill.desired_orientation()
        assert frame is not None, task
        expected_tool_offset = 0.15 if task == "light_switch" else FINGERTIP_OFFSET
        assert np.isclose(skill.tool_offset, expected_tool_offset), task
        assert skill.approach_gripper == "open", task
        assert skill.engage_gripper == "close", task
        assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-7), (task, frame)
        assert np.isclose(np.linalg.det(frame), 1.0, atol=1e-7), (task, frame)

        approach = skill.approach_axis() / np.linalg.norm(skill.approach_axis())
        assert np.dot(frame[:, 2], approach) > 1.0 - 1e-7, task
        if task != "light_switch":
            # Local y is the jaw-separation axis: it must cross the handle, while local x
            # follows the handle itself and local z points along the object-relative reach.
            handle = skill.handle_axis()
            handle = handle - np.dot(handle, approach) * approach
            handle /= np.linalg.norm(handle)
            assert np.dot(frame[:, 0], handle) > 1.0 - 1e-7, task
            assert abs(np.dot(frame[:, 1], handle)) < 1e-7, task

    # The compact profile *hooks* the microwave instead of pinching its D-bar: the grasp
    # frame is yawed about the bar so the free-edge pad sits behind it and the pull loads
    # that pad against the bar.  That deliberately breaks "local z == approach axis"
    # asserted above, so it is checked separately here.
    hooked = build_skill(sim, "microwave", KitchenPolicyConfig.fast_demo())
    hook_frame = hooked.desired_orientation()
    assert np.allclose(hook_frame.T @ hook_frame, np.eye(3), atol=1e-7), hook_frame
    assert np.isclose(np.linalg.det(hook_frame), 1.0, atol=1e-7), hook_frame
    bar = hooked.handle_axis() / np.linalg.norm(hooked.handle_axis())
    # The yaw is about the bar itself, so local x stays on it and handle_roll_action's
    # "keep the jaws orthogonal to the bar" correction is unaffected.
    assert np.dot(hook_frame[:, 0], bar) > 1.0 - 1e-7
    hook_approach = hooked.approach_axis() / np.linalg.norm(hooked.approach_axis())
    lead = float(np.dot(hook_frame[:, 1] * hooked.JAW_HALF_SPAN, hook_approach))
    assert np.isclose(lead, KitchenPolicyConfig.fast_demo().microwave_hook_depth, atol=1e-9)
    # It must be the pad on the door's free edge (+radial, away from the hinge) that ends
    # up behind the bar; hooking on the hinge side would fight the door's swing.
    assert np.dot(hook_frame[:, 1], hooked.door_radial()) > 0.0
    # Zero depth restores the square, friction-only frame the legacy profile uses.
    square = build_skill(sim, "microwave", KitchenPolicyConfig(microwave_hook_depth=0.0))
    square_approach = square.approach_axis() / np.linalg.norm(square.approach_axis())
    assert np.dot(square.desired_orientation()[:, 2], square_approach) > 1.0 - 1e-7

    assert isinstance(build_skill(sim, "kettle", cfg), KettleGraspSkill)
    slide = build_skill(sim, "slide_cabinet", cfg)
    drive = slide.drive_direction()
    handle = slide.handle_axis()
    approach = slide.approach_axis()
    assert drive[0] > 0.99
    assert np.dot(approach, FRONT_APPROACH_AXIS) > 1.0 - 1e-7
    assert abs(np.dot(approach, drive)) < 1e-7
    assert abs(np.dot(approach, handle)) < 1e-7
    assert abs(np.dot(drive, handle)) < 1e-7
    precontact_delta = slide.touch_point() - slide.precontact_point()
    assert np.dot(precontact_delta, approach) > 0.0
    assert abs(np.dot(precontact_delta, drive)) < 1e-7
    recede_delta = slide.touch_point() - slide.recede_point()
    assert slide.recede_after_manipulation
    assert np.isclose(np.linalg.norm(recede_delta), cfg.slide_recede_distance)
    assert np.dot(recede_delta, approach) > 0.0
    assert np.isclose(
        np.linalg.norm(slide.manipulate_point() - slide.touch_point()),
        cfg.slide_manipulate_lookahead,
    )
    assert np.isclose(slide.manipulation_step_scale(), cfg.slide_manipulation_step)

    microwave = build_skill(sim, "microwave", cfg)
    assert isinstance(microwave, MicrowavePullSkill)
    radial = microwave.door_radial()
    microwave_drive = microwave.drive_direction()
    microwave_approach = microwave.approach_axis()
    assert np.dot(microwave_approach, -microwave_drive) > 1.0 - 1e-7
    # Front-facing approach: a 90-degree yaw from the old radial/sideways reach.
    assert abs(np.dot(microwave_approach, radial)) < 1e-7
    assert abs(np.dot(microwave_approach, microwave.handle_axis())) < 1e-7
    assert np.dot(microwave.touch_point() - microwave.precontact_point(),
                  microwave_approach) > 0.0
    assert np.allclose(microwave.contact_point(), microwave.touch_point())
    assert microwave._handle_geoms
    assert set(microwave._finger_geoms.values()) == {"left_finger", "right_finger"}
    assert 0.0 < microwave.radial_bias < HandlePullSkill.radial_bias
    assert np.isclose(
        np.linalg.norm(microwave.touch_point() - sim.site_xpos("microhandle_site")),
        microwave.radial_bias,
    )
    assert microwave.manipulation_step_scale() > HandlePullSkill.manipulation_step
    microwave_roll = microwave.handle_roll_action()
    assert np.all(np.isfinite(microwave_roll))
    assert np.max(np.abs(microwave_roll)) <= microwave.rotation_step_scale() + 1e-9
    assert np.linalg.norm(np.cross(microwave_roll, sim.eef_approach_axis)) < 1e-9
    microwave._capture_offset = np.array([0.01, -0.005, 0.03])
    expected_pull = (microwave.touch_point()
                     + microwave.door_radial() * 0.01
                     + microwave.drive_direction()
                     * (microwave.manipulation_lookahead - 0.005))
    expected_pull[2] = microwave.touch_point()[2] + 0.03
    assert np.allclose(microwave.manipulate_point(), expected_pull)

    light = build_skill(sim, "light_switch", cfg)
    light_frame = light.desired_orientation()
    switch_axis = light.switch_axis()
    light_approach = light.approach_axis()
    light_drive = light.drive_direction()
    planar_approach = light_approach.copy()
    planar_approach[2] = 0.0
    planar_approach /= np.linalg.norm(planar_approach)
    assert np.dot(planar_approach, switch_axis) > 1.0 - 1e-7
    assert light_approach[2] < 0.0
    assert np.dot(light_frame[:, 0], VERTICAL_HANDLE_AXIS) > 0.98
    assert abs(np.dot(light_drive, switch_axis)) < 1e-7
    assert light_drive[0] < 0.0  # the light-on tangent is world right-to-left
    assert np.dot(light.contact_point() - light.touch_point(), switch_axis) > 0.0
    assert light._switch_geoms
    assert set(light._finger_geoms.values()) == {"left_finger", "right_finger"}
    light_recede_delta = light.touch_point() - light.recede_point()
    assert light.recede_after_manipulation
    assert np.isclose(np.linalg.norm(light_recede_delta), cfg.light_recede_distance)
    assert np.dot(light_recede_delta / cfg.light_recede_distance,
                  light_approach) > 1.0 - 1e-7

    kettle = build_skill(sim, "kettle", cfg)
    kettle_body_pos = sim.body_xpos("kettleroot")
    kettle_body_mat = sim.body_xmat("kettleroot")
    expected_left_bar = (kettle_body_pos
                         + kettle_body_mat @ kettle.LEFT_HANDLE_LOCAL_POS)
    expected_left_axis = kettle_body_mat @ kettle.LEFT_HANDLE_LOCAL_AXIS
    assert np.allclose(kettle.touch_point(), expected_left_bar)
    assert np.dot(kettle.handle_axis(), expected_left_axis) > 1.0 - 1e-7
    assert np.linalg.norm(kettle.touch_point() - sim.site_xpos("kettle_site")) > 0.05
    handle_axis = kettle.handle_axis()
    handle_axis /= np.linalg.norm(handle_axis)
    recede_delta = kettle.touch_point() - kettle.recede_point()
    assert kettle.recede_after_manipulation
    assert np.isclose(np.linalg.norm(recede_delta), cfg.kettle_recede_distance)
    assert abs(np.dot(recede_delta, handle_axis)) < 1e-9
    assert np.dot(recede_delta, kettle.approach_axis()) > 0.0
    recede_axis = recede_delta / np.linalg.norm(recede_delta)
    required_clearance = cfg.kettle_recede_distance - cfg.position_tolerance
    lateral_offset = handle_axis * (cfg.kettle_recede_tolerance + 0.01)
    kettle.tool_pos = lambda: kettle._recede_origin - recede_axis * (
        required_clearance - 1e-4) + lateral_offset
    assert not kettle.recede_complete()
    kettle.tool_pos = lambda: kettle._recede_origin - recede_axis * (
        required_clearance + 1e-4) + lateral_offset
    assert kettle.recede_complete()
    # A small axial IK residual can miss the strict projected-clearance threshold while
    # still physically reaching the fixed recede target closely enough.
    kettle.tool_pos = lambda: kettle.recede_point() + recede_axis * (
        cfg.kettle_recede_tolerance - 1e-4)
    assert kettle.recede_complete()

    # The kettle closes only after precise centering, then uses a short horizontal lead;
    # there are no lift/place stages or vertical transport commands.
    kettle = build_skill(sim, "kettle", cfg)
    assert kettle.contact_tolerance < KitchenSkill.contact_tolerance
    assert kettle.grasp_center_tolerance < KitchenSkill.precontact_tolerance
    assert kettle.grasp_contact_min_opening > 0.0
    assert np.allclose(
        sim.model.geom_pos[kettle._left_handle_geom_id],
        kettle.LEFT_HANDLE_LOCAL_POS,
        atol=0.01,
    )
    assert kettle._finger_pad_geoms
    assert all(
        sim.model.geom_type[geom_id] == int(mujoco.mjtGeom.mjGEOM_BOX)
        for geom_id in kettle._finger_pad_geoms
    )
    assert cfg.gripper_close <= kettle.gripper_command("close") < cfg.gripper_open
    assert kettle.gripper_command("open") == cfg.gripper_open
    assert np.isclose(kettle.manipulation_step_scale(), cfg.kettle_transport_step)
    assert cfg.yaw_tolerance <= kettle.orientation_tolerance_value() < 0.2
    assert kettle.rotation_step_scale() < cfg.max_rotation_step
    assert kettle.alignment_exit_tolerance_value() < kettle.orientation_tolerance_value()
    assert kettle.precontact_step_scale() <= cfg.free_space_step
    assert kettle.approach_step_scale() < cfg.contact_step
    assert not hasattr(cfg, "kettle_lift_height")
    assert not hasattr(cfg, "kettle_lift_tolerance")
    assert not hasattr(cfg, "kettle_place_planar_tolerance")
    assert not hasattr(cfg, "kettle_place_height_tolerance")
    goal = kettle._goal_pos().copy()
    root = goal + np.array([0.20, -0.10, 0.0])
    kettle._root_pos = lambda: root.copy()
    assert kettle.manipulation_stage() == "kettle_transport"

    tool = np.array([-0.3, 0.4, 1.85])
    kettle.tool_pos = lambda: tool.copy()
    transport_delta = kettle.manipulate_point() - tool
    assert np.dot(transport_delta[:2], (goal - root)[:2]) > 0.0
    assert abs(transport_delta[2]) < 1e-9
    assert np.isclose(np.linalg.norm(transport_delta), cfg.kettle_transport_lookahead)

    # Inside one lookahead the request must shrink with what is actually left, so the
    # loaded push decelerates onto the goal instead of driving at the step cap until the
    # stop predicate happens to fire.
    near_gap = cfg.kettle_transport_lookahead / 2.0
    near_root = goal + np.array([0.0, -near_gap, 0.0])
    kettle._root_pos = lambda: near_root.copy()
    assert np.isclose(np.linalg.norm(kettle.manipulate_point() - tool), near_gap)
    kettle._root_pos = lambda: root.copy()

    # The only orientation command a loaded transport may issue is a bounded world-z term
    # toward the fixed goal yaw; anything else re-enters the live-handle-frame feedback
    # loop that the manipulation branch deliberately stops driving.
    original_kettle_yaw = kettle._kettle_yaw
    # A one-pad hold is never turned: the torque has nothing to react against and levers
    # the bar out of the jaws instead of rotating the body.
    kettle.contacting_fingers = lambda: {"left_finger"}
    kettle._kettle_yaw = lambda: kettle._goal_yaw() - 0.5
    assert np.allclose(kettle.transport_rotation_action(), 0.0)
    kettle.contacting_fingers = lambda: {"left_finger", "right_finger"}
    yaw_command = kettle.transport_rotation_action()
    assert yaw_command.shape == (3,)
    assert np.allclose(yaw_command[:2], 0.0)
    assert abs(yaw_command[2]) <= cfg.kettle_transport_yaw_step
    assert yaw_command[2] > 0.0
    kettle._kettle_yaw = lambda: kettle._goal_yaw() + 0.5
    assert kettle.transport_rotation_action()[2] < 0.0
    kettle._kettle_yaw = original_kettle_yaw

    # Converting a live-frame transport target back to an EEF command must preserve the
    # measured EEF height too; this catches accidental vertical offsets in either frame.
    kettle = build_skill(sim, "kettle", cfg)
    transport_eef_delta = kettle.eef_target(kettle.manipulate_point()) - sim.eef_pos
    assert abs(transport_eef_delta[2]) < 1e-9

    # The hand may only descend to the low left bar inside the stand-off column: the
    # kettle's top-handle capsule sits directly above the body, and a diagonal reach from
    # anywhere else comes down onto it.
    corridor = build_skill(sim, "kettle", cfg)
    kettle_root = sim.body_xpos("kettleroot")
    transit = corridor.transit_point()
    assert transit is not None
    assert transit[2] > kettle_root[2] + corridor.TOP_HANDLE_GUARD_HEIGHT
    over_handle = kettle_root + np.array(
        [0.0, 0.0, corridor.TOP_HANDLE_GUARD_HEIGHT + 0.05])
    assert corridor.in_top_handle_corridor(over_handle)
    corridor.tool_pos = lambda: over_handle.copy()
    assert not corridor.transit_reached(transit)
    # Escaping the column is a sideways step at the height already reached, not a second
    # climb to the high waypoint.
    escape = corridor.transit_point()
    assert np.isclose(escape[2], over_handle[2])
    assert np.allclose(escape[:2], corridor.precontact_point()[:2])
    corridor.tool_pos = lambda: transit.copy()
    assert corridor.transit_reached(transit)
    # The left-bar approach height is below the guard, so a descent already under way is
    # never sent back up to the transit waypoint.
    low_bar = corridor.touch_point()
    assert not corridor.in_top_handle_corridor(low_bar)
    corridor.tool_pos = lambda: low_bar.copy()
    assert corridor.transit_reached(transit)
    # A captured kettle is legitimately inside the guarded volume and has no transit gate.
    corridor._capture_confirmed = True
    assert corridor.transit_point() is None

    original_task_distance = sim.task_distance
    completion_threshold = BONUS_THRESH * cfg.kettle_completion_margin
    sim.task_distance = lambda task: completion_threshold - 1e-4
    assert kettle.manipulation_done()
    sim.task_distance = lambda task: completion_threshold + 1e-4
    # Outside the pose margin, delivery of the body itself ends transport: a kettle already
    # at its goal position only spins further if it keeps being pushed.
    kettle._root_pos = lambda: root.copy()
    assert not kettle.manipulation_done()
    kettle._root_pos = lambda: goal.copy()
    assert kettle.manipulation_done()
    kettle._kettle_yaw = lambda: (kettle._goal_yaw()
                                  + cfg.kettle_goal_yaw_tolerance + 0.1)
    assert not kettle.manipulation_done()
    kettle._kettle_yaw = original_kettle_yaw
    sim.task_distance = original_task_distance
    env.close()
    print("  live side-grasp frames + safe recede paths  ok")


def test_initial_forward_alignment_precedes_task_motion():
    env = make_env(control_steps=1)
    env.reset(seed=0)
    policy = env.kt_policy
    action = policy.get_action()
    diag = policy.get_diagnostics()
    assert diag["controller_phase"] == "orient_forward", diag
    assert diag["selected_subtask"] is None, diag
    assert diag["initial_orientation_complete"] is False, diag
    assert np.linalg.norm(action[3:6]) > 0.0, action
    # The only initial translation is roomward clearance for the horizontal fingertips.
    assert np.dot(action[:3], -FRONT_APPROACH_AXIS) > 0.0, action
    assert np.allclose(diag["target_position"], policy._initial_orientation_target), diag
    env.close()
    print("  forward alignment precedes task motion .... ok")


def test_action_sanitisation():
    """A skill returning garbage must still produce a finite, in-box action."""
    env = make_env()
    env.reset(seed=0)
    policy = env.kt_policy
    broken = np.array([np.nan, np.inf, -np.inf, 1e9, -1e9, 0.0, 5.0])
    policy._compute_reactive_action = lambda: broken  # bypass the planner on purpose
    action = policy.get_action(None)
    assert np.all(np.isfinite(action)), action
    assert np.all(action >= -1.0) and np.all(action <= 1.0), action
    assert env.action_space.contains(action)
    env.close()
    print("  NaN/Inf/out-of-box actions sanitised ...... ok")


def test_completion_predicate_matches_env():
    """The policy's predicate must agree with BONUS_THRESH on the env's own achieved_goal."""
    env = make_env()
    env.reset(seed=0)
    sim = KitchenSim(env)
    raw = sim.kitchen._get_obs(sim.kitchen.robot_env._get_obs())
    for task in sim.goal:
        env_distance = float(np.linalg.norm(raw["achieved_goal"][task] - raw["desired_goal"][task]))
        assert abs(env_distance - sim.task_distance(task)) < 1e-9, task
        assert sim.task_complete(task) == (env_distance < BONUS_THRESH), task
    env.close()

    # The four burner joints have range [-0.009, 0] and a goal of -0.01, so they are inside
    # BONUS_THRESH before the robot moves.  Assert it, because the planner relies on it.
    burners = ["bottom_right_burner", "bottom_left_burner", "top_right_burner", "top_left_burner"]
    env = make_env(tasks_to_complete=burners)
    env.reset(seed=0)
    sim = KitchenSim(env)
    for burner in burners:
        assert sim.task_distance(burner) < BONUS_THRESH, burner
    env.close()
    print("  completion predicate == BONUS_THRESH ...... ok")


def test_planner_orders_and_reactivates_disturbed_task():
    # Exercise the documented precedence order explicitly; the workspace's default
    # profile may randomize the order for demonstration diversity.
    env = make_env(policy_config=KitchenPolicyConfig(randomize_task_order=False))
    env.reset(seed=0)
    policy = env.kt_policy
    order = policy._make_order()
    assert set(order) == set(env.kitchen_env.goal.keys()), "planner must drive the env's own tasks"
    assert order[0] == "slide_cabinet", "the slide cabinet must be opened first (it blocks the arm)"

    # Historical bookkeeping must not hide an actually incomplete task.  At reset the
    # slide distance is 0.37 (> BONUS_THRESH + hysteresis), so a stale "completed" marker
    # must be ignored and the physical task reselected.
    policy._completed_order.append("slide_cabinet")
    policy._task = None
    policy._skill = None
    policy._initial_orientation_complete = True
    _ = policy.get_action()
    assert policy.get_diagnostics()["selected_subtask"] == "slide_cabinet"
    env.close()
    print("  planner precedence + disturbance recovery . ok")


def test_randomized_task_order():
    cfg = KitchenPolicyConfig(randomize_task_order=True)
    env = make_env(policy_config=cfg)
    env.reset(seed=3)
    order = env.kt_policy._make_order()
    assert set(order) == set(env.kitchen_env.goal.keys())
    env.close()
    print("  randomized task order is a valid ordering . ok")


def test_randomized_microwave_before_kettle_rollout():
    """Microwave-first transitions must capture, exit, and leave later skills usable."""
    for seed in (4, 42, 45):
        cfg = KitchenPolicyConfig.fast_demo(randomize_task_order=True)
        env = make_env(policy_config=cfg)
        env.reset(seed=seed)
        order = list(env.kt_policy._order)
        assert order.index("microwave") < order.index("kettle"), order

        # Measured worst case across these seeds is 241 steps (seed 42), up from 220 since
        # the kettle approach stopped colliding with the top handle -- that collision used
        # to push the kettle most of the way to its goal for free.  See
        # test_expert_rollout for the per-phase breakdown of the difference.
        for _ in range(260):
            _, _, terminated, truncated, _ = env.step(env.expert_action)
            if terminated or truncated:
                break

        completed = list(env.kitchen_env.episode_task_completions)
        assert set(completed) == set(env.kitchen_env.goal), (
            f"seed {seed} microwave-before-kettle rollout stalled: order={order}, "
            f"completed={completed}, diagnostics={env.kt_policy.get_diagnostics()}")
        assert completed.index("microwave") < completed.index("kettle"), completed
        env.close()
    print("  microwave-before-kettle seeds 4/42/45 ..... ok")


# ---------------------------------------------------------------------------------------
# Multi-episode regression for KitchenEnv.reset not restoring tasks_to_complete (bug #1)
# ---------------------------------------------------------------------------------------

def test_two_episodes_still_score():
    # Both episodes use the *same* seed.  The bug under test is that `KitchenEnv.reset()`
    # does not restore `tasks_to_complete`, which silently costs the second episode every
    # subtask the first one banked; comparing one seed against itself isolates that from
    # how many tasks a given seed's task order happens to fit inside the step limit.
    env = make_env()
    kitchen = env.kitchen_env
    returns = []
    for episode in range(2):
        env.reset(seed=0)
        assert set(kitchen.tasks_to_complete) == set(kitchen.goal.keys()), (
            f"episode {episode} started with tasks_to_complete="
            f"{sorted(kitchen.tasks_to_complete)}; KitchenEnv.reset() does not restore it")
        assert kitchen.episode_task_completions == []
        total = 0.0
        for _ in range(280):
            _, reward, term, trunc, _ = env.step(env.expert_action)
            total += reward
            if term or trunc:
                break
        returns.append(total)
    assert returns[1] > 0, f"the second episode scored nothing: {returns}"
    assert returns[0] == returns[1], f"episodes disagree: {returns}"
    env.close()
    print(f"  two consecutive episodes scored {returns} .... ok")


# ---------------------------------------------------------------------------------------
# Rollout smoke test
# ---------------------------------------------------------------------------------------

def test_expert_rollout(episodes=1, seed=0):
    env = make_env()
    kitchen = env.kitchen_env
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        sim = KitchenSim(env)
        completion_step = {}
        total = 0.0
        steps = 0
        for step in range(260):
            _, reward, term, trunc, info = env.step(env.expert_action)
            total += reward
            steps += 1
            for task in kitchen.episode_task_completions:
                completion_step.setdefault(task, step)
            if term or trunc:
                break
        diag = env.kt_policy.get_diagnostics()
        print(f"\n  --- expert rollout, episode {episode} (seed {seed + episode}) ---")
        print(f"  steps               : {steps}")
        print(f"  total return        : {total}")
        print(f"  completed subtasks  : {list(kitchen.episode_task_completions)}")
        print(f"  completion order    : {completion_step}")
        print(f"  steps per subtask   : {diag['steps_per_subtask']}")
        print(f"  phase failures      : {diag['phase_failures']}")
        print(f"  retries (last task) : {diag['retry_count']}")
        print(f"  abandoned subtasks  : {diag['abandoned']}")
        print("  final distances     : " + ", ".join(
            f"{t}={sim.task_distance(t):.3f}" for t in sorted(kitchen.goal)))
        expected = set(DEFAULT_TASKS_TO_COMPLETE)
        assert set(kitchen.episode_task_completions) == expected, \
            f"the compact expert did not complete every task: expected {expected}"
        # 150 steps was measured when the kettle approach still came down on the top
        # handle: that collision shoved the kettle most of the way to its goal for free,
        # so transport was short and the arm receded from a shallow pose.  A collision-free
        # descent, a decelerated push that actually delivers the body, and the resulting
        # deeper recede cost about 120 steps more (measured seed 0: 129 -> 246).
        assert steps < 260, steps
    env.close()
    print("  expert rollout ............................ ok")


# ---------------------------------------------------------------------------------------

GROUPS = {
    "interface": [
        test_registration,
        test_wrapper_stack,
        test_reset_contract,
        test_step_contract,
        test_expert_action_is_valid,
        test_expert_action_is_memoized_per_step,
        test_repeated_policy_query_is_state_feedback,
        test_expert_survives_unexecuted_recommendations,
    ],
    "policy": [
        test_zero_error_gives_zero_translation,
        test_action_points_at_the_target,
        test_limit_to_box_preserves_direction,
        test_gripper_convention,
        test_ik_target_anti_windup_is_configurable,
        test_side_grasp_frames_follow_live_geometry,
        test_initial_forward_alignment_precedes_task_motion,
        test_action_sanitisation,
        test_completion_predicate_matches_env,
        test_planner_orders_and_reactivates_disturbed_task,
        test_randomized_task_order,
    ],
    "regression": [
        test_two_episodes_still_score,
        test_randomized_microwave_before_kettle_rollout,
    ],
    "rollout": [test_expert_rollout],
}


def main(only=None):
    failures = []
    for group, tests in GROUPS.items():
        if only and group != only:
            continue
        print(f"\n[{group}]")
        for test in tests:
            try:
                test()
            except AssertionError as exc:
                failures.append((test.__name__, exc))
                print(f"  {test.__name__} .... FAILED: {exc}")
    print("\n" + ("ALL PASSED" if not failures else f"{len(failures)} FAILED"))
    for name, exc in failures:
        print(f"  - {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(GROUPS), default=None)
    raise SystemExit(main(**vars(parser.parse_args())))
