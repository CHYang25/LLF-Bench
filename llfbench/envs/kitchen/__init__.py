import gymnasium as gym
import gymnasium_robotics
import mujoco
import warnings
import logging
from gymnasium.envs.registration import register
from llfbench.envs.kitchen.wrapper import KitchenWrapper
from llfbench.envs.kitchen.scripted_policy import KitchenPolicyConfig
from collections import defaultdict
import random
import time
from gymnasium.wrappers import TimeLimit
import numpy as np
import os
import types

gym.register_envs(gymnasium_robotics)

#: The classic 4-task relay-policy-learning set. Used whenever the caller does not name
#: tasks; `KitchenEnv.__init__` does `set(tasks_to_complete)`, so passing None raises.
#: Gymnasium-Robotics 1.2.1 spells the goal keys with spaces and merged the left/right
#: burner and hinge variants 1.2.0 scored separately; these are the 1.2.1 names.
DEFAULT_TASKS_TO_COMPLETE = ['microwave', 'kettle', 'light switch', 'slide cabinet']

#: `qpos` slice of the kettle's free joint: 3 translation + 4 quaternion (w, x, y, z).
#:
#: The kettle is the only goal object worth perturbing at reset. The microwave, slide-cabinet
#: and light-switch joints all start at one end of a tight range and are precisely what the
#: agent has to move, so jittering them would start episodes partially complete.
KETTLE_QPOS = slice(23, 30)

#: `qpos` slice of the seven Franka arm hinges. The two finger slides at `qpos[7:9]` are
#: left alone: their range is [0, 0.04] with the gripper closed at 0, so symmetric jitter
#: would clip to a half-degenerate distribution and randomly pre-open the gripper.
ARM_QPOS = slice(0, 7)

def make_env(env_name,
             tasks_to_complete=None,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             seed=0,
             warning=True,
             robot_noise_ratio=0.0,
             object_noise_ratio=0.0,
             terminate_on_tasks_completed=True,
             policy_config=None,
             episode_steps=500,
             debug=False,
             init_noise_arm=0.05,
             init_noise_kettle=0.02,
             init_noise_kettle_yaw=0.0,
             inference_expert_action=True,
             ):
    """Build the language-feedback Franka kitchen.

    ``init_noise_*`` randomize the *reset state*, which ``FrankaKitchen-v1`` otherwise does not
    do at all: ``FrankaEnv.reset_model`` assigns ``qpos = self.init_qpos`` unconditionally, and
    its ``robot_noise_ratio``/``object_noise_ratio`` are *observation* noise applied in
    ``_get_obs``, not state noise.  With no jitter every seed replays one trajectory, so a
    collected dataset holds a single initial state however many episodes are requested, and an
    evaluation over N seeds is really N copies of one rollout.  Pass 0 for the legacy behaviour.

    :param init_noise_arm: half-width, in radians, of the uniform jitter on each of the seven
        arm hinges. Clipped to the joint's own range.
    :param init_noise_kettle: half-width, in metres, of the uniform jitter on the kettle's x
        and y position.
    :param init_noise_kettle_yaw: half-width, in radians, of a yaw rotation composed onto the
        kettle's orientation. Defaults to 0, as the expert's grasp approach is yaw-sensitive.
    :param inference_expert_action: whether to run the scripted expert at all. Pass False to
        skip it, which drops every part of the feedback and of ``info`` derived from it; see
        :class:`llfbench.envs.kitchen.wrapper.KitchenWrapper`.

    The action space is the environment's own ``Box(-1, 1, (9,))`` of normalized joint
    velocities; the wrapper holds each action for ``policy_config.action_repeat`` env steps.
    """

    # The measured compact-demonstration profile. 1.2.0's `control_steps` knob is gone --
    # the env has no internal control loop any more -- so step granularity now lives in the
    # policy config's `action_repeat`, which the wrapper reads.
    if policy_config is None:
        policy_config = KitchenPolicyConfig.fast_demo()

    if tasks_to_complete is None:
        tasks_to_complete = DEFAULT_TASKS_TO_COMPLETE
    tasks_to_complete = list(tasks_to_complete)

    # One wrapper step spends `action_repeat` env steps, so the inner limit -- which counts
    # env steps, and which `gym.make` falls back to the spec's 280 for when passed None --
    # has to be scaled by the largest repeat the policy can ask for. Otherwise it truncates
    # every episode after 280/repeat wrapper steps. The limit that matters is the outer one
    # below, counted in the units the agent actually acts in.
    inner_repeat_bound = max(policy_config.action_repeat,
                             policy_config.microwave_action_repeat or 0)
    env = gym.make(env_name,
        max_episode_steps=episode_steps * inner_repeat_bound,
        tasks_to_complete=tasks_to_complete,
        terminate_on_tasks_completed=terminate_on_tasks_completed,
        object_noise_ratio=object_noise_ratio,
        robot_noise_ratio=robot_noise_ratio,
        render_mode='rgb_array',
        default_camera_config={
            "distance": 2.2,
            "azimuth": 70.0,
            "elevation": -35.0,
            "lookat": np.array([-0.2, 0.5, 2.0]),
        },
    )

    class Wrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._render_video = False
            self.visual = visual
            self.enhance_random = False
            self._init_rng = np.random.default_rng(seed)

        def render_video(self, value):
            self._render_video = value
            # `make_env` always builds the env with render_mode='rgb_array', so there is
            # normally nothing to do.  Assigning to `self.env.render_mode` (as the adroit
            # wrapper this was copied from does) raises AttributeError on gymnasium 0.29,
            # where `render_mode` is a read-only property of the wrapper stack.
            if value and self.env.render_mode != 'rgb_array':
                self.env.unwrapped.render_mode = 'rgb_array'
                self.env.unwrapped.robot_env.render_mode = 'rgb_array'
        
        @property
        def env_name(self):
            return env_name
        
        def reset(self, *, seed=None, options=None):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                self._init_rng = np.random.default_rng(seed)

            observation, info = self.env.reset(seed=seed, options=options)
            if init_noise_arm or init_noise_kettle or init_noise_kettle_yaw:
                observation = self._randomize_initial_state()
            return observation, info

        def _randomize_initial_state(self):
            """Jitter the reset state and recompute the observation it implies.

            See `make_env` for why this is needed at all.  Runs after `KitchenEnv.reset`, which
            has already restored the task bookkeeping, so only the physical state is touched.
            """
            kitchen = self.env.unwrapped
            robot = kitchen.robot_env
            rng = self._init_rng

            qpos = robot.data.qpos.copy()
            qvel = robot.data.qvel.copy()

            if init_noise_arm:
                arm = ARM_QPOS
                lo = robot.model.jnt_range[:7, 0]
                hi = robot.model.jnt_range[:7, 1]
                qpos[arm] = np.clip(
                    qpos[arm] + rng.uniform(-init_noise_arm, init_noise_arm, size=7), lo, hi)

            if init_noise_kettle:
                qpos[KETTLE_QPOS.start:KETTLE_QPOS.start + 2] += rng.uniform(
                    -init_noise_kettle, init_noise_kettle, size=2)

            if init_noise_kettle_yaw:
                # Compose a yaw onto the existing orientation rather than perturbing the four
                # quaternion components independently, which would denormalize it.
                yaw = rng.uniform(-init_noise_kettle_yaw, init_noise_kettle_yaw)
                spin = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                quat = slice(KETTLE_QPOS.start + 3, KETTLE_QPOS.stop)
                rotated = np.empty(4)
                mujoco.mju_mulQuat(rotated, spin, qpos[quat])
                qpos[quat] = rotated

            robot.set_state(qpos, qvel)  # calls mj_forward

            # The arm is position-servoed and `reset_model` left `data.ctrl` at `init_ctrl`.
            # Without this the servo would spend the episode pulling the jittered pose back
            # toward the nominal one.
            robot.data.ctrl[:7] = qpos[ARM_QPOS]

            return kitchen._get_obs(robot._get_obs())


    env = Wrapper(env)

    if not warning:
        gym.logger.set_level(gym.logger.ERROR)
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    return TimeLimit(KitchenWrapper(env,
                                    instruction_type=instruction_type,
                                    feedback_type=feedback_type,
                                    policy_config=policy_config,
                                    debug=debug,
                                    inference_expert_action=inference_expert_action),
                     max_episode_steps=episode_steps)

register(
    id=f"llf-kitchen-FrankaKitchen-v1",
    entry_point='llfbench.envs.kitchen:make_env',
    kwargs=dict(env_name="FrankaKitchen-v1", tasks_to_complete=DEFAULT_TASKS_TO_COMPLETE,
                feedback_type='a', instruction_type='b', visual=False, seed=0, warning=True)
)
