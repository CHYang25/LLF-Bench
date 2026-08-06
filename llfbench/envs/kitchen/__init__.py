import gymnasium as gym
import gymnasium_robotics
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
DEFAULT_TASKS_TO_COMPLETE = ['microwave', 'kettle', 'light_switch', 'slide_cabinet']

def make_env(env_name,
             tasks_to_complete=None,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             seed=0,
             warning=True,
             control_steps=None,
             robot_noise_ratio=0.0,
             object_noise_ratio=0.0,
             terminate_on_tasks_completed=True,
             policy_config=None,
             fast_expert=None,
             debug=False,
             ):

    # With no speed arguments, use the measured compact-demonstration profile. Explicit
    # control_steps keep the legacy policy unless fast_expert=True is also requested, so
    # control_steps=1 diagnostics and older experiments retain their original dynamics.
    if fast_expert is None:
        fast_expert = (control_steps is None
                       and (policy_config is None
                            or policy_config.microwave_control_steps is not None))
    if control_steps is None:
        control_steps = 15 if fast_expert else 5
    if policy_config is None:
        policy_config = (KitchenPolicyConfig.fast_demo()
                         if fast_expert else KitchenPolicyConfig())

    if tasks_to_complete is None:
        tasks_to_complete = DEFAULT_TASKS_TO_COMPLETE
    tasks_to_complete = list(tasks_to_complete)

    env = gym.make(env_name,
        max_episode_steps=280,
        tasks_to_complete=tasks_to_complete,
        terminate_on_tasks_completed=terminate_on_tasks_completed,
        object_noise_ratio=object_noise_ratio,
        robot_noise_ratio=robot_noise_ratio,
        control_steps=control_steps,
        render_mode='rgb_array',
        default_camera_config={
            "distance": 2.2,
            "azimuth": 70.0,
            "elevation": -35.0,
            "lookat": np.array([-0.2, 0.5, 2.0]),
        },
    )
    max_episode_steps = env._max_episode_steps

    class Wrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)            
            self._render_video = False
            self.visual = visual
            self.enhance_random = False

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

            return self.env.reset(seed=seed, options=options)
        
    env = Wrapper(env)

    if not warning:
        gym.logger.set_level(gym.logger.ERROR)
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    return TimeLimit(KitchenWrapper(env,
                                    instruction_type=instruction_type,
                                    feedback_type=feedback_type,
                                    policy_config=policy_config,
                                    debug=debug),
                     max_episode_steps=max_episode_steps)

register(
    id=f"llf-kitchen-FrankaKitchen-v1",
    entry_point='llfbench.envs.kitchen:make_env',
    kwargs=dict(env_name="FrankaKitchen-v1", tasks_to_complete=DEFAULT_TASKS_TO_COMPLETE,
                feedback_type='a', instruction_type='b', visual=False, seed=0, warning=True)
)
