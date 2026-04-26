import gymnasium as gym
import gymnasium_robotics
import warnings
import logging
from gymnasium.envs.registration import register
from llfbench.envs.adroit.wrapper import AdroitWrapper
from collections import defaultdict
import random
import time
from gymnasium.wrappers import TimeLimit
import numpy as np
import os
import types

gym.register_envs(gymnasium_robotics)

def custom_reset_model_adroit_hand_hammer_v1(self):
    target_bid = self._model_names.body_name2id["nail_board"]
    self.model.body_pos[target_bid, 2] = self.np_random.uniform(low=0.1, high=0.3)
    self.set_state(self.init_qpos, self.init_qvel)
    return self._get_obs()

def make_env(env_name,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             seed=0,
             warning=True,
             ):
    
    env = gym.make(env_name, max_episode_steps=150, render_mode='rgb_array')
    max_episode_steps = env._max_episode_steps  

    class Wrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)            
            self._render_video = False
            self.visual = visual
            self.enhance_random = False

        def render_video(self, value):
            self._render_video = value
            if value:
                self.env.render_mode = 'rgb_array'
        
        @property
        def env_name(self):
            return env_name
        
        def reset(self, *, seed=None, options=None):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

            if options is not None:
                if options.get("enhance_random", False):
                    if not self.enhance_random:
                        if self.env_name == "AdroitHandHammer-v1":
                            base_env = self.env.unwrapped
                            base_env.reset_model = types.MethodType(custom_reset_model_adroit_hand_hammer_v1, base_env)
                            self.enhance_random = True
                else:
                    base_env = self.env.unwrapped
                    base_env.reset_model = types.MethodType(gymnasium_robotics.envs.adroit_hand.adroit_hammer.AdroitHandHammerEnv.reset_model, base_env)
                    self.enhance_random = False

            return self.env.reset(seed=seed, options=options)
        
    env = Wrapper(env)

    if not warning:
        gym.logger.set_level(gym.logger.ERROR)
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    return TimeLimit(AdroitWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type), max_episode_steps=max_episode_steps)

register(
    id=f"llf-adroit-adroit-hand-door-v1",
    entry_point='llfbench.envs.adroit:make_env',
    kwargs=dict(env_name="AdroitHandDoor-v1", feedback_type='a', instruction_type='b', visual=False, seed=0, warning=True)
)

register(
    id=f"llf-adroit-adroit-hand-hammer-v1",
    entry_point='llfbench.envs.adroit:make_env',
    kwargs=dict(env_name="AdroitHandHammer-v1", feedback_type='a', instruction_type='b', visual=False, seed=0, warning=True)
)

register(
    id=f"llf-adroit-adroit-hand-relocate-v1",
    entry_point='llfbench.envs.adroit:make_env',
    kwargs=dict(env_name="AdroitHandRelocate-v1", feedback_type='a', instruction_type='b', visual=False, seed=0, warning=True)
)