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

gym.register_envs(gymnasium_robotics)

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