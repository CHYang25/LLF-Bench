import gymnasium as gym
import warnings
import logging
from gymnasium.envs.registration import register
import d4rl
from d4rl.pointmaze import maze_model
from llfbench.envs.pointmaze.wrapper import PointmazeWrapper
from llfbench.envs.block_pushing.utils.gym_to_gymnasium_converter import convert_gym_to_gymnasium_env
from collections import defaultdict
import random
import time
from gymnasium.wrappers import TimeLimit
import numpy as np
import os

def make_env(env_name,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             seed=0,
             warning=True,
             ):
    
    import gym as gym_old
    env = gym_old.make(env_name)
    max_episode_steps = env._max_episode_steps  

    class Wrapper(gym_old.Wrapper):
        def __init__(self, env):
            super().__init__(env)            
            self._render_video = False
            self.visual = visual
            self.env.render_mode = 'rgb_array'

        def render_video(self, value):
            self._render_video = value
            if value:
                self.env.render_mode = 'rgb_array'

        @property
        def env_name(self):
            return env_name
        
        # using seeding env.seed somehow would cause the q_pos and target in the same grid
        # rewrite the set_target function manually with np
        def set_target(self, target_location=None):
            if target_location is None:
                idx = np.random.choice(len(self.env.empty_and_goal_locations))
                reset_location = np.array(self.env.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
                target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            self.env.unwrapped._target = target_location

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                self.env.seed(seed)
            self.set_target()
            return self.env.reset(seed=seed, options=options, return_info=True) 
            
        def render(self):
            return self.env.render(mode=self.env.render_mode)
        
    env = Wrapper(env)

    if not warning:
        gym.logger.set_level(gym.logger.ERROR)
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    env = convert_gym_to_gymnasium_env(env)
    return TimeLimit(PointmazeWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type), max_episode_steps=max_episode_steps)

register(
    id=f"llf-pointmaze-maze2d-large-dense-v0",
    entry_point='llfbench.envs.pointmaze:make_env',
    kwargs=dict(env_name="maze2d-large-dense-v0", feedback_type='a', instruction_type='b', visual=False, seed=0, warning=True)
)