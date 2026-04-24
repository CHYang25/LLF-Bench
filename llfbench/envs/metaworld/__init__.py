import gymnasium as gym
from gymnasium.spaces import Box
import warnings
from gymnasium.envs.registration import register
from llfbench.utils import generate_combinations_dict
from llfbench.envs.metaworld.wrapper import MetaworldWrapper
from collections import defaultdict
import importlib
import metaworld
import random
import time
from gymnasium.wrappers import TimeLimit
import numpy as np

BENCHMARK = metaworld.MT1
ENVIRONMENTS = tuple(BENCHMARK.ENV_NAMES)
#VISUAL = '-visual'

def make_env(env_name,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             seed=0,
             warning=True,
             ):

    """ Make the original env and wrap it with the LLFWrapper. """
    benchmark = BENCHMARK(env_name, seed=seed) # This should pass in the seed for consistent reproduction
    env = benchmark.train_classes[env_name](render_mode=None) #'rgb_array')
    env.camera_name = 'corner'
    class Wrapper(gym.Wrapper):
         # a small wrapper to make sure the task is set
         # and to make the env compatible with the old gym api
        def __init__(self, env):
            super().__init__(env)
            self.env.max_path_length = float('inf')
            # We remove the internal time limit. We will redefine the time limit in the wrapper.
            self._render_video = False
            self.visual = visual
            if visual:
                self.env.render_mode = 'rgb_array'

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
            task = random.choice(benchmark.train_tasks)
            self.env.set_task(task)
            if options is not None and options.get("enhance_random", False):
                if self.env_name == 'box-close-v2':
                    obj_low = (-0.05, 0.48, 0.02)
                    obj_high = (0.05, 0.55, 0.02)
                    goal_low = (-0.1, 0.7, 0.133)
                    goal_high = (0.1, 0.8, 0.133)
                    self.env._random_reset_space = Box(
                        np.hstack((obj_low, goal_low)),
                        np.hstack((obj_high, goal_high)),
                    )
                    self.env._freeze_rand_vec = False
                elif self.env_name == 'sweep-v2':
                    obj_low = (-0.1, 0.59, 0.02)
                    obj_high = (0.1, 0.71, 0.02)
                    self.env._random_reset_space = Box(
                        np.array(obj_low),
                        np.array(obj_high),
                    )
                    self.env._freeze_rand_vec = False
            return self.env.reset(seed=seed, options=options)
        
    env = Wrapper(env)

    if not warning:
        gym.logger.set_level(gym.logger.ERROR)
        warnings.filterwarnings("ignore")

    return TimeLimit(MetaworldWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type), max_episode_steps=30)


for env_name in ENVIRONMENTS:
    # default version (backward compatibility)
    register(
        id=f"llf-metaworld-{env_name}",
        entry_point='llfbench.envs.metaworld:make_env',
        kwargs=dict(env_name=env_name, feedback_type='a', instruction_type='b', visual=False, seed=0, warning=True)
    )