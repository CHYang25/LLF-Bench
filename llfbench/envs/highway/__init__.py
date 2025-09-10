import random
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from llfbench.utils import generate_combinations_dict
from llfbench.envs.highway.wrapper import HighwayWrapper
from gymnasium.wrappers import TimeLimit
import sys

ENVIRONMENTS = (
    'parking-v0',
)


def make_env(env_name,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             seed=0,
             warning=True):
    # assert visual == False, "The highway environment has no visual observations"
    sys.path.insert(0, '/content/HighwayEnv/scripts/')

    if visual:
        from gymnasium.wrappers import RecordVideo
        env = gym.make(env_name, render_mode='rgb_array')
        env = RecordVideo(
            env, video_folder='./media/', episode_trigger=lambda e: False
        )
    else:
        env = gym.make(env_name)

    class Wrapper(gym.Wrapper):
         # a small wrapper to make sure the task is set
         # and to make the env compatible with the old gym api
        def __init__(self, env):
            super().__init__(env)
            self.env.max_path_length = float('inf')
            # We remove the internal time limit. We will redefine the time limit in the wrapper.
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
    
    return TimeLimit(HighwayWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type), max_episode_steps=1000)

for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"llf-highway-{env_name}",
        entry_point='llfbench.envs.highway:make_env',
        kwargs=dict(env_name=env_name, feedback_type='a', instruction_type='b', visual=False)
    )