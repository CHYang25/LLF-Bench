import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import random
# from gymnasium.wrappers import TimeLimit
import warnings
import logging
from llfbench.envs.block_pushing.wrapper import BlockPushingWrapper
from llfbench.envs.block_pushing.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from llfbench.envs.block_pushing.utils.gym_to_gymnasium_converter import convert_gym_to_gymnasium_env
from tf_agents.environments.gym_wrapper import GymWrapper
# from tf_agents.environments.wrappers import TimeLimit
from gymnasium.wrappers import TimeLimit

def make_env(env_name,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             seed=0,
             warning=True,
             ):
    env = GymWrapper(BlockPushMultimodal(control_frequency=10.0))

    class Wrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.env.max_path_length = float('inf') # We remove the internal time limit. We will redefine the time limit in the wrapper.
            
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
                # These are for pybullet environments reproduction
                # https://github.com/benelot/pybullet-gym/issues/34
                self.env.seed(seed)
                self.env.action_space.seed(seed)
                self.env.observation_space.seed(seed)
            return self.env.reset() # tf_agents GymWrapper reset has no argument
        
    env = Wrapper(env)
    
    if not warning:
        gym.logger.set_level(gym.logger.ERROR)
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    env = convert_gym_to_gymnasium_env(env)
    return TimeLimit(BlockPushingWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type), max_episode_steps=500)


register(
    id=f"llf-blockpushing-BlockPushMultimodal-v0",
    entry_point='llfbench.envs.block_pushing:make_env',
    kwargs=dict(env_name="BlockPushMultimodal-v0", feedback_type='a', instruction_type='b', visual=False, seed=0, warning=True)
)