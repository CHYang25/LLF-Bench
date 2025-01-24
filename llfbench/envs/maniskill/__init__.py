import gymnasium as gym
from gymnasium.envs.registration import register
import mani_skill.envs
from mani_skill.utils.download_demo import DATASET_SOURCES 
import numpy as np
import random
from gymnasium.wrappers import TimeLimit
import warnings
from llfbench.envs.maniskill.wrapper import ManiskillWrapper

def make_env(env_name,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             obs_mode='state',
             seed=0,
             warning=True,
             ):
    env = gym.make(
        env_name,
        obs_mode=obs_mode,
        control_mode="pd_ee_delta_pose", # This is fixed
        num_envs=1, 
    )

    class Wrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)

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

    return TimeLimit(ManiskillWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type), max_episode_steps=30)


# DATA_SOURCES is a dictionary that contains tasks that an expert planning policy is provided by the Maniskill Benchmark
for env_name in DATASET_SOURCES.keys():
    # default version (backward compatibility)
    register(
        id=f"llf-maniskill-{env_name}",
        entry_point='llfbench.envs.maniskill:make_env',
        kwargs=dict(env_name=env_name, feedback_type='a', instruction_type='b', visual=False, obs_mode='state', seed=0, warning=True)
    )