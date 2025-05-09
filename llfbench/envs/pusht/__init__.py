from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from llfbench.envs.pusht.wrapper import PushTWrapper
import random
import warnings
import llfbench.envs.pusht

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
        control_mode="pd_joint_delta_pos", # This is fixed
        num_envs=1, 
        render_mode="rgb_array",
    )

    class Wrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._render_video = False
            self.visual = visual
            assert self.env.render_mode == "rgb_array"

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

    return TimeLimit(PushTWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type), max_episode_steps=200)


register(
    id='llf-pusht-keypoints-v0',
    entry_point='llfbench.envs.pusht.pusht_keypoints_env:PushTKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0,
)