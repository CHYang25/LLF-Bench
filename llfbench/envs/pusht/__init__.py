from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from llfbench.envs.pusht.wrapper import PushTWrapper
import random
import warnings
from llfbench.envs.pusht.pusht.pusht_keypoints_env import PushTKeypointsEnv
from llfbench.envs.pusht.pusht.pusht_image_env import PushTImageEnv

def make_env(env_name,
             instruction_type='b',
             feedback_type='a',
             visual=False,
             seed=0,
             warning=True,
             ):
    if env_name == "llf-pusht-keypoints-v0":
        env = PushTKeypointsEnv(
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map=None, 
            color_map=None
        )
    else:  
        env = PushTImageEnv(
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map=None, 
            color_map=None
        )

    class Wrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._render_video = False
            self.visual = visual

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
    entry_point='llfbench.envs.pusht:make_env',
    max_episode_steps=200,
    reward_threshold=1.0,
    kwargs=dict(
        env_name="llf-pusht-keypoints-v0", 
        feedback_type='a', 
        instruction_type='b', 
        visual=False, 
        seed=0, 
        warning=True, 
    )
)

register(
    id='llf-pusht-image-v0',
    entry_point='llfbench.envs.pusht:make_env',
    max_episode_steps=200,
    reward_threshold=1.0,
    kwargs=dict(
        env_name="llf-pusht-image-v0", 
        feedback_type='a', 
        instruction_type='b', 
        visual=False, 
        seed=0, 
        warning=True, 
    )
)