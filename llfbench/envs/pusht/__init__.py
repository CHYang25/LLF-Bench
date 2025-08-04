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
             instruction_type = 'b',
             feedback_type = 'a',
             visual = False,
             seed = 0,
             warning = True,
             local_keypoint_map = None, 
             color_map = None,
             keypoint_visible_rate = 1.0,
             agent_keypoints = False,
             legacy = False,
             block_cog = None, 
             damping = None,
             render_size = 96,
             draw_keypoints = False,
             reset_to_state = None,
             render_action = True,
):
    if env_name == "pusht-pusht-keypoints-v0":
        env = PushTKeypointsEnv(
            seed = seed,
            legacy = legacy,
            block_cog = block_cog, 
            damping = damping,
            render_size = render_size,
            keypoint_visible_rate = keypoint_visible_rate, 
            agent_keypoints = agent_keypoints,
            draw_keypoints = draw_keypoints,
            reset_to_state = reset_to_state,
            render_action = render_action,
            local_keypoint_map = local_keypoint_map, 
            color_map = color_map
        )
    # else:  
    #     env = PushTImageEnv(
    #         legacy = False,
    #         block_cog = None, 
    #         damping = None,
    #         render_size = 96,
    #     )

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
        
        def render(self, mode='rgb_array'):
            if self.visual:
                return self.env.render(mode=mode)
            else:
                return None
        
    env = Wrapper(env)
    
    if not warning:
        gym.logger.set_level(gym.logger.ERROR)
        warnings.filterwarnings("ignore")

    return TimeLimit(PushTWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type), max_episode_steps=200)


register(
    id='llf-pusht-pusht-keypoints-v0',
    entry_point='llfbench.envs.pusht:make_env',
    max_episode_steps=200,
    reward_threshold=1.0,
    kwargs=dict(
        env_name = "pusht-pusht-keypoints-v0",
        instruction_type = 'b',
        feedback_type = 'a',
        visual = False,
        seed = 0,
        warning = True,
        local_keypoint_map = None, 
        color_map = None,
        keypoint_visible_rate = 1.0,
        agent_keypoints = False,
        legacy = False,
        block_cog = None, 
        damping = None,
        render_size = 96,
        draw_keypoints = False,
        reset_to_state = None,
        render_action = True,
    )
)

# register(
#     id='llf-pusht-pusht-image-v0',
#     entry_point='llfbench.envs.pusht:make_env',
#     max_episode_steps=200,
#     reward_threshold=1.0,
#     kwargs=dict(
#         env_name="llf-pusht-image-v0", 
#         feedback_type='a', 
#         instruction_type='b', 
#         visual=False, 
#         seed=0, 
#         warning=True, 
#     )
# )