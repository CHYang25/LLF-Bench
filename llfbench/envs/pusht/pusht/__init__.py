from gymnasium.envs.registration import register
import llfbench.envs.pusht.pusht

register(
    id='gym-pusht-keypoints-v0',
    entry_point='llfbench.envs.pusht.pusht.pusht_keypoints_env:PushTKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)

register(
    id='gym-pusht-image-v0',
    entry_point='llfbench.envs.pusht.pusht.pusht_image_env:PushTImageEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)