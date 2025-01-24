if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import click
import pathlib
import numpy as np
import gymnasium as gym
from llfbench.envs.block_pushing.block_pushing.oracles.multimodal_push_oracle import MultimodalOrientedPushOracle
from llfbench.envs.block_pushing.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments.gym_wrapper import GymWrapper
import torch
from PIL import Image

if __name__ == "__main__":
    # env = gym.make(
    #     "llf-blockpushing-BlockPushMultimodal-v0",
    # )
    # env = gym.make(
    #     "BlockPushMultimodal-v0",
    # )
    env = TimeLimit(GymWrapper(BlockPushMultimodal()), duration=350)
    frames = []
    policy = MultimodalOrientedPushOracle(env)
    time_step = env.reset()
    policy_state = policy.get_initial_state(1)
    step_cnt = 0
    while True:
        action_step = policy.action(time_step, policy_state)
        obs = np.concatenate(list(time_step.observation.values()), axis=-1)
        action = action_step.action
        print(time_step.observation['block_translation'])
        print("Obs:", time_step.observation)
        print("Action:", action)
        if time_step.step_type == 2:
            break

        # state = env.wrapped_env().gym.get_pybullet_state()
        time_step = env.step(action)
        frame = env.render()  # a display is required to render
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().detach().numpy().astype(np.uint8)[0]
        frames.append(Image.fromarray(frame))

        step_cnt += 1

    env.close()
    img = frames[0]
    img.save(f"./media/block_pushing_slow.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
    print("Timesteps:", step_cnt)