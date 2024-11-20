import gymnasium as gym
import mani_skill.envs
from PIL import Image
import torch
import numpy as np

# Tasks could be find here: https://maniskill.readthedocs.io/en/latest/tasks/index.html
env = gym.make(
    "PushCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="rgb_array"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

frames = []
obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    frame = env.render()  # a display is required to render
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().detach().numpy().astype(np.uint8)[0]
    frames.append(Image.fromarray(frame))
env.close()

img = frames[0]
img.save(f"./media/mani_skill.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)