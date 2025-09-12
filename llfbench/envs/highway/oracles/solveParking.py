import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from stable_baselines3 import SAC

from llfbench.envs.maniskill.oracles.rl_trainer.ppo_fast import Agent
import torch
import os

class solveParking:
    def __init__(self, env, device='cuda'):
        self.model = SAC.load(os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 
            'checkpoints/best_model.zip'
        ), env=env)
        self.device = device

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        action, _ = self.model.predict(state, deterministic=True)
        return action
