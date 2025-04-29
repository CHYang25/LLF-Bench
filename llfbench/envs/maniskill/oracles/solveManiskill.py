import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import sapien

from llfbench.envs.maniskill.oracles.rl_trainer.ppo_fast import Agent
import torch
import os

class solveManiskill:

    def __init__(self, env, checkpoint_dir, device='cuda', seed=None, debug=False, vis=False):
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        n_obs = spaces.flatten_space(self.observation_space).shape[0]
        n_action = self.action_space.shape[0]

        self.model = Agent(
            n_obs=n_obs,
            n_act=n_action,
            device=device
        )
        self.model.load_state_dict(torch.load(os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 
            checkpoint_dir
        )))

        self.n_obs = n_obs
        self.n_action = n_action
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        return self.model.actor_mean(state.view(-1, self.n_obs))
