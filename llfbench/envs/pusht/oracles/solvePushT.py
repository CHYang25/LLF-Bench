import gymnasium as gym
import gymnasium.spaces as spaces
from llfbench.envs.pusht.oracles.base_workspace import BaseWorkspace
import numpy as np
import torch
import hydra
import dill
import re

class solvePushT:

    def __init__(self, env, checkpoint_dir, device='cuda', seed=None, debug=False, vis=False):
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        n_obs = spaces.flatten_space(self.observation_space).shape[0]
        n_action = self.action_space.shape[0]

        # load checkpoint
        payload = torch.load(open(checkpoint_dir, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cfg._target_ = re.sub(r'diffusion_policy', 'llmbc', cfg._target_)
        cfg.policy._target_ = re.sub(r'diffusion_policy', 'llmbc', cfg.policy._target_)
        cfg.policy.model._target_ = re.sub(r'diffusion_policy', 'llmbc', cfg.policy.model._target_)
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        self.policy = workspace.model
        
        self.policy.to("cuda")
        self.policy.eval()
        device = torch.device(device)
        self.policy.to(device)

        self.n_obs = n_obs
        self.n_action = n_action
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = self.policy.to(self.device)

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
            state = {'obs': state[:20], 'obj_mask': state[20:]}
            if len(state['obs'].shape) == 1:
                state['obs'] = state['obs'].unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action_dict = self.policy.predict_action(state)
        return action_dict
