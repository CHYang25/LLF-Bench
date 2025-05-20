import gymnasium as gym
import gymnasium.spaces as spaces
from llfbench.envs.pusht.oracles.base_workspace import BaseWorkspace
from typing import Dict, Callable, List, Union
import numpy as np
import torch
import hydra
import dill
import re

def dict_apply(
        x: Dict[str, Union[torch.Tensor, np.ndarray]], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    if not isinstance(x, dict):
        return func(x)

    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            result[key] = func(value)
        else:
            result[key] = value
    return result

class solvePushT:

    def __init__(self, 
                 env, 
                 checkpoint_dir, 
                 device='cuda', 
                 seed=None, 
                 debug=False, 
                 vis=False,
                 past_action=False):
        
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
        self.n_obs_steps = cfg.n_obs_steps
        self.n_action_steps = cfg.n_action_steps
        self.n_latency_steps = cfg.n_latency_steps
        self.past_action = past_action

    def get_action(self, obs, past_action=None):
        obs = obs.copy()
        obs = obs.reshape(1, 1, -1)
        Do = obs.shape[-1] // 2
        # create obs dict
        np_obs_dict = {
            # handle n_latency_steps by discarding the last n_latency_steps
            'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
            'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
        }
        if self.past_action and (past_action is not None):
            # TODO: not tested
            np_obs_dict['past_action'] = past_action[
                :,-(self.n_obs_steps-1):].astype(np.float32)
        
        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=self.device))
        
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        
        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())
        # handle latency_steps, we discard the first n_latency_steps actions
            # to simulate latency
        action = np_action_dict['action'][:,self.n_latency_steps:]
        return action
