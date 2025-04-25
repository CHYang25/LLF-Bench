import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import sapien

from mani_skill.envs.tasks import PegInsertionSideEnv
from llfbench.envs.maniskill.oracles.motionplanner_oracle import PandaArmMotionPlanningOracle
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from enum import auto, Enum
from llfbench.envs.maniskill.oracles.rl_trainer.ppo_fast import Agent
import torch
import os

class PegInsertionSideStages(Enum):
    REACH = auto()
    GRASP = auto()
    ALIGN_PEG = auto()
    INSERT = auto()

class solvePegInsertionSide:

    def __init__(self, env:PegInsertionSideEnv, device='cuda', seed=None, debug=False, vis=False):
        self.env = PegInsertionSideEnv

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
            "rl_trainer/runs/ppo-PegInsertionSide-v1-state-42-walltime_efficient/ckpt_2151.pt"
        )))

        self.stage = PegInsertionSideStages.REACH
        self.n_obs = n_obs
        self.n_action = n_action
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        return self.model.actor_mean(state.view(-1, self.n_obs))
