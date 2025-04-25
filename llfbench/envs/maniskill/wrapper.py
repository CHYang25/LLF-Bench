import gymnasium.spaces as spaces
from typing import Dict, SupportsFloat, Union
import numpy as np
import torch
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.maniskill.prompts import *
# from llfbench.envs.maniskill.task_prompts import (
    
# )
from llfbench.envs.maniskill.utils_prompts.degree_prompts import degree_adverb_converter
from llfbench.envs.maniskill.utils_prompts.direction_prompts import direction_converter
from llfbench.envs.maniskill.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.maniskill.utils_prompts.recommend_prompts import recommend_templates_sampler
from llfbench.envs.metaworld.gains import P_GAINS, TERM_REWARDS
import mani_skill
import importlib
import json
from textwrap import dedent, indent
import llfbench.envs.maniskill.oracles as solve_policy
import re

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

class ManiskillWrapper(LLFWrapper):
    
    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env, instruction_type, feedback_type)
        # load the scripted policy
        self._policy_name = f"solve{self.env.env_name}"[:-3] # remove version postfix

        if 'fp' in feedback_type:
            self._policy = getattr(solve_policy, self._policy_name)(
                env = env,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ) # Expert policy is a function not class
        else:
            self._policy = None

        assert env.unwrapped.control_mode == "pd_joint_delta_pos"

        # self.p_control_time_out = 20 # timeout of the position tracking (for convergnece of P controller)
        # self.p_control_threshold = 1e-4 # the threshold for declaring goal reaching (for convergnece of P controller)
        self.control_relative_position = False
        self._current_observation = None
        self._prev_expert_action = None

        if self.env.env_name == 'PullCubeTool-v1':
            self._step = self._step_pull_cube_tool
        elif self.env.env_name == 'LiftPegUpright-v1':
            self._step = self._step_lift_peg_upright
        elif self.env.env_name == 'StackCube-v1':
            self._step = self._step_stack_cube
        elif self.env.env_name == 'PegInsertionSide-v1':
            self._step = self._step_peg_insertion_side
        elif self.env.env_name == 'PlugCharger-v1':
            self._step = self._step_plug_charger
        else:
            self._step = self._step_general

    @property
    def ms_policy(self): # maniskill policy
        return self._policy
    
    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        # Observation space: https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/concepts/observation.md
        return self._current_observation

    # Observation attributes
    ...

    # auxiliary functions for language feedback
    @property
    def expert_action(self):
        # Flatten current observation first
        obs_state = spaces.flatten(self.env.observation_space, self._current_observation)
        expert_action = self.ms_policy.get_action(obs_state).squeeze(0).cpu().detach().numpy()
        return expert_action


    # step functions
    def _step_general(self, action):
        pass

    def _step_pull_cube_tool(self, action):
        pass

    def _step_lift_peg_upright(self, action):
        pass

    def _step_stack_cube(self, action):
        pass

    def _step_peg_insertion_side(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        feedback_type = self._feedback_type
        
        self._current_observation = observation

        if 'fp' in feedback_type:
            expert_action = self.expert_action
            self._prev_expert_action = expert_action.copy()

        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type:
            _feedback = None
            feedback.hp = _feedback
        if 'hn' in feedback_type:
            _feedback = None
            feedback.hn = _feedback
        if 'fp' in feedback_type:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))

        observation = self._format_obs(observation)
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated.cpu().item(), truncated.cpu().item(), info

    def _step_plug_charger(self, action):
        pass

    def _reset(self, *, seed = None, options = None):
        self._current_observation, info = self.env.reset(seed=seed, options=options)
        if 'fp' in self._feedback_type:
            self._prev_expert_action = self.expert_action
        observation = self._format_obs(self._current_observation)
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        instruction = self.format(ms_instruction, task=task)
        info['success'] = False
        info['video'] = [self.env.render()[::-1]] if self.env._render_video else None
        return dict(instruction=instruction, observation=observation, feedback=None), info
    
    def _format_obs(self, observation):
        text = self.textualize_observation(observation)
        image = (self.env.render().squeeze(0).cpu().numpy() if self.env.visual else None)
        return text if image is None else dict(text=text, image=image)

    def textualize_expert_action(self, action):
        """ Parse action into text. """
        # The idea is to return something like
        # f"delta x: {action[0]:.2f}, delta y:{action[1]:.2f}, delta z:{action[2]:.2f}, gripper state:{action[3]:.1f}"
        # or another action text format if the action isn't a delta.
        # TODO should not be the raw action
        return np.array2string(action)

    def textualize_observation(self, observation):
        """ Parse np.ndarray observation into text. """
        obs_dict = {}
        observation = observation.copy()
        # convert np.ndarray to list
        for k,v in observation.items():
            if isinstance(v, dict):
                for vk,vv in v.items():
                    assert not isinstance(vv, dict)
                    if isinstance(vv, np.ndarray):
                        obs_dict[vk] = np.array2string(v)
                    elif isinstance(vv, torch.Tensor):
                        obs_dict[vk] = str(vv.flatten().tolist()).replace(',', '')
                    else: # it's a scalar
                        obs_dict[vk] = f"{vv}"
                    
            elif isinstance(v, np.ndarray):
                obs_dict[k] = np.array2string(v)
            elif isinstance(v, torch.Tensor):
                obs_dict[k] = str(v.flatten().tolist()).replace(',', '')
            else: # it's a scalar
                obs_dict[k] = f"{v}"
        observation_text = json.dumps(obs_dict)
        return observation_text