from typing import Dict, SupportsFloat, Union
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.pusht.prompts import *
from llfbench.envs.pusht.utils_prompts.degree_prompts import degree_adverb_converter
from llfbench.envs.pusht.utils_prompts.direction_prompts import direction_converter
from llfbench.envs.pusht.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.pusht.utils_prompts.recommend_prompts import recommend_templates_sampler
import importlib
import json
from textwrap import dedent, indent
import llfbench.envs.pusht.oracles as solve_policy
import torch
import re

# so that we won't get scientific notation
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)

class PushTWrapper(LLFWrapper):
    
    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type, debug: bool = False):
        super().__init__(env, instruction_type, feedback_type)
        # load the scripted policy
        self._policy_name = f"solve{self.env.env_name}"[:-3] # remove version postfix

        # if 'fp' in feedback_type:
        #     self._policy = getattr(solve_policy, self._policy_name)(
        #         env = env,
        #         device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #     ) # Expert policy is a function not class
        # else:
        #     self._policy = None
        self.debug = debug
        self.control_relative_position = False
        self._current_observation = None
        self._prev_expert_action = None

        if self.env.env_name == 'pusht-keypoints-v0':
            self._step = self._step_keypoints
        else:
            self._step = self._step_image

    @property
    def reward_range(self):
        return (0., 1.)
    
    @property
    def pt_env(self):
        return self.env.env

    @property
    def pt_policy(self): # push-T policy
        return self._policy
    
    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation
    
    def current_observation_keypoints(self):
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation[:18]
    
    def current_agent_position(self):
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation[18:20]
    
    # step functions for keypoints-based observation
    def _step_keypoints(self, action):
        # Env Step
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        feedback_type = self._feedback_type
        
        # Environment Features
        prev_agnet_pos = self.current_agent_position()
        prev_block_kps = self.current_observation_keypoints()
        self._current_observation = observation

        # Prerequisites for staging, action, and recommendation feedbacks.
        """
        1. End-effector does not touch the block: Move to the block
        2. End-effector touches the block: Move the block
        3. Success: Block is pushed to the goal
        """

        # Set actions
        if 'fp' in feedback_type:
            expert_action = self.expert_action
            self._prev_expert_action = expert_action.copy()
        
        feedback = Feedback()

        

        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated.cpu().item(), truncated.cpu().item(), info

    # step functions for image-based observation
    def _step_image(self, action):
        # Env Step
        observation, reward, terminated, truncated, info = self.env.step(action)

    def _reset(self, seed=None, options=None):
        # Env Reset
        observation, info = self.env.reset(seed=seed, options=options)
        self._current_observation = observation
        self._prev_expert_action = None
        observation = self._format_obs(self._current_observation)
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        instruction = self.format(pt_instruction, task=task)
        return dict(instruction=instruction, observation=observation, feedback=None), info

    def _format_obs(self, observation):
        text = self.textualize_observation(observation)
        image = (self.env.render().squeeze(0).cpu().numpy() if self.env.visual else None)
        return text if image is None else dict(text=text, image=image)
