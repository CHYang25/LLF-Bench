from typing import Dict, SupportsFloat, Union
import numpy as np
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
import mani_skill.examples.motionplanning.panda.solutions as solve_policy
import re

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

class ManiskillWrapper(LLFWrapper):
    
    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env, instruction_type, feedback_type)
        # load the scripted policy
        self._policy_name = f"solve{self.env.env_name}"
        self._policy = getattr(solve_policy, self._policy_name)()

        self.p_control_time_out = 20 # timeout of the position tracking (for convergnece of P controller)
        self.p_control_threshold = 1e-4 # the threshold for declaring goal reaching (for convergnece of P controller)
        self.control_relative_position = False
        self._current_observation = None
        self._prev_expert_action = None

    @property
    def ms_policy(self): # maniskill policy
        return self._policy
    
    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation

