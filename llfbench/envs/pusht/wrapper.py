from typing import Dict, SupportsFloat, Union
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.pusht import prompts, task_feedback
from llfbench.envs.pusht.utils_prompts.degree_prompts import degree_adverb_converter
from llfbench.envs.pusht.utils_prompts.direction_prompts import direction_converter
from llfbench.envs.pusht.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.pusht.utils_prompts.recommend_prompts import recommend_templates_sampler
# from llfbench.envs.pusht.gains import P_GAINS, TERM_REWARDS
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
        self.debug = debug
        self.control_relative_position = False
        self._current_observation = None
        self._prev_expert_action = None

    @property
    def policy(self): # push-T policy
        return self._policy
    
    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation

    # step functions
    def _step():
        pass

