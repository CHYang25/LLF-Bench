from typing import Dict, SupportsFloat, Union, List
import gymnasium.spaces as spaces
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.pusht.prompts import *
from llfbench.envs.pusht import task_feedback
from llfbench.envs.pusht.utils_prompts.degree_prompts import degree_adverb_converter
from llfbench.envs.pusht.utils_prompts.direction_prompts import direction_converter
from llfbench.envs.pusht.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.pusht.utils_prompts.recommend_prompts import recommend_templates_sampler, recommend_templates
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
        self._curr_block_to_goal_distance = 1e8
        self._curr_block_to_goal_angle = np.pi
        self.mode = 'rgb_array'

        if self.env.env_name == 'llf-pusht-keypoints-v0':
            self._step = self._step_keypoints
            print("Using keypoints-based observation")
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
    
    @property
    def _current_observation_keypoints(self):
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation[:18]
    
    @property
    def _current_agent_position(self):
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation[18:20]

    # auxiliary functions for language feedback
    @property
    def expert_action(self):
        # Flatten current observation first
        obs_state = spaces.flatten(self.env.observation_space, self._current_observation)
        expert_action = self.ms_policy.get_action(obs_state).squeeze(0).cpu().detach().numpy()
        return expert_action
    
    # step functions for keypoints-based observation
    def _step_keypoints(self, action):
        # Env Step
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        feedback_type = self._feedback_type
        
        # Environment Features
        prev_agent_posi = self._current_agent_position
        prev_block_kps = self._current_observation_keypoints
        prev_block_to_goal_distance = self._curr_block_to_goal_distance
        prev_block_to_goal_angle = self._curr_block_to_goal_angle
        self._current_observation = observation
        self._curr_block_to_goal_distance = np.linalg.norm(info['block_pose'][:-1] - info['goal_pose'][:-1])
        self._curr_block_to_goal_angle = min(info['goal_pose'][-1] - info['block_pose'][-1], 2 * np.pi - abs(info['goal_pose'][-1] - info['block_pose'][-1]))
        is_moved_to_goal = self._curr_block_to_goal_distance < prev_block_to_goal_distance
        is_aligned_with_goal = self._curr_block_to_goal_angle < prev_block_to_goal_angle

        # Prerequisites for staging, action, and recommendation feedbacks.
        """
        Stage feedback:
        1. End-effector moves the block towards the goal
        2. End-effector aligns the block with the goal region
        3. Success: achieve the desired coverage

        Recommendation feedback:
        1. Tranlate the block towards the goal
        2. Rotate the block towards the goal
        """
        
        if reward == 0:
            _stage_feedback = self.format(task_feedback.move_T_shaped_block_to_the_goal_feedback)
        else:
            _stage_feedback = self.format(task_feedback.align_T_shaped_block_and_the_goal_region_feedback)


        if is_moved_to_goal or is_aligned_with_goal:
            moving_axis = [cbhpp > pbhpp for cbhpp, pbhpp in zip(abs(self._current_agent_position), abs(prev_agent_posi))]
            direction = direction_converter(self._current_agent_position - prev_agent_posi)
            degree = degree_adverb_converter(self._current_agent_position - prev_agent_posi)
            _recommend_feedback = [
                self.format(recommend_templates, direction=direction, degree=degree)
                for away, direction, degree in zip(moving_axis, direction, degree) if away
            ]
        else:
            moving_axis = [cbhpp < pbhpp for cbhpp, pbhpp in zip(torch.abs(self._current_agent_position), torch.abs(prev_agent_posi))]
            direction = direction_converter(prev_agent_posi - self._current_agent_position)
            degree = degree_adverb_converter(prev_agent_posi - self._current_agent_position)
            _recommend_feedback = [
                self.format(recommend_templates, direction=direction, degree=degree)
                for away, direction, degree in zip(moving_axis, direction, degree) if away
            ]

        if is_aligned_with_goal and is_moved_to_goal:
            _reason_feedback_move = self.format(task_feedback.move_T_shaped_block_to_the_goal_reason)
            _reason_feedback_align = self.format(task_feedback.align_T_shaped_block_and_the_goal_region_reason)
            _reason_feedback = _reason_feedback_move[:-1] + ' and ' + _reason_feedback_align[17:]
        elif is_moved_to_goal:
            _reason_feedback = self.format(task_feedback.move_T_shaped_block_to_the_goal_reason)
        elif is_aligned_with_goal:
            _reason_feedback = self.format(task_feedback.align_T_shaped_block_and_the_goal_region_reason)
        elif not is_moved_to_goal:
            _reason_feedback = self.format(task_feedback.move_T_shaped_block_away_from_the_goal_reason)
        else:
            _reason_feedback = self.format(task_feedback.misaligned_T_shaped_block_reason)

        # Set actions
        # TODO: use the expert policy to get the action
        if 'fp' in feedback_type:
            expert_action = self.expert_action
            self._prev_expert_action = expert_action.copy()
        
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))

        if 'hp' in feedback_type:
            _feedback = self.concatenate_sentences(
                stage_feedback = _stage_feedback,
                action_feedback = self.format(hp_feedback, reason=_reason_feedback),
                reco_feedback = _recommend_feedback,
                action_positive = True,
            )
            feedback.hp = _feedback

        if 'hn' in feedback_type:
            _feedback = self.concatenate_sentences(
                stage_feedback = _stage_feedback,
                action_feedback = self.format(hn_feedback, reason=_reason_feedback),
                reco_feedback = _recommend_feedback,
                action_positive = False,
            )
            feedback.hn = _feedback

        if 'fp' in feedback_type:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))

        observation = self._format_obs()
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated.cpu().item(), truncated.cpu().item(), info

    # step functions for image-based observation
    def _step_image(self, action):
        # Env Step
        observation, reward, terminated, truncated, info = self.env.step(action)

        #TODO: implement feedback for image-based observation

    def _reset(self, seed=None, options=None):
        # Env Reset
        observation, info = self.env.reset(seed=seed, options=options)
        self._current_observation = observation
        self._prev_expert_action = None
        observation = self._format_obs()
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        instruction = self.format(pt_instruction, task=task)
        info['success'] = False
        return dict(instruction=instruction, observation=observation, feedback=None), info

    def _format_obs(self):
        obs_dict = {'keypoints': self._current_observation_keypoints, 'agent_posi': self._current_agent_position}
        text = self.textualize_observation(obs_dict)
        image = (self.env.render(mode=self.mode) if self.env.visual else None)
        return text if image is None else dict(text=text, image=image)

    def concatenate_sentences(
        self,
        stage_feedback: str, 
        action_feedback: str, 
        reco_feedback: List[str], 
        action_positive: bool,):

        res = stage_feedback
        res += (positive_conjunctions_sampler() if action_positive else negative_conjunctions_sampler()) + action_feedback

        for rec in reco_feedback:
            res += positive_conjunctions_sampler() + rec

        return res
    
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
    
    def textualize_expert_action(self, action):
        """ Parse action into text. """
        # The idea is to return something like
        # f"delta x: {action[0]:.2f}, delta y:{action[1]:.2f}, delta z:{action[2]:.2f}, gripper state:{action[3]:.1f}"
        # or another action text format if the action isn't a delta.
        # TODO should not be the raw action
        return np.array2string(action)