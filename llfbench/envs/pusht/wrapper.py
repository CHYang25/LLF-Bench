from typing import Dict, SupportsFloat, Union, List
import gymnasium.spaces as spaces
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.pusht.prompts import *
from llfbench.envs.pusht import task_feedback
from llfbench.envs.pusht.utils_prompts.degree_prompts import move_degree_adverb_converter, turn_degree_adverb_converter
from llfbench.envs.pusht.utils_prompts.direction_prompts import move_direction_converter, turn_direction_converter
from llfbench.envs.pusht.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.pusht.utils_prompts.recommend_prompts import move_recommend_templates_sampler, turn_recommend_templates_sampler
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
        self._policy_name = "solvePushTKeypoints" if self.env.env_name=='llf-pusht-keypoints-v0' else "solvePushTImage" # remove version postfix

        if 'fp' in feedback_type:
            self._policy = getattr(solve_policy, self._policy_name)(
                env = env,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ) # Expert policy is a function not class
        else:
            self._policy = None
        self.debug = debug
        self.control_relative_position = False
        self._current_observation = None
        self._prev_expert_action = None
        self._curr_agent_to_block_distance = None
        self._curr_block_to_goal_distance = None
        self._curr_block_to_goal_angle = None
        self._curr_navigation_steps = 0
        self._max_navigation_steps = 10  # maximum steps limit for navigating to the next contact point
        self.is_moved_to_goal = False
        self.is_aligned_with_goal = False
        self.mode = 'rgb_array'

        if self.env.env_name == 'llf-pusht-keypoints-v0':
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
    
    @property
    def current_observation_keypoints(self):
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation[:18]
    
    @property
    def current_agent_position(self):
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation[18:20]

    # auxiliary functions for language feedback
    def expert_action(self):
        # Flatten current observation first
        obs_state = spaces.flatten(self.env.observation_space, self._current_observation)
        expert_action = self.pt_policy.get_action(obs_state)
        return expert_action
    
    def compute_angle_diff(self, goal_angle, block_angle):
        return (goal_angle - block_angle + np.pi) % (2 * np.pi) - np.pi
    
    # step functions for keypoints-based observation
    def _step_keypoints(self, action):
        # Env Step
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        feedback_type = self._feedback_type
        
        # Environment Features
        # previous info
        prev_agent_posi = self.current_agent_position
        prev_block_kps = self.current_observation_keypoints
        prev_block_to_goal_distance = self._curr_block_to_goal_distance
        prev_block_to_goal_angle = self._curr_block_to_goal_angle
        prev_agent_to_block_distance = self._curr_agent_to_block_distance
        prev_is_moved_to_goal = self.is_moved_to_goal
        prev_is_aligned_with_goal = self.is_aligned_with_goal

        # current info
        self._current_observation = observation
        self._curr_block_to_goal_distance = np.linalg.norm(info['goal_pose'][:-1] - info['block_pose'][:-1])
        self._curr_block_to_goal_angle = self.compute_angle_diff(info['goal_pose'][-1], info['block_pose'][-1])
        self._curr_agent_to_block_distance = np.linalg.norm(info['block_pose'][:-1] - info['pos_agent'])
        self.is_moved_to_goal = False if prev_block_to_goal_distance == None else self._curr_block_to_goal_distance < prev_block_to_goal_distance
        self.is_moved_away_from_goal = False if prev_block_to_goal_distance == None else self._curr_block_to_goal_distance > prev_block_to_goal_distance
        self.is_aligned_with_goal =  False if prev_block_to_goal_angle == None else self._curr_block_to_goal_angle < prev_block_to_goal_angle
        self.is_misaligned_with_goal =  False if prev_block_to_goal_angle == None else self._curr_block_to_goal_angle > prev_block_to_goal_angle
        self.agent_move_to_the_block = True if prev_block_to_goal_angle == None else self._curr_agent_to_block_distance < prev_agent_to_block_distance
        self._goal_pose = info['goal_pose']
        self._curr_block_pose = info['block_pose']

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

        # set current total navigation steps as 0 if eef contacts the block
        if info['n_contacts'] > 0 and self._curr_navigation_steps > 0:
            self._curr_navigation_steps = 0
        
        # stage feedback
        if info['n_contacts'] == 0:
            _stage_feedback = self.format(task_feedback.move_to_proper_contact_posi_feedback)
        elif reward == 0:
            _stage_feedback = self.format(task_feedback.move_T_shaped_block_to_the_goal_feedback)
        else:
            _stage_feedback = self.format(task_feedback.align_T_shaped_block_and_the_goal_region_feedback)

        # recommend feedback
        move_direction = move_direction_converter(self._goal_pose[:-1] - self._curr_block_pose[:-1])
        move_degree = move_degree_adverb_converter(self._goal_pose[:-1] - self._curr_block_pose[:-1])

        turn_direction = turn_direction_converter(self._curr_block_to_goal_angle)
        turn_degree = turn_degree_adverb_converter(self._curr_block_to_goal_angle)

        _recommend_feedback = [
            move_recommend_templates_sampler().format(degree=degree, direction=direction)
            for direction, degree in zip(move_direction, move_degree)
        ] + [
            turn_recommend_templates_sampler().format(direction=turn_direction, degree=turn_degree)
        ]

        # reason feedback
        positive_hindsight = True
        if self.is_aligned_with_goal and self.is_moved_to_goal:
            _reason_feedback_move = self.format(task_feedback.move_T_shaped_block_to_the_goal_reason)
            _reason_feedback_align = self.format(task_feedback.align_T_shaped_block_and_the_goal_region_reason)
            _reason_feedback = _reason_feedback_move[:-1] + ' and ' + _reason_feedback_align[17:]
        elif self.is_moved_to_goal:
            _reason_feedback = self.format(task_feedback.move_T_shaped_block_to_the_goal_reason)
        elif self.is_aligned_with_goal:
            _reason_feedback = self.format(task_feedback.align_T_shaped_block_and_the_goal_region_reason)
        elif self.is_moved_away_from_goal:
            positive_hindsight = False
            _reason_feedback = self.format(task_feedback.move_T_shaped_block_away_from_the_goal_reason)
        elif self.is_misaligned_with_goal:
            positive_hindsight = False
            _reason_feedback = self.format(task_feedback.misaligned_T_shaped_block_reason)
        else:
            # determine hp or hn
            # 1. if the previous action is good and the eef leaves, it is bad to navigate to the new contact point
            # 2. Spending too much time navigating is also bad.
            if (prev_is_moved_to_goal or prev_is_aligned_with_goal):
                positive_hindsight = False
                _reason_feedback = self.format(task_feedback.leave_the_proper_contact_posi_reason)
            elif self._curr_navigation_steps > self._max_navigation_steps:
                positive_hindsight = False
                _reason_feedback = self.format(task_feedback.long_navigation_reason)
            else:
                _reason_feedback = self.format(task_feedback.move_to_proper_contact_posi_reason)
            # count navigation steps
            self._curr_navigation_steps += 1

        # Set actions
        if 'fp' in feedback_type:
            expert_action = self.expert_action()
            self._prev_expert_action = expert_action.copy()
        
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))

        if 'hp' in feedback_type and positive_hindsight:
            _feedback = self.concatenate_sentences(
                stage_feedback = _stage_feedback,
                action_feedback = self.format(hp_feedback, reason=_reason_feedback),
                reco_feedback = _recommend_feedback,
                action_positive = True,
            )
            feedback.hp = _feedback

        if 'hn' in feedback_type and not positive_hindsight:
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
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated, truncated, info

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
        obs_dict = {'keypoints': self.current_observation_keypoints, 'agent_posi': self.current_agent_position}
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