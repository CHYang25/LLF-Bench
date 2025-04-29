import gymnasium.spaces as spaces
from typing import Dict, SupportsFloat, Union, List
import numpy as np
import torch
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.maniskill.prompts import *
from llfbench.envs.maniskill.task_prompts import (
    peg_insertion_side_prompts,
)
from llfbench.envs.maniskill.utils_prompts.degree_prompts import move_degree_adverb_converter, turn_degree_adverb_converter
from llfbench.envs.maniskill.utils_prompts.direction_prompts import move_direction_converter, turn_direction_converter
from llfbench.envs.maniskill.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.maniskill.utils_prompts.recommend_prompts import move_recommend_templates, turn_recommend_templates
from llfbench.envs.maniskill.utils import quaternion_angle, quaternion_rotation_difference
import mani_skill
import importlib
import json
from textwrap import dedent, indent
import llfbench.envs.maniskill.oracles as solve_policy
import re

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

COMPARE_THRESHOLD = 1e-4

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
        elif self.env.env_name == 'StackCube-v1':
            self._step = self._step_stack_cube
        elif self.env.env_name == 'PegInsertionSide-v1':
            self._step = self._step_peg_insertion_side
            self.peg_reached_threshold = 5e-2
            self.peg_moved_threshold = 1e-6
            self.peg_aligned_threshold = 1e-1 
            self.box_hole_peg_reached_threshold = 2.5e-1 
            self.box_hole_peg_inserted_threshold = 1.25e-1
        elif self.env.env_name == 'PlugCharger-v1':
            self._step = self._step_plug_charger
        elif self.env.env_name == 'PushT-v1':
            self._step = self._step_pusht
        elif self.env.env_name == 'RollBall-v1':
            self._step = self._step_roll_ball
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
    @property
    def current_q_pos(self):
        """ This is the current joint positions. """
        return self._current_observation['agent']['q_pos']
    
    @property
    def current_q_vel(self):
        """ This is the current joint velocities. """
        return self._current_observation['agent']['q_vel']
    
    @property
    def current_tcp_pose(self):
        """ The gripper pose of the panda arm. """
        return self._current_observation['extra']['tcp_pose']
        
    @property
    def current_peg_pose(self):
        """ 
        The peg pose for the following tasks:
        - PegInsertionSide-v1
        """
        return self._current_observation['extra']['peg_pose']
    
    @property
    def current_peg_half_size(self):
        """ 
        The peg half size for the following tasks:
        - PegInsertionSide-v1
        """
        return self._current_observation['extra']['peg_half_size']
    
    @property
    def current_box_hole_pose(self):
        """ 
        The box hole pose for the following tasks:
        - PegInsertionSide-v1
        """
        return self._current_observation['extra']['box_hole_pose']

    @property
    def current_box_hole_radius(self):
        """ 
        The box hole radius for the following tasks:
        - PegInsertionSide-v1
        """
        return self._current_observation['extra']['box_hole_radius']
    
    @property
    def current_goal_pos(self):
        """ 
        The goal position for the following tasks:
        - PushT-v1: The goal T shaped position
        """
        return self._current_observation['extra']['goal_pos']
    
    @property
    def current_obj_pose(self):
        """ 
        The object position for the following tasks:
        - PushT-v1: The position of the T-shaped object 
        """
        return self._current_observation['extra']['obj_pose']
    
    @property
    def current_ball_pose(self):
        """
        The ball pose for the following tasks:
        - Rollball-v1
        """
        return self._current_observation['extra']['ball_pose']
    
    @property
    def current_ball_vel(self):
        """
        The ball velocity for the following tasks:
        - Rollball-v1
        """
        return self._current_observation['extra']['ball_vel']
    
    @property
    def current_tcp_to_ball_pos(self):
        """
        The relative position from the gripper to the ball for the following tasks:
        - Rollball-v1
        """
        return self._current_observation['extra']['tcp_to_ball_pos']
    
    @property
    def current_ball_to_goal_pos(self):
        """
        The relative position from the ball to the goal for the following tasks:
        - Rollball-v1
        """
        return self._current_observation['extra']['ball_to_goal_pos']

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
        # Env Step
        observation, reward, terminated, truncated, info = self.env.step(action)

        feedback_type = self._feedback_type

        # Environment Features
        prev_tcp_pose = self.current_tcp_pose.squeeze(0)
        prev_peg_pose = self.current_peg_pose.squeeze(0)

        self._current_observation = observation

        current_tcp_pose = self.current_tcp_pose.squeeze(0)
        current_peg_pose = self.current_peg_pose.squeeze(0)

        peg_half_size = self.current_peg_half_size.squeeze(0)
        box_hole_pose = self.current_box_hole_pose.squeeze(0)
        box_hole_radius = self.current_box_hole_radius.squeeze(0)
        
        # Prerequisites for staging, action, and recommendation feedbacks.
        """
        1. Peg Not gripped: Move to peg
        2. Peg gripped: Peg gripped, lift up and align the peg with the hole
        3. Peg Aligned: Move the peg to the hole
        4. Peg reached the hole: Rotate the wrist to align the peg to the hole.
        5. Peg Aligned and Peg reached the hole: Insert the peg into the hole firmly.
        """
        current_tcp_to_peg_pos = current_peg_pose[:3] - current_tcp_pose[:3]
        prev_tcp_to_peg_pos = prev_peg_pose[:3] - prev_tcp_pose[:3]
        current_tcp_to_peg_dist = torch.norm(current_tcp_to_peg_pos)
        prev_tcp_to_peg_dist = torch.norm(prev_tcp_to_peg_pos)

        peg_reached = torch.norm(current_peg_pose[:3] - prev_peg_pose[:3]) > self.peg_moved_threshold \
                        and current_tcp_to_peg_dist < self.peg_reached_threshold 
        peg_gripped = peg_reached and action[-1] < 0 # Gripper grasp, the last dimension of action is negative

        current_peg_hole_angle = abs(quaternion_angle(current_peg_pose[-4:], box_hole_pose[-4:]))
        prev_peg_hole_angle = abs(quaternion_angle(prev_peg_pose[-4:], box_hole_pose[-4:]))
        current_peg_hole_euler_rot_diff = quaternion_rotation_difference(current_peg_pose[-4:], box_hole_pose[-4:])
        prev_peg_hole_euler_rot_diff = quaternion_rotation_difference(prev_peg_pose[-4:], box_hole_pose[-4:])
        peg_aligned = peg_gripped and current_peg_hole_angle < self.peg_aligned_threshold

        current_box_hole_to_peg_pos = current_peg_pose[:3] - box_hole_pose[:3]
        prev_box_hole_to_peg_pos = prev_peg_pose[:3] - box_hole_pose[:3]
        current_box_hole_peg_dist = torch.norm(current_box_hole_to_peg_pos)
        prev_box_hole_peg_dist = torch.norm(prev_box_hole_to_peg_pos)
        box_hole_peg_reached = peg_gripped and current_box_hole_peg_dist < self.box_hole_peg_reached_threshold
        box_hole_peg_inserted = peg_aligned and box_hole_peg_reached and current_box_hole_peg_dist < self.box_hole_peg_inserted_threshold

        if not peg_reached:
            _stage_feedback = self.format(peg_insertion_side_prompts.reach_peg_feedback)

            moving_away = current_tcp_to_peg_dist > prev_tcp_to_peg_dist
            _reason_feedback = self.format(peg_insertion_side_prompts.moving_away_from_peg_reason) \
                                if moving_away else self.format(peg_insertion_side_prompts.moving_to_peg_reason)
            close_gripper = False

            moving_away_axis = [cttpp > pttpp + COMPARE_THRESHOLD for cttpp, pttpp in zip(current_tcp_to_peg_pos, prev_tcp_to_peg_pos)]
            moving_away_direction = move_direction_converter(prev_tcp_to_peg_pos - current_tcp_to_peg_pos)
            moving_away_degree = move_degree_adverb_converter(prev_tcp_to_peg_pos - current_tcp_to_peg_pos)

            turning_away_axis = [False, False, False]
            turning_away_direction = ["", "", ""]
            turning_away_degree = ["", "", ""]
        else:
            if not peg_gripped:
                _stage_feedback = self.format(peg_insertion_side_prompts.grip_peg_feedback)

                moving_away = current_tcp_to_peg_dist > prev_tcp_to_peg_dist
                _reason_feedback = self.format(peg_insertion_side_prompts.moving_away_from_peg_reason) \
                                    if moving_away else self.format(peg_insertion_side_prompts.moving_to_peg_reason)
                close_gripper = True

                moving_away_axis = [cttpp > pttpp + COMPARE_THRESHOLD for cttpp, pttpp in zip(torch.abs(current_tcp_to_peg_pos), torch.abs(prev_tcp_to_peg_pos))]
                moving_away_direction = move_direction_converter(prev_tcp_to_peg_pos - current_tcp_to_peg_pos)
                moving_away_degree = move_degree_adverb_converter(prev_tcp_to_peg_pos - current_tcp_to_peg_pos)

                turning_away_axis = [False, False, False]
                turning_away_direction = ["", "", ""]
                turning_away_degree = ["", "", ""]
            else:
                if not peg_aligned and not box_hole_peg_reached:
                    _stage_feedback = self.format(peg_insertion_side_prompts.align_peg_and_reach_hole_feedback)

                    # If it's not aligned and not reached, then if the current action helps in either way, it shouldn't be moving away
                    moving_away = current_peg_hole_angle > prev_peg_hole_angle and current_box_hole_peg_dist > prev_box_hole_peg_dist
                    _reason_feedback = self.format(peg_insertion_side_prompts.moving_away_from_hole_and_misalign_hole_reason) \
                                        if moving_away else self.format(peg_insertion_side_prompts.moving_to_hole_and_align_hole_reason)
                    close_gripper = action[-1] > 0

                    moving_away_axis = [cbhpp > pbhpp + COMPARE_THRESHOLD for cbhpp, pbhpp in zip(torch.abs(current_box_hole_to_peg_pos), torch.abs(prev_box_hole_to_peg_pos))]
                    moving_away_direction = move_direction_converter(prev_box_hole_to_peg_pos - current_box_hole_to_peg_pos)
                    moving_away_degree = move_degree_adverb_converter(prev_box_hole_to_peg_pos - current_box_hole_to_peg_pos)

                    turning_away_axis = [cpherd > ppherd + COMPARE_THRESHOLD for cpherd, ppherd in zip(torch.abs(current_peg_hole_euler_rot_diff), torch.abs(prev_peg_hole_euler_rot_diff))]
                    turning_away_direction = turn_direction_converter(prev_peg_hole_euler_rot_diff - current_peg_hole_euler_rot_diff)
                    turning_away_degree = turn_degree_adverb_converter(prev_peg_hole_euler_rot_diff - current_peg_hole_euler_rot_diff)
                elif not peg_aligned:
                    _stage_feedback = self.format(peg_insertion_side_prompts.align_peg_feedback)

                    moving_away = current_peg_hole_angle > prev_peg_hole_angle
                    _reason_feedback = self.format(peg_insertion_side_prompts.misalign_hole_reason) \
                                        if moving_away else self.format(peg_insertion_side_prompts.align_hole_reason)
                    close_gripper = action[-1] > 0

                    moving_away_axis = [False, False, False]
                    moving_away_direction = ["", "", ""]
                    moving_away_degree = ["", "", ""]

                    turning_away_axis = [cpherd > ppherd + COMPARE_THRESHOLD for cpherd, ppherd in zip(torch.abs(current_peg_hole_euler_rot_diff), torch.abs(prev_peg_hole_euler_rot_diff))]
                    turning_away_direction = turn_direction_converter(prev_peg_hole_euler_rot_diff - current_peg_hole_euler_rot_diff)
                    turning_away_degree = turn_degree_adverb_converter(prev_peg_hole_euler_rot_diff - current_peg_hole_euler_rot_diff)
                elif not box_hole_peg_reached:
                    _stage_feedback = self.format(peg_insertion_side_prompts.reach_hole_feedback)

                    moving_away = current_box_hole_peg_dist > prev_box_hole_peg_dist
                    _reason_feedback = self.format(peg_insertion_side_prompts.moving_away_from_hole_reason) \
                                        if moving_away else self.format(peg_insertion_side_prompts.moving_to_hole_reason)
                    close_gripper = action[-1] > 0

                    moving_away_axis = [cbhpp > pbhpp + COMPARE_THRESHOLD for cbhpp, pbhpp in zip(torch.abs(current_box_hole_to_peg_pos), torch.abs(prev_box_hole_to_peg_pos))]
                    moving_away_direction = move_direction_converter(prev_box_hole_to_peg_pos - current_box_hole_to_peg_pos)
                    moving_away_degree = move_degree_adverb_converter(prev_box_hole_to_peg_pos - current_box_hole_to_peg_pos)

                    turning_away_axis = [False, False, False]
                    turning_away_direction = ["", "", ""]
                    turning_away_degree = ["", "", ""]
                else:
                    if not box_hole_peg_inserted:
                        _stage_feedback = self.format(peg_insertion_side_prompts.insert_peg_to_hole_feedback)

                        moving_away = current_box_hole_peg_dist > prev_box_hole_peg_dist
                        _reason_feedback = self.format(peg_insertion_side_prompts.pulling_out_from_hole_reason) \
                                            if moving_away else self.format(peg_insertion_side_prompts.inserting_into_hole_reason)
                        close_gripper = action[-1] > 0

                        moving_away_axis = [cbhpp > pbhpp + COMPARE_THRESHOLD for cbhpp, pbhpp in zip(torch.abs(current_box_hole_to_peg_pos), torch.abs(prev_box_hole_to_peg_pos))]
                        moving_away_direction = move_direction_converter(prev_box_hole_to_peg_pos - current_box_hole_to_peg_pos)
                        moving_away_degree = move_degree_adverb_converter(prev_box_hole_to_peg_pos - current_box_hole_to_peg_pos)

                        turning_away_axis = [False, False, False]
                        turning_away_direction = ["", "", ""]
                        turning_away_degree = ["", "", ""]
                    else:
                        _stage_feedback = self.format(peg_insertion_side_prompts.succeed_feedback)

                        moving_away = False
                        _reason_feedback = ""
                        close_gripper = False

                        moving_away_axis = [False, False, False]
                        moving_away_direction = ["", "", ""]
                        moving_away_degree = ["", "", ""]

                        turning_away_axis = [False, False, False]
                        turning_away_direction = ["", "", ""]
                        turning_away_degree = ["", "", ""]

        _recommend_feedback = [
            self.format(move_recommend_templates, direction=direction, degree=degree)
            for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree) if away
        ] + [
            self.format(turn_recommend_templates, direction=direction, degree=degree)
            for away, direction, degree in zip(turning_away_axis, turning_away_direction, turning_away_degree) if away
        ]

        # Set actions
        if 'fp' in feedback_type:
            expert_action = self.expert_action
            self._prev_expert_action = expert_action.copy()

        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type and not moving_away:
            _feedback = self.concatenate_sentences(
                stage_feedback = _stage_feedback,
                action_feedback = self.format(hp_feedback, reason=_reason_feedback),
                reco_feedback = _recommend_feedback,
                action_positive = True,
                gripper_feedback = self.format(close_gripper_feedback) if close_gripper else None
            )
            feedback.hp = _feedback
        if 'hn' in feedback_type and moving_away:
            _feedback = self.concatenate_sentences(
                stage_feedback = _stage_feedback,
                action_feedback = self.format(hn_feedback, reason=_reason_feedback),
                reco_feedback = _recommend_feedback,
                action_positive = False,
                gripper_feedback = self.format(close_gripper_feedback) if close_gripper else None
            )
            feedback.hn = _feedback
        if 'fp' in feedback_type:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))

        observation = self._format_obs(observation)
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated.cpu().item(), truncated.cpu().item(), info

    def _step_plug_charger(self, action):
        pass

    def _step_pusht(self, action):
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

    def _step_roll_ball(self, action):
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
    
    def concatenate_sentences(
        self,
        stage_feedback: str, 
        action_feedback: str, 
        reco_feedback: List[str], 
        action_positive: bool,
        gripper_feedback: str = None):

        res = stage_feedback
        res += (positive_conjunctions_sampler() if action_positive else negative_conjunctions_sampler()) + action_feedback
        if gripper_feedback:
            res += positive_conjunctions_sampler() + gripper_feedback

        for rec in reco_feedback:
            res += positive_conjunctions_sampler() + rec

        return res