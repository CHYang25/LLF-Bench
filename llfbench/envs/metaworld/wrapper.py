from typing import Dict, SupportsFloat, Union
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.metaworld.prompts import *
from llfbench.envs.metaworld.task_prompts import (
    box_close_v2_prompts,
    door_close_v2_prompts,
    push_v2_prompts,
    push_back_v2_prompts,
    sweep_v2_prompts,
)
from llfbench.envs.metaworld.task_prompts.hp_feedback import hp_feedback as task_hp_feedback
from llfbench.envs.metaworld.task_prompts.hn_feedback import hn_feedback as task_hn_feedback
from llfbench.envs.metaworld.utils_prompts.degree_prompts import degree_adverb_converter
from llfbench.envs.metaworld.utils_prompts.direction_prompts import direction_converter
from llfbench.envs.metaworld.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.metaworld.utils_prompts.recommend_prompts import recommend_templates_sampler
from llfbench.envs.metaworld.gains import P_GAINS, TERM_REWARDS
import metaworld
import importlib
import json
from textwrap import dedent, indent
from metaworld.policies.policy import move
from metaworld.policies.action import Action
from metaworld.policies import SawyerDrawerOpenV1Policy, SawyerDrawerOpenV2Policy, SawyerReachV2Policy
import re

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

class MetaworldWrapper(LLFWrapper):

    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env, instruction_type, feedback_type)
        # load the scripted policy
        if self.env.env_name=='peg-insert-side-v2':
            module = importlib.import_module(f"metaworld.policies.sawyer_peg_insertion_side_v2_policy")
            self._policy_name = f"SawyerPegInsertionSideV2Policy"
            self._policy = getattr(module, self._policy_name)()
        else:
            module = importlib.import_module(f"metaworld.policies.sawyer_{self.env.env_name.replace('-','_')}_policy")
            self._policy_name = f"Sawyer{self.env.env_name.title().replace('-','')}Policy"
            self._policy = getattr(module, self._policy_name)()
        self.p_control_time_out = 20 # timeout of the position tracking (for convergnece of P controller)
        self.p_control_threshold = 1e-4 # the threshold for declaring goal reaching (for convergnece of P controller)
        self.control_relative_position = False
        self._current_observation = None
        self._prev_expert_action = None

        if self.env.env_name == 'box-close-v2':
            self._step = self._step_box_close_v2
            self.lid_gripped_threshold = 1.5e-1
            self.lid_lifted_threshold = 6e-2
            self.initial_lid_pos = None
        elif self.env.env_name == 'door-close-v2':
            self._step = self._step_door_close_v2
            self.door_is_moved_threshold = 5e-1
            self.initial_door_pos = None
        elif self.env.env_name == 'push-v2':
            self._step = self._step_push_v2
            self.puck_is_gripped_threshold = 1e-1
        elif self.env.env_name == 'push-back-v2':
            self._step = self._step_push_back_v2
            self.puck_is_gripped_threshold = 1e-1
        elif self.env.env_name == 'sweep-v2':
            self._step = self._step_sweep_v2
            self.cube_is_gripped_threshold = 1e-1
        else:
            self._step = self._step_general

    @property
    def reward_range(self):
        return (0,10)

    @property
    def mw_env(self):
        return self.env.env

    @property
    def mw_policy(self): # metaworld policy
        return self._policy

    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation

    @property
    def _current_pos(self):
        """ Curret position of the hand. """
        return self.mw_policy._parse_obs(self.current_observation)['hand_pos']
    
    @property
    def _current_puck_pos(self):
        return self.mw_policy._parse_obs(self.current_observation)['puck_pos']
    
    @property
    def _current_door_pos(self):
        return self.mw_policy._parse_obs(self.current_observation)['door_pos']
    
    @property
    def _current_lid_pos(self):
        return self.mw_policy._parse_obs(self.current_observation)['lid_pos']
    
    @property
    def _current_box_pos(self): 
        return self.mw_policy._parse_obs(self.current_observation)['box_pos']
    
    @property
    def _current_cube_pos(self): 
        return self.mw_policy._parse_obs(self.current_observation)['cube_pos']
    
    @property
    def _goal_pos(self):
        return self.mw_policy._parse_obs(self.current_observation)['goal_pos']

    @property
    def expert_action(self):
        """ Compute the desired xyz position and grab effort from the MW scripted policy.

            We want to compute the desired xyz position and grab effort instead of
            the low level action, so we cannot call directly
            self.mw_policy.get_aciton
        """
        # Get the desired xyz position from the MW scripted policy
        if type(self.mw_policy) in [SawyerDrawerOpenV1Policy, SawyerDrawerOpenV2Policy]:
            o_d = self.mw_policy._parse_obs(self.current_observation)
            # NOTE this policy looks different from the others because it must
            # modify its p constant part-way through the task
            pos_curr = o_d["hand_pos"]
            pos_drwr = o_d["drwr_pos"]
            # align end effector's Z axis with drawer handle's Z axis
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
                desired_xyz = pos_drwr + np.array([0.0, 0.0, 0.3])
            # drop down to touch drawer handle
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                desired_xyz = pos_drwr
            # push toward a point just behind the drawer handle
            # also increase p value to apply more force
            else:
                desired_xyz = pos_drwr + np.array([0.0, -0.06, 0.0])
        elif type(self.mw_policy) == SawyerReachV2Policy:
            desired_xyz = self.mw_policy._parse_obs(self.current_observation)['goal_pos']
        else:
            if hasattr(self.mw_policy,'_desired_xyz'):
                compute_goal = self.mw_policy._desired_xyz
            elif hasattr(self.mw_policy,'_desired_pos'):
                compute_goal = self.mw_policy._desired_pos
            elif hasattr(self.mw_policy,'desired_pos'):
                compute_goal = self.mw_policy.desired_pos
            else:
                raise NotImplementedError
            desired_xyz = compute_goal(self.mw_policy._parse_obs(self.current_observation))
        # Get the desired grab effort from the MW scripted policy
        desired_grab = self.mw_policy.get_action(self.current_observation)[-1]  # TODO should be getting the goal
        if self.control_relative_position:
            desired_xyz = desired_xyz - self._current_pos
        return np.concatenate([desired_xyz, np.array([desired_grab])])

    def control_mode(self, mode: str):
        assert mode in ('absolute', 'relative'), "The control mode should be either 'absolute' or 'relative'."
        self.control_relative_position = mode == 'relative'

    def p_control(self, action):
        """ Compute the desired control based on a position target (action[:3])
        using P controller provided in Metaworld."""
        assert len(action)==4, "The action should be a 4D vector."
        p_gain = P_GAINS[type(self.mw_policy)]
        if type(self.mw_policy) in [type(SawyerDrawerOpenV1Policy), type(SawyerDrawerOpenV2Policy)]:
            # This needs special cares. It's implemented differently.
            o_d = self.mw_policy._parse_obs(self.current_observation)
            pos_curr = o_d["hand_pos"]
            pos_drwr = o_d["drwr_pos"]
            # align end effector's Z axis with drawer handle's Z axis
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
                p_gain = 4.0
            # drop down to touch drawer handle
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                p_gain = 4.0
            # push toward a point just behind the drawer handle
            # also increase p value to apply more force
            else:
                p_gain= 50.0

        control = Action({"delta_pos": np.arange(3), "grab_effort": 3})
        control["delta_pos"] = move(self._current_pos, to_xyz=action[:3], p=p_gain)
        control["grab_effort"] = action[3]
        return control.array

    def _step_general(self, action):
        # Run P controller until convergence or timeout
        # action is viewed as the desired position + grab_effort
        previous_pos = self._current_pos  # the position of the hand before moving
        if self.control_relative_position:
            action = action.copy()
            action[:3] += self._current_pos  # turn relative position to absolute position

        video = []
        for _ in range(self.p_control_time_out):
            control = self.p_control(action)  # this controls the hand to move an absolute position
            observation, reward, terminated, truncated, info = self.env.step(control)
            self._current_observation = observation
            desired_pos = action[:3]
            video.append(self.env.render()[::-1] if self.env._render_video else None)
            if np.abs(desired_pos - self._current_pos).max() < self.p_control_threshold:
                break

        feedback_type = self._feedback_type
        # Some pre-computation of the feedback
        expert_action = self.expert_action  # absolute or relative
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action.copy()
        # Target pos is in absolute position
        if self.control_relative_position:
            target_pos = self._prev_expert_action.copy()
            target_pos[:3] += self._current_pos
        else:
            target_pos = self._prev_expert_action
        
        self._prev_expert_action = expert_action.copy()

        # Compute Recommend Target
        if self.control_relative_position:
            recommend_target_pos = expert_action
            recommend_target_pos[:3] += self._current_pos
        else:
            recommend_target_pos = expert_action

        moving_away = np.linalg.norm(target_pos[:3]-previous_pos) < np.linalg.norm(target_pos[:3]-self._current_pos)
        moving_away_axis = [
            target_pos[i] - previous_pos[i] < target_pos[i] - self._current_pos[i]
            for i in range(3)
        ]
        moving_away_direction = direction_converter(target_pos[:3] - self._current_pos)
        moving_away_degree = degree_adverb_converter(target_pos[:3] - self._current_pos)

        if target_pos[3] > 0.5 and action[3] < 0.5:  # the gripper should be closed instead.
            gripper_feedback = self.format(close_gripper_feedback)
        elif target_pos[3] < 0.5 and action[3] > 0.5:  #the gripper should be open instead.
            gripper_feedback = self.format(open_gripper_feedback)
        else:
            gripper_feedback = None
        # Compute feedback
        feedback = Feedback()
        if 'r' in  feedback_type:
            feedback.r = self.format(r_feedback, reward=reward)
        if 'hp' in feedback_type:  # moved closer to the expert goal
            first_conjunction_used = False
            _feedback = self.format(hp_feedback) if not moving_away else None
            if not moving_away:
                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + recommend_templates_sampler().format(degree=degree, direction=direction)
                        first_conjunction_used = True
            
                if gripper_feedback is not None:
                    if _feedback is not None:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + gripper_feedback[0].lower() + gripper_feedback[1:]
                    else:
                        _feedback = gripper_feedback

            feedback.hp = _feedback
        if 'hn' in feedback_type:  # moved away from the expert goal
            # position feedback
            _feedback = self.format(hn_feedback) if moving_away else None
            if moving_away:
                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        _feedback += positive_conjunctions_sampler() + recommend_templates_sampler().format(degree=degree, direction=direction)

                # gripper feedback
                if gripper_feedback is not None:
                    if _feedback is not None:
                        _feedback += positive_conjunctions_sampler() + gripper_feedback[0].lower() + gripper_feedback[1:]
                    else:
                        _feedback = gripper_feedback

            feedback.hn = _feedback
        if 'fp' in feedback_type:  # suggest the expert goal
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(recommend_target_pos))
        observation = self._format_obs(observation)
        info['success'] = bool(info['success'])
        info['video'] = video if self.env._render_video else None
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated or info['success'], truncated, info
    
    def _step_box_close_v2(self, action):
        # Run P controller until convergence or timeout
        # action is viewed as the desired position + grab_effort
        previous_pos = self._current_pos  # the position of the hand before moving
        if self.initial_lid_pos is None:
            self.initial_lid_pos = self._current_lid_pos
            
        if self.control_relative_position:
            action = action.copy()
            action[:3] += self._current_pos  # turn relative position to absolute position

        video = []
        for _ in range(self.p_control_time_out):
            control = self.p_control(action)  # this controls the hand to move an absolute position
            observation, reward, terminated, truncated, info = self.env.step(control)
            self._current_observation = observation
            desired_pos = action[:3]
            video.append(self.env.render()[::-1] if self.env._render_video else None)
            if np.abs(desired_pos - self._current_pos).max() < self.p_control_threshold:
                break

        feedback_type = self._feedback_type
        # Some pre-computation of the feedback
        expert_action = self.expert_action  # absolute or relative
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action.copy()
        # Target pos is in absolute position
        if self.control_relative_position:
            target_pos = self._prev_expert_action.copy()
            target_pos[:3] += self._current_pos
        else:
            target_pos = self._prev_expert_action
        
        self._prev_expert_action = expert_action.copy()

        # Compute Recommend Target
        if self.control_relative_position:
            recommend_target_pos = expert_action
            recommend_target_pos[:3] += self._current_pos
        else:
            recommend_target_pos = expert_action

        moving_away = np.linalg.norm(target_pos[:3]-previous_pos) < np.linalg.norm(target_pos[:3]-self._current_pos)
        moving_away_axis = [
            target_pos[i] - previous_pos[i] < target_pos[i] - self._current_pos[i]
            for i in range(3)
        ]
        moving_away_direction = direction_converter(target_pos[:3] - self._current_pos)
        moving_away_degree = degree_adverb_converter(target_pos[:3] - self._current_pos)

        lid_gripped = np.linalg.norm(self._current_lid_pos - self._current_pos) < self.lid_gripped_threshold\
                        and np.linalg.norm(self._current_lid_pos - self.initial_lid_pos) > 0\
                        and action[3] > 0.5
        
        lid_lifted = self._current_lid_pos[2] > self.lid_lifted_threshold

        if target_pos[3] > 0.5 and action[3] < 0.5:  # the gripper should be closed instead.
            gripper_feedback = self.format(close_gripper_feedback)
        elif target_pos[3] < 0.5 and action[3] > 0.5:  #the gripper should be open instead.
            gripper_feedback = self.format(open_gripper_feedback)
        else:
            gripper_feedback = None
        # Compute feedback
        feedback = Feedback()
        if 'r' in  feedback_type:
            feedback.r = self.format(r_feedback, reward=reward)
        if 'hp' in feedback_type:  # moved closer to the expert goal
            first_conjunction_used = False
            if not moving_away:
                if lid_gripped and lid_lifted:
                    _reason_goal = self.format(box_close_v2_prompts.move_to_box_feedback)
                elif lid_gripped and not lid_lifted:
                    _reason_goal = self.format(box_close_v2_prompts.lift_up_lid_feedback)
                elif not lid_gripped:
                    _reason_goal = self.format(box_close_v2_prompts.move_to_lid_feedback)

                _feedback = _reason_goal + ' ' + positive_conjunctions_sampler() + self.format(task_hp_feedback)

                if gripper_feedback is not None:
                    if _feedback is not None:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + gripper_feedback[0].lower() + gripper_feedback[1:]
                        first_conjunction_used = True
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + recommend_templates_sampler().format(degree=degree, direction=direction)
                        first_conjunction_used = True

            else:
                _feedback = None

            feedback.hp = _feedback 
        if 'hn' in feedback_type:  # moved away from the expert goal
            # position feedback
            if moving_away:
                if lid_gripped and lid_lifted:
                    _reason_goal = self.format(box_close_v2_prompts.move_to_box_feedback)
                elif lid_gripped and not lid_lifted:
                    _reason_goal = self.format(box_close_v2_prompts.lift_up_lid_feedback)
                elif not lid_gripped:
                    _reason_goal = self.format(box_close_v2_prompts.move_to_lid_feedback)

                _feedback = _reason_goal + ' ' + negative_conjunctions_sampler() + self.format(task_hn_feedback)

                # gripper feedback
                if gripper_feedback is not None:
                    if _feedback is not None:
                        _feedback += positive_conjunctions_sampler() + gripper_feedback[0].lower() + gripper_feedback[1:]
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        _feedback += positive_conjunctions_sampler() + recommend_templates_sampler().format(degree=degree, direction=direction)
            
            else:
                _feedback = None

            feedback.hn = _feedback
        if 'fp' in feedback_type:  # suggest the expert goal
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(recommend_target_pos))
        observation = self._format_obs(observation)
        info['success'] = bool(info['success'])
        info['video'] = video if self.env._render_video else None
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated or info['success'], truncated, info
    
    def _step_door_close_v2(self, action):
        # Run P controller until convergence or timeout
        # action is viewed as the desired position + grab_effort
        previous_pos = self._current_pos  # the position of the hand before moving
        if self.initial_door_pos is None:
            self.initial_door_pos = self._current_door_pos

        if self.control_relative_position:
                action = action.copy()
                action[:3] += self._current_pos  # turn relative position to absolute position

        video = []
        for _ in range(self.p_control_time_out):
            control = self.p_control(action)  # this controls the hand to move an absolute position
            observation, reward, terminated, truncated, info = self.env.step(control)
            self._current_observation = observation
            desired_pos = action[:3]
            video.append(self.env.render()[::-1] if self.env._render_video else None)
            if np.abs(desired_pos - self._current_pos).max() < self.p_control_threshold:
                break

        feedback_type = self._feedback_type
        # Some pre-computation of the feedback
        expert_action = self.expert_action  # absolute or relative
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action.copy()
        # Target pos is in absolute position
        if self.control_relative_position:
            target_pos = self._prev_expert_action.copy()
            target_pos[:3] += self._current_pos
        else:
            target_pos = self._prev_expert_action

        self._prev_expert_action = expert_action.copy()

        # Compute Recommend Target
        if self.control_relative_position:
            recommend_target_pos = expert_action
            recommend_target_pos[:3] += self._current_pos
        else:
            recommend_target_pos = expert_action

        # moving away metrics
        moving_away = np.linalg.norm(target_pos[:3]-previous_pos) < np.linalg.norm(target_pos[:3]-self._current_pos)
        moving_away_axis = [
            target_pos[i] - previous_pos[i] < target_pos[i] - self._current_pos[i]
            for i in range(3)
        ]
        moving_away_direction = direction_converter(target_pos[:3] - self._current_pos)
        moving_away_degree = degree_adverb_converter(target_pos[:3] - self._current_pos)

        door_reached = np.linalg.norm(self._current_door_pos - self.initial_door_pos) > self.door_is_moved_threshold
        if target_pos[3] > 0.5 and action[3] < 0.5:  # the gripper should be closed instead.
            gripper_feedback = self.format(close_gripper_feedback)
        elif target_pos[3] < 0.5 and action[3] > 0.5:  #the gripper should be open instead.
            gripper_feedback = self.format(open_gripper_feedback)
        else:
            gripper_feedback = None
        # Compute feedback
        feedback = Feedback()
        if 'r' in  feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type:  # moved closer to the expert goal
            first_conjunction_used = False
            if not moving_away:
                if door_reached:
                    _reason_goal = self.format(door_close_v2_prompts.move_to_goal_feedback)
                else:
                    _reason_goal = self.format(door_close_v2_prompts.move_to_door_feedback)

                _feedback = _reason_goal + ' ' + positive_conjunctions_sampler() + self.format(task_hp_feedback)
                
                if gripper_feedback is not None:
                    if _feedback is not None:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + gripper_feedback[0].lower() + gripper_feedback[1:]
                        first_conjunction_used = True
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + recommend_templates_sampler().format(degree=degree, direction=direction)
                        first_conjunction_used = True
                
            else:
                _feedback = None

            feedback.hp = _feedback
        if 'hn' in feedback_type:  # moved away from the expert goal
            if moving_away:
                if door_reached:
                    _reason_goal = self.format(door_close_v2_prompts.move_to_goal_feedback)
                else:
                    _reason_goal = self.format(door_close_v2_prompts.move_to_door_feedback)

                _feedback = _reason_goal + ' ' + negative_conjunctions_sampler() + self.format(task_hn_feedback)

                # gripper feedback
                if gripper_feedback is not None:
                    if _feedback is not None:
                        _feedback += positive_conjunctions_sampler() + gripper_feedback[0].lower() + gripper_feedback[1:]
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        _feedback += positive_conjunctions_sampler() + recommend_templates_sampler().format(degree=degree, direction=direction)
                
            else:
                _feedback = None

            feedback.hn = _feedback
        if 'fp' in feedback_type:  
            # suggest the expert goal
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(recommend_target_pos))
        observation = self._format_obs(observation)
        info['success'] = bool(info['success'])
        info['video'] = video if self.env._render_video else None
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated or info['success'], truncated, info

    def _step_push_v2(self, action):
        # Run P controller until convergence or timeout
        # action is viewed as the desired position + grab_effort
        previous_pos = self._current_pos  # the position of the hand before moving
        previous_puck_pos = self._current_puck_pos
        if self.control_relative_position:
                action = action.copy()
                action[:3] += self._current_pos  # turn relative position to absolute position

        video = []
        for _ in range(self.p_control_time_out):
            control = self.p_control(action)  # this controls the hand to move an absolute position
            observation, reward, terminated, truncated, info = self.env.step(control)
            self._current_observation = observation
            desired_pos = action[:3]
            video.append(self.env.render()[::-1] if self.env._render_video else None)
            if np.abs(desired_pos - self._current_pos).max() < self.p_control_threshold:
                break

        feedback_type = self._feedback_type
        # Some pre-computation of the feedback
        expert_action = self.expert_action  # absolute or relative
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action.copy()
        # Target pos is in absolute position
        if self.control_relative_position:
            target_pos = self._prev_expert_action.copy()
            target_pos[:3] += self._current_pos
        else:
            target_pos = self._prev_expert_action

        self._prev_expert_action = expert_action.copy()

        # Compute Recommend Target
        if self.control_relative_position:
            recommend_target_pos = expert_action
            recommend_target_pos[:3] += self._current_pos
        else:
            recommend_target_pos = expert_action

        # moving away metrics
        moving_away = np.linalg.norm(target_pos[:3]-previous_pos) < np.linalg.norm(target_pos[:3]-self._current_pos)
        moving_away_axis = [
            target_pos[i] - previous_pos[i] < target_pos[i] - self._current_pos[i]
            for i in range(3)
        ]
        moving_away_direction = direction_converter(target_pos[:3] - self._current_pos)
        moving_away_degree = degree_adverb_converter(target_pos[:3] - self._current_pos)

        puck_gripped = np.linalg.norm(self._current_puck_pos - previous_puck_pos) > 0 \
                        and np.linalg.norm(self._current_puck_pos - self._current_pos) < self.puck_is_gripped_threshold \
                            and action[3] > 0.5
        if target_pos[3] > 0.5 and action[3] < 0.5:  # the gripper should be closed instead.
            gripper_feedback = self.format(close_gripper_feedback)
        elif target_pos[3] < 0.5 and action[3] > 0.5:  #the gripper should be open instead.
            gripper_feedback = self.format(open_gripper_feedback)
        else:
            gripper_feedback = None
        # Compute feedback
        feedback = Feedback()
        if 'r' in  feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type:  # moved closer to the expert goal
            first_conjunction_used = False
            if not moving_away:
                if puck_gripped:
                    _reason_goal = self.format(push_v2_prompts.move_to_goal_feedback)
                else:
                    _reason_goal = self.format(push_v2_prompts.move_to_puck_feedback)
                # _feedback = self.format(hp_feedback)
                # _feedback = _reason_goal + " " + _feedback[0].lower() + _feedback[1:]
                _feedback = _reason_goal + ' ' + positive_conjunctions_sampler() + self.format(task_hp_feedback)
                
                if gripper_feedback is not None:
                    if _feedback is not None:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + gripper_feedback[0].lower() + gripper_feedback[1:]
                        first_conjunction_used = True
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + recommend_templates_sampler().format(degree=degree, direction=direction)
                        first_conjunction_used = True

                # _feedback += f"target: {target_pos[:3]}, cur: {self._current_pos}, puck: {self._current_puck_pos}, goal: {self._goal_pos}."
            else:
                _feedback = None

            feedback.hp = _feedback
        if 'hn' in feedback_type:  # moved away from the expert goal
            if moving_away:
                if puck_gripped:
                    _reason_goal = self.format(push_v2_prompts.move_to_goal_feedback)
                else:
                    _reason_goal = self.format(push_v2_prompts.move_to_puck_feedback)
                # _feedback = self.format(hn_feedback)
                # _feedback = _reason_goal + " " + _feedback[0].lower() + _feedback[1:]
                _feedback = _reason_goal + ' ' + negative_conjunctions_sampler() + self.format(task_hn_feedback)

                # gripper feedback
                if gripper_feedback is not None:
                    if _feedback is not None:
                        _feedback += positive_conjunctions_sampler() + gripper_feedback[0].lower() + gripper_feedback[1:]
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        _feedback += positive_conjunctions_sampler() + recommend_templates_sampler().format(degree=degree, direction=direction)

                # _feedback += f"target: {target_pos[:3]}, cur: {self._current_pos}, puck: {self._current_puck_pos}, goal: {self._goal_pos}."
            else:
                _feedback = None

            feedback.hn = _feedback
        if 'fp' in feedback_type:  
            # suggest the expert goal
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(recommend_target_pos))
        observation = self._format_obs(observation)
        info['success'] = bool(info['success'])
        info['video'] = video if self.env._render_video else None
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated or info['success'], truncated, info

    def _step_push_back_v2(self, action):
        # Run P controller until convergence or timeout
        # action is viewed as the desired position + grab_effort
        previous_pos = self._current_pos  # the position of the hand before moving
        previous_puck_pos = self._current_puck_pos
        if self.control_relative_position:
                action = action.copy()
                action[:3] += self._current_pos  # turn relative position to absolute position

        video = []
        for _ in range(self.p_control_time_out):
            control = self.p_control(action)  # this controls the hand to move an absolute position
            observation, reward, terminated, truncated, info = self.env.step(control)
            self._current_observation = observation
            desired_pos = action[:3]
            video.append(self.env.render()[::-1] if self.env._render_video else None)
            if np.abs(desired_pos - self._current_pos).max() < self.p_control_threshold:
                break

        feedback_type = self._feedback_type
        # Some pre-computation of the feedback
        expert_action = self.expert_action  # absolute or relative
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action.copy()
        # Target pos is in absolute position
        if self.control_relative_position:
            target_pos = self._prev_expert_action.copy()
            target_pos[:3] += self._current_pos
        else:
            target_pos = self._prev_expert_action

        self._prev_expert_action = expert_action.copy()

        # Compute Recommend Target
        if self.control_relative_position:
            recommend_target_pos = expert_action
            recommend_target_pos[:3] += self._current_pos
        else:
            recommend_target_pos = expert_action

        # moving away metrics
        moving_away = np.linalg.norm(target_pos[:3]-previous_pos) < np.linalg.norm(target_pos[:3]-self._current_pos)
        moving_away_axis = [
            target_pos[i] - previous_pos[i] < target_pos[i] - self._current_pos[i]
            for i in range(3)
        ]
        moving_away_direction = direction_converter(target_pos[:3] - self._current_pos)
        moving_away_degree = degree_adverb_converter(target_pos[:3] - self._current_pos)

        puck_gripped = np.linalg.norm(self._current_puck_pos - previous_puck_pos) > 0 \
                        and np.linalg.norm(self._current_puck_pos - self._current_pos) < self.puck_is_gripped_threshold \
                            and action[3] > 0.5
        if target_pos[3] > 0.5 and action[3] < 0.5:  # the gripper should be closed instead.
            gripper_feedback = self.format(close_gripper_feedback)
        elif target_pos[3] < 0.5 and action[3] > 0.5:  #the gripper should be open instead.
            gripper_feedback = self.format(open_gripper_feedback)
        else:
            gripper_feedback = None
        # Compute feedback
        feedback = Feedback()
        if 'r' in  feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type:  # moved closer to the expert goal
            first_conjunction_used = False
            if not moving_away:
                if puck_gripped:
                    _reason_goal = self.format(push_back_v2_prompts.move_to_goal_feedback)
                else:
                    _reason_goal = self.format(push_back_v2_prompts.move_to_puck_feedback)
                # _feedback = self.format(hp_feedback)
                # _feedback = _reason_goal + " " + _feedback[0].lower() + _feedback[1:]
                _feedback = _reason_goal + ' ' + positive_conjunctions_sampler() + self.format(task_hp_feedback)
                
                if gripper_feedback is not None:
                    if _feedback is not None:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + gripper_feedback[0].lower() + gripper_feedback[1:]
                        first_conjunction_used = True
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + recommend_templates_sampler().format(degree=degree, direction=direction)
                        first_conjunction_used = True
                
            else:
                _feedback = None

            feedback.hp = _feedback
        if 'hn' in feedback_type:  # moved away from the expert goal
            if moving_away:
                if puck_gripped:
                    _reason_goal = self.format(push_back_v2_prompts.move_to_goal_feedback)
                else:
                    _reason_goal = self.format(push_back_v2_prompts.move_to_puck_feedback)
                # _feedback = self.format(hn_feedback)
                # _feedback = _reason_goal + " " + _feedback[0].lower() + _feedback[1:]
                _feedback = _reason_goal + ' ' + negative_conjunctions_sampler() + self.format(task_hn_feedback)

                # gripper feedback
                if gripper_feedback is not None:
                    if _feedback is not None:
                        _feedback += positive_conjunctions_sampler() + gripper_feedback[0].lower() + gripper_feedback[1:]
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        _feedback += positive_conjunctions_sampler() + recommend_templates_sampler().format(degree=degree, direction=direction)
                
            else:
                _feedback = None

            feedback.hn = _feedback
        if 'fp' in feedback_type:  
            # suggest the expert goal
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(recommend_target_pos))
        observation = self._format_obs(observation)
        info['success'] = bool(info['success'])
        info['video'] = video if self.env._render_video else None
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated or info['success'], truncated, info

    def _step_sweep_v2(self, action):
        # Run P controller until convergence or timeout
        # action is viewed as the desired position + grab_effort
        previous_pos = self._current_pos  # the position of the hand before moving
        previous_cube_pos = self._current_cube_pos
        if self.control_relative_position:
                action = action.copy()
                action[:3] += self._current_pos  # turn relative position to absolute position

        video = []
        for _ in range(self.p_control_time_out):
            control = self.p_control(action)  # this controls the hand to move an absolute position
            observation, reward, terminated, truncated, info = self.env.step(control)
            self._current_observation = observation
            desired_pos = action[:3]
            video.append(self.env.render()[::-1] if self.env._render_video else None)
            if np.abs(desired_pos - self._current_pos).max() < self.p_control_threshold:
                break

        feedback_type = self._feedback_type
        # Some pre-computation of the feedback
        expert_action = self.expert_action  # absolute or relative
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action.copy()
        # Target pos is in absolute position
        if self.control_relative_position:
            target_pos = self._prev_expert_action.copy()
            target_pos[:3] += self._current_pos # FIXME
        else:
            target_pos = self._prev_expert_action

        self._prev_expert_action = expert_action.copy()

        # Compute Recommend Target
        if self.control_relative_position:
            recommend_target_pos = expert_action
            recommend_target_pos[:3] += self._current_pos
        else:
            recommend_target_pos = expert_action

        # moving away metrics
        moving_away = np.linalg.norm(target_pos[:3]-previous_pos) < np.linalg.norm(target_pos[:3]-self._current_pos)
        moving_away_axis = [
            target_pos[i] - previous_pos[i] < target_pos[i] - self._current_pos[i]
            for i in range(3)
        ]
        moving_away_direction = direction_converter(target_pos[:3] - self._current_pos) #FIXME: should be target - previous
        moving_away_degree = degree_adverb_converter(target_pos[:3] - self._current_pos)

        cube_gripped = np.linalg.norm(self._current_cube_pos - previous_cube_pos) > 0 \
                        and np.linalg.norm(self._current_cube_pos - self._current_pos) < self.cube_is_gripped_threshold \
                            and action[3] > 0.5
        if target_pos[3] > 0.5 and action[3] < 0.5:  # the gripper should be closed instead.
            gripper_feedback = self.format(close_gripper_feedback)
        elif target_pos[3] < 0.5 and action[3] > 0.5:  #the gripper should be open instead.
            gripper_feedback = self.format(open_gripper_feedback)
        else:
            gripper_feedback = None
        # Compute feedback
        feedback = Feedback()
        if 'r' in  feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type:  # moved closer to the expert goal
            first_conjunction_used = False
            if not moving_away:
                if cube_gripped:
                    _reason_goal = self.format(sweep_v2_prompts.move_to_goal_feedback)
                else:
                    _reason_goal = self.format(sweep_v2_prompts.move_to_cube_feedback)
                # _feedback = self.format(hp_feedback)
                # _feedback = _reason_goal + " " + _feedback[0].lower() + _feedback[1:]
                _feedback = _reason_goal + ' ' + positive_conjunctions_sampler() + self.format(task_hp_feedback)
                
                if gripper_feedback is not None:
                    if _feedback is not None:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + gripper_feedback[0].lower() + gripper_feedback[1:]
                        first_conjunction_used = True
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + recommend_templates_sampler().format(degree=degree, direction=direction)
                        first_conjunction_used = True
                
            else:
                _feedback = None

            feedback.hp = _feedback
        if 'hn' in feedback_type:  # moved away from the expert goal
            if moving_away:
                if cube_gripped:
                    _reason_goal = self.format(sweep_v2_prompts.move_to_goal_feedback)
                else:
                    _reason_goal = self.format(sweep_v2_prompts.move_to_cube_feedback)
                # _feedback = self.format(hn_feedback)
                # _feedback = _reason_goal + " " + _feedback[0].lower() + _feedback[1:]
                _feedback = _reason_goal + ' ' + negative_conjunctions_sampler() + self.format(task_hn_feedback)

                # gripper feedback
                if gripper_feedback is not None:
                    if _feedback is not None:
                        _feedback += positive_conjunctions_sampler() + gripper_feedback[0].lower() + gripper_feedback[1:]
                    else:
                        _feedback = gripper_feedback

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        _feedback += positive_conjunctions_sampler() + recommend_templates_sampler().format(degree=degree, direction=direction)
                
            else:
                _feedback = None

            feedback.hn = _feedback
        if 'fp' in feedback_type:  
            # suggest the expert goal
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(recommend_target_pos))
        observation = self._format_obs(observation)
        info['success'] = bool(info['success'])
        info['video'] = video if self.env._render_video else None
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated or info['success'], truncated, info

    def _reset(self, *, seed=None, options=None):
        self._current_observation, info = self.env.reset(seed=seed, options=options)
        self._prev_expert_action = None
        observation = self._format_obs(self._current_observation)
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        mode = 'relative' if self.control_relative_position else 'absolute'
        instruction = self.format(mw_instruction, task=task, mode=mode)
        info['success'] = False
        info['video'] = [self.env.render()[::-1]] if self.env._render_video else None
        return dict(instruction=instruction, observation=observation, feedback=None), info

    def _format_obs(self, observation):
        text = self.textualize_observation(observation)
        image = (self.env.render()[::-1] if self.env.visual else None)
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
        obs_dict = self.mw_policy._parse_obs(observation)
        # remove unused parts
        unused_keys = [k for k in obs_dict.keys() if 'unused' in k or 'extra_info' in k]
        for k in unused_keys:
            del obs_dict[k]
        # convert np.ndarray to list
        for k,v in obs_dict.items():
            if isinstance(v, np.ndarray):
                obs_dict[k] = np.array2string(v)
            else: # it's a scalar
                obs_dict[k] = f"{v:.3f}"
        observation_text = json.dumps(obs_dict)
        return observation_text