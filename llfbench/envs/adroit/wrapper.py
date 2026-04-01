from typing import Dict, SupportsFloat, Union, List
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.adroit.prompts import *
from llfbench.envs.adroit.task_prompts import (
    hammer_prompts,
    relocate_prompts,
)
from llfbench.envs.adroit.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.adroit.utils_prompts.recommend_prompts import (
    move_down_recommend, 
    move_up_recommend,
    move_right_recommend,
    move_left_recommend,
    move_forward_recommend,
    move_backward_recommend,
)
import importlib
import json
import re
import os
import torch
import pickle
import gymnasium.spaces as spaces

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

class AdroitWrapper(LLFWrapper):

    """
    Please refer to the following links:
    1. Hand Hammer
        - https://robotics.farama.org/envs/adroit_hand/adroit_hammer/
        - https://minari.farama.org/datasets/D4RL/hammer/expert-v2/
        - https://github.com/aravindr93/hand_dapg/blob/master/README.md
        - https://github.com/aravindr93/mjrl/blob/master/mjrl/policies/gaussian_mlp.py
    """

    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type, debug: bool = False):
        super().__init__(env, instruction_type, feedback_type)
        # load the scripted policy

        self.task_name = self.env.env_name
        if self.task_name == "AdroitHandDoor-v1":
            policy_name = "door-v0.pickle"
        elif self.task_name == "AdroitHandHammer-v1":
            policy_name = "hammer-v0.pickle"
            self._step = self._step_hand_hammer

            # Hammer-specific thresholds
            self.hammer_reach_threshold = 5.5e-2
            self.hammer_lift_threshold = 4e-2
            self.hammer_nail_align_threshold = 8e-2
            self.hammer_success_threshold = 1e-2
            self.hammer_retry_progress_threshold = 1e-3
            self.hammer_reach_progress_threshold = 1e-4
            self.hammer_force_threshold = 0.10
            self.expert_action_close_threshold = 0.14
            self.arm_reco_threshold = 0.12
            self.wrist_reco_threshold = 0.12
            self.finger_reco_threshold = 0.14
            self.recon_threshold = 1.0e-1

            self._prev_nail_displace = None
        elif self.task_name == "AdroitHandPen-v1":
            policy_name = "pen-v0.pickle"
        elif self.task_name == "AdroitHandRelocate-v1":
            policy_name = "relocate-v0.pickle"
            self._step = self._step_hand_relocate

            # Relocate-specific thresholds
            self.relocate_reach_threshold = 6.0e-2
            self.relocate_success_threshold = 5.0e-2
            self.relocate_reach_progress_threshold = 1.0e-4
            self.relocate_target_progress_threshold = -2.0e-3
            self.relocate_grip_threshold = 5.0e-3
            self.ball_lifted_threshold = 4.0e-2
            self.expert_action_close_threshold = 0.14
            self.arm_reco_threshold = 0.12
            self.wrist_reco_threshold = 0.12
            self.finger_reco_threshold = 0.14
            self.recon_threshold = 1.0e-2

        else:
            raise ValueError(f"Unrecognized task name from adroit hand: {self.task_name}.")

        if 'fp' in feedback_type:
            with open(os.path.join(os.path.realpath(os.path.dirname(__file__)), "hand_dapg/dapg/policies", policy_name), "rb") as f:
                self._policy = pickle.load(f)
        else:
            self._policy = None

        self.debug = debug
        self._current_observation = None
        self._prev_expert_action = None
        self.t = 0

    @property
    def ad_policy(self): # adroit hand policy
        return self._policy

    @property
    def current_observation(self):  # external interface
        return self._current_observation

    @property
    def base_env(self):
        return self.env.env
    
    # auxiliary functions for language feedback
    @property
    def expert_action(self):
        # Flatten current observation first
        obs_state = spaces.flatten(self.env.observation_space, self._current_observation)
        expert_action, action_info = self.ad_policy.get_action(obs_state)
        self._action_mean = action_info['mean']
        self._log_std = action_info['log_std']
        return expert_action

    # step functions
    def _step_general(self, action):
        pass

    def _step_hand_relocate(self, action):
        prev_expert_action = self._prev_expert_action.copy()
        prev_palm_to_ball_dist = self._prev_palm_to_ball_dist.copy()
        prev_ball_to_target_dist = self._prev_ball_to_target_dist.copy()
        prev_palm_to_target_diff = self._prev_palm_to_target_diff.copy()
        prev_ball_to_target_diff = self._prev_ball_to_target_diff.copy()

        self._current_observation, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)

        obs = self.current_observation
        arm_pos = obs[0:3]
        arm_apos = obs[3:6]
        wrist_apos = obs[6:8]
        forefinger_pos = obs[8:12]
        midfinger_pos = obs[12:16]
        ringfinger_pos = obs[16:20]
        litfinger_pos = obs[20:25]
        thumb_pos = obs[25:30]
        palm_to_ball_diff = obs[30:33]
        palm_to_target_diff = obs[33:36]
        ball_to_target_diff = obs[36:39]

        expert_action = self.expert_action
        self._prev_expert_action = expert_action.copy()

        palm_to_ball_dist = np.linalg.norm(palm_to_ball_diff)
        palm_to_target_dist = np.linalg.norm(palm_to_target_diff)
        ball_to_target_dist = np.linalg.norm(ball_to_target_diff)
        
        palm_to_ball_progress = prev_palm_to_ball_dist - palm_to_ball_dist
        ball_to_target_progress = prev_ball_to_target_dist - ball_to_target_dist

        ball_move = ball_to_target_diff - prev_ball_to_target_diff
        palm_move = palm_to_target_diff - prev_palm_to_target_diff
        palm_ball_move_diff = np.linalg.norm(ball_move - palm_move)

        env_state = self.base_env.get_env_state()
        ball_pos = env_state["obj_pos"].copy()

        # stage metrics
        ball_near_palm = bool(palm_to_ball_dist < self.relocate_reach_threshold)
        ball_gripped = palm_ball_move_diff < self.relocate_grip_threshold and ball_near_palm and ball_pos[2] > self.ball_lifted_threshold
        goal_reached = (ball_to_target_dist < self.relocate_success_threshold) and ball_gripped

        if goal_reached:
            stage_key = "move_to_target"
            _stage_feedback = self.format(relocate_prompts.move_to_target_feedback)
        elif not ball_near_palm:
            stage_key = "go_to_ball"
            _stage_feedback = self.format(relocate_prompts.go_to_ball_feedback)
        elif ball_near_palm and not ball_gripped:
            stage_key = "grip_ball"
            _stage_feedback = self.format(relocate_prompts.grip_ball_feedback)
        else:
            stage_key = "move_to_target"
            _stage_feedback = self.format(relocate_prompts.move_to_target_feedback)
        
        # action optimality
        action_error = np.linalg.norm(action - prev_expert_action) / np.sqrt(action.shape[0])
        action_optim = action_error < self.expert_action_close_threshold

        # movement guidance
        _recommend_feedback = []

        # action-vs-expert mismatch by actuator groups
        action_delta = prev_expert_action - action

        # relocate has 30 actions: 6 arm + 2 wrist + 22 finger/thumb
        arm_delta = action_delta[0:6]
        wrist_delta = action_delta[6:8]
        finger_delta = action_delta[8:30]

        arm_mismatch = np.linalg.norm(arm_delta) / np.sqrt(6)
        wrist_mismatch = np.linalg.norm(wrist_delta) / np.sqrt(2)
        finger_mismatch = np.linalg.norm(finger_delta) / np.sqrt(22)

        def _axis_recommend(vec):
            recon = []
            if vec[0] > self.recon_threshold:
                recon.append(self.format(move_right_recommend))
            elif vec[0] < -self.recon_threshold:
                recon.append(self.format(move_left_recommend))
            if vec[1] > self.recon_threshold:
                recon.append(self.format(move_forward_recommend))
            elif vec[1] < -self.recon_threshold:
                recon.append(self.format(move_backward_recommend))
            if vec[2] > self.recon_threshold:
                recon.append(self.format(move_up_recommend))
            elif vec[2] < -self.recon_threshold:
                recon.append(self.format(move_down_recommend))
            return recon
        
        desired_vec = np.zeros(3)

        if stage_key == "go_to_ball":
            no_progress = palm_to_ball_progress < self.relocate_reach_progress_threshold
            if no_progress:
                action_optim = False
            elif not no_progress and palm_to_ball_dist > 0.15:
                action_optim = True
            if no_progress or (arm_mismatch > self.arm_reco_threshold or wrist_mismatch > self.wrist_reco_threshold):
                desired_vec = -palm_to_ball_diff
                _recommend_feedback.extend(_axis_recommend(desired_vec))
            
        elif stage_key == "grip_ball":
            if arm_mismatch > self.arm_reco_threshold or wrist_mismatch > self.wrist_reco_threshold:
                desired_vec = -palm_to_ball_diff
                _recommend_feedback.extend(_axis_recommend(desired_vec))

            if finger_mismatch > self.finger_reco_threshold or not ball_near_palm:
                _recommend_feedback.append(self.format(relocate_prompts.grasp_ball_recommend))     

        elif stage_key == "move_to_target":
            no_progress = ball_to_target_progress < self.relocate_target_progress_threshold
            action_optim = not no_progress
            if not goal_reached and (no_progress or (arm_mismatch > self.arm_reco_threshold or wrist_mismatch > self.wrist_reco_threshold)):
                desired_vec = -ball_to_target_diff
                _recommend_feedback.extend(_axis_recommend(desired_vec))

            if not goal_reached and (no_progress or finger_mismatch > self.finger_reco_threshold):
                _recommend_feedback.append(self.format(relocate_prompts.grasp_ball_recommend))

        feedback_type = self._feedback_type
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type and action_optim:
            _feedback = self.concatenate_sentences(
                stage_feedback=_stage_feedback,
                action_feedback=self.format(hp_feedback),
                reco_feedback=_recommend_feedback,
                action_positive=True,
            )

            if self.debug:
                _feedback += (
                    f"\n[palm_to_ball_dist]={palm_to_ball_dist:.6f}"
                    f"\n[palm_to_target_dist]={palm_to_target_dist:.6f}"
                    f"\n[ball_to_target_dist]={ball_to_target_dist:.6f}"
                    f"\n[action_error]={action_error:.6f}"
                    f"\n[stage]={stage_key}"
                    f"\n[palm_to_ball_progress]={palm_to_ball_progress}"
                    f"\n[ball_to_target_progress]={ball_to_target_progress}"
                    f"\n[ball_move]={ball_move}"
                    f"\n[palm_move]={palm_move}"
                    f"\n[ball_pos]={ball_pos}"
                    f"\n[palm_ball_move_diff]={palm_ball_move_diff}"
                    f"\n[arm_mismatch]={arm_mismatch:.6f}"
                    f"\n[wrist_mismatch]={wrist_mismatch:.6f}"
                    f"\n[finger_mismatch]={finger_mismatch:.6f}"
                    f"\n[arm_reco_threshold]={self.arm_reco_threshold:.6f}"
                    f"\n[wrist_reco_threshold]={self.wrist_reco_threshold:.6f}"
                    f"\n[finger_reco_threshold]={self.finger_reco_threshold:.6f}"
                    f"\n[desired_vec]={desired_vec}"
                    f"\n[action]={action}"
                    f"\n[prev_expert_action]={prev_expert_action}."
                )

            feedback.hp = _feedback
        if 'hn' in feedback_type and not action_optim:
            _feedback = self.concatenate_sentences(
                stage_feedback=_stage_feedback,
                action_feedback=self.format(hn_feedback),
                reco_feedback=_recommend_feedback,
                action_positive=False,
            )

            if self.debug:
                _feedback += (
                    f"\n[palm_to_ball_dist]={palm_to_ball_dist:.6f}"
                    f"\n[palm_to_target_dist]={palm_to_target_dist:.6f}"
                    f"\n[ball_to_target_dist]={ball_to_target_dist:.6f}"
                    f"\n[action_error]={action_error:.6f}"
                    f"\n[stage]={stage_key}"
                    f"\n[palm_to_ball_progress]={palm_to_ball_progress}"
                    f"\n[ball_to_target_progress]={ball_to_target_progress}"
                    f"\n[ball_move]={ball_move}"
                    f"\n[palm_move]={palm_move}"
                    f"\n[ball_pos]={ball_pos}"
                    f"\n[palm_ball_move_diff]={palm_ball_move_diff}"
                    f"\n[arm_mismatch]={arm_mismatch:.6f}"
                    f"\n[wrist_mismatch]={wrist_mismatch:.6f}"
                    f"\n[finger_mismatch]={finger_mismatch:.6f}"
                    f"\n[arm_reco_threshold]={self.arm_reco_threshold:.6f}"
                    f"\n[wrist_reco_threshold]={self.wrist_reco_threshold:.6f}"
                    f"\n[finger_reco_threshold]={self.finger_reco_threshold:.6f}"
                    f"\n[desired_vec]={desired_vec}"
                    f"\n[action]={action}"
                    f"\n[prev_expert_action]={prev_expert_action}."
                )

            feedback.hn = _feedback
        if 'fp' in feedback_type:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))

        self._prev_palm_to_ball_dist = palm_to_ball_dist
        self._prev_ball_to_target_dist = ball_to_target_dist
        self._prev_palm_to_target_diff = palm_to_target_diff
        self._prev_ball_to_target_diff = ball_to_target_diff
        observation = self._format_obs(self.current_observation)
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated or goal_reached, truncated, info
        

    def _step_hand_hammer(self, action):
        prev_expert_action = self._prev_expert_action.copy()
        prev_nail_displace = self._prev_nail_displace
        prev_hammer_pos = self._prev_hammer_pos.copy()

        self._current_observation, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)

        obs = self.current_observation
        arm_apos = obs[0:2] # arm angular position
        wrist_apos = obs[2:4]
        forefinger_pos = obs[4:8]
        midfinger_pos = obs[8:12]
        ringfinger_pos = obs[12:16]
        litfinger_pos = obs[16:21]
        thumb_pos = obs[21:26]
        nail_displace = obs[26]
        hammer_vel = obs[27:30]
        hammer_avel = obs[30:33]
        palm_pos = obs[33:36]
        hammer_pos = obs[36:39]
        hammer_apos = obs[39:42]
        nail_pos = obs[42:45]
        nail_force = obs[45]

        expert_action = self.expert_action
        self._prev_expert_action = expert_action.copy()

        goal_pos = self.base_env.data.site_xpos[self.base_env.goal_site_id].ravel()
        hammer_head_pos = self.base_env.data.site_xpos[self.base_env.tool_site_id].ravel().copy()

        palm_to_hammer_dist = np.linalg.norm(palm_pos - hammer_pos)
        hammer_head_to_nail_dist = np.linalg.norm(hammer_head_pos - nail_pos)
        nail_to_goal_dist = np.linalg.norm(nail_pos - goal_pos)

        # deterime task stage and stage feedback
        hammer_lifted = bool(
            hammer_pos[2] > self.hammer_lift_threshold and hammer_head_pos[2] > self.hammer_lift_threshold
        )
        goal_reached = info['success'] or (nail_to_goal_dist < self.hammer_success_threshold)
        palm_above_hammer = bool(palm_pos[2] > hammer_pos[2] + 3.0e-2)
        has_impact = abs(nail_force) > 0

        if goal_reached:
            stage_key = "hammer_nail"
            _stage_feedback = self.format(hammer_prompts.hammer_nail_feedback)
        elif palm_to_hammer_dist > self.hammer_reach_threshold:
            stage_key = "get_hammer"
            _stage_feedback = self.format(hammer_prompts.get_hammer_feedback)
        elif not hammer_lifted:
            stage_key = "lift_hammer"
            _stage_feedback = self.format(hammer_prompts.lift_hammer_feedback)
        elif hammer_head_to_nail_dist > self.hammer_nail_align_threshold and not has_impact:
            stage_key = "swing_to_nail"
            _stage_feedback = self.format(hammer_prompts.swing_to_nail_feedback)
        else:
            stage_key = "hammer_nail"
            _stage_feedback = self.format(hammer_prompts.hammer_nail_feedback)

        # action optimality
        action_error = np.linalg.norm(action - prev_expert_action) / np.sqrt(action.shape[0])
        action_optim = action_error < self.expert_action_close_threshold

        # Movement guidance (Recommendation)
        _recommend_feedback = []

        # action-vs-expert mismatch by actuator groups
        action_delta = prev_expert_action - action

        arm_delta = action_delta[0:2]       # full arm
        wrist_delta = action_delta[2:4]     # wrist
        finger_delta = action_delta[4:26]   # all fingers + thumb

        arm_mismatch = np.linalg.norm(arm_delta) / np.sqrt(2)
        wrist_mismatch = np.linalg.norm(wrist_delta) / np.sqrt(2)
        finger_mismatch = np.linalg.norm(finger_delta) / np.sqrt(22)
        get_hammer_progress = np.linalg.norm(hammer_pos - palm_pos) - np.linalg.norm(prev_hammer_pos - palm_pos)
        nail_progress = nail_displace - prev_nail_displace

        # helper: convert a desired task-space vector into a language direction
        # heuristic convention here:
        #   +x -> forward, -x -> backward
        #   +y -> left,    -y -> right
        #   +z -> up,      -z -> down
        def _axis_recommend(vec):
            recon = []
            if vec[0] > self.recon_threshold:
                recon.append(self.format(move_left_recommend))
            elif vec[0] < -self.recon_threshold:
                recon.append(self.format(move_right_recommend))
            if vec[1] > self.recon_threshold:
                recon.append(self.format(move_up_recommend))
            elif vec[1] < -self.recon_threshold:
                recon.append(self.format(move_down_recommend))
            if vec[2] > self.recon_threshold:
                recon.append(self.format(move_forward_recommend))
            elif vec[2] < -self.recon_threshold:
                recon.append(self.format(move_backward_recommend))
            return recon
                
        desired_vec = 0
        if stage_key == "get_hammer":
            no_progress = get_hammer_progress > -self.hammer_reach_progress_threshold and self.t > 5
            if no_progress:
                action_optim = False
            # if arm / wrist action disagrees with expert, recommend a Cartesian correction
            if arm_mismatch > self.arm_reco_threshold or wrist_mismatch > self.wrist_reco_threshold:
                desired_vec = hammer_pos - palm_pos
                _recommend_feedback.extend(_axis_recommend(desired_vec))

        elif stage_key == "lift_hammer":
            # lifting is mainly upward motion; use geometry for wording
            if arm_mismatch > self.arm_reco_threshold or wrist_mismatch > self.wrist_reco_threshold:
                desired_vec = np.array([0.0, 0.0, 1.0])
                _recommend_feedback.extend(_axis_recommend(desired_vec))

        elif stage_key == "swing_to_nail":
            # guide the hammer head toward the nail
            if arm_mismatch > self.arm_reco_threshold or wrist_mismatch > self.wrist_reco_threshold:
                desired_vec = nail_pos - hammer_head_pos
                _recommend_feedback.extend(_axis_recommend(desired_vec))

        elif stage_key == "hammer_nail":
            no_progress = nail_progress < self.hammer_retry_progress_threshold

            # if arm / wrist motion disagrees with expert, still give directional correction
            if arm_mismatch > self.arm_reco_threshold or wrist_mismatch > self.wrist_reco_threshold:
                desired_vec = nail_pos - hammer_head_pos
                _recommend_feedback.extend(_axis_recommend(desired_vec))

            if not goal_reached and has_impact and no_progress:
                # if the strike is poor and not progressing, back off and try again
                _recommend_feedback.append(self.format(hammer_prompts.pull_back_recommend))
                    
        # if finger action disagrees with expert, recommend grasping
        if stage_key != "get_hammer" and finger_mismatch > self.finger_reco_threshold:
            _recommend_feedback.append(self.format(hammer_prompts.grasp_hammer_recommend))

        # put everything together
        feedback_type = self._feedback_type
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type and action_optim:
            _feedback = self.concatenate_sentences(
                stage_feedback = _stage_feedback,
                action_feedback = self.format(hp_feedback),
                reco_feedback = _recommend_feedback,
                action_positive = True,
            )

            if self.debug:
                _feedback += (
                    f"\n[palm_to_hammer_dist]={palm_to_hammer_dist:.6f}"
                    f"\n[hammer_head_to_nail_dist]={hammer_head_to_nail_dist:.6f}"
                    f"\n[nail_to_goal_dist]={nail_to_goal_dist:.6f}"
                    f"\n[nail_force]={nail_force:.6f}"
                    f"\n[action_error]={action_error:.6f}"
                    f"\n[palm_pos]={palm_pos}"
                    f"\n[hammer_pos]={hammer_pos}"
                    f"\n[stage]={stage_key}"
                    f"\nget_hammer_progress={get_hammer_progress}"
                    f"\narm_mismatch={arm_mismatch:.6f}"
                    f"\nwrist_mismatch={wrist_mismatch:.6f}"
                    f"\nfinger_mismatch={finger_mismatch:.6f}"
                    f"\narm_reco_threshold={self.arm_reco_threshold:.6f}"
                    f"\nwrist_reco_threshold={self.wrist_reco_threshold:.6f}"
                    f"\nfinger_reco_threshold={self.finger_reco_threshold:.6f}"
                    f"\nnail_progress={nail_progress}"
                    f"\ndesired_vec={desired_vec}"
                    f"\naction={action}"
                    f"\nprev_expert_action={prev_expert_action}"
                )

            feedback.hp = _feedback
        if 'hn' in feedback_type and not action_optim:
            _feedback = self.concatenate_sentences(
                stage_feedback = _stage_feedback,
                action_feedback = self.format(hn_feedback),
                reco_feedback = _recommend_feedback,
                action_positive = False,
            )

            if self.debug:
                _feedback += (
                    f"\n[palm_to_hammer_dist]={palm_to_hammer_dist:.6f}"
                    f"\n[hammer_head_to_nail_dist]={hammer_head_to_nail_dist:.6f}"
                    f"\n[nail_to_goal_dist]={nail_to_goal_dist:.6f}"
                    f"\n[nail_force]={nail_force:.6f}"
                    f"\n[action_error]={action_error:.6f}"
                    f"\n[palm_pos]={palm_pos}"
                    f"\n[hammer_pos]={hammer_pos}"
                    f"\n[stage]={stage_key}"
                    f"\nget_hammer_progress={get_hammer_progress}"
                    f"\narm_mismatch={arm_mismatch:.6f}"
                    f"\nwrist_mismatch={wrist_mismatch:.6f}"
                    f"\nfinger_mismatch={finger_mismatch:.6f}"
                    f"\narm_reco_threshold={self.arm_reco_threshold:.6f}"
                    f"\nwrist_reco_threshold={self.wrist_reco_threshold:.6f}"
                    f"\nfinger_reco_threshold={self.finger_reco_threshold:.6f}"
                    f"\nnail_progress={nail_progress}"
                    f"\ndesired_vec={desired_vec}"
                    f"\naction={action}"
                    f"\nprev_expert_action={prev_expert_action}"
                )
                
            feedback.hn = _feedback
        if 'fp' in feedback_type:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))

        self._prev_nail_displace = nail_displace
        self._prev_hammer_pos = hammer_pos
        observation = self._format_obs(self.current_observation)
        self.t += 1
        # info["video"] = [self.env.render()[::-1]] if self.env._render_video else None
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), terminated or info['success'], truncated, info

    def _reset(self, *, seed = None, options = None):
        self._current_observation, info = self.env.reset(seed=seed, options=options)

        if 'fp' in self._feedback_type:
            self._prev_expert_action = self.expert_action.copy()

        if self.task_name == "AdroitHandHammer-v1":
            self._prev_nail_displace = self.current_observation[26]
            self._prev_hammer_pos = self.current_observation[36:39].copy()
        elif self.task_name == "AdroitHandRelocate-v1":
            self._prev_palm_to_ball_dist = np.linalg.norm(self.current_observation[30:33])
            self._prev_ball_to_target_dist = np.linalg.norm(self.current_observation[36:39])
            self._prev_palm_to_target_diff = self.current_observation[33:36]
            self._prev_ball_to_target_diff = self.current_observation[36:39]

        observation = self._format_obs(self._current_observation)
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        instruction = self.format(ad_instruction, task=task)
        info['success'] = False
        info['video'] = [self.env.render()[::-1]] if self.env._render_video else None
        feedback = Feedback()
        self.t = 0
        if 'fp' in self._feedback_type:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(self._prev_expert_action))
        return dict(instruction=instruction, observation=observation, feedback=feedback), info

    def _format_obs(self, observation):
        text = self.textualize_observation(observation)
        image = (self.env.render() if self.env.visual else None)
        return text if image is None else dict(text=text, image=image)

    def textualize_expert_action(self, action):
        """ Parse action into text. """
        # The idea is to return something like
        # f"delta x: {action[0]:.2f}, delta y:{action[1]:.2f}, delta z:{action[2]:.2f}, gripper state:{action[3]:.1f}"
        # or another action text format if the action isn't a delta.
        # TODO should not be the raw action
        return np.array2string(action, precision=10)

    def textualize_observation(self, observation):
        """ Parse np.ndarray observation into text. """
        if isinstance(observation, np.ndarray):
            return json.dumps({'obs': np.array2string(observation, precision=10)})

        obs_dict = {}
        observation = observation.copy()
        # convert np.ndarray to list
        for k,v in observation.items():
            if isinstance(v, dict):
                for vk,vv in v.items():
                    assert not isinstance(vv, dict)
                    if isinstance(vv, np.ndarray):
                        obs_dict[vk] = np.array2string(v, precision=10)
                    elif isinstance(vv, torch.Tensor):
                        obs_dict[vk] = str(vv.flatten().tolist()).replace(',', '')
                    else: # it's a scalar
                        obs_dict[vk] = f"{vv:.10f}"

            elif isinstance(v, np.ndarray):
                obs_dict[k] = np.array2string(v, precision=10)
            elif isinstance(v, torch.Tensor):
                obs_dict[k] = str(v.flatten().tolist()).replace(',', '')
            else: # it's a scalar
                obs_dict[k] = f"{v:.10f}"
        observation_text = json.dumps(obs_dict)
        return observation_text

    def concatenate_sentences(
        self,
        stage_feedback: str,
        action_feedback: str,
        reco_feedback: List[str],
        action_positive: bool):

        res = stage_feedback
        res += (positive_conjunctions_sampler() if action_positive else negative_conjunctions_sampler()) + action_feedback

        for rec in reco_feedback:
            res += positive_conjunctions_sampler() + rec

        return res
    