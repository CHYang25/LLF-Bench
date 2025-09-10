import numpy as np
import torch
import json
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.highway.prompts import *
from llfbench.envs.highway.task_prompts import (
    highway_prompts,
)
from llfbench.envs.highway.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.highway.utils_prompts.recommend_prompts import *
import llfbench.envs.highway.oracles as solve_policy

class HighwayWrapper(LLFWrapper):

    """ This is a wrapper for highway-env. """

    INSTRUCTION_TYPES = ('b')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env, instruction_type, feedback_type)

        self._policy_name = 'solveParking'
        if 'fp' in feedback_type:
            self._policy = getattr(solve_policy, self._policy_name)(
                env = env,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            )
    
    @property
    def hw_env(self):
        return self.env.env

    @property
    def hw_policy(self): # highway policy
        return self._policy
    
    @property
    def current_observation(self):  # external interface
        return self._current_observation

    @property
    def reward_range(self):
        return ((self.env.config['collision_reward']-1)*self.env.config['controlled_vehicles'], 0.0)

    def _format_obs(self, observation):
        text = self.textualize_observation(observation)
        image = (self.env.render() if self.env.visual else None)
        return text if image is None else dict(text=text, image=image)

    def _reset(self, seed=None, options=None):
        options = options or {}
        self._current_observation, info = self.env.reset(seed=seed, options=options)
        self._prev_expert_action = None
        self._prev_angle_diff = None
        self._prev_dist = None
        observation = self._format_obs(self.current_observation)
        if 'action' in info:
            del info['action']
        info['success'] = bool(info['is_success'])
        instruction = self.format(hw_instruction)
        return dict(instruction=instruction, observation=observation, feedback=None), info

    def _step(self, action):
        # processed_action = self.extract_action(action)
        self._current_observation, reward, terminated, truncated, info = self.env.step(action)
        current_pos = self.current_observation['achieved_goal'][:2]
        current_vel = self.current_observation['achieved_goal'][2:4]
        current_angle = np.arccos(self.current_observation['achieved_goal'][4])
        current_angle = (current_angle + 2 * np.pi) % (2 * np.pi) # normalize angle to [0, 2pi)
        reward = float(reward)

        feedback_type = self._feedback_type

        # calculation for the feedback
        expert_action = self.hw_policy.get_action(self.current_observation)
        # expert_action = self.env.action_space.sample()
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action

        # decide in what direction the agent should move (8 directions)
        goal = self.current_observation['desired_goal'][:2]
        dist = np.linalg.norm(goal - current_pos)
        goal_theta = np.arccos(self.current_observation['desired_goal'][4])
        goal_theta = (goal_theta + 2 * np.pi) % (2 * np.pi) # normalize angle to [0, 2pi)
        angle_diff = goal_theta - current_angle
        need_turning = True if angle_diff > np.pi/2 else False

        is_crashed = info.get('crashed', False)
        
        # decide if the agent is moving away
        moving_away = (abs(angle_diff)> self._prev_angle_diff) and (dist > self._prev_dist) if self._prev_angle_diff is not None else False

        # decide what the agent shoud do. Go straight, turn right, turn left, deccelerate
        

        forward = expert_action[0] > 0
        backward = expert_action[0] < 0
        turn_left = expert_action[1] < 0
        turn_right = expert_action[1] > 0
        accel_x = abs(expert_action[0]) - abs(action[0])
        accel_y = abs(expert_action[1]) - abs(action[1]) 

        _feedback = ""

        # Compute feedback
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=reward)
        if 'hp' in feedback_type:  # moved closer to the expert goal
            if not moving_away and not is_crashed:
                _feedback += positive_conjunctions_sampler() + self.format(hp_feedback)
                if need_turning:
                    if turn_left:
                        _feedback += positive_conjunctions_sampler() + self.format(turn_left_recommend)
                    if turn_right:
                        _feedback += positive_conjunctions_sampler() + self.format(turn_right_recommend)
                    _feedback += positive_conjunctions_sampler() + self.format(forward_recommend)
                else:
                    if forward:
                        _feedback += positive_conjunctions_sampler() + self.format(forward_recommend)
                    if backward:
                        _feedback += positive_conjunctions_sampler() + self.format(backward_recommend)
                    if accel_x or accel_y > 0:
                        _feedback += positive_conjunctions_sampler() + self.format(accel_recommend)
                    else:
                        _feedback += positive_conjunctions_sampler() + self.format(deccel_recommend)
                    if turn_left:
                        _feedback += positive_conjunctions_sampler() + self.format(turn_left_recommend)
                    if turn_right:
                        _feedback += positive_conjunctions_sampler() + self.format(turn_right_recommend)
                feedback.hp = _feedback

        if 'hn' in feedback_type:
            if moving_away:
                _feedback += negative_conjunctions_sampler() + self.format(hn_feedback)
                if need_turning:
                    if turn_left:
                        _feedback += positive_conjunctions_sampler() + self.format(turn_left_recommend)
                    if turn_right:
                        _feedback += positive_conjunctions_sampler() + self.format(turn_right_recommend)
                    _feedback += positive_conjunctions_sampler() + self.format(forward_recommend)
                else:
                    if forward:
                        _feedback += positive_conjunctions_sampler() + self.format(forward_recommend)
                    if backward:
                        _feedback += positive_conjunctions_sampler() + self.format(backward_recommend)
                    if accel_x > 0 or accel_y > 0:
                        _feedback += positive_conjunctions_sampler() + self.format(accel_recommend)
                    else:
                        _feedback += positive_conjunctions_sampler() + self.format(deccel_recommend)
                    if turn_left:
                        _feedback += positive_conjunctions_sampler() + self.format(turn_left_recommend)
                    if turn_right:
                        _feedback += positive_conjunctions_sampler() + self.format(turn_right_recommend)
                feedback.hn = _feedback
            if is_crashed:
                feedback.hn = "The car is crashed, you should be more careful."

        if 'fp' in feedback_type and not is_crashed:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))

        observation = self._format_obs(self.current_observation)

        info["success"] = bool(info["is_success"])

        self._prev_dist = dist
        self._prev_angle_diff = abs(angle_diff)

        return dict(instruction=None, observation=observation, feedback=feedback), reward, terminated, truncated, info

    def textualize_observation(self, observation):
        """ Parse np.ndarray observation into text. """
        obs_dict = observation.copy()
        # convert np.ndarray to list
        for k,v in obs_dict.items():
            if isinstance(v, np.ndarray):
                obs_dict[k] = np.array2string(v)
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