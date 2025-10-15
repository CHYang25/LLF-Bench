import numpy as np
import torch
import json
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.highway.prompts import *
from llfbench.envs.highway.task_prompts import (
    parking_prompts,
)
from llfbench.envs.highway.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.highway.utils_prompts.recommend_prompts import *
import llfbench.envs.highway.oracles as solve_policy

class HighwayWrapper(LLFWrapper):

    """ This is a wrapper for highway-env. """

    INSTRUCTION_TYPES = ('b')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type, debug: bool=False):
        super().__init__(env, instruction_type, feedback_type)

        self._policy_name = 'solveParking'
        if 'fp' in feedback_type:
            self._policy = getattr(solve_policy, self._policy_name)(
                env = env,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            )
        self.accel_thres = 1.0e-4
        self.turn_thres = 1.0e-4
        self.goal_reached_thres = 7.0e-2
        self.debug = debug
    
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
    def current_pos(self):
        return self.current_observation['achieved_goal'][:2]
    
    @property
    def current_vel(self):
        return self.current_observation['achieved_goal'][2:4]
    
    @property
    def current_vel_angle(self):
        vy, vx = self.current_vel
        angle = np.arctan2(vy, vx)
        angle = np.mod(angle, 2 * np.pi)
        return angle

    @property
    def goal_pos(self):
        return self.current_observation['desired_goal'][:2]

    @property
    def reward_range(self):
        return ((self.env.config['collision_reward']-1)*self.env.config['controlled_vehicles'], 0.0)

    def sin_cos_to_rad(self, sin, cos):
        angle = np.mod(np.arctan2(sin, cos), 2 * np.pi)
        return angle

    @property
    def current_angle(self):
        return self.sin_cos_to_rad(self.current_observation['achieved_goal'][4], self.current_observation['achieved_goal'][5])

    @property
    def goal_angle(self):
        return self.sin_cos_to_rad(self.current_observation['desired_goal'][4], self.current_observation['desired_goal'][5])

    def angle_diff(self, theta1, theta2):
        diff = abs(theta1 - theta2)
        return min(diff, 2 * np.pi - diff)

    def _format_obs(self, observation):
        text = self.textualize_observation(observation)
        image = (self.env.render() if self.env.visual else None)
        return text if image is None else dict(text=text, image=image)

    def _reset(self, seed=None, options=None):
        options = options or {}
        self._current_observation, info = self.env.reset(seed=seed, options=options)
        expert_action = self.hw_policy.get_action(self.current_observation)
        self._prev_throttle, self._prev_steer = expert_action
        observation = self._format_obs(self.current_observation)
        if 'action' in info:
            del info['action']
        info['success'] = bool(info['is_success'])
        instruction = self.format(hw_instruction)
        feedback = Feedback()
        feedback.fp = self.format(fp_feedback, expert_action=expert_action)
        return dict(instruction=instruction, observation=observation, feedback=feedback), info

    def _step(self, action):
        # processed_action = self.extract_action(action)
        self._current_observation, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        # environment variables
        current_pos = self.current_pos
        current_vel = self.current_vel
        current_angle = self.current_angle
        current_vel_angle =self.current_vel_angle
        goal_pos = self.goal_pos
        goal_angle = self.goal_angle

        expert_action = self.hw_policy.get_action(self.current_observation)
        # calculation for the feedback
        car_aligned = self.angle_diff(goal_angle, current_angle) < np.pi * (0.4)
        goal_reached = np.linalg.norm(goal_pos - current_pos) < self.goal_reached_thres
        is_crashed = info.get('crashed', False)
        
        # decide if the agent is moving away
        # the controls are bad (regardless of the magnitude) if it's in different direction
        moving_away_throttle = self._prev_throttle * action[0] < 0 
        moving_away_steer = self._prev_steer * action[1] < 0
        moving_away = moving_away_throttle or moving_away_steer
        self._prev_throttle, self._prev_steer = expert_action

        # decide what the agent shoud do. Go straight, turn right, turn left, deccelerate
        forward = self.angle_diff(current_vel_angle, current_angle) < np.pi / 2
        backward = self.angle_diff(current_vel_angle, current_angle) > np.pi / 2
        accel = (expert_action[0] > self.accel_thres and forward) or (expert_action[0] < -self.accel_thres and backward)
        turn_right = expert_action[1] > self.turn_thres
        turn_left = expert_action[1] < -self.turn_thres

        feedback_type = self._feedback_type
        if not car_aligned:
            _stage_feedback = self.format(parking_prompts.align_car_prompts)
        elif car_aligned and not goal_reached:
            _stage_feedback = self.format(parking_prompts.reach_goal_prompts)
        elif car_aligned and goal_reached:
            _stage_feedback = self.format(parking_prompts.careful_park_prompts)
        else:
            raise ValueError

        # Compute feedback
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=reward)
        if 'hp' in feedback_type:  # moved closer to the expert goal
            if not moving_away and not is_crashed:
                _feedback = _stage_feedback
                _feedback += positive_conjunctions_sampler() + self.format(hp_feedback)

                if accel:
                    if forward:
                        _feedback += positive_conjunctions_sampler() + self.format(accel_forward_recommend)
                    elif backward:
                        _feedback += positive_conjunctions_sampler() + self.format(accel_backward_recommend)
                else:
                    _feedback += positive_conjunctions_sampler() + self.format(deccel_recommend)
                if turn_right:
                    _feedback += positive_conjunctions_sampler() + self.format(turn_right_recommend)
                elif turn_left:
                    _feedback += positive_conjunctions_sampler() + self.format(turn_left_recommend)
            else:
                _feedback = None

            feedback.hp = _feedback
        if 'hn' in feedback_type:
            if moving_away:
                _feedback = _stage_feedback
                _feedback += negative_conjunctions_sampler() + self.format(hn_feedback)

                if accel:
                    if forward:
                        _feedback += positive_conjunctions_sampler() + self.format(accel_forward_recommend)
                    elif backward:
                        _feedback += positive_conjunctions_sampler() + self.format(accel_backward_recommend)
                else:
                    _feedback += positive_conjunctions_sampler() + self.format(deccel_recommend)
                if turn_right:
                    _feedback += positive_conjunctions_sampler() + self.format(turn_right_recommend)
                elif turn_left:
                    _feedback += positive_conjunctions_sampler() + self.format(turn_left_recommend)
            else:
                _feedback = None

            if is_crashed:
                _feedback = self.format(hn_feedback).capitalize() + positive_conjunctions_sampler() + "the car is crashed, you should be more careful."
         
            if self.debug:
                _feedback = f"""
current_vel_angle: {current_vel_angle}
current_angle: {current_angle}
goal_angle: {goal_angle}
vel angle diff: {self.angle_diff(current_vel_angle, current_angle)}
goal angle diff: {self.angle_diff(goal_angle, current_angle)}
goal dist: {np.linalg.norm(goal_pos - current_pos)}
throttle: {expert_action[0]}
steer: {expert_action[1]}
Accelerate: {accel}
Turn right: {turn_right}
=. Happy day.
"""
            feedback.hn = _feedback
        if 'fp' in feedback_type and not is_crashed:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))

        observation = self._format_obs(self.current_observation)
        info["success"] = bool(info["is_success"])
        return dict(instruction=None, observation=observation, feedback=feedback), reward, terminated or info["success"], truncated, info

    def textualize_observation(self, observation):
        """ Parse np.ndarray observation into text. """
        obs_dict = observation.copy()
        # convert np.ndarray to list
        for k,v in obs_dict.items():
            if isinstance(v, np.ndarray):
                obs_dict[k] = np.array2string(v, precision=10)
            else: # it's a scalar
                obs_dict[k] = f"{v:.10f}"
        observation_text = json.dumps(obs_dict)
        return observation_text

    def textualize_expert_action(self, action):
        """ Parse action into text. """
        # The idea is to return something like
        # f"delta x: {action[0]:.2f}, delta y:{action[1]:.2f}, delta z:{action[2]:.2f}, gripper state:{action[3]:.1f}"
        # or another action text format if the action isn't a delta.
        # TODO should not be the raw action
        return np.array2string(action, precision=10)