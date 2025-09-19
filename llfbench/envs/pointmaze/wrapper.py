from typing import Dict, SupportsFloat, Union
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.pointmaze.prompts import *
from llfbench.envs.pointmaze.task_prompts import (
    pointmaze_prompts,
)
from llfbench.envs.pointmaze.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.pointmaze.utils_prompts.recommend_prompts import forward_recommend, deccel_recommend, turn_left_recommend, turn_right_recommend
from d4rl.pointmaze import waypoint_controller
import importlib
import json
from textwrap import dedent, indent
import re

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

class PointmazeWrapper(LLFWrapper):

    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env, instruction_type, feedback_type)

        self.maze = env.str_maze_spec
        self._policy = waypoint_controller.WaypointController(self.maze)
        self.moving_away_thres = np.pi / 10
        self.moving_forward_thres = np.pi / 8
        self.moving_deccel_thres = np.pi * 5 / 7

    @property
    def pm_env(self):
        return self.env.env

    @property
    def pm_policy(self): # metaworld policy
        return self._policy

    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        return {'q_pos': self.current_pos, 'q_vel': self.current_vel, 'goal': self.env_target}

    @property
    def current_pos(self):
        return self._current_observation[0:2]

    @property
    def current_vel(self):
        return self._current_observation[2:4]
        
    @property
    def env_target(self):
        return self.env.unwrapped._target
    
    @property
    def env_target_grid(self):
        return (int(round(self.env_target[0])), int(round(self.env_target[1])))
    
    @property
    def current_grid(self):
        return (int(round(self.current_pos[0])), int(round(self.current_pos[1])))
    
    @property
    def next_waypoint(self):
        return self.pm_policy._waypoints[self.pm_policy._waypoint_idx]
    
    @property
    def next_waypoint_grid(self):
        nw = self.next_waypoint
        return (int(round(nw[0])), int(round(nw[1])))
    
    @property
    def str_maze(self):
        # Replace '\' with newline and make goal visible
        clean_maze = self.maze.replace('\\', '\n').replace('G', 'O')
        l = len(clean_maze.split('\n')[0]) + 1
        clean_maze = list(clean_maze)

        agent_pos = self.current_grid[0] * l + self.current_grid[1]
        target_pos = self.env_target_grid[0] * l + self.env_target_grid[1]

        # Overlap case: mark with yellow #
        if agent_pos == target_pos:
            clean_maze[agent_pos] = "\033[33mX\033[0m"  # yellow
        else:
            # Mark agent (red A)
            clean_maze[agent_pos] = "\033[31mA\033[0m"  # red
            # Mark target (green G)
            clean_maze[target_pos] = "\033[32mG\033[0m"  # green

        maze = "".join(clean_maze)
        return maze
    
    def angle_between_vectors(self, a, b):
        # dot product and magnitudes
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # avoid floating-point errors outside [-1, 1]
        cos_theta = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
        theta = np.arccos(cos_theta)  # radians in [0, pi]
        return theta

    def signed_angle_between_vectors(self, a, b):
        dot = np.dot(a, b)
        cross = a[0]*b[1] - a[1]*b[0]  # scalar in 2D
        theta = np.arctan2(cross, dot)  # signed angle in (-pi, pi]
        return theta

    def _step(self, action):

        # environment step
        self._current_observation, reward, done, info = self.env.step(action)

        feedback_type = self._feedback_type

        # calculation for the feedback
        if self._prev_grid != self.current_grid and self.current_grid != self.next_waypoint_grid and 'fp' in feedback_type:
            # if the grid changed, and yet the point didn't progress to the next waypoint, it needs recalculation
            self.pm_policy._new_target(self.current_pos, self.env_target)

        expert_action, success = self.pm_policy.get_action(self.current_pos, self.current_vel, self.env_target)
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action

        # decide in what direction the agent should move (8 directions)
        next_waypoint = self.next_waypoint
        del_x, del_y = (next_waypoint - self.current_pos)
        theta = np.arctan2(del_y, del_x)
        theta = (theta + 2 * np.pi) % (2 * np.pi) # normalize angle to [0, 2pi)
        direction_idx = int(np.round(theta / (np.pi / 4))) % 8
        navi_dir = pointmaze_prompts.navigate_direction[direction_idx]
        _stage_feedback = self.format(pointmaze_prompts.navigate_prompts,direction=navi_dir)
        
        # decide if the agent is moving away
        moving_away = self.angle_between_vectors(action, self._prev_expert_action) > self.moving_away_thres
        self._prev_expert_action = expert_action
        self._prev_grid = self.current_grid

        # decide what the agent shoud do. Go straight, turn right, turn left, deccelerate
        acc_theta = self.signed_angle_between_vectors(self.current_vel, expert_action)

        forward = abs(acc_theta) < self.moving_forward_thres
        deccel = abs(acc_theta) > self.moving_deccel_thres
        turn_left = self.moving_forward_thres < acc_theta and acc_theta < np.pi - self.moving_forward_thres
        turn_right = -self.moving_forward_thres > acc_theta and acc_theta > -(np.pi - self.moving_forward_thres)

        # Compute feedback
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=reward)
        if 'hp' in feedback_type:  # moved closer to the expert goal
            if not moving_away:
                _feedback = _stage_feedback
                _feedback += positive_conjunctions_sampler() + self.format(hp_feedback)
                if forward:
                    _feedback += positive_conjunctions_sampler() + self.format(forward_recommend)
                if deccel:
                    _feedback += positive_conjunctions_sampler() + self.format(deccel_recommend)
                if turn_left:
                    _feedback += positive_conjunctions_sampler() + self.format(turn_left_recommend)
                if turn_right:
                    _feedback += positive_conjunctions_sampler() + self.format(turn_right_recommend)
            else:
                _feedback = None
            feedback.hp = _feedback
        if 'hn' in feedback_type:
            if moving_away:
                _feedback = _stage_feedback
                _feedback += negative_conjunctions_sampler() + self.format(hn_feedback)
                if forward:
                    _feedback += positive_conjunctions_sampler() + self.format(forward_recommend)
                if deccel:
                    _feedback += positive_conjunctions_sampler() + self.format(deccel_recommend)
                if turn_left:
                    _feedback += positive_conjunctions_sampler() + self.format(turn_left_recommend)
                if turn_right:
                    _feedback += positive_conjunctions_sampler() + self.format(turn_right_recommend)
            else:
                _feedback = None
            feedback.hn = _feedback
        if 'fp' in feedback_type:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))
        observation = self._format_obs(self.current_observation)
        info.update({'success': success, 'q_pos': self.current_pos, 'q_vel': self.current_vel, 'goal': self.env_target})
        # obs, reward, terminated, truncated, info
        # The current TimeLimit wrapper from tf_agents can't distinguish termination from truncation
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), success, not success and done, info 

    def _reset(self, seed=None, options=None): # there's no variables for the reset function
        self._current_observation, info = self.env.reset(seed=seed, options=options)
        self._prev_expert_action, _ = self.pm_policy.get_action(self.current_pos, self.current_vel, self.env_target)
        self._prev_grid = self.current_grid
        observation = self._format_obs(self.current_observation)
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        instruction = self.format(pm_instruction, task=task)
        info.update({'success': False, 'q_pos': self.current_pos, 'q_vel': self.current_vel, 'goal': self.env_target})
        info['video'] = [self.env.render()[::-1]] if self.env._render_video else None
        return dict(instruction=instruction, observation=observation, feedback=None), info

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
        obs_dict = observation.copy()
        # convert np.ndarray to list
        for k,v in obs_dict.items():
            if isinstance(v, np.ndarray):
                obs_dict[k] = np.array2string(v, precision=10)
            else: # it's a scalar
                obs_dict[k] = f"{v:.10f}"
        observation_text = json.dumps(obs_dict)
        return observation_text