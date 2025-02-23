from typing import Dict, SupportsFloat, Union
import numpy as np
from gymnasium import spaces
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.block_pushing.prompts import *
from llfbench.envs.block_pushing.utils_prompts.degree_prompts import degree_adverb_converter
from llfbench.envs.block_pushing.utils_prompts.direction_prompts import direction_converter
from llfbench.envs.block_pushing.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.block_pushing.utils_prompts.recommend_prompts import recommend_templates_sampler
from llfbench.envs.block_pushing.task_prompts import blockpushing_multimodal
import importlib
import json
from textwrap import dedent, indent
import re
import sys, string

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

class BlockPushingWrapper(LLFWrapper):
    
    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env, instruction_type, feedback_type)
        # load the scripted policy
        if self.env.env_name == "BlockPushMultimodal-v0":
            module = importlib.import_module("llfbench.envs.block_pushing.block_pushing.oracles.multimodal_push_oracle")
            self._policy_name = "MultimodalOrientedPushOracle"
        else:
            raise ValueError(f"The current environment {self.env.env_name} is not supported.")
        
        self._policy = getattr(module, self._policy_name)(self.env)
        self._current_time_step = None
        self._prev_expert_action = None

    @property
    def reward_range(self):
        pass

    @property
    def bp_env(self):
        return self.env.env
    
    @property
    def bp_policy(self):
        return self._policy
    
    @property
    def current_time_step(self):
        return self._current_time_step
    
    @property
    def current_observation(self):
        return self._current_time_step.observation

    # Observation attributes
    @property
    def _current_block_translation(self):
        return self.current_observation["block_translation"]
    
    @property
    def _current_block_orientation(self):
        return self.current_observation["block_orientation"]

    @property
    def _current_block2_translation(self):
        return self.current_observation["block2_translation"]

    @property
    def _current_block2_orientation(self):
        return self.current_observation["block2_orientation"]

    @property
    def _current_effector_translation(self):
        return self.current_observation["effector_translation"]

    @property
    def _current_effector_target_translation(self):
        return self.current_observation["effector_target_translation"]

    @property
    def _current_target_translation(self):
        return self.current_observation["target_translation"]
    
    @property
    def _current_target_orientation(self):
        return self.current_observation["target_orientation"]
    
    @property
    def _current_target2_translation(self):
        return self.current_observation["target2_translation"]
    
    @property
    def _current_target2_orientation(self):
        return self.current_observation["target2_orientation"]
    
    @property
    def _any_block_out_of_bounds(self):
        inbound = lambda x: x[0] >= self.env.workspace_bounds[0][0] and x[0] <= self.env.workspace_bounds[1][0] and x[1] >= self.env.workspace_bounds[0][1] and x[1] <= self.env.workspace_bounds[1][1] 
        return not (inbound(self._current_block_translation) and inbound(self._current_block2_translation))
    
    # auxiliary functions for language feedback
    def expert_action(self):
        """ 
        Compute the desired xy position from the BlockPushing scripted policy.

        So far we cannot implement the less frequent control over the robot arm.
        Thus, we use the default controlling frequency.
        """
        return self.bp_policy.action(self.current_time_step).action
    
    @property
    def num_reached_target_blocks(self):
        return sum([
            np.linalg.norm(self._current_block_translation - self._current_target_translation) < self.bp_policy.goal_dist_tolerance,
            np.linalg.norm(self._current_block_translation - self._current_target2_translation) < self.bp_policy.goal_dist_tolerance,
            np.linalg.norm(self._current_block2_translation - self._current_target_translation) < self.bp_policy.goal_dist_tolerance,
            np.linalg.norm(self._current_block2_translation - self._current_target2_translation) < self.bp_policy.goal_dist_tolerance,
        ])

    @property
    def _current_block_to_reach(self):
        return self.bp_policy.current_block
    
    @property
    def _current_target_to_reach(self):
        return self.bp_policy.current_target
    
    # step function
    def _step(self, action):
        previous_effector_translation = self._current_effector_translation

        action = action.copy()
        time_step = self.env.step(action)
        self._current_time_step = time_step

        feedback_type = self._feedback_type
        expert_action = self.expert_action()
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action.copy()

        target_effector_translation = self._prev_expert_action.copy() + previous_effector_translation

        self._prev_expert_action = expert_action.copy()
        recommend_action = expert_action.copy()

        moving_away = np.linalg.norm(target_effector_translation - previous_effector_translation) < np.linalg.norm(target_effector_translation - self._current_effector_translation)
        moving_away_axis = [
            tet - pet < tet - cet
            for tet, pet, cet in zip(target_effector_translation, previous_effector_translation, self._current_effector_translation)
        ]

        moving_away_direction = direction_converter(target_effector_translation - self._current_effector_translation)
        moving_away_degree = degree_adverb_converter(target_effector_translation - self._current_effector_translation)

        """
        Stages:
        - both blocks not in the target:
            - first reach the one that the oracle wants to reach:
                e.g. You should reach the first block
            - push it to the target
                e.g. You should push the first block to the first goal.
        - one of the blocks in the target:
            - reach to another block, which the oracle then wants to reach
                e.g. You should reach the second block
            - push it to the target        
                e.g. You should push the second block to the second goal.
        """
        num_reached_target_blocks = self.num_reached_target_blocks
        current_block_to_reach_reached = np.linalg.norm(
            self.current_observation[f"{self._current_block_to_reach}_translation"] - self._current_effector_translation
        ) < self.bp_policy.block_effector_dist_tolerance*1.2
        current_target_to_reach_reached = np.linalg.norm(
            self.current_observation[f"{self._current_target_to_reach}_translation"] - self.current_observation[f"{self._current_block_to_reach}_translation"]
        ) < self.bp_policy.goal_dist_tolerance

        if num_reached_target_blocks == 0:
            if not current_block_to_reach_reached:
                _stage_feedback = self.format(blockpushing_multimodal.move_to_first_block_feedback)
            elif current_block_to_reach_reached and not current_target_to_reach_reached:
                _stage_feedback = self.format(blockpushing_multimodal.push_first_block_to_first_goal_feedback)
            else:
                raise ValueError
        elif num_reached_target_blocks == 1:
            if not current_block_to_reach_reached:
                _stage_feedback = self.format(blockpushing_multimodal.move_to_second_block_feedback)
            elif current_block_to_reach_reached and not current_target_to_reach_reached:
                _stage_feedback = self.format(blockpushing_multimodal.push_second_block_to_second_goal_feedback)
            else:
                raise ValueError
        else:
            _stage_feedback = self.format(blockpushing_multimodal.push_second_block_to_second_goal_feedback)

        # _stage_feedback += str(np.linalg.norm(
        #     self.current_observation[f"{self._current_block_to_reach}_translation"] - self._current_effector_translation
        # ))

        # Compute Feedback
        feedback = Feedback()
        reward = self.current_time_step.reward
        if 'r' in  feedback_type:
            feedback.r = self.format(r_feedback, reward=np.round(reward, 3))
        if 'hp' in feedback_type:  # moved closer to the expert goal
            first_conjunction_used = False
            # _feedback = self.format(hp_feedback) if not moving_away else None
            if not moving_away:
                _feedback = _stage_feedback
                _feedback += ' ' + self.format(hp_feedback)

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        conj = positive_conjunctions_sampler() if first_conjunction_used else negative_conjunctions_sampler()
                        _feedback += conj + recommend_templates_sampler().format(degree=degree, direction=direction)
                        first_conjunction_used = True

                feedback.hp = _feedback

        if 'hn' in feedback_type:  # moved away from the expert goal
            # _feedback = self.format(hn_feedback) if moving_away else None
            if moving_away:
                _feedback = _stage_feedback
                _feedback += ' ' + self.format(hn_feedback)

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        _feedback += positive_conjunctions_sampler() + recommend_templates_sampler().format(degree=degree, direction=direction)

                feedback.hn = _feedback

        if 'fp' in feedback_type:  
            # suggest the expert goal
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(recommend_action))
        observation = self._format_obs(self.current_observation)
        info = {'success': self.current_time_step.step_type == 2, 'discount': self.current_time_step.discount}
        # obs, reward, terminated, truncated, info
        # The current TimeLimit wrapper from tf_agents can't distinguish termination from truncation
        return dict(instruction=None, observation=observation, feedback=feedback), float(reward), info['success'], info['success'] or self._any_block_out_of_bounds, info 

    def _reset(self, seed=None, options=None): # there's no variables for the reset function
        self._current_time_step = self.env.reset()
        self._prev_expert_action = self.expert_action() # This would also activate the oracle policy to choose the stages
        observation = self._format_obs(self.current_observation)
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        instruction = self.format(bp_instruction, task=task)
        info = {'success': False, 'discount': self.current_time_step.discount}
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
        return np.array2string(action)

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