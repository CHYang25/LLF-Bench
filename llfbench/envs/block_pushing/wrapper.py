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

STAGE_MOVE_TO_FIRST_BLOCK = 0
STAGE_PUSH_FIRST_BLOCK_TO_FIRST_GOAL = 1
STAGE_MOVE_TO_SECOND_BLOCK = 2
STAGE_PUSH_SECOND_BLOCK_TO_SECOND_GOAL = 3

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

class BlockPushingWrapper(LLFWrapper):
    
    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type, debug: bool = True, mani_oracle: bool = True):
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
        self._first_block = None
        self._first_target = None
        self._second_block = None
        self._second_target = None
        self.stage = STAGE_MOVE_TO_FIRST_BLOCK
        self.debug = debug
        self.mani_oracle = mani_oracle
        self.control_time_out = 20
        self.control_tolerance = 1e-2
        self.max_step_distance = 0.03

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
    def bp_policy_stage(self):
        return {
            'current_block': self.bp_policy.current_block,
            'current_target': self.bp_policy.current_target,
            'phase': self.bp_policy.phase,
            'has_switched': self.bp_policy.has_switched
        }
    
    @property
    def current_time_step(self):
        return self._current_time_step
    
    @property
    def current_observation(self):
        return self.current_time_step.observation

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
    def previous_time_step(self):
        return self._previous_time_step
    
    @property
    def previous_observation(self):
        return self.previous_time_step.observation

    # Observation attributes
    @property
    def _previous_block_translation(self):
        return self.previous_observation["block_translation"]
    
    @property
    def _previous_block_orientation(self):
        return self.previous_observation["block_orientation"]

    @property
    def _previous_block2_translation(self):
        return self.previous_observation["block2_translation"]

    @property
    def _previous_block2_orientation(self):
        return self.previous_observation["block2_orientation"]

    @property
    def _previous_effector_translation(self):
        return self.previous_observation["effector_translation"]

    @property
    def _previous_effector_target_translation(self):
        return self.previous_observation["effector_target_translation"]

    @property
    def _previous_target_translation(self):
        return self.previous_observation["target_translation"]
    
    @property
    def _previous_target_orientation(self):
        return self.previous_observation["target_orientation"]
    
    @property
    def _previous_target2_translation(self):
        return self.previous_observation["target2_translation"]
    
    @property
    def _previous_target2_orientation(self):
        return self.previous_observation["target2_orientation"]
    
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
        # print([
        #     np.linalg.norm(self._current_block_translation - self._current_target_translation) < self.env.goal_dist_tolerance,
        #     np.linalg.norm(self._current_block2_translation - self._current_target_translation) < self.env.goal_dist_tolerance,
        #     np.linalg.norm(self._current_block_translation - self._current_target2_translation) < self.env.goal_dist_tolerance,
        #     np.linalg.norm(self._current_block2_translation - self._current_target2_translation) < self.env.goal_dist_tolerance,
        # ])
        # Two blocks can't be in one target.
        return sum([
            np.linalg.norm(self._current_block_translation - self._current_target_translation) < self.env.goal_dist_tolerance or\
            np.linalg.norm(self._current_block2_translation - self._current_target_translation) < self.env.goal_dist_tolerance,
            np.linalg.norm(self._current_block_translation - self._current_target2_translation) < self.env.goal_dist_tolerance or\
            np.linalg.norm(self._current_block2_translation - self._current_target2_translation) < self.env.goal_dist_tolerance,
        ])

    @property
    def _current_block_to_reach(self):
        return self.bp_policy.current_block
    
    @property
    def _current_target_to_reach(self):
        return self.bp_policy.current_target
    
    @property
    def block_effector_dist_tolerance(self):
        return self.bp_policy.block_effector_dist_tolerance * 2.0
    
    def set_block_target_order(self):
        if self.mani_oracle:
            return
        """
        This function would determine the block target order. Here's the order
        STAGE_MOVE_TO_FIRST_BLOCK:
            1. The end effector must approach one of the blocks, or return negative feedback. Blocks order could be estimated.
        STAGE_PUSH_FIRST_BLOCK_TO_FIRST_GOAL:
            2. The end effector pushing one of the blocks, so blocks order could be determined.
            3. The first block must be approaching one of the targets, or return negative feedback. Targets order could be estimated.
        STAGE_MOVE_TO_SECOND_BLOCK, STAGE_PUSH_SECOND_BLOCK_TO_SECOND_GOAL:
            4. The first block entered one of targets, so targets order could be determiend.
            5. The second block and second target can be determiend.

        block2 is green
        """
        if self.stage == STAGE_MOVE_TO_FIRST_BLOCK:
            # consider the distance to the two blocks, and where the ee is heading to
            prev_dist_to_block = np.linalg.norm(self._previous_effector_translation - self._previous_block_translation)
            prev_dist_to_block2 = np.linalg.norm(self._previous_effector_translation - self._previous_block2_translation)
            # if self.debug:
            #     assert np.linalg.norm(self._previous_block_translation - self._current_block_translation) < self.bp_policy.block_effector_dist_tolerance
            #     assert np.linalg.norm(self._previous_block2_translation - self._current_block2_translation) < self.bp_policy.block_effector_dist_tolerance
            cur_dist_to_block = np.linalg.norm(self._current_effector_translation - self._current_block_translation)
            cur_dist_to_block2 = np.linalg.norm(self._current_effector_translation - self._current_block2_translation)

            if prev_dist_to_block - cur_dist_to_block > prev_dist_to_block2 - cur_dist_to_block2 \
                or cur_dist_to_block < self.block_effector_dist_tolerance:
                # Set current block to "block"
                self._first_block = "block"
                self._second_block = "block2"
            else:
                # Set current block to "block2"
                self._first_block = "block2"
                self._second_block = "block"
            
            self.bp_policy.set_phase("move_to_pre_block")
            self.bp_policy.set_block_target_order(
                first_block = self._first_block,
                first_target = self.bp_policy.first_target,
                second_block = self._second_block,
                second_target = self.bp_policy.second_target,
            )
        
        elif self.stage == STAGE_PUSH_FIRST_BLOCK_TO_FIRST_GOAL:
            # consider the distance to the two blocks, and where the ee is heading to
            cur_dist_to_block = np.linalg.norm(self._current_effector_translation - self._current_block_translation)
            cur_dist_to_block2 = np.linalg.norm(self._current_effector_translation - self._current_block2_translation)
            if cur_dist_to_block < cur_dist_to_block2:
                self._first_block = "block"
                self._second_block = "block2"
                # assert self._current_block_to_reach == "block"
                # assert self._first_block == "block"
            else:
                # assert self._current_block_to_reach == "block2"
                # assert self._first_block == "block2"
                self._first_block = "block2"
                self._second_block = "block"

            # self.bp_policy.set_phase("move_to_pre_block")
            self.bp_policy.set_block_target_order(
                first_block = self._first_block,
                first_target = self.bp_policy.first_target,
                second_block = self._second_block,
                second_target = self.bp_policy.second_target,
            )

            prev_dist_to_target = np.linalg.norm(self.previous_observation[f"{self._current_block_to_reach}_translation"] - self._previous_target_translation)
            prev_dist_to_target2 = np.linalg.norm(self.previous_observation[f"{self._current_block_to_reach}_translation"] - self._previous_target2_translation)
            # if self.debug:
            #     assert np.linalg.norm(self._previous_target_translation - self._current_target_translation) == 0
            #     assert np.linalg.norm(self._previous_target2_translation - self._current_target2_translation) == 0
            cur_dist_to_target = np.linalg.norm(self.current_observation[f"{self._current_block_to_reach}_translation"] - self._current_target_translation)
            cur_dist_to_target2 = np.linalg.norm(self.current_observation[f"{self._current_block_to_reach}_translation"] - self._current_target2_translation)

            if prev_dist_to_target - cur_dist_to_target > prev_dist_to_target2 - cur_dist_to_target2 \
                or cur_dist_to_target < self.env.goal_dist_tolerance:
                # Set current target to "target"
                self._first_target = "target"
                self._second_target = "target2"
            else:
                # Set current target to "target2"
                self._first_target = "target2"
                self._second_target = "target"


            if "orient" not in self.bp_policy.phase and self.bp_policy.phase != "move_to_pre_block":
                self.bp_policy.set_phase("push_block")

            self.bp_policy.set_block_target_order(
                first_block = self.bp_policy.first_block,
                first_target = self._first_target,
                second_block = self.bp_policy.second_block,
                second_target = self._second_target,
            )

            # FIXME: Change this before evaluating the new stage
            # print(cur_dist_to_target)
            # print(cur_dist_to_target2)
            if cur_dist_to_target < self.env.goal_dist_tolerance or cur_dist_to_target2 < self.env.goal_dist_tolerance:
                self.bp_policy.set_current_block(self.bp_policy.second_block)
                self.bp_policy.set_current_target(self.bp_policy.second_target)
                self.bp_policy.set_has_switched(True)
                self.bp_policy.set_phase("return_to_first_preblock")

        elif self.stage == STAGE_MOVE_TO_SECOND_BLOCK:
            first_block_target_dist = np.linalg.norm(self.current_observation[f"{self._first_block}_translation"] - self.current_observation[f"{self._first_target}_translation"])
            # if self.debug:
            #     assert first_block_target_dist < self.env.goal_dist_tolerance

            if self._current_block_to_reach == self._first_block:
                self.bp_policy.set_current_block(self.bp_policy.second_block)
                self.bp_policy.set_current_target(self.bp_policy.second_target)
                self.bp_policy.set_has_switched(True)
                self.bp_policy.set_phase("return_to_first_preblock")

            if not self.bp_policy.phase == 'return_to_origin' and not self.bp_policy.phase == 'move_to_pre_block':
                self.bp_policy.set_phase("return_to_first_preblock")

        elif self.stage == STAGE_PUSH_SECOND_BLOCK_TO_SECOND_GOAL:
            first_block_target_dist = np.linalg.norm(self.current_observation[f"{self._first_block}_translation"] - self.current_observation[f"{self._first_target}_translation"])

            if first_block_target_dist >= self.env.goal_dist_tolerance:
                """
                This happens when the first block pushes the second block into the first goal.
                Thus, switch the block order manually
                """
                tmp = self._second_block
                self._second_block = self._first_block
                self._first_block = tmp
                self.bp_policy.set_block_target_order(
                    first_block = self._first_block,
                    first_target = self.bp_policy.first_target,
                    second_block = self._second_block,
                    second_target = self.bp_policy.second_target,
                )
                self.bp_policy.set_current_block(self.bp_policy.second_block)
                self.bp_policy.set_current_target(self.bp_policy.second_target)
                self.bp_policy.set_has_switched(True)
                self.bp_policy.set_phase("return_to_first_preblock")

            first_block_target_dist = np.linalg.norm(self.current_observation[f"{self._first_block}_translation"] - self.current_observation[f"{self._first_target}_translation"])
            # if self.debug:
            #     assert first_block_target_dist < self.env.goal_dist_tolerance

            if "orient" not in self.bp_policy.phase and self.bp_policy.phase != "move_to_pre_block":
                self.bp_policy.set_phase("push_block")
        
        else:
            raise ValueError("Invalid Stage to evaluate block target order.")

    def control(self, desired_pos):
        assert len(desired_pos) == 2, "The action should be a 2D vector."
        control = desired_pos - self._current_effector_translation
        length = np.linalg.norm(control)
        # if length * 0.3 > self.max_step_distance:
        #     return (control / length) * self.max_step_distance
        if length > self.max_step_distance:
            return control * 0.1
        else:
            return control

    # step function
    def _step(self, action):
        # step
        self._previous_time_step = self._current_time_step
        previous_effector_translation = self._previous_effector_translation

        action = action.copy()

        desired_pos = action + previous_effector_translation
        for _ in range(self.control_time_out):
            control = self.control(desired_pos)
            time_step = self.env.step(control)
            self._current_time_step = time_step

            if np.abs(desired_pos - self._current_effector_translation).max() < self.control_tolerance:
                break
        # set stages
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
        ) < self.block_effector_dist_tolerance
        current_target_to_reach_reached = np.linalg.norm(
            self.current_observation[f"{self._current_target_to_reach}_translation"] - self.current_observation[f"{self._current_block_to_reach}_translation"]
        ) < self.env.goal_dist_tolerance

        if num_reached_target_blocks == 0:
            if not current_block_to_reach_reached:
                self.stage = STAGE_MOVE_TO_FIRST_BLOCK
                _stage_feedback = self.format(blockpushing_multimodal.move_to_first_block_feedback)
            elif current_block_to_reach_reached and not current_target_to_reach_reached:
                self.stage = STAGE_PUSH_FIRST_BLOCK_TO_FIRST_GOAL
                _stage_feedback = self.format(blockpushing_multimodal.push_first_block_to_first_goal_feedback)
            else:
                raise ValueError
        elif num_reached_target_blocks == 1:
            if not current_block_to_reach_reached:
                self.stage = STAGE_MOVE_TO_SECOND_BLOCK
                _stage_feedback = self.format(blockpushing_multimodal.move_to_second_block_feedback)
            elif current_block_to_reach_reached and not current_target_to_reach_reached:
                self.stage = STAGE_PUSH_SECOND_BLOCK_TO_SECOND_GOAL
                _stage_feedback = self.format(blockpushing_multimodal.push_second_block_to_second_goal_feedback)
            else:
                # if self.debug:
                #     assert self._current_block_to_reach == self._first_block and self._current_target_to_reach == self._first_target
                self.stage = STAGE_MOVE_TO_SECOND_BLOCK
                _stage_feedback = self.format(blockpushing_multimodal.move_to_second_block_feedback)
        else:
            _stage_feedback = self.format(blockpushing_multimodal.push_second_block_to_second_goal_feedback)

        # _stage_feedback += str(np.linalg.norm(
        #     self.current_observation[f"{self._current_block_to_reach}_translation"] - self._current_effector_translation
        # ))

        self.set_block_target_order()

        # After the correct stage is set, copy the correct expert action
        feedback_type = self._feedback_type
        expert_action = self.expert_action()
        # if self._prev_expert_action is None:
        # self._prev_expert_action = expert_action.copy()

        target_effector_translation = self._prev_expert_action.copy() + previous_effector_translation

        self._prev_expert_action = expert_action.copy()
        recommend_action = expert_action.copy()

        # moving_away = np.linalg.norm(target_effector_translation - previous_effector_translation) < np.linalg.norm(target_effector_translation - self._current_effector_translation)
        moving_away_axis = [
            tet - pet < tet - cet
            for tet, pet, cet in zip(target_effector_translation, previous_effector_translation, self._current_effector_translation)
        ]

        moving_away_direction = direction_converter(target_effector_translation - self._current_effector_translation)
        moving_away_degree = degree_adverb_converter(target_effector_translation - self._current_effector_translation)

        if self.stage == STAGE_MOVE_TO_FIRST_BLOCK:
            """
            If the stage is to move to the first block, 
            then as soon as the effector is approaching one of the blocks, moving away shall be false.
            Since, it couldn't be determined which one is the first block to reach.
            It should at least approaching one of the blocks.
            """
            moving_away = ((np.linalg.norm(self._previous_effector_translation - self._previous_block_translation) \
                            <= np.linalg.norm(self._current_effector_translation - self._current_block_translation)) \
                                and (np.linalg.norm(self._previous_effector_translation - self._previous_block2_translation) \
                                     <= np.linalg.norm(self._current_effector_translation - self._current_block2_translation))) 
        
        if self.stage == STAGE_PUSH_FIRST_BLOCK_TO_FIRST_GOAL:
            """
            If the stage is to push to the first goal, 
            then as soon as the effector is pushing the block to one of the targets, moving away shall be false.
            Since, it couldn't be determined which one is the first block to reach.
            The current block should be at least approaching one of the targets.
            """
            moving_away = ((np.linalg.norm(self.previous_observation[f"{self._current_block_to_reach}_translation"] - self._previous_target_translation) \
                            <= np.linalg.norm(self.current_observation[f"{self._current_block_to_reach}_translation"] - self._current_target_translation)) \
                                and (np.linalg.norm(self.previous_observation[f"{self._current_block_to_reach}_translation"] - self._previous_target2_translation) \
                                    <= np.linalg.norm(self.current_observation[f"{self._current_block_to_reach}_translation"] - self._current_target2_translation))) 

        if self.stage == STAGE_MOVE_TO_SECOND_BLOCK:
            """
            If the stage is to move to the second block, 
            we need to specify the stages to first move to the left,
            then approach the block.
            """
            if self.bp_policy.phase == 'return_to_first_preblock':
                moving_away = (np.linalg.norm(self.bp_policy.first_preblock - self._previous_effector_translation) \
                                <= np.linalg.norm(self.bp_policy.first_preblock - self._current_effector_translation))
            elif self.bp_policy.phase == 'return_to_origin':
                moving_away = (np.linalg.norm(self.bp_policy.origin - self._previous_effector_translation) \
                                <= np.linalg.norm(self.bp_policy.origin - self._current_effector_translation))
            else:
                moving_away = (np.linalg.norm(self._previous_effector_translation - self.previous_observation[f"{self._current_block_to_reach}_translation"]) \
                                <= np.linalg.norm(self._current_effector_translation - self.current_observation[f"{self._current_block_to_reach}_translation"]))
        
        if self.stage == STAGE_PUSH_SECOND_BLOCK_TO_SECOND_GOAL:
            """
            If the stage is to push the second block,
            just see whether the block is closer to the target
            """
            moving_away = (np.linalg.norm(self.previous_observation[f"{self._current_block_to_reach}_translation"] - self.previous_observation[f"{self._current_target_to_reach}_translation"]) \
                                <= np.linalg.norm(self.current_observation[f"{self._current_block_to_reach}_translation"] - self.current_observation[f"{self._current_target_to_reach}_translation"]))

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
                _feedback += positive_conjunctions_sampler() + self.format(hp_feedback)

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        conj = negative_conjunctions_sampler()
                        _feedback += conj + recommend_templates_sampler().format(degree=degree, direction=direction)

                a = self.bp_policy_stage
                if self.debug:
                    feedback.hp = f"""
{_feedback} 

action: {action},

expert_action: {recommend_action},

curblock: {a['current_block']}

curtarget: {a['current_target']}

phase: {a['phase']}

has_switched: {a['has_switched']}

stage: {self.stage}."""
                else:
                    feedback.hp = _feedback

        if 'hn' in feedback_type:  # moved away from the expert goal
            # _feedback = self.format(hn_feedback) if moving_away else None
            if moving_away:
                _feedback = _stage_feedback
                _feedback += positive_conjunctions_sampler() + self.format(hn_feedback)

                for away, direction, degree in zip(moving_away_axis, moving_away_direction, moving_away_degree):
                    if away:
                        _feedback += positive_conjunctions_sampler() + recommend_templates_sampler().format(degree=degree, direction=direction)

                a = self.bp_policy_stage
                if self.debug:
                    feedback.hn = f"""
{_feedback} 

action: {action},

expert_action: {recommend_action},

curblock: {a['current_block']}

curtarget: {a['current_target']}

phase: {a['phase']}

has_switched: {a['has_switched']}

stage: {self.stage}."""
                else:
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
        self._current_time_step = self.env.reset(seed=seed, options=options)
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