from typing import Dict, SupportsFloat, Union, List
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.kitchen.prompts import *
# from llfbench.envs.kitchen.task_prompts import (
#)
from llfbench.envs.kitchen.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.kitchen.utils_prompts.recommend_prompts import (
    move_down_recommend,
    move_up_recommend,
    move_right_recommend,
    move_left_recommend,
    move_forward_recommend,
    move_backward_recommend,
)
from llfbench.envs.kitchen.scripted_policy import (
    KitchenPolicyConfig,
    KitchenSim,
    ScriptedKitchenPolicy,
)
import importlib
import json
import random
import re
import os
import torch
import pickle
import gymnasium.spaces as spaces

# so that we won't get scientific notation
np.set_printoptions(suppress=True)

#: World axis -> paraphrase pools, used to turn a Cartesian correction into a hint.
#: The kitchen is laid out with +x to the robot's right, +y away from the robot (into the
#: counter) and +z up.
_DIRECTION_PROMPTS = (
    (move_right_recommend, move_left_recommend),      # x
    (move_forward_recommend, move_backward_recommend),  # y
    (move_up_recommend, move_down_recommend),         # z
)


class KitchenWrapper(LLFWrapper):

    """
    Useful links:
    1. https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/kitchen/diffusion_policy_cnn/train_1/checkpoints/epoch%3D3400-test_mean_score%3D0.589.ckpt
    2. https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/kitchen/base.py
    3. https://robotics.farama.org/envs/franka_kitchen/

    Control granularity: one wrapper step is exactly one env step.  In the pinned
    Gymnasium-Robotics 1.2.0 environment, each Cartesian IK iteration advances 0.1 s of
    simulated time; an external step therefore advances ``0.1 * control_steps`` seconds.
    The compact expert profile uses 15 updates for the first three tasks and 8 for the
    contact-sensitive microwave grasp. The wrapper does not add another hidden macro
    loop. Pass ``fast_expert=False`` (or an explicit legacy config/control_steps pair) to
    reproduce the conservative profile; set ``KitchenPolicyConfig.ik_target_leak=0`` for
    bit-for-bit upstream controller dynamics.
    """

    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type, debug: bool = False,
                 policy_config: KitchenPolicyConfig = None):
        super().__init__(env, instruction_type, feedback_type)

        self.task_name = self.env.env_name

        # The scripted expert. There is no pickled mjrl policy for kitchen (the Adroit
        # wrapper this file was copied from loads one); the expert is written in
        # llfbench/envs/kitchen/scripted_policy.py and reads the sim directly.
        self._sim = KitchenSim(self.env)
        self._policy = ScriptedKitchenPolicy(self.env, policy_config)

        self.debug = True
        self._current_observation = None
        self._prev_expert_action = None
        self._expert_action = None
        self._expert_action_t = None
        self.t = 0

    @property
    def kt_policy(self): # franka kitchen policy
        return self._policy

    @property
    def current_observation(self):  # external interface
        return self._current_observation

    @property
    def base_env(self):
        """The gym wrapper stack under this wrapper (TimeLimit/OrderEnforcing/...).

        This is *not* the ``KitchenEnv``; use :attr:`kitchen_env` for that.
        """
        return self.env.env

    @property
    def kitchen_env(self):
        """The ``gymnasium_robotics`` ``KitchenEnv``.

        Every model/data access in this package goes through
        :class:`llfbench.envs.kitchen.scripted_policy.KitchenSim`; this property exists so
        callers can reach ``goal`` / ``tasks_to_complete`` / ``episode_task_completions``.
        """
        return self.env.unwrapped

    @property
    def reward_range(self):
        """The env's reward is the number of tasks completed on a step."""
        return (0.0, float(len(self.kitchen_env.goal)))

    # auxiliary functions for language feedback
    @property
    def expert_action(self):
        """One scripted-expert action for the current state, in the env's action space.

        Shape ``(7,)``, finite, inside ``Box(-1, 1)``: the same normalized Cartesian delta
        plus gripper command the agent emits.

        The reactive expert derives its task and waypoint from the live simulator state;
        querying it does not count as progress.  The result is also memoized per env step
        so feedback and the learner see exactly the same recommendation.
        """
        if self._expert_action is None or self._expert_action_t != self.t:
            self._expert_action = self._policy.get_action(self._current_observation)
            self._expert_action_t = self.t
        return self._expert_action.copy()

    # step functions
    def _step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)

        # 1. one wrapper step == one env step (see the class docstring).
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._current_observation = observation
        self.t += 1
        video = [self.env.render()] if self.env._render_video else None

        feedback_type = self._feedback_type

        # 2. recompute the expert action from the *new* state.
        #
        # Pairing convention (same as MetaworldWrapper._step_general): the feedback about
        # the step that was just taken is judged against `self._prev_expert_action`, which
        # is the expert's action for the state the agent acted *from*. The `fp` suggestion
        # instead uses `expert_action`, the expert's action for the state the agent has
        # arrived *at*, i.e. what to do next.
        expert_action = self.expert_action
        if self._prev_expert_action is None:
            self._prev_expert_action = expert_action.copy()
        target_action = self._prev_expert_action
        self._prev_expert_action = expert_action.copy()

        # 3. build the Feedback object.
        #
        # NOTE this is deliberately a placeholder until the real feedback design lands;
        # the `features` / `feature_reward` signals (cf. _step_box_close_v2 /
        # _step_sweep_v2 in the metaworld wrapper) are not computed here yet.
        agreement = self._action_agreement(action, target_action)
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=reward)
        if 'hp' in feedback_type:
            feedback.hp = self.format(hp_feedback) if agreement else None
        if 'hn' in feedback_type:
            if agreement:
                feedback.hn = None
            else:
                hint = self._direction_hint(target_action[:3] - action[:3])
                feedback.hn = self.format(hn_feedback) + (
                    positive_conjunctions_sampler() + hint if hint is not None else '')
        if 'fp' in feedback_type:
            feedback.fp = self.format(fp_feedback,
                                      expert_action=self.textualize_expert_action(expert_action))

        # 4. assemble the return values.
        kitchen = self.kitchen_env
        expert_diagnostics = self._policy.get_diagnostics()
        self._append_debug_policy_feedback(
            feedback,
            expert_diagnostics,
            action=action,
            expert_action=expert_action,
        )
        info['success'] = bool(len(kitchen.episode_task_completions) == len(kitchen.goal))
        info['video'] = video if self.env._render_video else None
        info['tasks_to_complete'] = self._normalized_tasks_to_complete(info)
        info['expert_diagnostics'] = expert_diagnostics
        observation = self._format_obs(observation)
        return (dict(instruction=None, observation=observation, feedback=feedback),
                float(reward), bool(terminated or info['success']), bool(truncated), info)

    def _reset(self, *, seed = None, options = None):
        # Bug workaround: KitchenEnv.reset() clears `episode_task_completions` but never
        # restores `tasks_to_complete`, which `step` empties as tasks are completed. From
        # the second episode on, already-completed tasks are silently missing from the set
        # `compute_reward` iterates over and can never be scored again. Restore it here.
        kitchen = self.kitchen_env
        kitchen.tasks_to_complete = set(kitchen.goal.keys())
        kitchen.step_task_completions.clear()

        self._current_observation, info = self.env.reset(seed=seed, options=options)

        self.t = 0
        self._expert_action = None
        self._expert_action_t = None
        self._policy.seed(seed)
        self._policy.reset(self._current_observation, info)
        self._prev_expert_action = self.expert_action.copy()

        observation = self._format_obs(self._current_observation)
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        instruction = self.format(kt_instruction, task=task)
        info['success'] = False
        # NOTE unlike metaworld, the kitchen renderer already returns upright frames, so
        # there is no [::-1] flip here (verified by saving a frame at reset).
        info['video'] = [self.env.render()] if self.env._render_video else None
        info['tasks_to_complete'] = self._normalized_tasks_to_complete(info)
        expert_diagnostics = self._policy.get_diagnostics()
        info['expert_diagnostics'] = expert_diagnostics
        feedback = Feedback()
        if 'fp' in self._feedback_type:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(self._prev_expert_action))
        self._append_debug_policy_feedback(
            feedback,
            expert_diagnostics,
            expert_action=self._prev_expert_action,
        )
        return dict(instruction=instruction, observation=observation, feedback=feedback), info

    def _append_debug_policy_feedback(self, feedback, diagnostics, *, action=None,
                                      expert_action=None):
        """Append a readable snapshot of the scripted expert in debug mode.

        Hindsight feedback is not present on every step, so the snapshot is attached to
        the first populated feedback channel.  If the configured feedback types produced
        no text, debug mode still emits the snapshot through ``fp``; otherwise the policy
        state would disappear exactly on the steps that are often most useful to inspect.
        """
        if not self.debug:
            return

        def array_text(value):
            if value is None:
                return None
            return np.array2string(np.asarray(value), precision=6)

        lines = [
            "[scripted_policy]",
            f"[selected_subtask]={diagnostics['selected_subtask']}",
            f"[controller_phase]={diagnostics['controller_phase']}",
            f"[manipulation_stage]={diagnostics['manipulation_stage']}",
            f"[kettle_contacting_fingers]={diagnostics['kettle_contacting_fingers']}",
            f"[kettle_grasp_retained]={diagnostics['kettle_grasp_retained']}",
            f"[microwave_contacting_fingers]={diagnostics['microwave_contacting_fingers']}",
            f"[microwave_contact_depth]={diagnostics['microwave_contact_depth']}",
            f"[microwave_grasp_captured]={diagnostics['microwave_grasp_captured']}",
            f"[microwave_grasp_retained]={diagnostics['microwave_grasp_retained']}",
            f"[microwave_radial_bias]={diagnostics['microwave_radial_bias']}",
            f"[microwave_handle_position]="
            f"{array_text(diagnostics['microwave_handle_position'])}",
            f"[microwave_contact_position]="
            f"{array_text(diagnostics['microwave_contact_position'])}",
            f"[microwave_tool_position]="
            f"{array_text(diagnostics['microwave_tool_position'])}",
            f"[phase_steps]={diagnostics['phase_steps']}",
            f"[target_position]={array_text(diagnostics['target_position'])}",
            f"[position_error]={diagnostics['position_error']:.6f}",
            f"[tool_error]={diagnostics['tool_error']:.6f}",
            f"[orientation_error]={diagnostics['orientation_error']:.6f}",
            f"[finger_opening]={diagnostics['finger_opening']:.6f}",
            f"[task_distance]={diagnostics['task_distance']}",
            f"[subtask_complete]={diagnostics['subtask_complete']}",
            f"[retry_count]={diagnostics['retry_count']}",
            f"[completed_order]={diagnostics['completed_order']}",
            f"[abandoned]={diagnostics['abandoned']}.",
        ]
        if action is not None:
            lines.append(f"[action]={array_text(action)}.")
        if expert_action is not None:
            lines.append(f"[expert_action]={array_text(expert_action)}. Good day.")
        # Keep following feedback channels (usually ``fp``) on a separate line after the
        # multi-line snapshot when LLFWrapper verbalizes the Feedback object.
        debug_text = "\n".join(lines) + "\n"

        for feedback_name in ('hp', 'hn', 'fp', 'r'):
            current = getattr(feedback, feedback_name)
            if current is not None:
                setattr(feedback, feedback_name, f"{current}\n{debug_text}")
                break
        else:
            feedback.fp = debug_text

    @staticmethod
    def _normalized_tasks_to_complete(info):
        """`info['tasks_to_complete']` is a dict at reset and a set at step; normalize.

        (At reset `KitchenEnv` reports `self.task_to_complete` -- note the missing `s`, a
        different attribute -- which is a copy of the goal *dict*.)
        """
        tasks = info.get('tasks_to_complete', ())
        return sorted(str(t) for t in tasks)

    @staticmethod
    def _action_agreement(action, expert_action, tolerance: float = 0.0):
        """Whether the agent's action broadly agrees with the expert's.

        Placeholder heuristic: the commanded translation must not point away from the
        expert's, and the gripper command must have the same sign.
        """
        agent_xyz = np.asarray(action[:3], dtype=np.float64)
        expert_xyz = np.asarray(expert_action[:3], dtype=np.float64)
        if np.linalg.norm(expert_xyz) < 1e-8:
            aligned = True
        else:
            aligned = float(np.dot(agent_xyz, expert_xyz)) > tolerance
        gripper_ok = np.sign(action[6]) == np.sign(expert_action[6]) or abs(expert_action[6]) < 1e-8
        return bool(aligned and gripper_ok)

    @staticmethod
    def _direction_hint(correction):
        """Turn the largest component of a Cartesian correction into a recommendation."""
        correction = np.asarray(correction, dtype=np.float64)
        axis = int(np.argmax(np.abs(correction)))
        if abs(correction[axis]) < 1e-6:
            return None
        positive, negative = _DIRECTION_PROMPTS[axis]
        return random.choice(positive if correction[axis] > 0 else negative)

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
                    # Namespace the key: the kitchen observation nests 'achieved_goal' and
                    # 'desired_goal' dicts under the *same* task names, so a flat key would
                    # silently drop one of them.
                    key = f"{k}/{vk}"
                    if isinstance(vv, np.ndarray):
                        obs_dict[key] = np.array2string(vv, precision=10)
                    elif isinstance(vv, torch.Tensor):
                        obs_dict[key] = str(vv.flatten().tolist()).replace(',', '')
                    else: # it's a scalar
                        obs_dict[key] = f"{vv:.10f}"

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
    
