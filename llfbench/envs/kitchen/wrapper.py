from typing import Dict, SupportsFloat, Union, List
import numpy as np
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.kitchen.prompts import *
from llfbench.envs.kitchen.task_prompts import franka_kitchen_prompts as kt_prompts
from llfbench.envs.kitchen.utils_prompts.conjunction_prompts import positive_conjunctions_sampler, negative_conjunctions_sampler
from llfbench.envs.kitchen.utils_prompts.degree_prompts import (
    move_degree_adverb_converter,
    turn_degree_adverb_converter,
)
from llfbench.envs.kitchen.utils_prompts.direction_prompts import (
    move_direction_converter,
    turn_direction_converter,
)
from llfbench.envs.kitchen.utils_prompts.recommend_prompts import (
    move_recommend_templates,
    turn_recommend_templates,
    close_gripper_recommend,
    open_gripper_recommend,
)
from llfbench.envs.kitchen.scripted_policy import (
    ACTION_VELOCITY_RANGE,
    FINGER_RANGE,
    MAX_CARTESIAN_DISPLACEMENT,
    MAX_ROTATION_DISPLACEMENT,
    KitchenPolicyConfig,
    KitchenSim,
    ScriptedKitchenPolicy,
)
import mujoco
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

#: Below this the remaining Cartesian error on an axis is not worth a sentence (metres).
_MOVE_DEADBAND = 5e-3
#: Below this the expert is not asking for a meaningful wrist rotation (radians).
_TURN_DEADBAND = 2e-2
#: A translation axis only counts as "moved away from" once it got measurably worse
#: (metres). Set at the damped-least-squares IK's own per-axis jitter: the controller does
#: not travel exactly along the commanded direction, so a perfectly tracked expert reach
#: still regresses a little on some axis (measured over 150 expert steps: median 2.3 mm,
#: p95 13 mm). Genuinely bad actions are far past this (random actions: median 72 mm).
_MOVE_REGRESSION_EPS = 2e-3
#: A commanded direction below this is treated as "no preference" in the agreement test.
_AGREEMENT_EPS = 1e-8

#: Scripted-policy FSM phase -> stage prompt pool, and whether that pool names an object.
#: Kept as a table rather than a chain of ifs because the phase set is fixed by
#: ``scripted_policy.PHASE_SEQUENCE`` plus the three phases outside it.
_PHASE_PROMPTS = {
    'orient_forward': (kt_prompts.orient_forward_feedback, ()),
    'select_subtask': (kt_prompts.select_subtask_feedback, ()),
    'move_to_precontact': (kt_prompts.move_to_precontact_feedback, ('goal', 'object')),
    'align': (kt_prompts.align_feedback, ('goal', 'object')),
    'approach': (kt_prompts.approach_feedback, ('goal', 'object')),
    'contact_or_grasp': (kt_prompts.contact_or_grasp_feedback, ('goal', 'object')),
    'manipulate': (kt_prompts.manipulate_feedback, ('goal', 'manipulation')),
    'kettle_transport': (kt_prompts.kettle_transport_feedback, ('goal',)),
    'recede': (kt_prompts.recede_feedback, ('goal', 'object')),
    'verify': (kt_prompts.verify_feedback, ('goal',)),
    'retreat': (kt_prompts.retreat_feedback, ('object',)),
    'idle': (kt_prompts.idle_feedback, ()),
}


class KitchenWrapper(LLFWrapper):

    """
    Useful links:
    1. https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/kitchen/diffusion_policy_cnn/train_1/checkpoints/epoch%3D3400-test_mean_score%3D0.589.ckpt
    2. https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/kitchen/base.py
    3. https://robotics.farama.org/envs/franka_kitchen/

    Action space: Gymnasium-Robotics 1.2.1's own ``Box(-1, 1, (9,))`` of normalized joint
    velocities, seven arm hinges then two finger slides.  Nothing is swapped or wrapped.

    Control granularity: one wrapper step holds the action for
    ``KitchenPolicyConfig.action_repeat`` env steps, each 0.08 s.  The env has no internal
    control loop to set this with -- 1.2.0's ``control_steps`` went away with its Cartesian
    IK action space -- so the repeat lives here, and the expert asks for a shorter one
    during the contact-sensitive microwave grasp via ``microwave_action_repeat``.
    """

    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type, debug: bool = False,
                 policy_config: KitchenPolicyConfig = None,
                 inference_expert_action: bool = True):
        """
        :param inference_expert_action: whether to run the scripted expert at all. With
            ``False`` the expert's FSM is never queried, so everything derived from it is
            dropped: :attr:`expert_action` and :attr:`expert_cartesian_action` are ``None``,
            ``fp``/``hp``/``hn`` feedback and the debug snapshot are not emitted, the expert
            -derived entries of ``info['features']`` are zero and ``info['expert_diagnostics']``
            is ``None``. Only the reward feedback (``r``), the observation and the env's own
            reward and ``success`` survive. Use it to collect or evaluate rollouts without
            paying for the expert, or to withhold expert supervision from a learner.
        """
        super().__init__(env, instruction_type, feedback_type)

        self.task_name = self.env.env_name

        # The scripted expert. There is no pickled mjrl policy for kitchen (the Adroit
        # wrapper this file was copied from loads one); the expert is written in
        # llfbench/envs/kitchen/scripted_policy.py and reads the sim directly.
        self._sim = KitchenSim(self.env)
        self._policy = ScriptedKitchenPolicy(self.env, policy_config)
        # The policy object is built either way -- it owns the sim handle and the control
        # profile the wrapper reads `action_repeat` from -- but it is only *planned with*
        # when this is set.
        self.inference_expert_action = bool(inference_expert_action)

        self.debug = debug
        self._current_observation = None
        # The Cartesian expert action for the state the agent last acted *from*. All the
        # language feedback is phrased in task space, so the geometry is kept in Cartesian
        # even though the action space the agent and the learner see is joint space.
        self._prev_expert_cartesian = None
        self._expert_cartesian = None
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
    def _refresh_expert_action(self):
        """Recompute and memoize the expert's recommendation for the current state.

        The reactive expert derives its task and waypoint from the live simulator state, so
        querying it does not count as progress.  Memoizing per env step keeps the plan and the
        joint action derived from it consistent, so the feedback and the learner never see a
        recommendation and a rationale computed from different states.

        A no-op when the expert is switched off, which leaves both memoized plans ``None``.
        """
        if not self.inference_expert_action:
            return
        if self._expert_action is None or self._expert_action_t != self.t:
            self._expert_cartesian = self._policy.get_cartesian_plan(self._current_observation)
            self._expert_action = self._policy.cartesian_plan_to_action(self._expert_cartesian)
            self._expert_action_t = self.t

    @property
    def expert_action(self):
        """One scripted-expert action for the current state, in the env's action space.

        Shape ``(9,)``, finite, inside ``Box(-1, 1)``: the normalized joint-position delta the
        agent emits, seven arm hinges then two finger slides.  This is what ``fp`` verbalizes
        and what a learner imitates.

        ``None`` when the wrapper was built with ``inference_expert_action=False``.
        """
        self._refresh_expert_action()
        return None if self._expert_action is None else self._expert_action.copy()

    @property
    def expert_cartesian_action(self):
        """The same recommendation as the expert itself expressed it: a 7-dim Cartesian delta.

        Used only to word the feedback, which talks about where the gripper should move and
        how the wrist should turn -- quantities the joint-space action does not name directly.

        ``None`` when the wrapper was built with ``inference_expert_action=False``.
        """
        self._refresh_expert_action()
        return None if self._expert_cartesian is None else self._expert_cartesian.copy()

    # step functions
    def _step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)

        # 1. one wrapper step == one env step (see the class docstring).
        #
        # The gripper position before the step is the reference the movement guidance uses
        # to decide which axes the agent made *worse*; it has to be read before `env.step`
        # advances the sim.
        eef_before = self._sim.eef_pos.copy()
        # Which way the joint command points in task space, so the Cartesian feedback below can
        # judge it against the expert's Cartesian recommendation. The Jacobian this uses is
        # evaluated at the current pose, so it too has to be read before the step.
        # Nothing consumes it once the expert is off: every sentence and feature it feeds is
        # phrased *against* the expert's plan.
        action_cartesian = (self._cartesian_reading_of(action)
                            if self.inference_expert_action else None)
        # One wrapper step holds the action for `action_repeat` env steps (see the class
        # docstring). Rewards are summed because the kitchen reward counts subtasks completed
        # on a step, so a repeat that finishes two of them must report both.
        reward = 0.0
        terminated = truncated = False
        for _ in range(self._policy.action_repeat):
            observation, step_reward, terminated, truncated, info = self.env.step(action)
            reward += float(step_reward)
            if terminated or truncated:
                break
        self._current_observation = observation
        self.t += 1
        video = [self.env.render()] if self.env._render_video else None

        feedback_type = self._feedback_type

        # 2. recompute the expert action from the *new* state.
        #
        # Pairing convention (same as MetaworldWrapper._step_general): the feedback about
        # the step that was just taken is judged against `self._prev_expert_cartesian`, which
        # is the expert's action for the state the agent acted *from*. The `fp` suggestion
        # instead uses `expert_action`, the expert's action for the state the agent has
        # arrived *at*, i.e. what to do next.
        #
        # With the expert switched off none of this is computed: the FSM is never advanced,
        # so there is no plan to pair, no diagnostics to report the stage from, and no
        # recommendation to judge the agent against. Every part below degrades to its empty
        # value rather than to a stale one.
        expert_action = self.expert_action
        expert_cartesian = self.expert_cartesian_action
        if self.inference_expert_action:
            if self._prev_expert_cartesian is None:
                self._prev_expert_cartesian = expert_cartesian.copy()
            target_action = self._prev_expert_cartesian
            self._prev_expert_cartesian = expert_cartesian.copy()

            # Querying `expert_action` re-ran the reactive FSM against the new state, so the
            # diagnostics below (selected subtask, phase, live waypoint) describe where the
            # agent is *now* -- which is what the stage and guidance sentences must report.
            expert_diagnostics = self._policy.get_diagnostics()
        else:
            target_action = None
            expert_diagnostics = None
        eef_after = self._sim.eef_pos.copy()

        # 3. the three language-feedback parts.
        #
        # 3a. Task progress: which goal subtask is active and how far into it the
        #     manipulation has got. The kitchen goal is an unordered *set* and the expert
        #     picks its order at random, so -- as in BlockPushingWrapper._step -- the stage
        #     is read off the live state rather than assumed. Here the reactive policy has
        #     already re-derived it from the simulator (`cfg.reactive`), so its selected
        #     subtask and FSM phase are that state-derived stage.
        #
        # 3b. Action optimality: does the action the agent just took agree with what the
        #     expert would have done from the same state, over all 7 dimensions?
        #     `None` (rather than True/False) when there is no expert to agree with, which
        #     suppresses both hindsight channels below.
        #
        # 3c. Movement guidance: where the gripper still has to go from here, in Cartesian
        #     terms plus the wrist and the gripper. The env's action space is already a
        #     Cartesian delta (see scripted_policy.validate_action_contract), so no
        #     conversion is needed -- the expert's own waypoint is a world-frame position.
        if self.inference_expert_action:
            _stage_feedback = self._stage_feedback(expert_diagnostics)
            agreement = self._action_agreement(action_cartesian, target_action)
            _recommend_feedback, _gripper_feedback, move_residual, turn_residual = \
                self._movement_guidance(expert_diagnostics, eef_before, eef_after,
                                        action_cartesian, expert_cartesian, target_action)
        else:
            _stage_feedback = ''
            agreement = None
            _recommend_feedback, _gripper_feedback = [], None
            move_residual = np.zeros(3)
            turn_residual = np.zeros(3)

        # 4. raw signals underlying the language feedback: the distance and orientation
        # error to the expert's live waypoint and the active subtask's goal-joint distance
        # (which together pick the stage), the number of subtasks already banked, the
        # signed per-axis Cartesian residual and per-axis wrist residual the guidance is
        # worded from, and how much closer to the waypoint the step actually got
        # (the hp/hn "moved away" signal).
        #
        # All of these but the completion count are read off the expert's plan, so with the
        # expert off they are reported as 0.0 -- the keys stay, so consumers that index the
        # dict keep working, and the feature reward degrades to the completion count alone.
        kitchen = self.kitchen_env
        if self.inference_expert_action:
            target_position = expert_diagnostics['target_position']
            dist_delta = (0.0 if target_position is None else
                          float(np.linalg.norm(np.asarray(target_position) - eef_before)
                                - np.linalg.norm(np.asarray(target_position) - eef_after)))
            task_distance = expert_diagnostics['task_distance']
            waypoint_dist = float(expert_diagnostics['position_error'])
            orientation_error = float(expert_diagnostics['orientation_error'])
            gripper_delta = float(expert_cartesian[6] - action_cartesian[6])
        else:
            dist_delta = 0.0
            task_distance = 0.0
            waypoint_dist = 0.0
            orientation_error = 0.0
            gripper_delta = 0.0
        features = dict(
            waypoint_dist=waypoint_dist,
            orientation_error=orientation_error,
            task_distance=float(0.0 if task_distance is None else task_distance),
            num_completed_subtasks=float(len(kitchen.episode_task_completions)),
            waypoint_delta_x=float(move_residual[0]),
            waypoint_delta_y=float(move_residual[1]),
            waypoint_delta_z=float(move_residual[2]),
            wrist_delta_x=float(turn_residual[0]),
            wrist_delta_y=float(turn_residual[1]),
            wrist_delta_z=float(turn_residual[2]),
            gripper_delta=gripper_delta,
            dist_delta=dist_delta,
        )
        # Alternative reward summing the features (error terms negative, progress terms
        # positive), exposed via info; the env reward remains the step reward.
        feature_reward = (
            - (features['waypoint_dist'] + features['orientation_error']
               + features['task_distance'])
            - (abs(features['waypoint_delta_x']) + abs(features['waypoint_delta_y'])
               + abs(features['waypoint_delta_z']))
            - (abs(features['wrist_delta_x']) + abs(features['wrist_delta_y'])
               + abs(features['wrist_delta_z']))
            - abs(features['gripper_delta'])
            + features['num_completed_subtasks']
            + features['dist_delta']
        )

        # 5. build the Feedback object.
        feedback = Feedback()
        if 'r' in feedback_type:
            feedback.r = self.format(r_feedback, reward=reward)
        if 'hp' in feedback_type and agreement:
            feedback.hp = self.concatenate_sentences(
                stage_feedback=_stage_feedback,
                action_feedback=self.format(hp_feedback),
                reco_feedback=_recommend_feedback,
                action_positive=True,
                gripper_feedback=_gripper_feedback,
            )
        if 'hn' in feedback_type and agreement is False:
            feedback.hn = self.concatenate_sentences(
                stage_feedback=_stage_feedback,
                action_feedback=self.format(hn_feedback),
                reco_feedback=_recommend_feedback,
                action_positive=False,
                gripper_feedback=_gripper_feedback,
            )
        # `fp` names an action only the expert can supply, so it is left unset (rather than
        # set to an empty sentence) when the expert is off: `Feedback` spells "this channel
        # has nothing to say" as None, and `LLFWrapper._verbalize_feedback` indexes the last
        # character of every string it is given, so an empty one would raise there.
        if 'fp' in feedback_type and expert_action is not None:
            feedback.fp = self.format(fp_feedback,
                                      expert_action=self.textualize_expert_action(expert_action))

        # 6. assemble the return values.
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
        info['features'] = features
        info['feature_reward'] = float(feature_reward)
        observation = self._format_obs(observation)
        return (dict(instruction=None, observation=observation, feedback=feedback),
                float(reward), bool(terminated or info['success']), bool(truncated), info)

    def _cartesian_reading_of(self, action):
        """First-order Cartesian reading of a 9-D joint-velocity action, via the Jacobian.

        The feedback words its guidance in task space -- "move left", "turn the wrist" -- and
        judges the agent by comparing against the expert's Cartesian plan, so it needs to know
        which way in task space a joint command actually points.  ``J @ dq`` answers that, and
        is the linearization the policy's own IK inverts to go the other way.

        Must be called *before* the step, since the Jacobian is taken at the current pose.
        """
        robot = self.kitchen_env.robot_env
        model, data = robot.model, robot.data
        action = np.asarray(action, dtype=np.float64).reshape(-1)

        # Joint displacement this action produces over the whole wrapper step.
        horizon = self._policy.action_repeat * robot.dt
        dq_full = np.zeros(model.nv)
        dq_full[:7] = action[:7] * ACTION_VELOCITY_RANGE * horizon

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, self._policy.sim.controller.eef_id)

        # The gripper axis is a command, not a displacement: report the opening this action
        # is driving the fingers toward, on the same [-1, 1] scale the expert's plan uses.
        finger_target = np.clip(
            data.qpos[7:9].mean() + action[7:9].mean() * ACTION_VELOCITY_RANGE * horizon,
            0.0, FINGER_RANGE)
        gripper = 2.0 * finger_target / FINGER_RANGE - 1.0

        return np.clip(np.concatenate([
            (jacp @ dq_full) / MAX_CARTESIAN_DISPLACEMENT,
            (jacr @ dq_full) / MAX_ROTATION_DISPLACEMENT,
            [gripper],
        ]), -1.0, 1.0)

    # -- language feedback parts -------------------------------------------------------
    def _stage_feedback(self, diagnostics):
        """Part 1: what the arm is doing now, as subtask + manipulation phase.

        The subtask is whichever goal element the reactive expert has selected for the
        current state, and the phase is that subtask's point in
        ``scripted_policy.PHASE_SEQUENCE`` (reach -> align -> approach -> grasp ->
        manipulate -> recede -> verify -> retreat).
        """
        task = diagnostics['selected_subtask']
        prompts, slots = _PHASE_PROMPTS.get(
            diagnostics['controller_phase'],
            (kt_prompts.select_subtask_feedback, ()))

        # A phase template that names the subtask is unusable without one; fall back to
        # the subtask-selection sentence rather than emitting a dangling "You are None.".
        if task is None and slots:
            prompts, slots = kt_prompts.select_subtask_feedback, ()

        kwargs = {}
        if 'goal' in slots:
            kwargs['goal'] = self.format(
                kt_prompts.GOAL_PHRASES.get(task, kt_prompts.unknown_goal_phrase))
        if 'object' in slots:
            kwargs['object'] = self.format(
                kt_prompts.OBJECT_PHRASES.get(task, kt_prompts.unknown_object_phrase))
        if 'manipulation' in slots:
            kwargs['manipulation'] = self.format(
                kt_prompts.MANIPULATION_PHRASES.get(
                    task, kt_prompts.unknown_manipulation_phrase))
        return self.format(prompts, **kwargs)

    def _movement_guidance(self, diagnostics, eef_before, eef_after, action, expert_action,
                           prev_expert_action):
        """Part 3: Cartesian, wrist and gripper guidance.

        All three action arguments are in the *Cartesian* 7-dim convention, not the joint-space
        action space the agent emits: the agent's command arrives already mapped through the
        end-effector Jacobian (:meth:`_cartesian_reading_of`), and the two expert
        arguments are the scripted policy's own output before it is converted to joints. This
        section reasons about where the gripper should travel and how the wrist should turn,
        neither of which a joint-position delta names directly.

        Returns ``(recommendation sentences, gripper sentence or None, move_residual,
        turn_residual)``.

        The Cartesian content is the residual to the expert's live waypoint,
        ``target_position - eef_pos``, so a degree adverb maps to a real distance in
        metres instead of to a step-capped command. A translation axis is only mentioned
        when the step just taken made that axis *worse* -- the same "moving away" filter
        ManiskillWrapper applies -- which keeps the advice corrective and the sentence
        short.

        The wrist works the same way. ``rotation_action`` recomputes the full orientation
        error every step and only then caps it, so the expert's own ``action[3:6]`` is the
        remaining rotation correction; comparing the expert's command for the state the
        agent arrived at against its command for the state the agent acted from is the
        exact rotational analogue of the translation regression test.
        """
        target_position = diagnostics['target_position']
        if target_position is None:
            # IDLE and any phase without a waypoint: nothing to steer toward.
            move_residual = np.zeros(3)
            moving_away_axis = [False, False, False]
        else:
            target_position = np.asarray(target_position, dtype=np.float64)
            move_residual = target_position - eef_after
            residual_before = target_position - eef_before
            moving_away_axis = [
                bool(abs(move_residual[i]) > abs(residual_before[i]) + _MOVE_REGRESSION_EPS
                     and abs(move_residual[i]) > _MOVE_DEADBAND)
                for i in range(3)
            ]

        # Radians of world-frame wrist rotation the expert still wants from here.
        turn_residual = np.asarray(expert_action[3:6], dtype=np.float64) * MAX_ROTATION_DISPLACEMENT
        # A wrist axis is faulted by comparing the agent's command against the expert's
        # command *for the same state*, which is what the hp/hn verdict compares too. The
        # translation test cannot be posed this way -- there the sim gives a real
        # before/after position -- but a rotation regression test would misfire instead,
        # because the desired frame itself changes when a skill advances a phase, so the
        # expert's own command can grow through no fault of the agent (measured: it does so
        # on 36 of 150 perfectly-tracked steps).
        turn_shortfall = ((np.asarray(prev_expert_action[3:6], dtype=np.float64)
                           - np.asarray(action[3:6], dtype=np.float64))
                          * MAX_ROTATION_DISPLACEMENT)
        turning_away_axis = [
            bool(abs(turn_shortfall[i]) > _TURN_DEADBAND
                 and abs(turn_residual[i]) > _TURN_DEADBAND)
            for i in range(3)
        ]

        move_direction = move_direction_converter(move_residual)
        move_degree = move_degree_adverb_converter(move_residual)
        turn_direction = turn_direction_converter(turn_residual)
        turn_degree = turn_degree_adverb_converter(turn_residual)

        recommendations = [
            self.format(move_recommend_templates, direction=direction, degree=degree)
            for away, direction, degree in zip(moving_away_axis, move_direction, move_degree)
            if away
        ] + [
            self.format(turn_recommend_templates, direction=direction, degree=degree)
            for away, direction, degree in zip(turning_away_axis, turn_direction, turn_degree)
            if away
        ]

        # The gripper clause is corrective only: it fires when the command the agent just
        # issued disagrees with what the expert wants from the state now reached.
        gripper_feedback = None
        if abs(expert_action[6]) > _AGREEMENT_EPS and np.sign(action[6]) != np.sign(expert_action[6]):
            gripper_feedback = self.format(
                open_gripper_recommend if expert_action[6] > 0 else close_gripper_recommend)

        return recommendations, gripper_feedback, move_residual, turn_residual

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
        self._expert_cartesian = None
        self._expert_action_t = None
        self._policy.seed(seed)
        # The policy is reset even when it will not be planned with: it owns the control
        # profile `action_repeat` is read from, and `kt_policy` stays a usable handle.
        self._policy.reset(self._current_observation, info)
        self._prev_expert_cartesian = (self.expert_cartesian_action.copy()
                                       if self.inference_expert_action else None)

        observation = self._format_obs(self._current_observation)
        task = re.search(r'(.*)-v[0-9]', self.env.env_name).group(1)
        instruction = self.format(kt_instruction, task=task)
        info['success'] = False
        # NOTE unlike metaworld, the kitchen renderer already returns upright frames, so
        # there is no [::-1] flip here (verified by saving a frame at reset).
        info['video'] = [self.env.render()] if self.env._render_video else None
        info['tasks_to_complete'] = self._normalized_tasks_to_complete(info)
        expert_diagnostics = (self._policy.get_diagnostics()
                              if self.inference_expert_action else None)
        info['expert_diagnostics'] = expert_diagnostics
        feedback = Feedback()
        # `fp` names the action in the space the agent acts in, so it is the joint-space
        # recommendation that is verbalized here, not the Cartesian one kept for the geometry.
        # It is `None` with the expert off, and then no channel is populated at all.
        expert_action = self.expert_action
        if 'fp' in self._feedback_type and expert_action is not None:
            feedback.fp = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))
        self._append_debug_policy_feedback(
            feedback,
            expert_diagnostics,
            expert_action=expert_action,
        )
        return dict(instruction=instruction, observation=observation, feedback=feedback), info

    def _append_debug_policy_feedback(self, feedback, diagnostics, *, action=None,
                                      expert_action=None):
        """Append a readable snapshot of the scripted expert in debug mode.

        Hindsight feedback is not present on every step, so the snapshot is attached to
        the first populated feedback channel.  If the configured feedback types produced
        no text, debug mode still emits the snapshot through ``fp``; otherwise the policy
        state would disappear exactly on the steps that are often most useful to inspect.

        Nothing is appended when the expert is off: there are no diagnostics to report, and
        an abridged snapshot would still end in a bracketed array, which downstream parsers
        read as the expert action (see ``KitchenLowdimRunner._expert_action_from_feedback``).
        """
        if not self.debug or diagnostics is None:
            return

        def _fmt(value):
            # These are None until the microwave skill freezes its exit line, and the
            # snapshot has to stay readable on every other step and every other subtask.
            return "n/a" if value is None else f"{float(value):.6f}"

        def array_text(value):
            if value is None:
                return None
            return np.array2string(np.asarray(value), precision=6)

        lines = [
            f"[scripted_policy] [eef_position]="
            f"{array_text(diagnostics['eef_position'])}",
            f"[kettle_position]={array_text(diagnostics['kettle_position'])}",
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
            f"[microwave_recede_progress]="
            f"{_fmt(diagnostics['microwave_recede_progress'])}"
            f" of {_fmt(diagnostics['microwave_recede_progress_target'])}",
            f"[microwave_recede_error]="
            f"{_fmt(diagnostics['microwave_recede_error'])}"
            f" (recede also exits below "
            f"{_fmt(diagnostics['microwave_recede_tolerance'])})",
            f"[phase_steps]={diagnostics['phase_steps']}",
            f"[target_position]={array_text(diagnostics['target_position'])}",
            f"[position_error]={diagnostics['position_error']:.6f}",
            f"[tool_error]={diagnostics['tool_error']:.6f}",
            f"[orientation_error]={diagnostics['orientation_error']:.6f}"
            f" (orient_forward exits below {diagnostics['yaw_tolerance']:.3f})",
            f"[orient_forward_clearance_error]="
            f"{diagnostics['orient_forward_clearance_error']:.6f}"
            f" (orient_forward exits below "
            f"{diagnostics['orient_forward_position_tolerance']:.3f})",
            f"[initial_orientation_complete]="
            f"{diagnostics['initial_orientation_complete']}",
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
    def _cosine_agreement(agent, expert, tolerance: float):
        """Whether ``agent`` points the same way as ``expert``, by cosine similarity.

        A near-zero expert command expresses no preference and so always agrees; cosine
        rather than a raw dot product keeps the threshold independent of how large the
        expert's own step happens to be.
        """
        agent = np.asarray(agent, dtype=np.float64)
        expert = np.asarray(expert, dtype=np.float64)
        expert_norm = float(np.linalg.norm(expert))
        if expert_norm < _AGREEMENT_EPS:
            return True
        agent_norm = float(np.linalg.norm(agent))
        if agent_norm < _AGREEMENT_EPS:
            return False
        return bool(float(np.dot(agent, expert)) / (agent_norm * expert_norm) > tolerance)

    @classmethod
    def _action_agreement(cls, action, expert_action, tolerance: float = 0.0,
                          rotation_tolerance: float = 0.0):
        """Part 2: whether the agent's action agrees with the expert's, over all 7 dims.

        Positive requires all three of: the commanded translation pointing the same way as
        the expert's, the commanded wrist rotation pointing the same way as the expert's,
        and matching gripper signs. Any dimension on which the expert expresses no
        preference (a near-zero command) is treated as agreeing.
        """
        translation_ok = cls._cosine_agreement(action[:3], expert_action[:3], tolerance)
        rotation_ok = cls._cosine_agreement(action[3:6], expert_action[3:6], rotation_tolerance)
        gripper_ok = (abs(expert_action[6]) < _AGREEMENT_EPS
                      or np.sign(action[6]) == np.sign(expert_action[6]))
        return bool(translation_ok and rotation_ok and gripper_ok)

    def _format_obs(self, observation):
        text = self.textualize_observation(observation)
        image = (self.env.render() if self.env.visual else None)
        return text if image is None else dict(text=text, image=image)

    def textualize_expert_action(self, action):
        """ Parse action into text.  There is no action to name when the expert is off. """
        if action is None:
            return ""
        # The idea is to return something like
        # f"delta x: {action[0]:.2f}, delta y:{action[1]:.2f}, delta z:{action[2]:.2f}, gripper state:{action[3]:.1f}"
        # or another action text format if the action isn't a delta.
        # TODO should not be the raw action
        return np.array2string(action, precision=10)

    def textualize_observation(self, observation):
        """Parse the kitchen observation into text: the 59-D robot/object state block.

        ``KitchenEnv`` returns a goal dict of ``observation`` / ``achieved_goal`` /
        ``desired_goal``.  Only the first is exposed -- 9 robot joint positions, 9 robot joint
        velocities, 21 object positions and 20 object velocities.  ``achieved_goal`` is a
        re-slicing of that same state, and ``desired_goal`` is constant for a fixed task set,
        so neither adds information; emitting them only made every consumer slice them back
        off (which the consumers then had to agree on).
        """
        if isinstance(observation, dict):
            observation = observation['observation']
        return json.dumps({'obs': np.array2string(np.asarray(observation), precision=10)})

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
    
