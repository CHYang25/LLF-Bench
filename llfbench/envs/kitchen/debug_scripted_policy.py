"""Rollout / visualisation harness for the scripted Franka Kitchen expert.

    python -m llfbench.envs.kitchen.debug_scripted_policy --render --episodes 5 --seed 0

Prints, per step, the selected subtask and FSM phase, the commanded EEF target and its
error, and any task completions.  ``--render`` saves one video per episode
(``render_mode='rgb_array'`` is already set by ``make_env``; kitchen frames come out
upright, so unlike metaworld they are *not* flipped with ``[::-1]``).
"""

import argparse
import json
import os

import numpy as np

import llfbench
from llfbench.envs.kitchen.scripted_policy import KitchenPolicyConfig, KitchenSim


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tasks', type=str, default=None,
                        help='comma separated task list, e.g. microwave,kettle')
    parser.add_argument('--task-order', type=str, default=None,
                        help='comma separated execution order for the planner')
    parser.add_argument('--randomize-order', action='store_true')
    parser.add_argument('--include-burners', action='store_true',
                        help='run the knob skill even though the burner tasks are already '
                             'within BONUS_THRESH at reset')
    parser.add_argument('--kettle-strategy', choices=('push', 'grasp'), default='grasp')
    parser.add_argument('--free-space-step', type=float, default=None,
                        help='override normalized free-space translation limit')
    parser.add_argument('--contact-step', type=float, default=None,
                        help='override normalized contact translation limit')
    parser.add_argument('--slide-lookahead', type=float, default=None,
                        help='override live rightward slide target offset in metres')
    parser.add_argument('--slide-step', type=float, default=None,
                        help='override normalized slide-manipulation action cap')
    parser.add_argument('--slide-recede-distance', type=float, default=None,
                        help='override post-release roomward clearance in metres')
    parser.add_argument('--leveling', action='store_true',
                        help='enable experimental wrist-leveling commands')
    parser.add_argument('--nominal-tool-frame', action='store_true',
                        help='ignore live wrist tilt when converting tool points to EEF targets')
    parser.add_argument('--task-step-budget', type=int, default=None,
                        help='override per-task policy-call budget')
    parser.add_argument('--align-timeout', type=int, default=None)
    parser.add_argument('--max-rotation-step', type=float, default=None)
    parser.add_argument('--ik-target-leak', type=float, default=None,
                        help='override actuator-target anti-windup gain (0 disables it)')
    parser.add_argument('--no-terminate', action='store_true',
                        help='do not end the episode when every task is complete; needed to '
                             'watch a skill whose task the env already scores at reset')
    parser.add_argument('--render', action='store_true', help='save a video per episode')
    parser.add_argument('--video-dir', type=str, default='./media')
    parser.add_argument('--max-steps', type=int, default=280)
    parser.add_argument('--control-steps', type=int, default=None,
                        help='base DLS IK iterations per environment step (fast default: 15)')
    parser.add_argument('--legacy-speed', action='store_true',
                        help='use the older 5-control-step / conservative-motion profile')
    parser.add_argument('--verbose', action='store_true', help='print every step')
    parser.add_argument('--print-every', type=int, default=10)
    return parser.parse_args(argv)


def build_env(args):
    config_type = KitchenPolicyConfig if args.legacy_speed else KitchenPolicyConfig.fast_demo
    config = config_type(
        task_order=args.task_order.split(',') if args.task_order else None,
        randomize_task_order=args.randomize_order,
        skip_pre_completed_tasks=not args.include_burners,
        kettle_strategy=args.kettle_strategy,
        keep_gripper_down=args.leveling,
        use_measured_tool_frame=not args.nominal_tool_frame,
    )
    if args.free_space_step is not None:
        config.free_space_step = args.free_space_step
    if args.contact_step is not None:
        config.contact_step = args.contact_step
    if args.slide_lookahead is not None:
        config.slide_manipulate_lookahead = args.slide_lookahead
    if args.slide_step is not None:
        config.slide_manipulation_step = args.slide_step
    if args.slide_recede_distance is not None:
        config.slide_recede_distance = args.slide_recede_distance
    if args.task_step_budget is not None:
        config.task_step_budget = args.task_step_budget
    if args.align_timeout is not None:
        config.align_timeout = args.align_timeout
    if args.max_rotation_step is not None:
        config.max_rotation_step = args.max_rotation_step
    if args.ik_target_leak is not None:
        config.ik_target_leak = args.ik_target_leak
    kwargs = dict(policy_config=config, control_steps=args.control_steps,
                  fast_expert=not args.legacy_speed)
    if args.tasks:
        kwargs['tasks_to_complete'] = args.tasks.split(',')
    if args.no_terminate:
        kwargs['terminate_on_tasks_completed'] = False
    env = llfbench.make('llf-kitchen-FrankaKitchen-v1', **kwargs)
    if args.render:
        env.render_video(True)
    return env


def save_video(frames, path):
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - optional dependency
        print(f'  [video] Pillow not available, skipping {path}')
        return
    images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames if f is not None]
    if not images:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)
    print(f'  [video] {len(images)} frames -> {path}')


def run_episode(env, args, episode):
    seed = args.seed + episode
    obs, info = env.reset(seed=seed)
    sim = KitchenSim(env)
    kitchen = sim.kitchen

    print(f'\n=== episode {episode} (seed {seed}) ===')
    print(f'  tasks in goal      : {sorted(kitchen.goal.keys())}')
    print(f'  tasks_to_complete  : {sorted(kitchen.tasks_to_complete)}')
    print('  distance at reset  : ' + ', '.join(
        f'{t}={sim.task_distance(t):.3f}{"*" if sim.task_complete(t) else ""}'
        for t in sorted(kitchen.goal.keys())))
    print('  (* = already within BONUS_THRESH before the robot moves)')

    frames = list(info.get('video') or [])
    total_return = 0.0
    completion_steps = {}
    prev_phase = None
    steps = 0

    for step in range(args.max_steps):
        expert_action = env.expert_action
        obs, reward, terminated, truncated, info = env.step(expert_action)
        steps += 1
        total_return += float(reward)
        frames.extend(info.get('video') or [])

        diag = info['expert_diagnostics']
        for task in kitchen.episode_task_completions:
            if task not in completion_steps:
                completion_steps[task] = step
                print(f'  [{step:3d}] COMPLETED {task}  (reward {reward:.1f})')

        phase_changed = (diag['selected_subtask'], diag['controller_phase']) != prev_phase
        prev_phase = (diag['selected_subtask'], diag['controller_phase'])
        if args.verbose or phase_changed or step % args.print_every == 0:
            target = diag['target_position']
            target_str = 'None' if target is None else np.array2string(np.asarray(target), precision=3)
            print(f'  [{step:3d}] task={str(diag["selected_subtask"]):>20s} '
                  f'phase={diag["controller_phase"]:<18s} '
                  f'target={target_str:<26s} '
                  f'pos_err={diag["position_error"]:.3f} tool_err={diag["tool_error"]:.3f} '
                  f'rot_err={diag["orientation_error"]:.3f} grip={diag["finger_opening"]:.3f} '
                  f'ik_gap={diag["ik_joint_target_error"]:.3f} '
                  f'eef={np.array2string(sim.eef_pos, precision=3)}')

        # `--no-terminate` also has to override the wrapper, which ends the episode as
        # soon as info['success'] is True regardless of terminate_on_tasks_completed.
        if (terminated and not args.no_terminate) or truncated:
            break

    print(f'  --- episode summary ---')
    print(f'  steps                : {steps}')
    print(f'  total return         : {total_return}')
    print(f'  completed subtasks   : {kitchen.episode_task_completions}')
    print(f'  completion order/step: {completion_steps}')
    print(f'  final distances      : ' + ', '.join(
        f'{t}={sim.task_distance(t):.3f}' for t in sorted(kitchen.goal.keys())))
    diag = env.kt_policy.get_diagnostics()
    print(f'  steps per subtask    : {diag["steps_per_subtask"]}')
    print(f'  phase failures       : {diag["phase_failures"]}')
    print(f'  abandoned subtasks   : {diag["abandoned"]}')
    print(f'  info["success"]      : {info["success"]}')

    if args.render:
        save_video(frames, os.path.join(args.video_dir, f'kitchen_expert_ep{episode}.gif'))

    return dict(steps=steps, total_return=total_return,
                completed=list(kitchen.episode_task_completions),
                completion_steps=completion_steps,
                steps_per_subtask=diag['steps_per_subtask'],
                phase_failures=diag['phase_failures'],
                abandoned=diag['abandoned'],
                success=bool(info['success']))


def main(argv=None):
    args = parse_args(argv)
    env = build_env(args)
    results = [run_episode(env, args, ep) for ep in range(args.episodes)]
    env.close()

    print('\n=== aggregate ===')
    print(json.dumps(results, indent=2, default=str))
    returns = [r['total_return'] for r in results]
    print(f'mean return over {len(results)} episode(s): {np.mean(returns):.2f}')
    return results


if __name__ == '__main__':
    main()
