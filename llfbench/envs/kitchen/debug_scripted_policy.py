"""Run the scripted Franka Kitchen expert and report what it actually achieves.

This is the validation entry point for the expert: it drives the real env loop through the
llfbench wrapper -- the same path ``scripts/gen_dataset.py`` collects through -- and scores
episodes with the environment's own completion predicate rather than with any of the
policy's internal ones.

Per-skill success is the number that matters when changing a skill, because the four tasks
in a combined episode interfere: each starts from wherever the previous one left the arm,
and a task that stalls spends budget the rest of the episode then does not have.

    # every skill on its own, which is how to tell whether a change helped
    python -m llfbench.envs.kitchen.debug_scripted_policy --per-skill --episodes 8

    # the full four-task goal
    python -m llfbench.envs.kitchen.debug_scripted_policy --episodes 8

    # one episode, step by step, for a skill that is misbehaving
    python -m llfbench.envs.kitchen.debug_scripted_policy \
        --tasks "microwave" --episodes 1 --trace

    # save a video per episode
    python -m llfbench.envs.kitchen.debug_scripted_policy --episodes 2 --render
"""

import argparse
import os
from dataclasses import replace

import numpy as np

import llfbench
from llfbench.envs.kitchen import DEFAULT_TASKS_TO_COMPLETE
from llfbench.envs.kitchen.scripted_policy import KitchenPolicyConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0,
                        help='first seed; episode i uses seed + i')
    parser.add_argument('--tasks', type=str, default=None,
                        help=f'comma separated subset of {DEFAULT_TASKS_TO_COMPLETE}')
    parser.add_argument('--per-skill', action='store_true',
                        help='run each task alone, in its own episode, and report per task')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='step budget per episode (default: 400 alone, 1000 combined)')
    parser.add_argument('--randomize-order', action='store_true',
                        help='shuffle the task order, as the collection default does')
    parser.add_argument('--action-repeat', type=int, default=None)
    parser.add_argument('--trace', action='store_true',
                        help='print every phase transition with its live errors')
    parser.add_argument('--render', action='store_true',
                        help='save one mp4 per episode to --video-dir')
    parser.add_argument('--video-dir', type=str, default='data/debug/kitchen')
    return parser.parse_args(argv)


def build_config(args):
    overrides = {'randomize_task_order': bool(args.randomize_order)}
    if args.action_repeat is not None:
        overrides['action_repeat'] = args.action_repeat
    return replace(KitchenPolicyConfig.fast_demo(), **overrides)


def run_episode(env, seed, max_steps, trace=False, render=False):
    """One episode. Returns (completed task names, steps taken, frames)."""
    env.reset(seed=seed)
    wrapper = env.env
    policy = wrapper.kt_policy
    if render:
        wrapper.render_video(True)

    frames = []
    previous = None
    steps = 0
    for steps in range(1, max_steps + 1):
        _, _, terminated, truncated, _ = env.step(wrapper.expert_action)
        if render:
            frames.append(env.render())
        if trace:
            diagnostics = policy.get_diagnostics()
            current = (diagnostics['selected_subtask'], diagnostics['controller_phase'])
            if current != previous:
                distance = diagnostics['task_distance']
                print(f"    step {steps:4d}  {str(current[0]):<14} {current[1]:<18} "
                      f"waypoint_err={diagnostics['position_error']:.3f} "
                      f"frame_err={diagnostics['orientation_error']:.3f} "
                      f"task_dist={'-' if distance is None else f'{distance:.3f}'}")
                previous = current
        if terminated or truncated:
            break
    # `episode_task_completions` is the environment's own record, not the policy's.
    return list(wrapper.kitchen_env.episode_task_completions), steps, frames


def save_video(frames, path):
    try:
        import imageio
    except ImportError:
        print(f"    (imageio not installed; skipping {path})")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=12)
    print(f"    saved {path}")


def evaluate(tasks, args, config, label):
    """Run `args.episodes` episodes on `tasks` and print a one-line summary."""
    # A single skill is bounded by its own budget; the four-task goal needs room for
    # all of them plus the retreats between, and the microwave alone runs ~364 steps.
    max_steps = args.max_steps or (400 if len(tasks) == 1 else 1000)
    env = llfbench.make('llf-kitchen-FrankaKitchen-v1',
                        instruction_type='b', feedback_type=('hp', 'hn', 'fp'),
                        visual=False, seed=args.seed, warning=False,
                        policy_config=config, episode_steps=max_steps,
                        tasks_to_complete=tasks)

    solved, lengths, completions = 0, [], []
    for episode in range(args.episodes):
        seed = args.seed + episode
        if args.trace:
            print(f"  seed {seed}")
        done, steps, frames = run_episode(env, seed, max_steps,
                                          trace=args.trace, render=args.render)
        completions.append(len(done))
        if len(done) == len(tasks):
            solved += 1
            lengths.append(steps)
        if args.render and frames:
            save_video(frames, os.path.join(args.video_dir, f"{label}_seed{seed}.mp4"))
        if not args.per_skill:
            print(f"  seed {seed}: {len(done)}/{len(tasks)} {sorted(done)}  steps {steps}")

    median = int(np.median(lengths)) if lengths else None
    print(f"{label:<16} {solved}/{args.episodes} episodes fully solved   "
          f"mean {np.mean(completions):.2f}/{len(tasks)} tasks   "
          f"median steps when solved: {median if median else '-'}")
    return solved


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    tasks = ([t.strip() for t in args.tasks.split(',')] if args.tasks
             else list(DEFAULT_TASKS_TO_COMPLETE))

    if args.per_skill:
        print(f"Per-skill, {args.episodes} seeds each, each task alone in its own episode.\n")
        for task in tasks:
            evaluate([task], args, config, task)
        return 0

    print(f"Combined goal {tasks}, {args.episodes} seeds.\n")
    evaluate(tasks, args, config, 'combined')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
