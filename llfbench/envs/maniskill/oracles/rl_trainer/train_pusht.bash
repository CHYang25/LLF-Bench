#! /bin/bash
seed=42

CUDA_VISIBLE_DEVICES=1 python ppo_fast.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num_eval_steps=100 --gamma=0.99 \
    --num_eval_envs=16 \
    --exp-name="ppo-PushT-v1-state-${seed}-walltime_efficient" \
    --track \
    --control_mode="pd_joint_delta_pos" # control mode is fixed