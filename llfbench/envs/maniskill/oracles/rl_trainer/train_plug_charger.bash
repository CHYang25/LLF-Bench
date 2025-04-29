#! /bin/bash
seed=42

CUDA_VISIBLE_DEVICES=1 python ppo_fast.py --env_id="PlugCharger-v1" --seed=${seed} \
    --num_envs=2048 --update_epochs=8 --num_minibatches=32 --gamma=0.97 --gae_lambda=0.95 \
    --total_timesteps=75_000_000 --num-steps=16 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --exp-name="ppo-PlugCharger-v1-state-${seed}-walltime_efficient" \
    --track \
    --control_mode="pd_joint_delta_pos" # control mode is fixed