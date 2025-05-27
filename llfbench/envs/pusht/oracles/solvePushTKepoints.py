from llfbench.envs.pusht.pusht.pusht_keypoints_env import PushTKeypointsEnv
from llfbench.envs.pusht.oracles.solvePushT import solvePushT
import os

class solvePushTKeypoints(solvePushT):

    def __init__(self, env:PushTKeypointsEnv, device='cuda', seed=None, debug=False, vis=False):
        self.env: PushTKeypointsEnv

        super().__init__(
            env, 
            # checkpoint_dir=os.getcwd()+"/LLF-Bench/llfbench/envs/pusht/oracles/pusht_keypoints_checkpoints/epoch=1100-test_mean_score=0.878.ckpt",
            checkpoint_dir=os.getcwd()+"/LLF-Bench/llfbench/envs/pusht/oracles/pusht_keypoints_checkpoints/epoch=0100-test_mean_score=0.578.ckpt",
            # checkpoint_dir=os.getcwd()+"/LLF-Bench/llfbench/envs/pusht/oracles/pusht_keypoints_checkpoints/epoch=0050-test_mean_score=0.143.ckpt",
            device=device,
            seed=seed,
            debug=debug,
            vis=vis
        )