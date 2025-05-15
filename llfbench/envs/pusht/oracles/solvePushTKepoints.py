from llfbench.envs.pusht.pusht.pusht_keypoints_env import PushTKeypointsEnv
from llfbench.envs.pusht.oracles.solvePushT import solvePushT

class solvePushTKeypoints(solvePushT):

    def __init__(self, env:PushTKeypointsEnv, device='cuda', seed=None, debug=False, vis=False):
        self.env: PushTKeypointsEnv

        super().__init__(
            env, 
            checkpoint_dir="LLF-Bench/llfbench/envs/pusht/oracles/pusht_keypoints_checkpoints/epoch=3000-test_mean_score=0.903.ckpt",
            device=device,
            seed=seed,
            debug=debug,
            vis=vis
        )