from llfbench.envs.pusht.pusht.pusht_image_env import PushTImageEnv
from llfbench.envs.pusht.oracles.solvePushT import solvePushT

class solvePushTImage(solvePushT):

    def __init__(self, env:PushTImageEnv, device='cuda', seed=None, debug=False, vis=False):
        self.env: PushTImageEnv

        super().__init__(
            env, 
            checkpoint_dir=None,
            # checkpoint_dir="pusht_image_checkpoints/epoch=1850-test_mean_score=0.898.ckpt",
            device=device,
            seed=seed,
            debug=debug,
            vis=vis
        )