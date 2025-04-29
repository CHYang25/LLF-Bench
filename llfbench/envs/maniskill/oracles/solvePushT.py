from mani_skill.envs.tasks import PushTEnv
from enum import auto, Enum
from llfbench.envs.maniskill.oracles.solveManiskill import solveManiskill

class PushT(Enum):
    pass

class solvePushT(solveManiskill):

    def __init__(self, env:PushTEnv, device='cuda', seed=None, debug=False, vis=False):
        self.env: PushTEnv

        super().__init__(
            env, 
            checkpoint_dir="rl_trainer/runs/ppo-PushT-v1-state-42-walltime_efficient/ckpt_1976.pt",
            device=device,
            seed=seed,
            debug=debug,
            vis=vis
        )

        #TODO: Define PushT stages
