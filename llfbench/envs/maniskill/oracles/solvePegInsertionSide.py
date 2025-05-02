from mani_skill.envs.tasks import PegInsertionSideEnv
from enum import auto, Enum
from llfbench.envs.maniskill.oracles.solveManiskill import solveManiskill

class solvePegInsertionSide(solveManiskill):

    def __init__(self, env:PegInsertionSideEnv, device='cuda', seed=None, debug=False, vis=False):
        self.env: PegInsertionSideEnv

        super().__init__(
            env, 
            checkpoint_dir="rl_trainer/runs/ppo-PegInsertionSide-v1-state-42-walltime_efficient/ckpt_2151.pt",
            device=device,
            seed=seed,
            debug=debug,
            vis=vis
        )