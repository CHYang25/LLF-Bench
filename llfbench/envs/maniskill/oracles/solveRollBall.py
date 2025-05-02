from mani_skill.envs.tasks import RollBallEnv
from enum import auto, Enum
from llfbench.envs.maniskill.oracles.solveManiskill import solveManiskill

class solveRollBall(solveManiskill):

    def __init__(self, env:RollBallEnv, device='cuda', seed=None, debug=False, vis=False):
        self.env: RollBallEnv

        super().__init__(
            env, 
            checkpoint_dir="rl_trainer/runs/ppo-RollBall-v1-state-42-walltime_efficient/ckpt_651.pt",
            device=device,
            seed=seed,
            debug=debug,
            vis=vis
        )
