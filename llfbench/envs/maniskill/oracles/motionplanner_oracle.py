import gymnasium as gym
import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

class PandaArmMotionPlanningOracle(PandaArmMotionPlanningSolver):

    def __init__(self, 
            env, 
            debug = False, 
            vis = True, 
            base_pose = None, 
            visualize_target_grasp_pose = True, 
            print_env_info = True, 
            joint_vel_limits=0.9, 
            joint_acc_limits=0.9):
        
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits)
        self._action = None

    def follow_path(self, result, refine_steps = 0):
        qpos = result["position"][0]
        if self.control_mode == "pd_joint_pos_vel":
            qvel = result["velocity"][0]
            action = np.hstack([qpos, qvel, self.gripper_state])
        else:
            action = np.hstack([qpos, self.gripper_state])
            
        if self.vis:
            self.base_env.render_human()
        return action