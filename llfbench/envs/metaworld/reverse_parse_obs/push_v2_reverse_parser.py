import numpy as np
from .base_reverse_parser import BaseReverseParser

class SawyerPushV2PolicyReverseParser(BaseReverseParser):
    
    @classmethod
    def _reverse_parse_obs(self, obs):
        observation = np.zeros(self.OBS_SPACE)
        hand_pos = obs[:3]
        puck_pos = obs[3:6]
        goal_pos = obs[6:9]
        observation[:3] = hand_pos
        observation[4:7] = puck_pos
        observation[-3:] = goal_pos
        return observation