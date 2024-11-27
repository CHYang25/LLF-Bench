import numpy as np

class BaseReverseParser:
    OBS_SPACE = 39

    @classmethod
    def _reverse_parse_obs(obs):
        raise NotImplementedError()