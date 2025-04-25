import importlib
from llfbench.envs import gridworld
from llfbench.envs import bandits
from llfbench.envs import optimization
from llfbench.envs import reco
from llfbench.envs import poem
from llfbench.envs import highway
from llfbench.envs import block_pushing
from llfbench.envs import maniskill

if importlib.util.find_spec('metaworld'):
    from llfbench.envs import metaworld

if importlib.util.find_spec('alfworld'):
    from llfbench.envs import alfworld