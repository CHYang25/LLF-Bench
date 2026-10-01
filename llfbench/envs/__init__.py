import importlib
from llfbench.envs import gridworld
from llfbench.envs import bandits
from llfbench.envs import optimization
from llfbench.envs import reco
from llfbench.envs import poem
from llfbench.envs import highway
try:
    from llfbench.envs import block_pushing
except ImportError as exc:  # tf_agents/Keras is not importable in every process (e.g. after
    # transformers/datasets set up TensorFlow); the other envs and the multistep mergers must
    # still be usable there, so block pushing is simply left unregistered.
    import warnings
    warnings.warn(f"llfbench: block_pushing envs not registered ({exc})")
from llfbench.envs import maniskill
from llfbench.envs import pusht
from llfbench.envs import pointmaze
from llfbench.envs import adroit

if importlib.util.find_spec('gymnasium_robotics'):
    from llfbench.envs import kitchen

if importlib.util.find_spec('metaworld'):
    from llfbench.envs import metaworld

if importlib.util.find_spec('alfworld'):
    from llfbench.envs import alfworld