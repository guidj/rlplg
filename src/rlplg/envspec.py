import dataclasses

from tf_agents.environments import py_environment

from rlplg import envdesc
from rlplg.learning.tabular import markovdp


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """
    Class holds environment variables.
    """

    name: str
    level: str
    environment: py_environment.PyEnvironment
    discretizer: markovdp.MdpDiscretizer
    env_desc: envdesc.EnvDesc
