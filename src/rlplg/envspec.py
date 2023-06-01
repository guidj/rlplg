"""
This module has spec definition for environments.
"""

import dataclasses

from rlplg import core, envdesc
from rlplg.learning.tabular import markovdp


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """
    Class holds environment variables.
    """

    name: str
    level: str
    environment: core.PyEnvironment
    discretizer: markovdp.MdpDiscretizer
    env_desc: envdesc.EnvDesc
