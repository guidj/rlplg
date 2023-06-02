"""
This module has spec definition for environments.
"""

import dataclasses

import gymnasium as gym

from rlplg import envdesc
from rlplg.learning.tabular import markovdp


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """
    Class holds environment variables.
    """

    name: str
    level: str
    environment: gym.Env
    discretizer: markovdp.MdpDiscretizer
    env_desc: envdesc.EnvDesc
