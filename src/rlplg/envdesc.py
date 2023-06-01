"""
This module has description definition for environments.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class EnvDesc:
    """
    Class contains properties of the environment.
    """

    num_states: int
    num_actions: int
