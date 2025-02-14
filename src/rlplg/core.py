"""
This module defines core abstractions.
"""

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from gymnasium.core import ObsType, RenderFrame, SupportsFloat

NestedArray = Union[Mapping, np.ndarray]
TimeStep = Tuple[ObsType, SupportsFloat, bool, bool, Mapping[str, Any]]
InitState = Tuple[ObsType, Mapping[str, Any]]
RenderType = Optional[Union[RenderFrame, Sequence[RenderFrame]]]
StateTransition = Mapping[int, Sequence[Tuple[float, int, float, bool]]]
# Type: Mapping[state, Mapping[action, Sequence[Tuple[prob, next_state, reward, terminated]]]]
EnvTransition = Mapping[int, StateTransition]
MutableStateTransition = Dict[int, List[Tuple[float, int, float, bool]]]
MutableEnvTransition = Dict[int, MutableStateTransition]
