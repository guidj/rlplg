"""
This module has utilities to load environments,
defined in either `rlplg` or gym.
"""


import functools
from typing import Any, Callable, Mapping

from rlplg import envspec, gymenv
from rlplg.environments import abcseq, gridworld, randomwalk, redgreen

TAXI = "Taxi-v3"
FROZEN_LAKE = "FrozenLake-v1"

SUPPORTED_RLPLG_ENVS = frozenset(
    (abcseq.ENV_NAME, gridworld.ENV_NAME, randomwalk.ENV_NAME, redgreen.ENV_NAME)
)
SUPPORTED_GYM_ENVS = frozenset((TAXI, FROZEN_LAKE))


def load(name: str, **args) -> envspec.EnvSpec:
    """
    Creates an environment with the given arguments.

    Args:
        name: unique identifier.
        args: parameters that are passed to an environment constructor.

    Returns:
        An instantiated environment.

    Raises:
        A ValueError is the environment is unsupported.

    """
    constructors = __environment_spec_constructors()
    if name not in constructors:
        raise ValueError(f"Unsupported environment: {name}.")
    return constructors[name](**args)


def __environment_spec_constructors() -> Mapping[str, Callable[..., envspec.EnvSpec]]:
    """
    Creates a mapping of rlplg and gym environment names to their constructors.

    Returns:
        A mapping from a unique string identifier to a constructor.

    """
    rlplg_envs: Mapping[str, Callable[..., envspec.EnvSpec]] = {
        abcseq.ENV_NAME: abcseq.create_env_spec,
        gridworld.ENV_NAME: gridworld.create_envspec_from_grid,
        randomwalk.ENV_NAME: randomwalk.create_env_spec,
        redgreen.ENV_NAME: redgreen.create_env_spec,
    }
    gym_envs = {
        name: __gym_environment_spec_constructor(name) for name in SUPPORTED_GYM_ENVS
    }
    return {**rlplg_envs, **gym_envs}


def __gym_environment_spec_constructor(
    name: str,
) -> Callable[..., envspec.EnvSpec]:
    """
    Partially initiatializes a gym environment spec with its name.

    Args:
        name: unique identifier.

    Returns:
        A callable that creates a gym environment.
    """

    return functools.partial(__gym_environment_spec, name)


def __gym_environment_spec(name: str, **kwargs: Mapping[str, Any]) -> envspec.EnvSpec:
    """
    Creates a gym environment spec.
    """
    return gymenv.create_env_spec(name, **kwargs)
