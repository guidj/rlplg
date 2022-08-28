import functools
from typing import Any, Callable, Mapping

from rlplg import envspec, gymenv
from rlplg.environments.alphabet import constants as alphabet_constants
from rlplg.environments.alphabet import env as alphabet_env
from rlplg.environments.gridworld import constants as gridworld_constants
from rlplg.environments.gridworld import utils as gridworld_utils
from rlplg.environments.redgreen import constants as redgreen_constants
from rlplg.environments.redgreen import env as redgreen_env

ABC = alphabet_constants.ENV_NAME
GRID_WORLD = gridworld_constants.ENV_NAME
RED_GREEN = redgreen_constants.ENV_NAME
TAXI = "Taxi-v3"
FROZEN_LAKE = "FrozenLake-v1"

SUPPORTED_RLPLG_ENVS = frozenset((ABC, GRID_WORLD, RED_GREEN))
SUPPORTED_GYM_ENVS = frozenset((TAXI, FROZEN_LAKE))


def load(name: str, **args: Mapping[str, Any]) -> envspec.EnvSpec:
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
    rlplg_envs = {
        ABC: alphabet_env.create_env_spec,
        GRID_WORLD: gridworld_utils.create_envspec_from_grid,
        RED_GREEN: redgreen_env.create_env_spec,
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
