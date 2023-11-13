"""
This module has utilities to load environments,
defined in either `rlplg` or gymnasium.
"""


import base64
import functools
import hashlib
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlplg import core, npsci
from rlplg.core import EnvTransition
from rlplg.environments import abcseq, gridworld, randomwalk, redgreen

TAXI = "Taxi-v3"
FROZEN_LAKE = "FrozenLake-v1"

SUPPORTED_RLPLG_ENVS = frozenset(
    (abcseq.ENV_NAME, gridworld.ENV_NAME, randomwalk.ENV_NAME, redgreen.ENV_NAME)
)
SUPPORTED_GYM_ENVS = frozenset((TAXI, FROZEN_LAKE))


class GymEnvMdpDiscretizer(core.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def state(self, observation: Any) -> int:
        """
        Maps an observation to a state ID.
        """
        del self
        state_: int = npsci.item(observation)
        return state_

    def action(self, action: Any) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        action_: int = npsci.item(action)
        return action_


def load(name: str, **args) -> core.EnvSpec:
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


def __environment_spec_constructors() -> Mapping[str, Callable[..., core.EnvSpec]]:
    """
    Creates a mapping of rlplg and gym environment names to their constructors.

    Returns:
        A mapping from a unique string identifier to a constructor.

    """
    rlplg_envs: Mapping[str, Callable[..., core.EnvSpec]] = {
        abcseq.ENV_NAME: abcseq.create_env_spec,
        gridworld.ENV_NAME: gridworld.create_envspec_from_grid_file,
        randomwalk.ENV_NAME: randomwalk.create_env_spec,
        redgreen.ENV_NAME: redgreen.create_env_spec,
    }
    gym_envs = {
        name: __gym_environment_spec_constructor(name) for name in SUPPORTED_GYM_ENVS
    }
    return {**rlplg_envs, **gym_envs}


def __gym_environment_spec_constructor(
    name: str,
) -> Callable[..., core.EnvSpec]:
    """
    Partially initiatializes a gym environment spec with its name.

    Args:
        name: unique identifier.

    Returns:
        A callable that creates a gym environment.
    """

    return functools.partial(__gym_environment_spec, name)


def __gym_environment_spec(name: str, **kwargs: Mapping[str, Any]) -> core.EnvSpec:
    """
    Creates a gym environment spec.
    """
    environment = gym.make(name, **kwargs)
    discretizer = GymEnvMdpDiscretizer()
    env_desc = parse_gym_env_desc(environment=environment)
    return core.EnvSpec(
        name=name,
        level=__encode_env(**kwargs),
        environment=environment,
        discretizer=discretizer,
        env_desc=env_desc,
    )


def parse_gym_env_desc(environment: gym.Env) -> core.EnvDesc:
    """
    Infers the EnvDesc from a `gym.Env`.
    """
    num_actions = (
        environment.action_space.n
        if isinstance(environment.action_space, spaces.Discrete)
        else np.inf
    )
    num_states = (
        environment.observation_space.n
        if isinstance(environment.action_space, spaces.Discrete)
        else np.inf
    )
    return core.EnvDesc(num_states=num_states, num_actions=num_actions)


def parse_gym_env_transition(environment: gym.Env) -> EnvTransition:
    """
    Parses transition data from a `gym.Env`.
    """
    for key in ("P", "transition"):
        if hasattr(environment, key):
            return getattr(environment, key)
    raise ValueError("No `P` or `transition` attribute in environment")


def parse_gym_env_mdp(environment: gym.Env) -> core.Mdp:
    """
    Parses an Mdp from a `gym.Env`.
    """
    env_desc = parse_gym_env_desc(environment)
    transition = parse_gym_env_transition(environment)
    return core.EnvMdp(env_desc=env_desc, transition=transition)


def __encode_env(**kwargs: Mapping[str, Any]) -> str:
    """
    Encodes environment into a unique hash.
    """
    keys = []
    values = []
    for key, value in sorted(kwargs.items()):
        keys.append(key)
        values.append(value)

    hash_key = tuple(keys) + tuple(values)
    hashing = hashlib.sha512(str(hash_key).encode("UTF-8"))
    return base64.b32encode(hashing.digest()).decode("UTF-8")
