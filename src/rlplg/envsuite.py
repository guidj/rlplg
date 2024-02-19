"""
This module has utilities to load environments,
defined in either `rlplg` or gymnasium.
"""

import functools
from typing import Any, Callable, Mapping, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from rlplg import core, npsci
from rlplg.core import EnvTransition
from rlplg.environments import abcseq, gridworld, randomwalk, redgreen, towerhanoi

TAXI = "Taxi-v3"
FROZEN_LAKE = "FrozenLake-v1"
CLIFF_WALKING = "CliffWalking-v0"
TARIFF_FROZEN_LAKE = "TariffFrozenLake-v1"

SUPPORTED_RLPLG_ENVS = frozenset(
    (
        abcseq.ENV_NAME,
        gridworld.ENV_NAME,
        randomwalk.ENV_NAME,
        redgreen.ENV_NAME,
        towerhanoi.ENV_NAME,
        TARIFF_FROZEN_LAKE,
    )
)
SUPPORTED_GYM_ENVS = frozenset((TAXI, FROZEN_LAKE, CLIFF_WALKING, TARIFF_FROZEN_LAKE))


class DefaultGymEnvMdpDiscretizer(core.MdpDiscretizer):
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


class ShiftRewardWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env[ObsType, ActType], delta: float):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self.delta = delta
        self.reward_range = (env.reward_range[0] + delta, env.reward_range[1] + delta)

        # Update transition data, if existing
        if hasattr(self.env, "P"):
            transitions = getattr(self.env, "P")
            new_transitions = {}
            for state, action_transitions in transitions.items():
                new_transitions[state] = {}
                for action, transitions in action_transitions.items():
                    new_transitions[state][action] = []
                    for prob, next_state, reward, done in transitions:
                        new_transitions[state][action].append(
                            (prob, next_state, reward + self.delta, done)
                        )
            setattr(self.env, "P", new_transitions)

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return reward + self.delta


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
        gridworld.ENV_NAME: gridworld.create_envspec_from_grid_text,
        randomwalk.ENV_NAME: randomwalk.create_env_spec,
        redgreen.ENV_NAME: redgreen.create_env_spec,
        towerhanoi.ENV_NAME: towerhanoi.create_env_spec,
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
    environment = __make_gym_environment(name, **kwargs)
    discretizer = __gym_environment_discretizer(name)
    mdp = core.EnvMdp(
        env_desc=__parse_gym_env_desc(environment=environment),
        transition=__parse_gym_env_transition(environment),
    )
    return core.EnvSpec(
        name=name,
        level=__encode_env(**kwargs),
        environment=environment,
        discretizer=discretizer,
        mdp=mdp,
    )


def __make_gym_environment(name: str, **kwargs: Mapping[str, Any]) -> gym.Env:
    """
    Creates discretizers for supported environments.
    """
    if name == TARIFF_FROZEN_LAKE:
        return ShiftRewardWrapper(gym.make(FROZEN_LAKE, **kwargs), delta=-1.0)
    return gym.make(name, **kwargs)


def __parse_gym_env_desc(environment: gym.Env) -> core.EnvDesc:
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


def __parse_gym_env_transition(environment: gym.Env) -> EnvTransition:
    """
    Parses transition data from a `gym.Env`.
    """
    transition: EnvTransition = getattr(environment, "P")
    return transition


def __gym_environment_discretizer(name: str) -> core.MdpDiscretizer:
    """
    Creates discretizers for supported environments.
    """
    del name
    return DefaultGymEnvMdpDiscretizer()


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
    return core.encode_env(signature=hash_key)
