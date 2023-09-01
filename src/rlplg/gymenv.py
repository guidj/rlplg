import base64
import hashlib
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlplg import envdesc, envspec, npsci
from rlplg.learning.tabular import markovdp


class GymEnvMdpDiscretizer(markovdp.MdpDiscretizer):
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


def create_env_spec(name: str, **kwargs: Mapping[str, Any]) -> envspec.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = gym.make(name, **kwargs)
    discretizer = GymEnvMdpDiscretizer()
    env_desc = parse_gym_env_desc(environment=environment)
    return envspec.EnvSpec(
        name=name,
        level=__encode_env(**kwargs),
        environment=environment,
        discretizer=discretizer,
        env_desc=env_desc,
    )


def parse_gym_env_desc(environment: gym.Env) -> envdesc.EnvDesc:
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
    return envdesc.EnvDesc(num_states=num_states, num_actions=num_actions)


def __encode_env(**kwargs: Mapping[str, Any]) -> str:
    keys = []
    values = []
    for key, value in sorted(kwargs.items()):
        keys.append(key)
        values.append(value)

    hash_key = tuple(keys) + tuple(values)
    hashing = hashlib.sha512(str(hash_key).encode("UTF-8"))
    return base64.b32encode(hashing.digest()).decode("UTF-8")
