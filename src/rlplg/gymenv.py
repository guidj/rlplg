import base64
import hashlib
from typing import Any, Mapping

import gym

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
    num_states = (
        environment.observation_spec().maximum
        - environment.observation_spec().minimum
        + 1
    )
    num_actions = (
        environment.action_spec().maximum - environment.action_spec().minimum + 1
    )
    env_desc = envdesc.EnvDesc(num_states=num_states, num_actions=num_actions)
    return envspec.EnvSpec(
        name=name,
        level=__encode_env(**kwargs),
        environment=environment,
        discretizer=discretizer,
        env_desc=env_desc,
    )


def __encode_env(**kwargs: Mapping[str, Any]) -> str:
    keys = []
    values = []
    for key, value in sorted(kwargs.items()):
        keys.append(key)
        values.append(value)

    hash_key = tuple(keys) + tuple(values)
    hashing = hashlib.sha512(str(hash_key).encode("UTF-8"))
    return base64.b32encode(hashing.digest()).decode("UTF-8")
