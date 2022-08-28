import base64
import hashlib
from typing import Any, Mapping

from tf_agents.environments import suite_gym

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
        return npsci.item(observation)

    def action(self, action: Any) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        return npsci.item(action)


def create_env_spec(name: str, **kwargs: Mapping[str, Any]) -> envspec.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = suite_gym.load(name, **kwargs)
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
    for key, value in sorted(kwargs):
        keys.append(key)
        values.append(value)

    hash_key = tuple(keys) + tuple(values)
    hashing = hashlib.sha512(str(hash_key).encode("UTF-8"))
    return base64.b32encode(hashing.digest()).decode("UTF-8")
