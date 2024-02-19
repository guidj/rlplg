import math
import uuid
from typing import Any, Mapping

import numpy as np
import pytest

from rlplg import core, envsuite
from rlplg.core import TimeStep


@pytest.mark.parametrize(
    "env_name", envsuite.SUPPORTED_RLPLG_ENVS | envsuite.SUPPORTED_GYM_ENVS
)
def test_envsuite_load(env_name: str, args: Mapping[str, Mapping[str, Any]]):
    env_spec = envsuite.load(name=env_name, **args[env_name])
    assert isinstance(env_spec, core.EnvSpec)
    assert env_spec.name == env_name
    terminal_states = core.infer_env_terminal_states(env_spec.mdp.transition)
    # reset env and state (get initial values)
    # play for one episode
    obs, _ = env_spec.environment.reset()
    time_step: TimeStep = obs, math.nan, False, False, {}
    assert 0 <= env_spec.discretizer.state(obs) <= env_spec.mdp.env_desc.num_states

    while True:
        obs, _, terminated, truncated, _ = time_step
        action = np.random.randint(0, env_spec.mdp.env_desc.num_actions)
        next_time_step = env_spec.environment.step(action)
        next_obs, next_reward, _, _, _ = next_time_step
        assert 0 <= env_spec.discretizer.state(obs) <= env_spec.mdp.env_desc.num_states
        assert (
            0
            <= env_spec.discretizer.action(action)
            <= env_spec.mdp.env_desc.num_actions
        )
        if env_spec.discretizer.state(obs) in terminal_states:
            assert next_reward == 0.0
            assert env_spec.discretizer.state(obs) == env_spec.discretizer.state(
                next_obs
            )
        if terminated or truncated:
            break
        time_step = next_time_step

    env_spec.environment.close()


def test_envsuite_load_with_unsupported_env():
    with pytest.raises(ValueError):
        envsuite.load(str(uuid.uuid4()))


@pytest.fixture
def args() -> Mapping[str, Mapping[str, Any]]:
    return {
        "ABCSeq": {"length": 3},
        "CliffWalking-v0": {"max_episode_steps": 100},
        "FrozenLake-v1": {"is_slippery": False},
        "GridWorld": {"grid": "xsog"},
        "RedGreenSeq": {"cure": ["red", "green", "wait", "green"]},
        "StateRandomWalk": {"steps": 3},
        "TariffFrozenLake-v1": {"is_slippery": False},
        "Taxi-v3": {"max_episode_steps": 100},
        "TowerOfHanoi": {"num_disks": 4},
    }