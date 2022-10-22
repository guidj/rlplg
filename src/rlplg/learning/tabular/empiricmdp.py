"""
This module has classes that pertain to emperic decision processes.
"""
import collections
import copy
import dataclasses
import logging
import os.path
import tempfile
from typing import Any, Callable, DefaultDict, Mapping, Sequence, Tuple, TypeVar

import h5py
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArray

from rlplg import envdesc
from rlplg.learning.tabular import markovdp

KREF_TRANSITIONS = "transitions"
KREF_REWARDS = "rewards"
KREF_KEYS = "keys"
KREF_VALUES = "values"

FILE_EXT = "h5"

T = TypeVar("T")

StateAction = Tuple[int, int]
TransitionStats = Mapping[StateAction, Mapping[int, int]]
RewardStats = Mapping[StateAction, Mapping[int, float]]
MdpFn = Mapping[Tuple[int, int, int], float]


@dataclasses.dataclass(frozen=True)
class MdpStats:
    """
    Class contains collected transition and aggregate rewards for (S, A, S') tuples.
    """

    transitions: TransitionStats
    rewards: RewardStats


@dataclasses.dataclass(frozen=True)
class MdpFunctions:
    """
    Class contains functions for Markov decision process.
    """

    transition: MdpFn
    reward: MdpFn


class InferredMdp(markovdp.MDP):
    """
    Class for a markov decision process inferred from transitions of a policy pi.
    """

    def __init__(self, mdp_functions: MdpFunctions, env_desc: envdesc.EnvDesc):
        self._mdp_functions = mdp_functions
        self._env_desc = env_desc

    def transition_probability(
        self, state: NestedArray, action: NestedArray, next_state: NestedArray
    ) -> float:
        """
        Given a state s, action a, and next state s' returns a transition probability.
        Args:
            state: starting state
            action: agent's action
            next_state: state transition into after taking the action.

        Returns:
            A transition probability.
        """
        key = (state, action, next_state)
        return self._mdp_functions.transition.get(key, 0)

    def reward(
        self, state: NestedArray, action: NestedArray, next_state: NestedArray
    ) -> float:
        """
        Given a state s, action a, and next state s' returns the expected reward.

        Args:
            state: starting state
            action: agent's action
            next_state: state transition into after taking the action.
        Returns
            A transition probability.
        """
        key = (state, action, next_state)
        return self._mdp_functions.reward.get(key, 0.0)

    def env_desc(self) -> envdesc.EnvDesc:
        """
        Returns:
            An instance of EnvDesc with properties of the environment.
        """
        return self._env_desc


def collect_mdp_stats(
    environment: py_environment.PyEnvironment,
    policy: py_policy.PyPolicy,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    num_episodes: int,
    logging_frequency_episodes: int,
) -> MdpStats:
    """
    Collections emperical statistics from an environment through play.
    """
    environment.reset()
    transitions = collections.defaultdict(_create_trasitions)
    rewards = collections.defaultdict(_create_rewards)
    logging_enabled = logging_frequency_episodes > 0
    for episode in range(1, num_episodes + 1):
        environment.reset()
        while True:
            time_step = environment.current_time_step()
            policy_step = policy.action(time_step)
            next_time_step = environment.step(policy_step.action)

            state = state_id_fn(time_step.observation)
            action = action_id_fn(policy_step.action)
            next_state = state_id_fn(next_time_step.observation)
            transitions[(state, action)][next_state] += 1
            rewards[(state, action)][next_state] += next_time_step.reward

            if time_step.step_type == ts.StepType.LAST:
                break

        # non-positive logging frequency disables logging
        if logging_enabled and episode % logging_frequency_episodes == 0:
            logging.info("Episode %d/%d", episode, num_episodes)

    return MdpStats(transitions=transitions, rewards=rewards)


def aggregate_stats(elements: Sequence[MdpStats]) -> MdpStats:
    """
    Aggregates multiple instances of MDPStats into one.
    """

    transitions = collections.defaultdict(_create_trasitions)
    rewards = collections.defaultdict(_create_rewards)

    for element in elements:
        transitions = _accumulate(transitions, element.transitions)
        rewards = _accumulate(rewards, element.rewards)
    return MdpStats(transitions=transitions, rewards=rewards)


def _accumulate(
    collector: DefaultDict[StateAction, DefaultDict[int, T]],
    element: DefaultDict[StateAction, DefaultDict[int, T]],
) -> Mapping[StateAction, Mapping[int, T]]:
    """
    Accumulates stats from element into collector.
    """
    new_collector = copy.deepcopy(collector)
    for (state, action), next_state_values in element.items():
        for next_state, value in next_state_values.items():
            new_collector[(state, action)][next_state] += value
    return new_collector


def _create_trasitions() -> DefaultDict[int, int]:
    """
    Returns:
        A defaultdict of integers.
    Created to bypass serialization issues with lambdas.
    """
    return collections.defaultdict(int)


def _create_rewards() -> DefaultDict[int, float]:
    """
    A defaultdict of floats.
    Created to bypass serialization issues with lambdas.
    """
    return collections.defaultdict(float)


def export_stats(
    path: str, filename: str, mdp_stats: MdpStats, overwrite: bool = True
) -> None:
    """
    Saves stas in hdf5 format.
    """
    file_path = os.path.join(path, f"{filename}.{FILE_EXT}")
    with tempfile.NamedTemporaryFile() as tmp_file:
        logging.info("Exporting stats to %s", file_path)
        with h5py.File(tmp_file.file, "w") as database:
            for name, dtype, data in zip(
                (
                    KREF_TRANSITIONS,
                    KREF_REWARDS,
                ),
                (int, float),
                (mdp_stats.transitions, mdp_stats.rewards),
            ):
                keys, values = _flatten_mapping(data)
                # nested numpys arrays can't be built from tuples
                database.create_dataset(
                    f"{name}.{KREF_KEYS}",
                    data=np.array(list(keys), dtype=(np.int32, 3)),
                )
                database.create_dataset(
                    f"{name}.{KREF_VALUES}", data=np.array(values, dtype=dtype)
                )

        tf.io.gfile.copy(src=tmp_file.name, dst=file_path, overwrite=overwrite)


def _flatten_mapping(
    mapping: Mapping[Tuple[int, int], Mapping[int, Any]]
) -> Tuple[Sequence[Tuple[int, int, int]], Sequence[Any]]:
    keys = []
    values = []
    for key in tuple(mapping):
        state, action = key
        for next_state, value in mapping[key].items():
            keys.append((state, action, next_state))
            values.append(value)
    return tuple(keys), tuple(values)


def load_stats(path: str, filename: str) -> MdpStats:
    """
    Loads stas from an hdf5 file.
    """
    file_path = os.path.join(path, f"{filename}.{FILE_EXT}")
    with tempfile.NamedTemporaryFile() as tmp_file:
        logging.info("Loading stats from %s", file_path)
        tf.io.gfile.copy(src=file_path, dst=tmp_file.name, overwrite=True)
        with h5py.File(tmp_file.name, "r") as database:
            data = {}
            for name, dtype in zip(
                (
                    KREF_TRANSITIONS,
                    KREF_REWARDS,
                ),
                (int, float),
            ):
                keys = tuple(np.array(database[f"{name}.{KREF_KEYS}"]).tolist())
                values = tuple(np.array(database[f"{name}.{KREF_VALUES}"]).tolist())

                # unnest
                data[name] = collections.defaultdict(
                    lambda constructor=dtype: collections.defaultdict(
                        lambda: constructor
                    )
                )
                for key, value in zip(keys, values):
                    state, action, next_state = key
                    data[name][(state, action)][next_state] = value
            return MdpStats(**data)


def create_mdp_functions(mdp_stats: MdpStats) -> MdpFunctions:
    """
    Converts MdpStats into MdpFunctions.
        - Normalizes state visits
        - Averages rewards.

    Args:
        mdp_stats: an instance of MdpStats.

    Returns:
        An instance of MdpFunctions.
    """

    transitions = {}
    rewards = {}

    # only computes for visited states - so min visits = 1.
    for state_action, next_state_values in mdp_stats.transitions.items():
        state, action = state_action
        max_visits = max(next_state_values.values())
        for next_state, visits in next_state_values.items():
            entry_key = (state, action, next_state)
            # normalize visits
            transitions[entry_key] = visits / max_visits
            # average rewards
            rewards[entry_key] = mdp_stats.rewards[state_action][next_state] / visits
    return MdpFunctions(transition=transitions, reward=rewards)
