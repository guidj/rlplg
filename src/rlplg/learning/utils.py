"""
Policy utility functions.
"""

import logging
from typing import Any, Callable, Iterable, Optional, Set, Union

import numpy as np
from numpy.typing import DTypeLike

from rlplg import core


def policy_prob_fn(policy: core.PyPolicy, traj: core.TrajectoryStep) -> float:
    """The policy we're evaluating is assumed to be greedy w.r.t. Q(s, a).
    So the best action has probability 1.0, and all the others 0.0.
    """
    policy_step = policy.action(traj.observation)
    prob: float = np.where(
        np.array_equal(policy_step.action, traj.action), 1.0, 0.0
    ).item()
    return prob


def collect_policy_prob_fn(policy: core.PyPolicy, traj: core.TrajectoryStep) -> float:
    """The behavior policy is assumed to be fixed over the evaluation window.
    We log probabilities when choosing actions, so we can just use that information.
    For a random policy on K arms, log_prob = log(1/K).
    We just have to return exp(log_prob).
    """
    del policy
    prob: float = np.exp(traj.policy_info["log_probability"])
    return prob


def initial_table(
    num_states: int,
    num_actions: int,
    dtype: DTypeLike = np.float32,
    random: bool = False,
    terminal_states: Optional[Set[int]] = None,
) -> np.ndarray:
    """
    The value of terminal states should be zero.
    """
    if random:
        if terminal_states is None:
            logging.warning("Creating Q-table with no terminal states")

        qtable = np.random.rand(num_states, num_actions)
        qtable[list(terminal_states or []), :] = 0.0
        return qtable.astype(dtype)
    return np.zeros(shape=(num_states, num_actions), dtype=dtype)


def chain_map(inputs: Any, funcs: Iterable[Callable[[Any], Any]]):
    """
    Applies a chain of functions to an element.
    """

    funcs_iterator = iter(funcs)
    while True:
        try:
            func = next(funcs_iterator)
            inputs = func(inputs)
        except StopIteration:
            return inputs


def nan_or_inf(array: Union[np.ndarray, int, float]) -> bool:
    """
    Checks if an array has `nan` or `inf` values.
    """
    is_nan: bool = np.any(np.isnan(array)).item()
    is_inf: bool = np.any(np.isinf(array)).item()
    return is_nan or is_inf
