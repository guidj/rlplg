"""
Policy utility functions.
"""

import logging
from typing import Set

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


def policy_prob_fn(policy: py_policy.PyPolicy, traj: trajectory.Trajectory) -> float:
    """The policy we're evaluating is assumed to be greedy w.r.t. Q(s, a).
    So the best action has probability 1.0, and all the others 0.0.
    """

    time_step = ts.TimeStep(
        step_type=traj.step_type,
        reward=traj.reward,
        discount=traj.discount,
        observation=traj.observation,
    )
    policy_step = policy.action(time_step)
    return np.where(np.array_equal(policy_step.action, traj.action), 1.0, 0.0)


def collect_policy_prob_fn(
    policy: py_policy.PyPolicy, traj: trajectory.Trajectory
) -> float:
    """The behavior policy is assumed to be fixed over the evaluation window.
    We log probabilities when choosing actions, so we can just use that information.
    For a random policy on K arms, log_prob = log(1/K).
    We just have to return exp(log_prob).
    """
    del policy
    return np.math.exp(traj.policy_info.log_probability)


def initial_table(
    num_states: int,
    num_actions: int,
    dtype: np.dtype = np.float32,
    random: bool = False,
    terminal_states: Set[int] = set(),
) -> np.ndarray:
    """
    The value of terminal states should be zero.
    """
    if random:
        if not terminal_states:
            logging.warning("Creating Q-table with no terminal states")

        qtable = np.random.rand(num_states, num_actions)
        qtable[list(terminal_states), :] = 0.0
        return qtable.astype(dtype)
    return np.zeros(shape=(num_states, num_actions), dtype=dtype)
