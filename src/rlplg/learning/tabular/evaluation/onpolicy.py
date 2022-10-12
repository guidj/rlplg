"""
Policy evaluation methods.
"""
import collections
import copy
from typing import Any, Callable, Generator, Tuple

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import Array

from rlplg import envplay
from rlplg.learning.tabular import policies


def first_visit_monte_carlo_action_values(
    policy: policies.PyQGreedyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    gamma: float,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episodes: Callable[
        [
            py_environment.PyEnvironment,
            py_policy.PyPolicy,
            int,
        ],
        Generator[trajectory.Trajectory, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, Array, float], None, None]:
    """
    First-Visit Monte Carlo Prediction.
    Estimates Q(s, a) for a fixed policy pi.
    Source: http://www.incompleteideas.net/book/ebook/node52.html

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        alpha: The learning rate.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(S, A) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(S, A) table.
        initial_qtable: A prior belief of Q(S, A) estimates.
        event_mapper: A function that generates trajectories from a given trajectory.
            This is useful in cases where the caller whishes to apply
            a transformation to experience logs.
            Defautls to `envplay.identity_replay`.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """

    def visit_key(experience) -> Tuple[int, int]:
        return state_id_fn(experience.observation), action_id_fn(experience.action)

    # first state and reward come from env reset
    qtable = copy.deepcopy(initial_qtable)
    state_action_updates = collections.defaultdict(int)
    state_action_visits_remaining = collections.defaultdict(int)

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        _experiences = list(generate_episodes(environment, policy, num_episodes=1))
        # reverse list and ammortize state visits
        experiences = []
        while len(_experiences) > 0:
            experience = _experiences.pop()
            state_action_visits_remaining[visit_key(experience)] += 1
            experiences.append(experience)

        episode_return = 0
        for experience in experiences:
            key = visit_key(experience)
            state_action_visits_remaining[key] -= 1
            state_id, action_id = key
            reward = experience.reward
            episode_return = gamma * episode_return + reward

            if state_action_visits_remaining[key] == 0:
                state_action_updates[key] += 1
                if state_action_updates[key] == 1:
                    # first value
                    qtable[state_id, action_id] = episode_return
                else:
                    qtable[state_id, action_id] = qtable[state_id, action_id] + (
                        (episode_return - qtable[state_id, action_id])
                        / state_action_updates[key]
                    )

        # need to copy values because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(qtable)


def sarsa_action_values(
    policy: policies.PyQGreedyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    alpha: float,
    gamma: float,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episodes: Callable[
        [
            py_environment.PyEnvironment,
            py_policy.PyPolicy,
            int,
        ],
        Generator[trajectory.Trajectory, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, Array, float], None, None]:
    """
    On-policy Sarsa Prediction.
    Estimates Q(s, a) for a fixed policy pi.
    Source: https://homes.cs.washington.edu/~bboots/RL-Spring2020/Lectures/TD_notes.pdf
    In the document, they refer to Algorithm 15 as Algorithm 16.

    Note to self: As long you don't use the table you're updating,
    the current approach is fine

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        alpha: The learning rate.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(S, A) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(S, A) table.
        initial_qtable: A prior belief of Q(S, A) estimates.
        event_mapper: A function that generates trajectories from a given trajectory.
            This is useful in cases where the caller whishes to apply
            a transformation to experience logs.
            Defautls to `envplay.identity_replay`.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    # first state and reward come from env reset
    qtable = copy.deepcopy(initial_qtable)

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episodes(environment, policy, num_episodes=1))
        for step in range(len(experiences) - 1):
            state_id = state_id_fn(experiences[step].observation)
            action_id = action_id_fn(experiences[step].action)
            reward = experiences[step].reward

            next_state_id = state_id_fn(experiences[step + 1].observation)
            next_action_id = action_id_fn(experiences[step + 1].action)

            qtable[state_id, action_id] += alpha * (
                reward
                + gamma * qtable[next_state_id, next_action_id]
                - qtable[state_id, action_id]
            )

        # need to copy qtable because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(qtable)


def first_visit_monte_carlo_state_values(
    policy: policies.PyQGreedyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    gamma: float,
    state_id_fn: Callable[[Any], int],
    initial_values: np.ndarray,
    generate_episodes: Callable[
        [
            py_environment.PyEnvironment,
            py_policy.PyPolicy,
            int,
        ],
        Generator[trajectory.Trajectory, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, Array, float], None, None]:
    """
    First-Visit Monte Carlo Prediction.
    Estimates V(s) for a fixed policy pi.
    Source: http://www.incompleteideas.net/book/ebook/node51.html

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        alpha: The learning rate.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(S, A) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(S, A) table.
        initial_qtable: A prior belief of Q(S, A) estimates.
        event_mapper: A function that generates trajectories from a given trajectory.
            This is useful in cases where the caller whishes to apply
            a transformation to experience logs.
            Defautls to `envplay.identity_replay`.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    # first state and reward come from env reset
    values = copy.deepcopy(initial_values)
    state_updates = collections.defaultdict(int)
    state_visits_remaining = collections.defaultdict(int)

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        _experiences = list(generate_episodes(environment, policy, num_episodes=1))
        # reverse list and ammortize state visits
        experiences = []
        while len(_experiences) > 0:
            experience = _experiences.pop()
            state_visits_remaining[state_id_fn(experience.observation)] += 1
            experiences.append(experience)

        episode_return = 0
        for experience in experiences:
            state_id = state_id_fn(experience.observation)
            reward = experience.reward
            episode_return = gamma * episode_return + reward
            state_visits_remaining[state_id] -= 1

            if state_visits_remaining[state_id] == 0:
                state_updates[state_id] += 1
                if state_updates[state_id] == 1:
                    # first value
                    values[state_id] = episode_return
                else:
                    values[state_id] = values[state_id] + (
                        (episode_return - values[state_id]) / state_updates[state_id]
                    )

        # need to copy values because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(values)


def one_step_td_state_values(
    policy: policies.PyQGreedyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    alpha: float,
    gamma: float,
    state_id_fn: Callable[[Any], int],
    initial_values: np.ndarray,
    generate_episodes: Callable[
        [
            py_environment.PyEnvironment,
            py_policy.PyPolicy,
            int,
        ],
        Generator[trajectory.Trajectory, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, Array, float], None, None]:
    """
    TD(0) or one-step TD.
    Estimates V(s) for a fixed policy pi.
    Source: https://en.wikipedia.org/wiki/Temporal_difference_learning

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        alpha: The learning rate.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(S, A) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(S, A) table.
        initial_qtable: A prior belief of Q(S, A) estimates.
        event_mapper: A function that generates trajectories from a given trajectory.
            This is useful in cases where the caller whishes to apply
            a transformation to experience logs.
            Defautls to `envplay.identity_replay`.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    # first state and reward come from env reset
    values = copy.deepcopy(initial_values)

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episodes(environment, policy, num_episodes=1))
        for step in range(len(experiences) - 1):
            state_id = state_id_fn(experiences[step].observation)
            next_state_id = state_id_fn(experiences[step + 1].observation)
            values[state_id] += alpha * (
                experiences[step].reward
                + gamma * values[next_state_id]
                - values[state_id]
            )

        # need to copy values because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(values)
