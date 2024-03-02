"""
Policy evaluation methods.
"""

import copy
from typing import Any, Callable, Generator, List, Tuple

import gymnasium as gym
import numpy as np

from rlplg import core, envplay
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies


def sarsa_action_values(
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episodes: Callable[
        [
            gym.Env,
            core.PyPolicy,
            int,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    On-policy Control Sarsa.
    Learns a policy to maximize rewards in an environment.

    Note to self: As long you don't use the table you're updating,
    the current approach is fine

    Args:
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        lrs: The learning rate schedule.
        gamma: The discount rate.
        epsilon: exploration rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(s,a) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(s,a) table.
        initial_qtable: Initial action-value estimates.
        generate_episodes: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (Sutton & Barto) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    egreedy_policy = policies.PyEpsilonGreedyPolicy(
        policy=policies.PyQGreedyPolicy(
            state_id_fn=state_id_fn, action_values=initial_qtable
        ),
        num_actions=initial_qtable.shape[1],
        epsilon=epsilon,
    )
    qtable = copy.deepcopy(initial_qtable)
    steps_counter = 0
    for episode in range(num_episodes):
        experiences: List[core.TrajectoryStep] = []
        for step, traj_step in enumerate(
            generate_episodes(environment, egreedy_policy, 1)
        ):
            experiences.append(traj_step)
            if step - 1 > 0:
                state_id = state_id_fn(experiences[step - 1].observation)
                action_id = action_id_fn(experiences[step - 1].action)
                reward = experiences[step - 1].reward

                next_state_id = state_id_fn(experiences[step].observation)
                next_action_id = action_id_fn(experiences[step].action)
                alpha = lrs(episode=episode, step=steps_counter)
                qtable[state_id, action_id] += alpha * (
                    reward
                    + gamma * qtable[next_state_id, next_action_id]
                    - qtable[state_id, action_id]
                )
                steps_counter += 1
            # update the qtable before generating the
            # next step in the trajectory
            setattr(egreedy_policy.exploit_policy, "_state_action_value_table", qtable)

        # need to copy qtable because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(qtable)


def qlearning(
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episodes: Callable[
        [
            gym.Env,
            core.PyPolicy,
            int,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Implements Q-learning, using epsilon-greedy as a collection (behavior) policy.

    Args:
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        lrs: The learning rate schedule.
        gamma: The discount rate.
        epsilon: exploration rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(s,a) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(s,a) table.
        initial_qtable: Initial action-value estimates.
        generate_episodes: A function that generates episodic
            trajectories given an environment and policy.

    Yields:
        A tuple of steps (count) and q-table.
    """
    egreedy_policy = policies.PyEpsilonGreedyPolicy(
        policy=policies.PyQGreedyPolicy(
            state_id_fn=state_id_fn, action_values=initial_qtable
        ),
        num_actions=initial_qtable.shape[1],
        epsilon=epsilon,
    )
    qtable = copy.deepcopy(initial_qtable)
    steps_counter = 0
    for episode in range(num_episodes):
        experiences: List[core.TrajectoryStep] = []
        for step, traj_step in enumerate(
            generate_episodes(environment, egreedy_policy, 1)
        ):
            experiences.append(traj_step)
            if step - 1 > 0:
                state_id = state_id_fn(experiences[step - 1].observation)
                action_id = action_id_fn(experiences[step - 1].action)
                reward = experiences[step - 1].reward

                next_state_id = state_id_fn(experiences[step].observation)
                alpha = lrs(episode=episode, step=steps_counter)
                # Q-learning uses the next best action's
                # value
                qtable[state_id, action_id] += alpha * (
                    reward
                    + gamma * np.max(qtable[next_state_id])
                    - qtable[state_id, action_id]
                )
                steps_counter += 1
            # update the qtable before generating the
            # next step in the trajectory
            setattr(egreedy_policy.exploit_policy, "_state_action_value_table", qtable)

        # need to copy qtable because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(qtable)
