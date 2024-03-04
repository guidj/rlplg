"""
Policy evaluation methods.
"""

import copy
import dataclasses
from typing import Any, Callable, Generator, List

import gymnasium as gym
import numpy as np

from rlplg import core, envplay
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies


@dataclasses.dataclass(frozen=True)
class PolicyControlSnapshot:
    steps: int
    returns: float
    action_values: np.ndarray


def onpolicy_sarsa_control(
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episode: Callable[
        [
            gym.Env,
            core.PyPolicy,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episode,
) -> Generator[PolicyControlSnapshot, None, None]:
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
        A `PolicyControlSnapshot` for each episode.

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
        returns = 0.0
        for step, traj_step in enumerate(generate_episode(environment, egreedy_policy)):
            experiences.append(traj_step)
            returns += traj_step.reward
            if step - 1 >= 0:
                state_id = state_id_fn(experiences[0].observation)
                action_id = action_id_fn(experiences[0].action)
                reward = experiences[0].reward

                next_state_id = state_id_fn(experiences[1].observation)
                next_action_id = action_id_fn(experiences[1].action)
                alpha = lrs(episode=episode, step=steps_counter)
                qtable[state_id, action_id] += alpha * (
                    reward
                    + gamma * qtable[next_state_id, next_action_id]
                    - qtable[state_id, action_id]
                )
                experiences = experiences[1:]
                steps_counter += 1
            # update the qtable before generating the
            # next step in the trajectory
            setattr(egreedy_policy.exploit_policy, "_state_action_value_table", qtable)

        # need to copy qtable because it's a mutable numpy array
        yield PolicyControlSnapshot(
            steps=step + 1, returns=returns, action_values=copy.deepcopy(qtable)
        )


def onpolicy_qlearning_control(
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episode: Callable[
        [
            gym.Env,
            core.PyPolicy,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episode,
) -> Generator[PolicyControlSnapshot, None, None]:
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
        A `PolicyControlSnapshot` for each episode.
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
        returns = 0.0
        for step, traj_step in enumerate(generate_episode(environment, egreedy_policy)):
            experiences.append(traj_step)
            returns += traj_step.reward
            if step - 1 > 0:
                state_id = state_id_fn(experiences[0].observation)
                action_id = action_id_fn(experiences[0].action)
                reward = experiences[0].reward

                next_state_id = state_id_fn(experiences[1].observation)
                alpha = lrs(episode=episode, step=steps_counter)
                # Q-learning uses the next best action's
                # value
                qtable[state_id, action_id] += alpha * (
                    reward
                    + gamma * np.max(qtable[next_state_id])
                    - qtable[state_id, action_id]
                )
                experiences = experiences[1:]
                steps_counter += 1
            # update the qtable before generating the
            # next step in the trajectory
            setattr(egreedy_policy.exploit_policy, "_state_action_value_table", qtable)

        # need to copy qtable because it's a mutable numpy array
        yield PolicyControlSnapshot(
            steps=step + 1, returns=returns, action_values=copy.deepcopy(qtable)
        )


def onpolicy_nstep_sarsa_control(
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    nstep: int,
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episode: Callable[
        [
            gym.Env,
            core.PyPolicy,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episode,
) -> Generator[PolicyControlSnapshot, None, None]:
    """
    n-step SARSA learning for policy control.
    Source: https://en.wikipedia.org/wiki/Temporal_difference_learning

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
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
        A `PolicyControlSnapshot` for each episode.

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract reward access references by one.
    """
    qtable = copy.deepcopy(initial_qtable)
    egreedy_policy = policies.PyEpsilonGreedyPolicy(
        policy=policies.PyQGreedyPolicy(
            state_id_fn=state_id_fn, action_values=initial_qtable
        ),
        num_actions=initial_qtable.shape[1],
        epsilon=epsilon,
    )
    steps_counter = 0
    for episode in range(num_episodes):
        final_step = np.iinfo(np.int64).max
        experiences: List[core.TrajectoryStep] = []
        returns = 0.0
        for step, traj_step in enumerate(generate_episode(environment, egreedy_policy)):
            experiences.append(traj_step)
            returns += traj_step.reward
            if step - 1 > 0:
                if step < final_step:
                    if experiences[1].terminated or experiences[1].truncated:
                        final_step = step + 1
                tau = step - nstep + 1
                if tau >= 0:
                    # tau + 1
                    min_idx = 1
                    # min(tau + nstep, final_step)
                    max_idx = min(nstep, len(experiences))
                    nstep_returns = 0.0

                    for i in range(min_idx, max_idx + 1):
                        # gamma ** (i - tau - 1); experiences[i - 1]
                        nstep_returns += (gamma ** (i - 1)) * experiences[i - 1].reward
                    if tau + nstep < final_step:
                        # tau + nstep
                        nstep_returns += (gamma**nstep) * qtable[
                            state_id_fn(experiences[nstep - 1].observation),
                            action_id_fn(experiences[nstep - 1].action),
                        ]
                    state_id = state_id_fn(experiences[0].observation)
                    action_id = action_id_fn(experiences[0].action)
                    alpha = lrs(episode=episode, step=steps_counter)
                    qtable[state_id, action_id] += alpha * (
                        nstep_returns - qtable[state_id, action_id]
                    )
                    experiences = experiences[1:]
                    # update the qtable before generating the
                    # next step in the trajectory
                    setattr(
                        egreedy_policy.exploit_policy,
                        "_state_action_value_table",
                        qtable,
                    )
                steps_counter += 1
        # need to copy qtable because it's a mutable numpy array
        yield PolicyControlSnapshot(
            steps=step + 1, returns=returns, action_values=copy.deepcopy(qtable)
        )
