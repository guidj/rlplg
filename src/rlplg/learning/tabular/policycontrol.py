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
    generate_episodes: Callable[
        [
            gym.Env,
            core.PyPolicy,
            int,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episodes,
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
        returns = 0.0
        for step, traj_step in enumerate(
            generate_episodes(environment, egreedy_policy, 1)
        ):
            experiences.append(traj_step)
            returns += (gamma**step) * traj_step.reward
            if step - 1 > 0:
                # this will be updated in the next `step`
                # we modify it to avoid changing indeces below.
                step -= 1
                state_id = state_id_fn(experiences[step].observation)
                action_id = action_id_fn(experiences[step].action)
                reward = experiences[step].reward

                next_state_id = state_id_fn(experiences[step + 1].observation)
                next_action_id = action_id_fn(experiences[step + 1].action)
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
        yield PolicyControlSnapshot(
            steps=len(experiences), returns=returns, action_values=copy.deepcopy(qtable)
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
    generate_episodes: Callable[
        [
            gym.Env,
            core.PyPolicy,
            int,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episodes,
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
        returns = 0.0
        for step, traj_step in enumerate(
            generate_episodes(environment, egreedy_policy, 1)
        ):
            experiences.append(traj_step)
            returns += (gamma**step) * traj_step.reward
            if step - 1 > 0:
                # this will be updated in the next `step`
                # we modify it to avoid changing indeces below.
                step -= 1
                state_id = state_id_fn(experiences[step].observation)
                action_id = action_id_fn(experiences[step].action)
                reward = experiences[step].reward

                next_state_id = state_id_fn(experiences[step + 1].observation)
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
        yield PolicyControlSnapshot(
            steps=len(experiences), returns=returns, action_values=copy.deepcopy(qtable)
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
    generate_episodes: Callable[
        [
            gym.Env,
            core.PyPolicy,
            int,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episodes,
) -> Generator[PolicyControlSnapshot, None, None]:
    """
    n-step TD learning.
    Estimates V(s) for a fixed policy pi.
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
        A tuple of steps (count) and v-table.

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
        for step, traj_step in enumerate(
            generate_episodes(environment, egreedy_policy, 1)
        ):
            experiences.append(traj_step)
            returns += (gamma**step) * traj_step.reward
            if step - 1 > 0:
                # this will be updated in the next `step`
                # we modify it to avoid changing indeces below.
                step -= 1
                if step < final_step:
                    if (
                        experiences[step + 1].terminated
                        or experiences[step + 1].truncated
                    ):
                        final_step = step + 1
                tau = step - nstep + 1
                if tau >= 0:
                    min_idx = tau + 1
                    max_idx = min(tau + nstep, final_step)
                    nstep_returns = 0.0

                    for i in range(min_idx, max_idx + 1):
                        nstep_returns += (gamma ** (i - tau - 1)) * experiences[
                            i - 1
                        ].reward
                    if tau + nstep < final_step:
                        nstep_returns += (gamma**nstep) * qtable[
                            state_id_fn(experiences[tau + nstep].observation),
                            action_id_fn(experiences[tau + nstep].action),
                        ]
                    state_id = state_id_fn(experiences[tau].observation)
                    action_id = action_id_fn(experiences[tau].action)
                    alpha = lrs(episode=episode, step=steps_counter)
                    qtable[state_id, action_id] += alpha * (
                        nstep_returns - qtable[state_id, action_id]
                    )
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
            steps=len(experiences), returns=returns, action_values=copy.deepcopy(qtable)
        )
