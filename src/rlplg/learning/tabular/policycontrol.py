"""
Policy control methods.
"""

import copy
import dataclasses
from typing import Any, Callable, Dict, Generator

import gymnasium as gym
import numpy as np

from rlplg import core, envplay
from rlplg.learning import utils
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies

EntityIdFn = Callable[[Any], int]


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
    create_egreedy_policy: Callable[
        [np.ndarray, EntityIdFn, float], policies.PyEpsilonGreedyPolicy
    ] = utils.create_egreedy_policy,
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
    qtable = copy.deepcopy(initial_qtable)
    egreedy_policy = create_egreedy_policy(qtable, state_id_fn, epsilon)
    steps_counter = 0
    for episode in range(num_episodes):
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, egreedy_policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        returns = 0.0
        while True:
            try:
                traj_step = next(trajectory)
            except StopIteration:
                break
            else:
                experiences[traj_step_idx] = traj_step
                traj_step_idx += 1
                returns += traj_step.reward

            # SARSA requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - 2, None)

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
            # update the qtable before generating the
            # next step in the trajectory
            egreedy_policy.exploit_policy.set_action_values(qtable)
            steps_counter += 1
            step += 1

        # need to copy qtable because it's a mutable numpy array
        yield PolicyControlSnapshot(
            steps=traj_step_idx, returns=returns, action_values=copy.deepcopy(qtable)
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
    create_egreedy_policy: Callable[
        [np.ndarray, EntityIdFn, float], policies.PyEpsilonGreedyPolicy
    ] = utils.create_egreedy_policy,
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
    qtable = copy.deepcopy(initial_qtable)
    egreedy_policy = create_egreedy_policy(qtable, state_id_fn, epsilon)
    steps_counter = 0
    for episode in range(num_episodes):
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, egreedy_policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        returns = 0.0
        while True:
            try:
                traj_step = next(trajectory)
            except StopIteration:
                break
            else:
                experiences[traj_step_idx] = traj_step
                traj_step_idx += 1
                returns += traj_step.reward

            # SARSA requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - 2, None)

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
            # update the qtable before generating the
            # next step in the trajectory
            egreedy_policy.exploit_policy.set_action_values(qtable)
            steps_counter += 1
            step += 1

        # need to copy qtable because it's a mutable numpy array
        yield PolicyControlSnapshot(
            steps=traj_step_idx, returns=returns, action_values=copy.deepcopy(qtable)
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
    create_egreedy_policy: Callable[
        [np.ndarray, EntityIdFn, float], policies.PyEpsilonGreedyPolicy
    ] = utils.create_egreedy_policy,
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
    egreedy_policy = create_egreedy_policy(qtable, state_id_fn, epsilon)
    steps_counter = 0
    for episode in range(num_episodes):
        final_step = np.iinfo(np.int64).max
        experiences: Dict[int, core.TrajectoryStep] = {}
        trajectory = generate_episode(environment, egreedy_policy)
        step = 0
        # `traj_step_idx` tracks the step in the traj
        traj_step_idx = 0
        # In the absence of a step as terminal
        # or truncated, `empty_steps` prevents
        # infinite loops
        empty_steps = 0
        returns = 0.0
        while True:
            if step > final_step or empty_steps > nstep:
                break
            try:
                traj_step = next(trajectory)
            except StopIteration:
                empty_steps += 1
            else:
                experiences[traj_step_idx] = traj_step
                traj_step_idx += 1
                returns += traj_step.reward

            # SARSA requires at least one next state
            if len(experiences) < 2:
                continue
            # keep the last n steps
            experiences.pop(step - nstep, None)

            if step < final_step:
                if experiences[step + 1].terminated or experiences[step + 1].truncated:
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
                egreedy_policy.exploit_policy.set_action_values(qtable)
                steps_counter += 1
            step += 1
        # need to copy qtable because it's a mutable numpy array
        yield PolicyControlSnapshot(
            steps=traj_step_idx, returns=returns, action_values=copy.deepcopy(qtable)
        )
