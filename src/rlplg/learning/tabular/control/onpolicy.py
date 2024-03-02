"""
Policy evaluation methods.
"""

import collections
import copy
from typing import Any, Callable, DefaultDict, Generator, List, Tuple

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


def first_visit_monte_carlo_action_values(
    environment: gym.Env,
    num_episodes: int,
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
    On-policy with First-Visit Monte Carlo (for e-soft policies).
    Learns a policy to maximize rewards in an environment.

    Note to self: As long you don't use the table you're updating,
    the current approach is fine

    Args:
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        alpha: The learning rate.
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

    Note: the first reward (in Sutton & Barto, 2018) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """

    def visit_key(experience) -> Tuple[int, int]:
        return state_id_fn(experience.observation), action_id_fn(experience.action)

    egreedy_policy = policies.PyEpsilonGreedyPolicy(
        policy=policies.PyQGreedyPolicy(
            state_id_fn=state_id_fn, action_values=initial_qtable
        ),
        num_actions=initial_qtable.shape[1],
        epsilon=epsilon,
    )
    qtable = copy.deepcopy(initial_qtable)
    state_action_updates: DefaultDict[Tuple[int, int], int] = collections.defaultdict(
        int
    )
    state_action_visits_remaining: DefaultDict[
        Tuple[int, int], int
    ] = collections.defaultdict(int)

    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        _experiences = list(generate_episodes(environment, egreedy_policy, 1))
        experiences: List[core.TrajectoryStep] = []
        while len(_experiences) > 0:
            experience = _experiences.pop()
            state_action_visits_remaining[visit_key(experience)] += 1
            experiences.append(experience)

        episode_return = 0.0
        for experience in experiences:
            key = visit_key(experience)
            state_action_visits_remaining[key] -= 1
            reward = experience.reward
            episode_return = gamma * episode_return + reward

            if state_action_visits_remaining[key] == 0:
                state_action_updates[key] += 1
                state_id, action_id = key
                if state_action_updates[key] == 1:
                    # first value
                    qtable[state_id, action_id] = episode_return
                else:
                    # average returns
                    qtable[state_id, action_id] = qtable[state_id, action_id] + (
                        (episode_return - qtable[state_id, action_id])
                        / state_action_updates[key]
                    )
        setattr(egreedy_policy.exploit_policy, "_state_action_value_table", qtable)
        yield len(experiences), copy.deepcopy(qtable)
