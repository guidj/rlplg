"""
Policy evaluation methods.
"""
import collections
import copy
from typing import Any, Callable, Generator, Tuple

import numpy as np

from rlplg import core, envplay
from rlplg.learning.opt import schedules

MCUpdate = collections.namedtuple("MCUpdate", ["returns", "cu_sum", "value", "weight"])


def monte_carlo_action_values(
    policy: core.PyPolicy,
    collect_policy: core.PyPolicy,
    environment: core.PyEnvironment,
    num_episodes: int,
    gamma: float,
    policy_probability_fn: Callable[
        [core.PyPolicy, core.Trajectory],
        float,
    ],
    collect_policy_probability_fn: Callable[
        [core.PyPolicy, core.Trajectory],
        float,
    ],
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episodes: Callable[
        [
            core.PyEnvironment,
            core.PyPolicy,
            int,
        ],
        Generator[core.Trajectory, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Off-policy MC Prediction.
    Estimates Q (table) for a fixed policy pi.

    Questions: when do you update the propensity of each action?
    At each step, do you use the propensity at that step the one
    reached at the end of the episode?

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        collect_policy: A behavior policy, used to generate episodes.
        environment: The environment used to generate episodes for evaluation.
        event_buffer: A buffer to store episode steps.
        num_episodes: The number of episodes to generate for evaluation.
        gamma: The discount rate.
        state_id_fn: A function that maps observations to an int ID for
            the Q(S, A) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(S, A) table.
        initial_qtable: A prior belief of Q(S, A) estimates.
            Initialized to randomly if None.
        collect_episodes: A function that generates trajectories.
            This is useful in cases where the caller whishes to apply
            a transformation to experience logs.
            Defautls to `envplay.collect_episodes`.

    Yields:
        A tuple of steps (count) and q-table.
    """
    for name, _policy in zip(("policy", "collect_policy"), (policy, collect_policy)):
        if not _policy.emit_log_probability:
            raise ValueError(
                f"{name} must have emit_log_probability=True. They are used for importance sampling."
            )

    qtable = copy.deepcopy(initial_qtable)
    cu_sum = np.zeros_like(initial_qtable)

    for _ in range(num_episodes):
        experiences = list(generate_episodes(environment, collect_policy, 1))
        num_steps = len(experiences)
        returns, weight = 0.0, 1.0
        # process from the ending
        iterator = reversed(experiences)
        while weight != 0.0:
            try:
                experience = next(iterator)
                state_id, action_id, reward = (
                    state_id_fn(experience.observation),
                    action_id_fn(experience.action),
                    experience.reward,
                )
                policy_prob = policy_probability_fn(policy, experience)
                collect_policy_prob = collect_policy_probability_fn(
                    collect_policy, experience
                )
                rho = policy_prob / collect_policy_prob
                (
                    returns,
                    cu_sum[state_id, action_id],
                    qtable[state_id, action_id],
                    weight,
                ) = monte_carlo_action_values_step(
                    reward=reward,
                    returns=returns,
                    cu_sum=cu_sum[state_id, action_id],
                    weight=weight,
                    value=qtable[state_id, action_id],
                    rho=rho,
                    gamma=gamma,
                )
            except StopIteration:
                break
        # need to copy qtable because it's a mutable numpy array
        yield num_steps, copy.deepcopy(qtable)


def monte_carlo_action_values_step(
    reward: float,
    returns: float,
    cu_sum: float,
    weight: float,
    value: float,
    rho: float,
    gamma: float,
) -> MCUpdate:
    """Single step update for Off-policy MC prediction for estimating Q_{pi}.
    G <- gamma * G + R_{t+1}
    C(S_{t},A_{t}) <- C(S_{t},A_{t}) + W
    Q(S_{t},A_{t}) <- Q(S_{t},A_{t}) + W/C(S_{t},A_{t}) * (G - Q(S_{t},A_{t}))
    rho <- pi(S_{t}|A_{t}) / b(S_{t}|A_{t})
    W <- W * rho
    """
    new_returns = gamma * returns + reward
    new_cu_sum = cu_sum + weight
    new_value = value + (weight / new_cu_sum) * (new_returns - value)
    new_weight = weight * rho
    return MCUpdate(
        returns=new_returns, cu_sum=new_cu_sum, value=new_value, weight=new_weight
    )


def nstep_sarsa_action_values(
    policy: core.PyPolicy,
    collect_policy: core.PyPolicy,
    environment: core.PyEnvironment,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    gamma: float,
    nstep: int,
    policy_probability_fn: Callable[
        [core.PyPolicy, core.Trajectory],
        float,
    ],
    collect_policy_probability_fn: Callable[
        [core.PyPolicy, core.Trajectory],
        float,
    ],
    state_id_fn: Callable[[Any], int],
    action_id_fn: Callable[[Any], int],
    initial_qtable: np.ndarray,
    generate_episodes: Callable[
        [
            core.PyEnvironment,
            core.PyPolicy,
            int,
        ],
        Generator[core.Trajectory, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Off-policy n-step Sarsa Prediction.
    Estimates Q (table) for a fixed policy pi.

    Args:
        policy: A target policy, pi, whose value function we wish to evaluate.
        collect_policy: A behavior policy, used to generate episodes.
        environment: The environment used to generate episodes for evaluation.
        num_episodes: The number of episodes to generate for evaluation.
        lrs: The learning rate schedule.
        gamma: The discount rate.
        nstep: The number of steps before value updates in the MDP sequence.
        policy_probability_fn: returns action propensity for the target policy,
            given a trajectory.
        collect_policy_probability_fn: returns action propensity for the collect policy,
            given a trajectory.
        state_id_fn: A function that maps observations to an int ID for
            the Q(S, A) table.
        action_id_fn: A function that maps actions to an int ID for
            the Q(S, A) table.
        initial_qtable: A prior belief of Q(S, A) estimates.
        generate_episodes: A function that generates trajectories.
            This is useful in cases where the caller whishes to apply
            a transformation to experience logs.
            Defautls to `envplay.collect_episodes`.

    Yields:
        A tuple of steps (count) and q-table.

    Note: the first reward (in the book) is R_{1} for R_{0 + 1};
    So index wise, we subtract them all by one.
    """
    if nstep < 1:
        raise ValueError(f"nstep must be > 1: {nstep}")
    # first state and reward come from env reset
    qtable = copy.deepcopy(initial_qtable)
    steps_counter = 0
    for episode in range(num_episodes):
        final_step = np.iinfo(np.int64).max
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episodes(environment, collect_policy, 1))
        for step, _ in enumerate(experiences):
            if step < final_step:
                # we don't need to transition because we already collected the experience
                # a better way to determine the next state is terminal one
                if np.array_equal(experiences[step].step_type, core.StepType.LAST):
                    final_step = step + 1

            tau = step - nstep + 1
            if tau >= 0:
                min_idx = tau + 1
                max_idx = min(tau + nstep, final_step - 1)
                rho = 1.0
                returns = 0.0

                for i in range(min_idx, max_idx + 1):
                    rho *= policy_probability_fn(
                        policy, experiences[i]
                    ) / collect_policy_probability_fn(collect_policy, experiences[i])
                    returns += (gamma ** (i - tau - 1)) * experiences[i - 1].reward
                if tau + nstep < final_step:
                    returns += (gamma**nstep) * qtable[
                        state_id_fn(experiences[tau + nstep].observation),
                        action_id_fn(experiences[tau + nstep].action),
                    ]

                state_id = state_id_fn(experiences[tau].observation)
                action_id = action_id_fn(experiences[tau].action)
                alpha = lrs(episode=episode, step=steps_counter)
                qtable[state_id, action_id] += (
                    alpha * rho * (returns - qtable[state_id, action_id])
                )
            steps_counter += 1
            if tau == final_step - 1:
                break

        # need to copy qtable because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(qtable)
