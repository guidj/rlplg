import copy
import logging
from typing import Callable, Sequence, Tuple

import numpy as np

from rlplg import core
from rlplg.core import NestedArray
from rlplg.learning.tabular import policies


def control(
    environment: core.PyEnvironment,
    num_episodes: int,
    state_id_fn: Callable[[NestedArray], int],
    initial_qtable: np.ndarray,
    epsilon: float,
    gamma: float,
    alpha: float,
    log_step: int = 100,
) -> Tuple[core.PyPolicy, np.ndarray]:
    """
    Implements Q-learning, using epsilon-greedy as a collection (behavior) policy.
    """
    qtable = copy.deepcopy(initial_qtable)
    policy, collect_policy = _target_and_collect_policies(
        environment=environment, state_id_fn=state_id_fn, qtable=qtable, epsilon=epsilon
    )
    episode = 0
    while episode < num_episodes:
        environment.reset()
        step = 0
        transitions = []
        while True:
            time_step = environment.current_time_step()
            policy_step = collect_policy.action(time_step)
            next_time_step = environment.step(policy_step.action)
            traj = core.Trajectory.from_transition(
                time_step, policy_step, next_time_step
            )
            transitions.append(traj)
            step += 1

            if len(transitions) == 2:
                _qlearing_step(
                    qtable,
                    state_id_fn,
                    gamma=gamma,
                    alpha=alpha,
                    experiences=transitions,
                )

                # update policies
                policy, collect_policy = _target_and_collect_policies(
                    environment=environment,
                    state_id_fn=state_id_fn,
                    qtable=qtable,
                    epsilon=epsilon,
                )

                # remove earliest step
                transitions.pop(0)

            if time_step.step_type == core.StepType.LAST:
                break

        episode += 1
        if episode % log_step == 0:
            logging.info(
                "Trained with %d steps, episode %d/%d",
                step,
                episode,
                num_episodes,
            )
    return policy, qtable


def _qlearing_step(
    qtable: np.ndarray,
    state_id_fn: Callable[[NestedArray], int],
    gamma: float,
    alpha: float,
    experiences: Sequence[core.Trajectory],
) -> None:
    steps = len(experiences)

    if steps < 2:
        logging.warning("Q-learning requires at least two steps per update - skipping")
        return
    for step in range(steps - 1):
        state_id = state_id_fn(experiences[step].observation)
        next_state_id = state_id_fn(experiences[step + 1].observation)

        state_action_value = qtable[state_id, experiences[step].action]  # type: ignore
        next_state_actions_values = qtable[next_state_id]
        next_best_action = np.random.choice(
            np.flatnonzero(next_state_actions_values == next_state_actions_values.max())
        )

        delta = (
            experiences[step].reward + gamma * qtable[next_state_id, next_best_action]
        ) - state_action_value
        state_action_value = state_action_value + alpha * delta
        qtable[state_id, experiences[step].action] = state_action_value


def _target_and_collect_policies(
    environment: core.PyEnvironment,
    state_id_fn: Callable[[np.ndarray], int],
    qtable: np.ndarray,
    epsilon: float,
) -> Tuple[core.PyPolicy, core.PyPolicy]:
    _, num_actions = qtable.shape
    policy = policies.PyQGreedyPolicy(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        state_id_fn=state_id_fn,
        action_values=qtable,
    )
    collect_policy = policies.PyEpsilonGreedyPolicy(
        policy=policy,
        num_actions=num_actions,
        epsilon=epsilon,
    )
    return policy, collect_policy
