"""
Policy evaluation methods with approximation.
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
from rlplg.learning.approx import modelspec


def gradient_monte_carlo_state_values(
    policy: py_policy.PyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    alpha: float,
    # epoch_scheduler: modelspec.EpochScheduler,
    estimator: modelspec.ApproxFn,
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
    Gradient monte-carlo based uses returns to
    approximate the value function of a policy.
    """
    for _ in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episodes(environment, policy, num_episodes=1))
        # reverse list and ammortize state visits
        episode_return = 0
        # while step isn't last
        for experience in experiences:
            episode_return += experience.reward
            # state = np.array([1, experience.observation], np.float32)
            # gradient = state
            # TODO: Keep alpha here; have gradient provider; and weight update function
            state_value = estimator.predict(experience.observation)
            gradients = estimator.gradients(experience.observation)
            weights = estimator.weights()
            new_weights = weights + alpha * (episode_return - state_value) * gradients
            delta = np.sum(np.abs(weights - new_weights))
            estimator.assign_weights(new_weights)
            # TODO: need a way to indicate state is terminal in its repr .e.g just zeros
        # need to copy values because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(estimator.weights()), delta
