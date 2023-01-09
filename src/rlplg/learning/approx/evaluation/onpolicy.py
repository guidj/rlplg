"""
Policy evaluation methods with approximation.
"""


import copy
from typing import Callable, Generator, Tuple

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import NestedArray

from rlplg import envplay
from rlplg.learning.approx import modelspec
from rlplg.learning.opt import schedules


def gradient_monte_carlo_state_values(
    policy: py_policy.PyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    estimator: modelspec.ApproxFn,
    generate_episodes: Callable[
        [
            py_environment.PyEnvironment,
            py_policy.PyPolicy,
            int,
        ],
        Generator[trajectory.Trajectory, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, NestedArray, float], None, None]:
    """
    Gradient monte-carlo based uses returns to
    approximate the value function of a policy.
    """
    step = 0
    for episode in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episodes(environment, policy, num_episodes=1))
        episode_return = np.sum([experience.reward for experience in experiences])
        # while step isn't last
        for experience in experiences:
            alpha = lrs(episode=episode, step=step)
            state_value = estimator.predict(experience.observation)
            gradients = estimator.gradients(experience.observation)
            weights = estimator.weights()
            new_weights = weights + alpha * (episode_return - state_value) * gradients
            delta = np.sum(np.abs(weights - new_weights))
            estimator.assign_weights(new_weights)
            # update returns for the next state
            episode_return -= experience.reward
            step += 1
            # TODO: need a way to indicate state is terminal in its repr .e.g just zeros
        # need to copy values because it's a mutable numpy array
        yield len(experiences), copy.deepcopy(estimator.weights()), delta
