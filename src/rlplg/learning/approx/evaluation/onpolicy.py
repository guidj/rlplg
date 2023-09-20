"""
Policy evaluation methods with approximation.
"""


import copy
from typing import Callable, Generator, Tuple

import gymnasium as gym
import numpy as np

from rlplg import core, envplay
from rlplg.learning import utils
from rlplg.learning.approx import modelspec
from rlplg.learning.opt import schedules


def gradient_monte_carlo_state_values(
    policy: core.PyPolicy,
    environment: gym.Env,
    num_episodes: int,
    lrs: schedules.LearningRateSchedule,
    estimator: modelspec.ApproxFn,
    generate_episodes: Callable[
        [
            gym.Env,
            core.PyPolicy,
            int,
        ],
        Generator[core.TrajectoryStep, None, None],
    ] = envplay.generate_episodes,
) -> Generator[Tuple[int, modelspec.ApproxFn], None, None]:
    """
    Gradient monte-carlo based uses returns to
    approximate the value function of a policy.
    """
    steps_counter = 0
    for episode in range(num_episodes):
        # This can be memory intensive, for long episodes and large state/action representations.
        experiences = list(generate_episodes(environment, policy, 1))
        episode_return = np.sum([experience.reward for experience in experiences])
        # while step isn't last
        for experience in experiences:
            alpha = lrs(episode=episode, step=steps_counter)
            state_value = estimator.predict(experience.observation)
            gradients = estimator.gradients(experience.observation)
            weights = estimator.weights()
            new_weights = weights + alpha * (episode_return - state_value) * gradients

            assert not utils.nan_or_inf(state_value), "Value estimate is NaN or Inf"
            assert not utils.nan_or_inf(gradients), "Gradients are NaN or Inf"
            estimator.assign_weights(new_weights)
            # update returns for the next state
            episode_return -= experience.reward
            steps_counter += 1
        # need to copy values because they can be mutable np.ndarrays or tf.tensors
        # we use shallow copy because tf doesn't play nicely with deepcopy
        yield len(experiences), copy.copy(estimator)
