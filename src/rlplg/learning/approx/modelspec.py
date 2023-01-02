"""
This module constaints APIs and support
classes for function approximation in RL.
"""

import abc
from typing import Callable, Sequence, Tuple

from tf_agents.typing.types import Array

from rlplg.learning import utils


class EpochScheduler:
    """
    This class updates epoch based on the current episode,
    based on a given function, `schedule`.
    """

    # TODO: delete

    def __init__(self, schedule: Callable[[int], int], verbose: bool = False):
        """
        Args:
            schedule: A functiont that takes the current episode
            and returns a new epoch.
            verbose: If true, logs updates.
        """
        self._schedule = schedule
        self._verbose = verbose

    def __call__(self, episode: int):
        return self._schedule(episode=episode)


class LearningRateScheduler:
    """
    This class updates the learning rate on every epoch according
    to a given function, `schedule`.
    """

    def __init__(self, schedule: Callable[[int, float], float], verbose: bool = False):
        """
        Args:
            schedule: A functiont that takes the current epoch and learning rate
            and returns a new learning rate.
            verbose: If true, logs updates.
        """
        self._schedule = schedule
        self._verbose = verbose

    def __call__(self, epoch: int, learning_rate: float):
        return self._schedule(epoch, learning_rate)


class ValueFnModel(abc.ABC):
    """
    This class defines an API for value functions
    using approximate solutions.

    Models should be differentiable.
    """

    @abc.abstractmethod
    def predict(self, features: Array) -> float:
        """
        Computes the value for a given context,
        i.e. state or (state, action)
        """

    @abc.abstractmethod
    def gradients(self, features: Array) -> Sequence[Array]:
        """
        Computes the gradients of the output
        with respect to the weights.
        """

    @abc.abstractmethod
    def predict_and_gradients(self, features: Array) -> Tuple[float, Sequence[Array]]:
        """
        Computes prediction and gradients of the
        weights for the prediction.
        """

    @abc.abstractmethod
    def weights(self) -> Sequence[Array]:
        """
        Returns the current weights of the model.
        """

    @abc.abstractmethod
    def assign_weights(self, weights: Sequence[Array]) -> None:
        """
        Assigns new values to the model weights.
        """


class ApproxFn:
    """
    Approximation of state or (state, action) functions.
    Wraps calls to a `ValueFnModel` by applying
    a chain of transformations to the inputs
    (e.g. feature data encoding).
    """

    def __init__(
        self,
        model: ValueFnModel,
        pre_procs: Sequence[Callable[[Array], Array]] = (),
    ):
        """
        Initialises instance.
        """
        self._model = model
        self._pre_procs = pre_procs

    def predict(self, features: Array) -> float:
        """
        Computes the value for a given context,
        i.e. state or (state, action)
        """
        return self._model.predict(utils.chain_map(features, self._pre_procs))

    def gradients(self, features: Array) -> Sequence[Array]:
        """
        Computes the gradients of the output
        with respect to the weights.
        """
        return self._model.gradients(utils.chain_map(features, self._pre_procs))

    def predict_and_gradients(self, features: Array) -> Tuple[float, Sequence[Array]]:
        """
        Computes prediction and gradients of the
        weights for the prediction.
        """
        return self._model.predict_and_gradients(
            utils.chain_map(features, self._pre_procs)
        )

    def weights(self) -> Sequence[Array]:
        """
        Returns the current weights of the model.
        """
        return self._model.weights()

    def assign_weights(self, weights: Sequence[Array]) -> None:
        """
        Assigns new values to the model weights.
        """
        self._model.assign_weights(weights)
