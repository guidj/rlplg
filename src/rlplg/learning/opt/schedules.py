from typing import Callable


class LearningRateSchedule:
    """
    This class updates the learning rate on every epoch according
    to a given function, `schedule`.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        schedule: Callable[[float, int], float],
        verbose: bool = False,
    ):
        """
        Args:
            schedule: A functiont that takes the current step
            (e.g. time step, episode, round) and learning rate
            and returns a new learning rate.
            verbose: If true, logs updates.
        """
        self._initial_learning_rate = initial_learning_rate
        self._schedule = schedule
        self._verbose = verbose

    def __call__(self, step: int):
        return self._schedule(self._initial_learning_rate, step)
