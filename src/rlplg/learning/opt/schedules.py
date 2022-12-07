from typing import Callable


class EpochSchedule:
    """
    This class updates epoch based on the current episode,
    based on a given function, `schedule`.
    """

    def __init__(self, schedule: Callable[[int], int], verbose: bool = False):
        """
        Args:
            schedule: A functiont that takes the current step (e.g. time step, episode, round)
            and returns a new epoch.
            verbose: If true, logs updates.
        """
        self._schedule = schedule
        self._verbose = verbose

    def __call__(self, step: int):
        return self._schedule(step)


class LearningRateSchedule:
    """
    This class updates the learning rate on every epoch according
    to a given function, `schedule`.
    """

    def __init__(self, schedule: Callable[[int, float], float], verbose: bool = False):
        """
        Args:
            schedule: A functiont that takes the current step
            (e.g. time step, episode, round) and learning rate
            and returns a new learning rate.
            verbose: If true, logs updates.
        """
        self._schedule = schedule
        self._verbose = verbose

    def __call__(self, step: int, learning_rate: float):
        return self._schedule(step, learning_rate)
