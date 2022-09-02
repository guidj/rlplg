"""
Module for classes and functions to log stats and episodic information,
to std output or files.
"""


import contextlib
import json
import os.path
import types
from typing import Any, Mapping, Optional, Type, Union

import tensorflow as tf


class EpisodeStats:
    """
    Tracks episode states - success, attempts, rewards.
    """

    def __init__(self):
        self._episode_count = 0
        self._episode_reward = 0.0
        self._total_reward = 0.0
        self._successful_episodes = 0

    def new_reward(self, reward: float) -> None:
        """
        Increments the reward.
        """
        self._episode_reward += reward
        self._total_reward += reward

    def end_episode(self, success: bool) -> None:
        """
        Resets the reward accumulator, and increments the episode.
        If success is True, it increments the success count.
        """
        self._episode_count += 1
        self._successful_episodes += int(success)
        self._episode_reward = 0.0

    @property
    def episodic_reward(self) -> float:
        """
        Returns cumulative reward for the current episode.
        """
        return self._episode_reward

    @property
    def total_reward(self) -> float:
        """
        Returns the total reward across episodes.
        """
        return self._total_reward

    @property
    def episode_count(self) -> int:
        """
        Returns the episode count.
        """
        return self._episode_count

    @property
    def successful_episodes(self) -> int:
        """
        Returns the successful episode count.
        """
        return self._successful_episodes

    def __str__(self) -> str:
        """
        Class reprensted as a logging message.
        """
        return f"Episode: {self.episode_count}, Episode(r): {self.episodic_reward}, Total(r): {self.total_reward}, Successful Attempts: {self.successful_episodes}"


class ExperimentLogger(contextlib.AbstractContextManager):
    """
    Logs info for an experiment for given episodes.
    """

    LOG_FILE_NAME = "experiment-logs.json"
    PARAM_FILE_NAME = "experiment-params.json"

    def __init__(
        self, log_dir: str, name: str, params: Mapping[str, Union[int, float, str]]
    ):
        self.log_file = os.path.join(log_dir, self.LOG_FILE_NAME)
        self.param_file = os.path.join(log_dir, self.PARAM_FILE_NAME)
        if not tf.io.gfile.exists(log_dir):
            tf.io.gfile.makedirs(log_dir)

        with tf.io.gfile.GFile(self.param_file, "w") as writer:
            writer.write(json.dumps(dict(params, name=name)))

        self._writer: Optional[Any] = None

    def open(self) -> None:
        """
        Opens the log file for writing.
        """
        self._writer = tf.io.gfile.GFile(self.log_file, "w")

    def close(self) -> None:
        """
        Closes the log file.
        """
        self._writer.close()

    def __enter__(self) -> "ExperimentLogger":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> bool:
        self.close()
        return super().__exit__(exc_type, exc_value, traceback)

    def log(
        self,
        episode: int,
        steps: int,
        returns: float,
        metadata: Mapping[str, Any] = None,
    ):
        """
        Logs an experiment entry for an episode.
        """
        entry = {
            "episode": episode,
            "steps": steps,
            "returns": returns,
            "metadata": metadata if metadata else {},
        }
        try:
            self._writer.write(f"{json.dumps(entry)}\n")
        except AttributeError as err:
            raise RuntimeError("ExperimentLogger is unitialized") from err
