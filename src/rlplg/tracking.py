import contextlib
import json
import os.path
import types
from typing import Any, Mapping, Optional, Type, Union

import tensorflow as tf


class EpisodeStats:
    def __init__(self):
        self._episode_count = 0
        self._episode_reward = 0.0
        self._total_reward = 0.0
        self._successful_episodes = 0

    def new_reward(self, reward: float) -> None:
        self._episode_reward += reward
        self._total_reward += reward

    def end_episode(self, success: bool) -> None:
        self._episode_count += 1
        self._successful_episodes += int(success)
        self._episode_reward = 0.0

    @property
    def episodic_reward(self) -> float:
        return self._episode_reward

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def successful_episodes(self) -> int:
        return self._successful_episodes

    def __str__(self) -> str:
        return f"Episode: {self.episode_count}, Episode(r): {self.episodic_reward}, Total(r): {self.total_reward}, Successful Attempts: {self.successful_episodes}"


class ExperimentLogger(contextlib.AbstractContextManager):
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

    def __enter__(self) -> "ExperimentLogger":
        self._writer = tf.io.gfile.GFile(self.log_file, "w")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> bool:
        self._writer.close()
        return super().__exit__(exc_type, exc_value, traceback)

    def log(
        self,
        episode: int,
        steps: int,
        returns: float,
        metadata: Mapping[str, Any] = {},
    ):
        entry = {
            "episode": episode,
            "steps": steps,
            "returns": returns,
            "metadata": metadata,
        }
        self._writer.write(f"{json.dumps(entry)}\n")
