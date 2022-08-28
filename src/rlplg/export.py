import abc
import json
import logging
import os.path
import queue
import time
from typing import Any, Generator, Mapping, Optional, Sequence, Text

import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

TYPES = (
    np.dtype("float16"),
    np.dtype("float32"),
    np.dtype("float64"),
    np.dtype("complex64"),
    np.dtype("complex128"),
    np.dtype("int8"),
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("bool8"),
    np.dtype("string_"),
)

TYPE_MAPPING = {dtype: str(dtype) for dtype in TYPES}
INVERSE_TYPE_MAPPING = {name: value for value, name in TYPE_MAPPING.items()}

SCHEMA_FILE = "_traj_schema.json"


class TrajectoryWriter(abc.ABC):
    """
    Saves trajectory data to disk.
    Note: does not support policy_info, since it's a custom object of any type.
    """

    def __init__(
        self,
        output_dir: Text,
        time_step_spec: ts.TimeStep,
        action_spec: tensor_spec.BoundedTensorSpec,
        partition_size: int = 2**16,
    ):
        """
        Args:
            output_dir (str): directory to save trajectories
            time_step_spec (ts.TimeStep): used to infer trajectory types
            action_spec (tensor_spec.BoundedTensorSpec): used to infer action type
            partition_size (int): the number of trajectory records per file partition

        Returns:
            An instance of TrajectoryExporter
        """
        self._queue = queue.Queue(maxsize=partition_size)
        self._partition_count = 0
        self._save_dir = os.path.join(output_dir, str(int(time.time())))
        self._types = infer_trajectory_field_types(time_step_spec, action_spec)

        if not tf.io.gfile.exists(self._save_dir):
            tf.io.gfile.makedirs(self._save_dir)
        save_schema(self._types, path=os.path.join(self._save_dir, SCHEMA_FILE))

    def save(self, traj: trajectory.Trajectory) -> None:
        if self._queue.full():
            self._write_partition()
        # TODO: deep copy? in case ref changes; check cost
        self._queue.put(traj)

    def sync(self) -> None:
        if not self._queue.empty():
            self._write_partition()

    @abc.abstractmethod
    def _write_partition(self) -> None:
        """
        When implemented, it should update `self._partition_count` for each
        partition.
        """
        raise NotImplementedError()


class TrajectoryTFRecordWriter(TrajectoryWriter):
    """
    Saves trajectory data to disk in TFRecords format.
    Note: does not support policy_info, since it's a custom object of any type.
    """

    def _write_partition(self) -> None:
        self._partition_count += 1
        partition_file_name = os.path.join(
            self._save_dir, f"partition-{self._partition_count:07d}.tfrecords"
        )

        with tf.io.TFRecordWriter(partition_file_name) as writer:
            logging.info(f"Writing {partition_file_name}")
            while not self._queue.empty():
                traj = self._queue.get()
                example_proto = trajectory_example_proto(traj)
                writer.write(example_proto.SerializeToString())


class TrajectoryJsonWriter(TrajectoryWriter):
    """
    Saves trajectory as json data to disk.
    Note: does not support policy_info, since it's a custom object of any type.
    """

    def _write_partition(self) -> None:
        self._partition_count += 1
        partition_file_name = os.path.join(
            self._save_dir, f"partition-{self._partition_count:07d}.json"
        )

        with tf.io.gfile.GFile(partition_file_name, "w") as writer:
            logging.info("Writing %s", partition_file_name)
            while not self._queue.empty():
                traj = self._queue.get()
                traj_json = json.dumps(self.serialize(traj))
                writer.write(traj_json)
                writer.write("\n")

    @classmethod
    def serialize(
        cls,
        traj: trajectory.Trajectory,
    ) -> Mapping[Text, Any]:
        if traj.policy_info not in ((), None):
            policy_info = {
                "log_probability": traj.policy_info.log_probability.tolist(),
            }
        else:
            policy_info = traj.policy_info
        return {
            "step_type": traj.step_type.tolist(),
            "observation": traj.observation.tolist(),
            "action": traj.action.tolist(),
            "policy_info": policy_info,
            "next_step_type": traj.next_step_type.tolist(),
            "reward": traj.reward.tolist(),
            "discount": traj.discount.tolist(),
        }


class TrajectoryReader(abc.ABC):
    """
    Saves trajectory data to disk.
    Note: does not support policy_info, since it's a custom object of any type.
    """

    def as_dataset(self) -> tf.data.Dataset:
        return self._as_dataset()

    def as_iterator(self) -> Generator[trajectory.Trajectory, None, None]:
        return self._as_iterator()

    @abc.abstractmethod
    def _as_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError()

    @abc.abstractmethod
    def _as_iterator(self) -> Generator[trajectory.Trajectory, None, None]:
        raise NotImplementedError()


class TrajectoryTFRecordReader(TrajectoryReader):
    """
    Saves trajectory data to disk.
    Note: does not support policy_info, since it's a custom object of any type.
    """

    def __init__(self, data_dir: Text):
        """
        Args:
            data_dir (str): directory to save trajectories
        """
        dirs = [os.path.join(data_dir, path) for path in tf.io.gfile.listdir(data_dir)]
        self._dirs = sorted([_dir for _dir in dirs if tf.io.gfile.isdir(str(_dir))])
        if not self._dirs:
            raise ValueError(
                f"Directory {data_dir} has no subdirs with trajectory data"
            )
        self._files = resolve_files(self._dirs, "partition-**.tfrecords")
        # read schema from one of the subdirs
        first_dir = next(iter(self._dirs))
        self._types = load_schema(path=os.path.join(first_dir, SCHEMA_FILE))

    def _as_dataset(self) -> tf.data.Dataset:
        def parse_example(serialized: tf.Tensor):
            return tf.io.parse_single_example(
                serialized, features=parsing_feature_spec()
            )

        def deserialize_traj(serialized: Mapping[str, bytes]) -> trajectory.Trajectory:
            return serialized_tensors_to_trajectory(serialized, self._types)

        dataset = tf.data.TFRecordDataset(
            filenames=tf.constant(
                [str(file_path) for file_path in self._files], dtype=tf.string
            ),
            buffer_size=1000000,
            # Reader files in order
            num_parallel_reads=None,
        )
        return dataset.map(parse_example).map(deserialize_traj)

    def _as_iterator(self) -> Generator[trajectory.Trajectory, None, None]:
        return self._as_dataset().as_numpy_iterator()


class TrajectoryJsonReader(TrajectoryReader):
    """
    Saves trajectory data to disk.
    Note: does not support policy_info, since it's a custom object of any type.
    """

    def __init__(self, data_dir: Text):
        """
        Args:
            data_dir (str): directory to save trajectories
        """
        dirs = [os.path.join(data_dir, path) for path in tf.io.gfile.listdir(data_dir)]
        self._dirs = sorted([_dir for _dir in dirs if tf.io.gfile.isdir(str(_dir))])
        if not self._dirs:
            raise ValueError(
                f"Directory {data_dir} has no subdirs with trajectory data"
            )
        self._files = resolve_files(self._dirs, "partition-**.json")
        # read schema from one of the subdirs
        first_dir = next(iter(self._dirs))
        self._types = load_schema(path=os.path.join(first_dir, SCHEMA_FILE))
        self._generator_types = dict(self._types, policy_info=(np.float32,))

    def _as_dataset(self) -> tf.data.Dataset:
        def parse_trajectory(record: Mapping[Text, Any]):
            return trajectory.Trajectory(**record)

        return tf.data.Dataset.from_generator(
            lambda: self._create_generator(),
            output_types=self._generator_types,
        ).map(parse_trajectory)

    def _as_iterator(self) -> Generator[trajectory.Trajectory, None, None]:
        return self._as_dataset().as_numpy_iterator()

    def _create_generator(self) -> Generator[Mapping[Text, Any], None, None]:
        for file_path in self._files:
            with tf.io.gfile.GFile(file_path, "r") as reader:
                for row in reader:
                    payload = json.loads(row)
                    if (
                        "policy_info" in payload
                        and "log_probability" in payload["policy_info"]
                    ):
                        policy_info = trajectory.policy_step.PolicyInfo(
                            log_probability=tf.convert_to_tensor(
                                payload["policy_info"]["log_probability"], np.float32
                            )
                        )
                    else:
                        # Create an instance to make the output consistent for tf.data.Dataset
                        policy_info = trajectory.policy_step.PolicyInfo(
                            log_probability=()
                        )

                    record = {
                        key: tf.convert_to_tensor(payload[key], dtype)
                        for key, dtype in self._types.items()
                    }
                    record["policy_info"] = policy_info
                    yield record


def infer_trajectory_field_types(
    time_step_spec: ts.TimeStep, action_spec: tensor_spec.BoundedTensorSpec
) -> Mapping[Text, np.dtype]:
    return {
        "action": action_spec.dtype,
        "reward": time_step_spec.reward.dtype,
        "discount": time_step_spec.discount.dtype,
        "observation": time_step_spec.observation.dtype,
        "step_type": np.dtype("int32"),
        "next_step_type": np.dtype("int32"),
    }


def save_schema(schema: Mapping[Text, np.dtype], path: Text) -> None:
    payload = {key: TYPE_MAPPING[dtype] for key, dtype in schema.items()}
    with tf.io.gfile.GFile(path, "w") as writer:
        json.dump(payload, fp=writer)


def trajectory_example_proto(traj: trajectory.Trajectory) -> tf.train.Example:
    serialized_tensors = {
        "action": tf.io.serialize_tensor(traj.action),
        "reward": tf.io.serialize_tensor(traj.reward),
        "discount": tf.io.serialize_tensor(traj.discount),
        "observation": tf.io.serialize_tensor(traj.observation),
        "step_type": tf.io.serialize_tensor(traj.step_type),
        "next_step_type": tf.io.serialize_tensor(traj.next_step_type),
    }
    feature = {
        field: bytes_feature(tensor.numpy())
        for field, tensor in serialized_tensors.items()
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def resolve_files(dirs: Sequence[Text], pattern: Text) -> Sequence[Text]:
    files = []
    for _dir in dirs:
        file_paths = tf.io.gfile.glob(os.path.join(_dir, pattern))
        files.extend(file_paths)
    return files


def load_schema(path: Text) -> Optional[Mapping[Text, np.dtype]]:
    try:
        with tf.io.gfile.GFile(path, "r") as reader:
            serialized_schema = json.load(reader)
        return {
            field: INVERSE_TYPE_MAPPING[type_name]
            for field, type_name in serialized_schema.items()
        }
    except tf.errors.NotFoundError as err:
        raise ValueError(f"Path '{path}' does not exist") from err
    except KeyError as err:
        raise ValueError(
            f"Schema file '{path}' has an invalid type value. Should be one of: {sorted(list(INVERSE_TYPE_MAPPING.keys()))}",
        ) from err


def parsing_feature_spec() -> Mapping[Text, tf.io.FixedLenFeature]:
    return {
        "action": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "reward": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "discount": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "observation": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "step_type": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "next_step_type": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    }


def serialized_tensors_to_trajectory(
    serialized_tensors: Mapping[Text, bytes], types: Mapping[Text, np.dtype]
) -> trajectory.Trajectory:
    traj = {
        key: tf.io.parse_tensor(serialized, out_type=types[key])
        for key, serialized in serialized_tensors.items()
    }
    # Set default value for policy_info, because it's not supported
    return trajectory.Trajectory(**traj, policy_info=())
