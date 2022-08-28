from typing import Text

import numpy as np
import pytest
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from rlplg import export
from tests import defaults


def test_tf_record_serdes(
    tmpdir: Text,
    time_step_spec: ts.TimeStep,
    action_spec: array_spec.BoundedArraySpec,
):
    _input = trajectory.Trajectory(
        step_type=defaults.batch(
            ts.StepType.FIRST,
            ts.StepType.MID,
            ts.StepType.MID,
            ts.StepType.MID,
        ),
        observation=defaults.batch(0, 1, 2, 3),
        action=defaults.batch(0, 2, 4, 6),
        policy_info=policy_step.PolicyInfo(
            log_probability=defaults.batch(
                np.log(0.3),
                np.log(0.8),
                np.log(0.7),
                np.log(0.2),
            )
        ),
        next_step_type=defaults.batch(
            ts.StepType.MID,
            ts.StepType.MID,
            ts.StepType.MID,
            ts.StepType.LAST,
        ),
        reward=defaults.batch(-1.0, -7.0, 5.0, 7.0),
        discount=defaults.batch(1.0, 1.0, 1.0, 1.0),
    )

    writer = export.TrajectoryTFRecordWriter(
        output_dir=tmpdir, time_step_spec=time_step_spec, action_spec=action_spec
    )
    writer.save(_input)
    writer.sync()

    reader = export.TrajectoryTFRecordReader(data_dir=tmpdir)
    outputs = [value for value in reader.as_dataset()]
    assert len(outputs) == 1
    output = next(iter(outputs))
    np.testing.assert_array_equal(output.step_type, _input.step_type)
    np.testing.assert_array_equal(output.observation, _input.observation)
    np.testing.assert_array_equal(output.action, _input.action)
    np.testing.assert_array_equal(output.next_step_type, _input.next_step_type)
    np.testing.assert_array_equal(output.reward, _input.reward)
    np.testing.assert_array_equal(output.discount, _input.discount)


def test_json_serdes(
    tmpdir: Text,
    time_step_spec: ts.TimeStep,
    action_spec: array_spec.BoundedArraySpec,
):
    inputs = [
        trajectory.Trajectory(
            step_type=defaults.batch(
                ts.StepType.FIRST,
                ts.StepType.MID,
                ts.StepType.MID,
                ts.StepType.MID,
            ),
            observation=defaults.batch(0, 1, 2, 3),
            action=defaults.batch(0, 2, 4, 6),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.3),
                    np.log(0.8),
                    np.log(0.7),
                    np.log(0.2),
                )
            ),
            next_step_type=defaults.batch(
                ts.StepType.MID,
                ts.StepType.MID,
                ts.StepType.MID,
                ts.StepType.LAST,
            ),
            reward=defaults.batch(-1.0, -7.0, 5.0, 7.0),
            discount=defaults.batch(1.0, 1.0, 1.0, 1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(
                ts.StepType.FIRST,
                ts.StepType.MID,
            ),
            observation=defaults.batch(0, 1),
            action=defaults.batch(0, 2),
            policy_info=(),
            next_step_type=defaults.batch(
                ts.StepType.MID,
                ts.StepType.MID,
            ),
            reward=defaults.batch(-1.0, -7.0),
            discount=defaults.batch(1.0, 1.0),
        ),
    ]

    expectations = [
        trajectory.Trajectory(
            step_type=defaults.batch(
                ts.StepType.FIRST,
                ts.StepType.MID,
                ts.StepType.MID,
                ts.StepType.MID,
            ),
            observation=defaults.batch(0, 1, 2, 3),
            action=defaults.batch(0, 2, 4, 6),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(
                    np.log(0.3),
                    np.log(0.8),
                    np.log(0.7),
                    np.log(0.2),
                )
            ),
            next_step_type=defaults.batch(
                ts.StepType.MID,
                ts.StepType.MID,
                ts.StepType.MID,
                ts.StepType.LAST,
            ),
            reward=defaults.batch(-1.0, -7.0, 5.0, 7.0),
            discount=defaults.batch(1.0, 1.0, 1.0, 1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(
                ts.StepType.FIRST,
                ts.StepType.MID,
            ),
            observation=defaults.batch(0, 1),
            action=defaults.batch(0, 2),
            # tf.data.Dataset expects outputs of the same shape
            policy_info=tf.zeros([1, 0], tf.float32),
            next_step_type=defaults.batch(
                ts.StepType.MID,
                ts.StepType.MID,
            ),
            reward=defaults.batch(-1.0, -7.0),
            discount=defaults.batch(1.0, 1.0),
        ),
    ]
    writer = export.TrajectoryJsonWriter(
        output_dir=tmpdir, time_step_spec=time_step_spec, action_spec=action_spec
    )
    for _input in inputs:
        writer.save(_input)
    writer.sync()

    reader = export.TrajectoryJsonReader(data_dir=tmpdir)
    outputs = [value for value in reader.as_dataset()]
    assert len(outputs) == 2
    for expected, output in zip(expectations, outputs):
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)
        np.testing.assert_almost_equal(output.policy_info, expected.policy_info)


@pytest.fixture
def time_step_spec():
    observation_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=100,
        name="observation",
    )
    reward_spec = array_spec.BoundedArraySpec(
        shape=(),
        minimum=-100.0,
        maximum=100.0,
        dtype=np.float32,
        name="action",
    )

    return ts.time_step_spec(observation_spec=observation_spec, reward_spec=reward_spec)


@pytest.fixture
def action_spec():
    return array_spec.BoundedArraySpec(
        shape=(),
        minimum=1,
        maximum=100,
        dtype=np.int32,
        name="action",
    )
