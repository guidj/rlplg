import numpy as np
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from rlplg import envplay
from tests import defaults


def test_unroll_trajectory():
    input = trajectory.Trajectory(
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

    expectations = [
        trajectory.Trajectory(
            step_type=defaults.batch(ts.StepType.FIRST),
            observation=defaults.batch(0),
            action=defaults.batch(0),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.3))
            ),
            next_step_type=defaults.batch(ts.StepType.MID),
            reward=defaults.batch(-1.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(ts.StepType.MID),
            observation=defaults.batch(1),
            action=defaults.batch(2),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.8))
            ),
            next_step_type=defaults.batch(ts.StepType.MID),
            reward=defaults.batch(-7.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(ts.StepType.MID),
            observation=defaults.batch(2),
            action=defaults.batch(4),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.7))
            ),
            next_step_type=defaults.batch(ts.StepType.MID),
            reward=defaults.batch(5.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(ts.StepType.MID),
            observation=defaults.batch(3),
            action=defaults.batch(6),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.2))
            ),
            next_step_type=defaults.batch(ts.StepType.LAST),
            reward=defaults.batch(7.0),
            discount=defaults.batch(1.0),
        ),
    ]

    outputs = list(envplay.unroll_trajectory(input))

    assert len(outputs) == 4
    for output, expected in zip(outputs, expectations):
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)


def test_slice_trajectory():
    input = trajectory.Trajectory(
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

    expectations = [
        trajectory.Trajectory(
            step_type=defaults.batch(ts.StepType.FIRST),
            observation=defaults.batch(0),
            action=defaults.batch(0),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.3))
            ),
            next_step_type=defaults.batch(ts.StepType.MID),
            reward=defaults.batch(-1.0),
            discount=defaults.batch(1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(ts.StepType.MID, ts.StepType.MID),
            observation=defaults.batch(1, 2),
            action=defaults.batch(2, 4),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.8), np.log(0.7))
            ),
            next_step_type=defaults.batch(ts.StepType.MID, ts.StepType.MID),
            reward=defaults.batch(-7.0, 5.0),
            discount=defaults.batch(1.0, 1.0),
        ),
        trajectory.Trajectory(
            step_type=defaults.batch(ts.StepType.MID),
            observation=defaults.batch(3),
            action=defaults.batch(6),
            policy_info=policy_step.PolicyInfo(
                log_probability=defaults.batch(np.log(0.2))
            ),
            next_step_type=defaults.batch(ts.StepType.LAST),
            reward=defaults.batch(7.0),
            discount=defaults.batch(1.0),
        ),
    ]

    outputs = [
        envplay.slice_trajectory(input, start=0, size=1),
        envplay.slice_trajectory(input, start=1, size=2),
        envplay.slice_trajectory(input, start=3, size=1),
    ]

    for output, expected in zip(outputs, expectations):
        np.testing.assert_array_equal(output.step_type, expected.step_type)
        np.testing.assert_array_equal(output.observation, expected.observation)
        np.testing.assert_array_equal(output.action, expected.action)
        np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
        np.testing.assert_array_equal(output.reward, expected.reward)
        np.testing.assert_array_equal(output.discount, expected.discount)


def test_slice_trajectory_with_out_of_range_index():
    input = trajectory.Trajectory(
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

    expected = trajectory.Trajectory(
        step_type=np.empty((), dtype=np.int64),
        observation=np.empty((), dtype=np.int64),
        action=np.empty((), dtype=np.int64),
        policy_info=policy_step.PolicyInfo(
            log_probability=np.empty((), dtype=np.float32)
        ),
        next_step_type=np.empty((), dtype=np.int64),
        reward=np.empty((), dtype=np.float32),
        discount=np.empty((), dtype=np.float32),
    )

    output = envplay.slice_trajectory(input, start=4, size=1)
    np.testing.assert_array_equal(output.step_type, expected.step_type)
    np.testing.assert_array_equal(output.observation, expected.observation)
    np.testing.assert_array_equal(output.action, expected.action)
    np.testing.assert_array_equal(output.next_step_type, expected.next_step_type)
    np.testing.assert_array_equal(output.reward, expected.reward)
    np.testing.assert_array_equal(output.discount, expected.discount)
