from typing import Any

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from gymnasium import spaces

from rlplg.core import TimeStep
from rlplg.environments import randomwalk


@hypothesis.given(steps=st.integers(min_value=3, max_value=100))
def test_state_randomwalk_init(steps: int):
    environment = randomwalk.StateRandomWalk(steps=steps)
    assert environment.steps == steps
    assert environment.left_end_reward == 0
    assert environment.right_end_reward == 1
    assert environment.step_reward == 0
    assert environment.action_space == spaces.Box(low=0, high=1, dtype=np.int64)
    assert environment.observation_space == spaces.Dict(
        {
            "position": spaces.Box(low=0, high=steps - 1, dtype=np.int64),
            "steps": spaces.Box(low=steps, high=steps, dtype=np.int64),
            "right_end_reward": spaces.Box(low=1, high=1, dtype=np.float32),
            "left_end_reward": spaces.Box(low=0, high=0, dtype=np.float32),
            "step_reward": spaces.Box(low=0, high=0, dtype=np.float32),
        }
    )


@hypothesis.given(steps=st.integers(max_value=2))
def test_state_randomwalk_with_invalid_steps(steps: int):
    with pytest.raises(AssertionError):
        randomwalk.StateRandomWalk(steps=steps)


def test_state_randomwalk_end_left_sequence():
    environment = randomwalk.StateRandomWalk(5)
    obs, info = environment.reset()
    assert_observation(
        obs,
        {
            "position": np.array(2, dtype=np.int64),
            "steps": np.array(5, dtype=np.int64),
            "right_end_reward": np.array(1.0, dtype=np.float32),
            "left_end_reward": np.array(0.0, dtype=np.float32),
            "step_reward": np.array(0.0, dtype=np.float32),
        },
    )
    assert info == {}

    # go left
    assert_time_step(
        environment.step(0),
        (
            {
                "position": np.array(1, dtype=np.int64),
                "steps": np.array(5, dtype=np.int64),
                "right_end_reward": np.array(1.0, dtype=np.float32),
                "left_end_reward": np.array(0.0, dtype=np.float32),
                "step_reward": np.array(0.0, dtype=np.float32),
            },
            0.0,
            False,
            False,
            {},
        ),
    )
    # go left (terminal state)
    assert_time_step(
        environment.step(0),
        (
            {
                "position": np.array(0, dtype=np.int64),
                "steps": np.array(5, dtype=np.int64),
                "right_end_reward": np.array(1.0, dtype=np.float32),
                "left_end_reward": np.array(0.0, dtype=np.float32),
                "step_reward": np.array(0.0, dtype=np.float32),
            },
            0.0,
            True,
            False,
            {},
        ),
    )
    # go right - no change (terminal state)
    assert_time_step(
        environment.step(1),
        (
            {
                "position": np.array(0, dtype=np.int64),
                "steps": np.array(5, dtype=np.int64),
                "right_end_reward": np.array(1.0, dtype=np.float32),
                "left_end_reward": np.array(0.0, dtype=np.float32),
                "step_reward": np.array(0.0, dtype=np.float32),
            },
            0.0,
            True,
            False,
            {},
        ),
    )


def test_state_randomwalk_end_right_sequence():
    environment = randomwalk.StateRandomWalk(5)
    obs, info = environment.reset()

    assert_observation(
        obs,
        {
            "position": np.array(2, dtype=np.int64),
            "steps": np.array(5, dtype=np.int64),
            "right_end_reward": np.array(1.0, dtype=np.float32),
            "left_end_reward": np.array(0.0, dtype=np.float32),
            "step_reward": np.array(0.0, dtype=np.float32),
        },
    )
    assert info == {}
    # go left
    assert_time_step(
        environment.step(0),
        (
            {
                "position": np.array(1, dtype=np.int64),
                "steps": np.array(5, dtype=np.int64),
                "right_end_reward": np.array(1.0, dtype=np.float32),
                "left_end_reward": np.array(0.0, dtype=np.float32),
                "step_reward": np.array(0.0, dtype=np.float32),
            },
            0.0,
            False,
            False,
            {},
        ),
    )
    # go right
    assert_time_step(
        environment.step(1),
        (
            {
                "position": np.array(2, dtype=np.int64),
                "steps": np.array(5, dtype=np.int64),
                "right_end_reward": np.array(1.0, dtype=np.float32),
                "left_end_reward": np.array(0.0, dtype=np.float32),
                "step_reward": np.array(0.0, dtype=np.float32),
            },
            0.0,
            False,
            False,
            {},
        ),
    )
    # go right
    assert_time_step(
        environment.step(1),
        (
            {
                "position": np.array(3, dtype=np.int64),
                "steps": np.array(5, dtype=np.int64),
                "right_end_reward": np.array(1.0, dtype=np.float32),
                "left_end_reward": np.array(0.0, dtype=np.float32),
                "step_reward": np.array(0.0, dtype=np.float32),
            },
            0.0,
            False,
            False,
            {},
        ),
    )
    # go right - terminal state
    assert_time_step(
        environment.step(1),
        (
            {
                "position": np.array(4, dtype=np.int64),
                "steps": np.array(5, dtype=np.int64),
                "right_end_reward": np.array(1.0, dtype=np.float32),
                "left_end_reward": np.array(0.0, dtype=np.float32),
                "step_reward": np.array(0.0, dtype=np.float32),
            },
            1.0,
            True,
            False,
            {},
        ),
    )


def test_state_randomwalk_render():
    environment = randomwalk.StateRandomWalk(steps=3, render_mode="rgb_array")
    environment.reset()
    # starting point
    np.testing.assert_array_equal(environment.render(), np.array([0, 1, 0]))
    # one move left
    environment.step(0)
    np.testing.assert_array_equal(environment.render(), np.array([1, 0, 0]))


@hypothesis.given(
    steps=st.integers(min_value=3, max_value=100),
)
def test_state_randomwalk_render_with_invalid_modes(steps: int):
    modes = ("human",)
    for mode in modes:
        environment = randomwalk.StateRandomWalk(steps=steps, render_mode=mode)
        environment.reset()
        with pytest.raises(NotImplementedError):
            environment.render()


@hypothesis.given(
    state=st.integers(min_value=0, max_value=100),
    action=st.integers(min_value=0, max_value=100),
)
def test_state_randomwalk_discretizer(state: int, action: int):
    discretizer = randomwalk.StateRandomWalkMdpDiscretizer()
    assert (
        discretizer.state(
            {
                "position": np.array(state, dtype=np.int64),
            }
        )
        == state
    )
    assert discretizer.action(action) == action


@hypothesis.given(steps=st.integers(min_value=3, max_value=100))
def test_is_finished(steps: int):
    assert randomwalk.is_finished(
        {
            "position": np.array(0, dtype=np.int64),
            "steps": np.array(steps, dtype=np.int64),
        }
    )
    assert randomwalk.is_finished(
        {
            "position": np.array(steps - 1, dtype=np.int64),
            "steps": np.array(steps, dtype=np.int64),
        }
    )
    for step in range(0 + 1, steps - 1):
        assert not randomwalk.is_finished(
            {
                "position": np.array(step, dtype=np.int64),
                "steps": np.array(steps, dtype=np.int64),
            }
        )


@hypothesis.given(steps=st.integers(min_value=3, max_value=100))
def test_create_env_spec(steps: int):
    env_spec = randomwalk.create_env_spec(steps=steps)

    assert env_spec.name == "StateRandomWalk"
    assert isinstance(env_spec.level, str)
    assert len(env_spec.level) > 0
    assert env_spec.env_desc.num_states == steps
    assert env_spec.env_desc.num_actions == 2
    assert isinstance(env_spec.environment, randomwalk.StateRandomWalk)
    assert isinstance(env_spec.discretizer, randomwalk.StateRandomWalkMdpDiscretizer)


@hypothesis.given(pos=st.integers(min_value=3, max_value=100))
def test_get_state_id(pos: int):
    # other fields are unucessary
    assert (
        randomwalk.get_state_id(
            {
                "position": np.array(pos, dtype=np.int64),
            }
        )
        == pos
    )


def test_state_representation():
    np.testing.assert_array_equal(
        randomwalk.state_representation({"position": 1, "steps": 3}),
        np.array([0, 1, 0]),
    )
    np.testing.assert_array_equal(
        randomwalk.state_representation({"position": 0, "steps": 3}),
        np.array([1, 0, 0]),
    )


def assert_time_step(output: TimeStep, expected: TimeStep) -> None:
    assert_observation(output[0], expected[0])
    assert output[1] == expected[1]
    assert output[2] is expected[2]
    assert output[3] is expected[3]
    assert output[4] == expected[4]


def assert_observation(output: Any, expected: Any) -> None:
    assert len(output) == 5
    for key, value in expected.items():  # type: ignore
        assert key in output
        np.testing.assert_allclose(output[key], value)
