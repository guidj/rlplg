import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from gymnasium import spaces

from rlplg.core import InitState, TimeStep
from rlplg.environments import abcseq


@hypothesis.given(length=st.integers(min_value=1, max_value=100))
def test_abcseq_init(length: int):
    environment = abcseq.ABCSeq(length)
    assert environment.length == length
    assert environment.action_space == spaces.Box(
        low=0, high=length - 1, dtype=np.int64
    )
    assert environment.observation_space == spaces.Box(
        low=0, high=1, shape=(length + 1,), dtype=np.int64
    )


def test_abcseq_simple_sequence():
    length = 4
    environment = abcseq.ABCSeq(length)
    assert_init_state(
        environment.reset(),
        (np.array([1, 0, 0, 0, 0]), {}),
    )
    # final step, prematurely
    assert_time_step(
        environment.step(3),
        (np.array([1, 0, 0, 0, 0]), -4.0, False, False, {}),
    )
    # first letter
    assert_time_step(
        environment.step(0),
        (np.array([1, 1, 0, 0, 0]), -1.0, False, False, {}),
    )
    # second letter
    assert_time_step(
        environment.step(1),
        (np.array([1, 1, 1, 0, 0]), -1.0, False, False, {}),
    )
    # skip ahead
    assert_time_step(
        environment.step(3),
        (np.array([1, 1, 1, 0, 0]), -2.0, False, False, {}),
    )
    # going backwards
    assert_time_step(
        environment.step(0),
        (np.array([1, 1, 1, 0, 0]), -4.0, False, False, {}),
    )
    # continue, third letter
    assert_time_step(
        environment.step(2),
        (np.array([1, 1, 1, 1, 0]), -1.0, False, False, {}),
    )
    # complete
    assert_time_step(
        environment.step(3),
        (np.array([1, 1, 1, 1, 1]), -1.0, True, False, {}),
    )
    # move in the terminal state
    assert_time_step(
        environment.step(0),
        (np.array([1, 1, 1, 1, 1]), 0.0, True, False, {}),
    )
    # another move in the terminal state
    assert_time_step(
        environment.step(4),
        (np.array([1, 1, 1, 1, 1]), 0.0, True, False, {}),
    )


@hypothesis.given(
    length=st.integers(min_value=1, max_value=100),
)
def test_abcseq_render(length: int):
    environment = abcseq.ABCSeq(length, render_mode="rgb_array")
    environment.reset(),
    # starting point
    np.testing.assert_array_equal(environment.render(), np.array([1] + [0] * length))  # type: ignore
    # one move
    environment.step(0)
    np.testing.assert_array_equal(
        environment.render(), np.array([1, 1] + [0] * (length - 1))  # type: ignore
    )


@hypothesis.given(
    length=st.integers(min_value=1, max_value=100),
)
def test_abcseq_render_with_invalid_modes(length: int):
    modes = ("human",)
    for mode in modes:
        environment = abcseq.ABCSeq(length, render_mode=mode)
        environment.reset()
        with pytest.raises(NotImplementedError):
            environment.render()


@hypothesis.given(
    completed=st.integers(min_value=0, max_value=10),
    missing=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_unmastered_letters_and_action_is_the_next(
    completed: int, missing: int
):
    obs = np.array([1] + [1] * completed + [0] * missing)
    expected = np.array([1] + [1] * (completed + 1) + [0] * (missing - 1))
    action = completed
    output = abcseq.apply_action(obs, action)
    np.testing.assert_array_equal(output, expected)


@hypothesis.given(
    completed=st.integers(min_value=0, max_value=10),
    missing=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_unmastered_letters_and_action_is_skipped_head(
    completed: int, missing: int
):
    obs = np.array([1] + [1] * completed + [0] * missing)
    skipped_steps = np.random.randint(low=1, high=missing + 1)
    action = completed + skipped_steps
    output = abcseq.apply_action(obs, action)
    np.testing.assert_array_equal(output, obs)


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
    missing=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_unmastered_letters_and_action_is_behind(
    completed: int, missing: int
):
    obs = np.array([1] + [1] * completed + [0] * missing)
    action = np.random.randint(low=0, high=completed)
    # num letters ahead + behind + 1 for rotation
    output = abcseq.apply_action(obs, action)
    np.testing.assert_array_equal(output, obs)


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_completed_sequence_action_is_end(completed: int):
    obs = np.array([1] + [1] * completed)
    action = completed
    output = abcseq.apply_action(obs, action)
    np.testing.assert_array_equal(output, obs)


@hypothesis.given(
    completed=st.integers(min_value=0, max_value=10),
    missing=st.integers(min_value=2, max_value=10),
)
def test_action_reward_with_unmastered_letters_and_action_is_the_next(
    completed: int, missing: int
):
    obs = np.array([1] + [1] * completed + [0] * missing)
    action = completed
    output = abcseq.action_reward(obs, action)
    assert output == -1.0


@hypothesis.given(
    completed=st.integers(min_value=0, max_value=10),
)
def test_action_reward_with_one_unmastered_letters_and_action_is_the_next(
    completed: int,
):
    obs = np.array([1] + [1] * completed + [0])
    action = completed
    output = abcseq.action_reward(obs, action)
    assert output == -1.0


@hypothesis.given(
    completed=st.integers(min_value=0, max_value=10),
    missing=st.integers(min_value=1, max_value=10),
)
def test_action_reward_with_unmastered_letters_and_action_is_skipped_ahead(
    completed: int, missing: int
):
    obs = np.array([1] + [1] * completed + [0] * missing)
    skipped_steps = np.random.randint(low=1, high=missing + 1)
    action = completed + skipped_steps
    output = abcseq.action_reward(obs, action)
    moving_penantly = 1
    assert output == -(skipped_steps + moving_penantly)


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
    missing=st.integers(min_value=1, max_value=10),
)
def test_action_reward_with_unmastered_letters_and_action_is_behind(
    completed: int, missing: int
):
    obs = np.array([1] + [1] * completed + [0] * missing)
    action = np.random.randint(low=0, high=completed)
    # num letters ahead + behind + 1 for rotation
    moving_penalty = 1
    expected = missing + action + moving_penalty + 1
    output = abcseq.action_reward(obs, action)
    assert output == -expected


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
)
def test_action_reward_with_completed_sequence_action_is_end(completed: int):
    obs = np.array([1] + [1] * completed)
    action = completed
    output = abcseq.action_reward(obs, action)
    assert output == 0.0


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
)
def test_action_reward_with_completed_sequence_action_is_behind(completed: int):
    obs = np.array([1] + [1] * completed)
    action = np.random.randint(low=0, high=completed)
    output = abcseq.action_reward(obs, action)
    assert output == 0.0


@hypothesis.given(
    length=st.integers(min_value=1, max_value=100),
)
def test_beginning_state(length: int):
    np.testing.assert_array_equal(
        abcseq.beginning_state(length), np.array([1] + [0] * length)
    )


def test_is_finished():
    assert not abcseq.is_finished(np.array([1, 0]), 0)
    assert not abcseq.is_finished(np.array([1, 0]), 1)
    assert abcseq.is_finished(np.array([1, 1]), 0)
    assert not abcseq.is_finished(np.array([1, 0, 0]), 0)
    assert not abcseq.is_finished(np.array([1, 1, 0]), 0)
    assert abcseq.is_finished(np.array([1, 1, 1]), 1)


def test_state_id():
    # (starting state)
    assert abcseq.get_state_id([1, 0, 0]) == 0
    assert abcseq.get_state_id([1, 1, 0]) == 1
    assert abcseq.get_state_id([1, 1, 1]) == 2


def assert_time_step(output: TimeStep, expected: TimeStep) -> None:
    np.testing.assert_array_equal(output[0], expected[0])
    assert output[1] == expected[1]
    assert output[2] == expected[2]
    assert output[3] == expected[3]
    assert output[4] == expected[4]


def assert_init_state(output: InitState, expected: InitState) -> None:
    np.testing.assert_array_equal(output[0], expected[0])
    assert output[1] == expected[1]
