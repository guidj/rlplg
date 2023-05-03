from typing import Any

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest

from rlplg import core
from rlplg.environments import abcseq


@hypothesis.given(length=st.integers(min_value=1, max_value=100))
def test_abcseq_init(length: int):
    environment = abcseq.ABCSeq(length)
    assert environment.length == length
    assert environment.action_spec() == action_spec(length)
    assert environment.observation_spec() == observation_spec(length)


def test_abcseq_simple_sequence():
    length = 4
    environment = abcseq.ABCSeq(length)
    assert_time_step(
        environment.reset(),
        core.TimeStep(
            step_type=core.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=np.array([1, 0, 0, 0, 0]),
        ),
    )
    # final step, prematurely
    assert_time_step(
        environment.step(3),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-4.0,
            discount=1.0,
            observation=np.array([1, 0, 0, 0, 0]),
        ),
    )
    # first letter
    assert_time_step(
        environment.step(0),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation=np.array([1, 1, 0, 0, 0]),
        ),
    )
    # second letter
    assert_time_step(
        environment.step(1),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 0, 0]),
        ),
    )
    # skip ahead
    assert_time_step(
        environment.step(3),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-2.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 0, 0]),
        ),
    )
    # going backwards
    assert_time_step(
        environment.step(0),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-4.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 0, 0]),
        ),
    )
    # continue, third letter
    assert_time_step(
        environment.step(2),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 1, 0]),
        ),
    )
    # complete
    assert_time_step(
        environment.step(3),
        core.TimeStep(
            step_type=core.StepType.LAST,
            reward=-1.0,
            discount=0.0,
            observation=np.array([1, 1, 1, 1, 1]),
        ),
    )
    # move in the terminal state
    assert_time_step(
        environment.step(0),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 1, 1]),
        ),
    )
    # another move in the terminal statethe
    assert_time_step(
        environment.step(4),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 1, 1]),
        ),
    )


@hypothesis.given(
    length=st.integers(min_value=1, max_value=100),
)
def test_abcseq_render(length: int):
    environment = abcseq.ABCSeq(length)
    environment.reset(),
    # starting point
    np.testing.assert_array_equal(
        environment.render("rgb_array"), np.array([1] + [0] * length)
    )
    # one move
    environment.step(0)
    np.testing.assert_array_equal(
        environment.render("rgb_array"), np.array([1, 1] + [0] * (length - 1))
    )


@hypothesis.given(
    length=st.integers(min_value=1, max_value=100),
)
def test_abcseq_render_with_invalid_modes(length: int):
    modes = ("human",)
    environment = abcseq.ABCSeq(length)
    for mode in modes:
        with pytest.raises(NotImplementedError):
            environment.render(mode)


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


def action_spec(length: int) -> Any:
    return ()


def observation_spec(length: int) -> Any:
    return ()


def assert_time_step(output: core.TimeStep, expected: core.TimeStep) -> None:
    assert output.step_type == expected.step_type
    assert output.reward == expected.reward
    assert output.discount == expected.discount
    np.testing.assert_array_equal(output.observation, expected.observation)  # type: ignore
