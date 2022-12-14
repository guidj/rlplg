import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from rlplg.environments.alphabet import env


@hypothesis.given(length=st.integers(min_value=1, max_value=100))
def test_abcseq_init(length: int):
    environment = env.ABCSeq(length)
    assert environment.length == length
    assert environment.action_spec() == action_spec(length)
    assert environment.observation_spec() == observation_spec(length)


def test_abcseq_simple_sequence():
    length = 4
    environment = env.ABCSeq(length)
    assert_time_step(
        environment.reset(),
        ts.TimeStep(
            step_type=ts.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=np.array([1, 0, 0, 0, 0]),
        ),
    )
    # final step, prematurely
    assert_time_step(
        environment.step(3),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-4.0,
            discount=1.0,
            observation=np.array([1, 0, 0, 0, 0]),
        ),
    )
    # first letter
    assert_time_step(
        environment.step(0),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation=np.array([1, 1, 0, 0, 0]),
        ),
    )
    # second letter
    assert_time_step(
        environment.step(1),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 0, 0]),
        ),
    )
    # skip ahead
    assert_time_step(
        environment.step(3),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-2.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 0, 0]),
        ),
    )
    # going backwards
    assert_time_step(
        environment.step(0),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-4.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 0, 0]),
        ),
    )
    # continue, third letter
    assert_time_step(
        environment.step(2),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 1, 0]),
        ),
    )
    # complete
    assert_time_step(
        environment.step(3),
        ts.TimeStep(
            step_type=ts.StepType.LAST,
            reward=-1.0,
            discount=0.0,
            observation=np.array([1, 1, 1, 1, 1]),
        ),
    )
    # move in the terminal state
    assert_time_step(
        environment.step(0),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 1, 1]),
        ),
    )
    # another move in the terminal statethe
    assert_time_step(
        environment.step(4),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=np.array([1, 1, 1, 1, 1]),
        ),
    )


@hypothesis.given(
    length=st.integers(min_value=1, max_value=100),
)
def test_abcseq_render(length: int):
    environment = env.ABCSeq(length)
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
    environment = env.ABCSeq(length)
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
    output = env.apply_action(obs, action)
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
    output = env.apply_action(obs, action)
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
    output = env.apply_action(obs, action)
    np.testing.assert_array_equal(output, obs)


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_completed_sequence_action_is_end(completed: int):
    obs = np.array([1] + [1] * completed)
    action = completed
    output = env.apply_action(obs, action)
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
    output = env.action_reward(obs, action)
    assert output == -1.0


@hypothesis.given(
    completed=st.integers(min_value=0, max_value=10),
)
def test_action_reward_with_one_unmastered_letters_and_action_is_the_next(
    completed: int,
):
    obs = np.array([1] + [1] * completed + [0])
    action = completed
    output = env.action_reward(obs, action)
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
    output = env.action_reward(obs, action)
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
    output = env.action_reward(obs, action)
    assert output == -expected


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
)
def test_action_reward_with_completed_sequence_action_is_end(completed: int):
    obs = np.array([1] + [1] * completed)
    action = completed
    output = env.action_reward(obs, action)
    assert output == 0.0


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
)
def test_action_reward_with_completed_sequence_action_is_behind(completed: int):
    obs = np.array([1] + [1] * completed)
    action = np.random.randint(low=0, high=completed)
    output = env.action_reward(obs, action)
    assert output == 0.0


@hypothesis.given(
    length=st.integers(min_value=1, max_value=100),
)
def test_beginning_state(length: int):
    np.testing.assert_array_equal(
        env.beginning_state(length), np.array([1] + [0] * length)
    )


def test_is_finished():
    assert not env.is_finished(np.array([1, 0]), 0)
    assert not env.is_finished(np.array([1, 0]), 1)
    assert env.is_finished(np.array([1, 1]), 0)
    assert not env.is_finished(np.array([1, 0, 0]), 0)
    assert not env.is_finished(np.array([1, 1, 0]), 0)
    assert env.is_finished(np.array([1, 1, 1]), 1)


def test_state_id():
    # (starting state)
    assert env.get_state_id([1, 0, 0]) == 0
    assert env.get_state_id([1, 1, 0]) == 1
    assert env.get_state_id([1, 1, 1]) == 2


def action_spec(length: int) -> array_spec.BoundedArraySpec:
    return array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=length - 1,
        name="action",
    )


def observation_spec(length: int) -> array_spec.BoundedArraySpec:
    return array_spec.BoundedArraySpec(
        shape=(length + 1,),
        dtype=np.int32,
        minimum=np.array([1] + [0] * length),
        maximum=np.array([1] * (length + 1)),
        name="observation",
    )


def assert_time_step(output: ts.TimeStep, expected: ts.TimeStep) -> None:
    assert output.step_type == expected.step_type
    assert output.reward == expected.reward
    assert output.discount == expected.discount
    np.testing.assert_array_equal(output.observation, expected.observation)
