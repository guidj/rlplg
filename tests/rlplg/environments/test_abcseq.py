import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from gymnasium import spaces

from rlplg import core
from rlplg.core import InitState, TimeStep
from rlplg.environments import abcseq
from tests.rlplg import dynamics


@hypothesis.given(length=st.integers(min_value=1, max_value=10))
@hypothesis.settings(deadline=None)
def test_abcseq_init(length: int):
    environment = abcseq.ABCSeq(length)
    assert environment.length == length
    assert environment.action_space == spaces.Discrete(length)
    assert environment.observation_space == spaces.Dict(
        {
            "length": spaces.Box(low=length, high=length, dtype=np.int64),
            "pos": spaces.Discrete(length + 1),
        }
    )
    dynamics.assert_transition_mapping(
        environment.transition,
        env_desc=core.EnvDesc(num_states=length + 1, num_actions=length),
    )


@hypothesis.given(length=st.integers(min_value=27, max_value=100))
def test_abcseq_init_with_invalid_length(length: int):
    with pytest.raises(ValueError):
        abcseq.ABCSeq(length)


def test_abcseq_simple_sequence():
    length = 4
    environment = abcseq.ABCSeq(length)
    assert_init_state(
        environment.reset(),
        ({"length": 4, "pos": 0}, {}),
    )
    # final step, prematurely
    assert_time_step(
        environment.step(3),
        ({"length": 4, "pos": 0}, -4.0, False, False, {}),
    )
    # first token
    assert_time_step(
        environment.step(0),
        ({"length": 4, "pos": 1}, -1.0, False, False, {}),
    )
    # second token
    assert_time_step(
        environment.step(1),
        ({"length": 4, "pos": 2}, -1.0, False, False, {}),
    )
    # skip ahead
    assert_time_step(
        environment.step(3),
        ({"length": 4, "pos": 2}, -2.0, False, False, {}),
    )
    # going backwards
    assert_time_step(
        environment.step(0),
        ({"length": 4, "pos": 2}, -4.0, False, False, {}),
    )
    # continue, third token
    assert_time_step(
        environment.step(2),
        ({"length": 4, "pos": 3}, -1.0, False, False, {}),
    )
    # complete
    assert_time_step(
        environment.step(3),
        ({"length": 4, "pos": 4}, -1.0, True, False, {}),
    )
    # move in the terminal state
    assert_time_step(
        environment.step(0),
        ({"length": 4, "pos": 4}, 0.0, True, False, {}),
    )
    # another move in the terminal state
    assert_time_step(
        environment.step(4),
        ({"length": 4, "pos": 4}, 0.0, True, False, {}),
    )


@hypothesis.given(
    length=st.integers(min_value=1, max_value=10),
)
@hypothesis.settings(deadline=None)
def test_abcseq_render(length: int):
    environment = abcseq.ABCSeq(length, render_mode="rgb_array")
    environment.reset()
    # starting point
    np.testing.assert_array_equal(environment.render(), np.array([1] + [0] * length))  # type: ignore
    # one move
    environment.step(0)
    np.testing.assert_array_equal(
        environment.render(), np.array([1, 1] + [0] * (length - 1))  # type: ignore
    )


@hypothesis.given(
    length=st.integers(min_value=1, max_value=10),
)
@hypothesis.settings(deadline=None)
def test_abcseq_render_with_invalid_modes(length: int):
    modes = ("human",)
    for mode in modes:
        environment = abcseq.ABCSeq(length, render_mode=mode)
        environment.reset()
        with pytest.raises(NotImplementedError):
            environment.render()


def test_abcseqmdpdiscretizer():
    discretizer = abcseq.ABCSeqMdpDiscretizer()
    assert discretizer.state({"length": 4, "pos": 0}) == 0
    assert discretizer.state({"length": 4, "pos": 1}) == 1
    assert discretizer.state({"length": 4, "pos": 2}) == 2
    assert discretizer.state({"length": 4, "pos": 3}) == 3

    assert discretizer.action(0) == 0
    assert discretizer.action(1) == 1
    assert discretizer.action(3) == 3


@hypothesis.given(
    completed=st.integers(min_value=0, max_value=10),
    missing=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_unsorted_tokens_and_action_is_the_next(
    completed: int, missing: int
):
    output_obs, output_reward = abcseq.apply_action(
        {"length": completed + missing, "pos": completed}, action=completed
    )
    output_obs == {"length": completed + missing, "pos": completed}
    assert output_reward == -1.0


@hypothesis.given(
    completed=st.integers(min_value=0, max_value=10),
    missing=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_unsorted_tokens_and_action_is_skipped_head(
    completed: int, missing: int
):
    obs = {"length": completed + missing, "pos": completed}
    skipped_steps = np.random.randint(low=1, high=missing + 1)
    output_obs, output_reward = abcseq.apply_action(
        obs, action=completed + skipped_steps
    )
    assert output_obs == obs
    assert output_reward == -(skipped_steps + 1.0)


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
    missing=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_unsorted_tokens_and_action_is_behind(
    completed: int, missing: int
):
    obs = {"length": completed + missing, "pos": completed}
    action = np.random.randint(low=0, high=completed)
    # num tokens ahead + behind + 1 for rotation
    output_obs, output_reward = abcseq.apply_action(obs, action=action)
    assert output_obs == obs
    assert output_reward == -(missing + action + 2.0)


@hypothesis.given(
    completed=st.integers(min_value=1, max_value=10),
)
def test_apply_action_with_completed_sequence_action_is_end(completed: int):
    obs = {"length": completed, "pos": completed}
    output_obs, output_reward = abcseq.apply_action(obs, action=completed)
    assert output_obs == obs
    assert output_reward == 0.0


@hypothesis.given(
    length=st.integers(min_value=1, max_value=100),
)
def test_beginning_state(length: int):
    abcseq.beginning_state(length) == {"length": length, "pos": 0}


def test_is_terminal_state():
    assert not abcseq.is_terminal_state({"length": 1, "pos": 0})
    assert abcseq.is_terminal_state({"length": 1, "pos": 1})
    assert not abcseq.is_terminal_state({"length": 2, "pos": 1})
    assert abcseq.is_terminal_state({"length": 2, "pos": 2})


def test_state_id():
    # (starting state)
    assert abcseq.get_state_id({"pos": 0}) == 0
    assert abcseq.get_state_id({"pos": 1}) == 1
    assert abcseq.get_state_id({"pos": 2}) == 2


@hypothesis.given(length=st.integers(min_value=1, max_value=10))
@hypothesis.settings(deadline=None)
def test_abcseq_create_env_spec(length: int):
    env_spec = abcseq.create_env_spec(length)
    assert env_spec.name == "ABCSeq"
    assert env_spec.level == str(length)
    assert isinstance(env_spec.environment, abcseq.ABCSeq)
    assert isinstance(env_spec.discretizer, abcseq.ABCSeqMdpDiscretizer)
    assert env_spec.mdp.env_desc.num_states == length + 1
    assert env_spec.mdp.env_desc.num_actions == length
    assert len(env_spec.mdp.transition) == length + 1


def assert_time_step(output: TimeStep, expected: TimeStep) -> None:
    np.testing.assert_array_equal(output[0], expected[0])
    assert output[1] == expected[1]
    assert output[2] == expected[2]
    assert output[3] == expected[3]
    assert output[4] == expected[4]


def assert_init_state(output: InitState, expected: InitState) -> None:
    np.testing.assert_array_equal(output[0], expected[0])
    assert output[1] == expected[1]
