from typing import Any, Sequence

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from gymnasium import spaces

from rlplg.core import TimeStep
from rlplg.environments import redgreen
from tests.rlplg import dynamics

VALID_ACTIONS = ["red", "green", "wait"]


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
@hypothesis.settings(deadline=None)
def test_redgreen_init(cure: Sequence[str]):
    cure_sequence = [redgreen.ACTION_NAME_MAPPING[step] for step in cure]
    environment = redgreen.RedGreenSeq(cure)
    assert environment.cure_sequence == tuple(cure_sequence)
    assert environment.action_space == spaces.Discrete(3)
    assert environment.observation_space == spaces.Dict(
        {
            "id": spaces.Discrete(len(cure_sequence) + 1),
            "cure_sequence": spaces.Sequence(spaces.Discrete(3)),
            "pos": spaces.Discrete(len(cure_sequence) + 1),
        }
    )
    dynamics.assert_transition_mapping(
        environment.transition,
        env_dim=(len(cure_sequence) + 1, 3),
    )


def test_redgreen_simple_sequence():
    cure = ["red", "green", "wait"]
    environment = redgreen.RedGreenSeq(cure)
    obs, info = environment.reset()
    assert_observation(
        obs,
        {
            "cure_sequence": (0, 1, 2),
            "pos": 0,
            "id": 0,
        },
    )
    assert info == {}
    # last treatment step, prematurely
    assert_time_step(
        environment.step(2),
        (
            {
                "cure_sequence": (0, 1, 2),
                "pos": 0,
                "id": 0,
            },
            -1.0,
            False,
            False,
            {},
        ),
    )
    # first treatment step
    assert_time_step(
        environment.step(0),
        (
            {
                "cure_sequence": (0, 1, 2),
                "pos": 1,
                "id": 1,
            },
            -1.0,
            False,
            False,
            {},
        ),
    )
    # second treatment step
    assert_time_step(
        environment.step(1),
        (
            {
                "cure_sequence": (0, 1, 2),
                "pos": 2,
                "id": 2,
            },
            -1.0,
            False,
            False,
            {},
        ),
    )
    # wrong treatment step
    assert_time_step(
        environment.step(0),
        (
            {
                "cure_sequence": (0, 1, 2),
                "pos": 2,
                "id": 2,
            },
            -1.0,
            False,
            False,
            {},
        ),
    )
    # third and final treatment step
    assert_time_step(
        environment.step(2),
        (
            {
                "cure_sequence": (0, 1, 2),
                "pos": 3,
                "id": 3,
            },
            -1.0,
            True,
            False,
            {},
        ),
    )

    # another treatment step in the terminal state
    assert_time_step(
        environment.step(0),
        (
            {
                "cure_sequence": (0, 1, 2),
                "pos": 3,
                "id": 3,
            },
            0.0,
            True,
            False,
            {},
        ),
    )


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_render(cure: Sequence[str]):
    environment = redgreen.RedGreenSeq(cure, render_mode="rgb_array")
    environment.reset()
    # starting point
    np.testing.assert_array_equal(
        environment.render(),  # type: ignore
        [0] * len(cure),
    )
    # one move
    environment.step(redgreen.ACTION_NAME_MAPPING[cure[0]])
    np.testing.assert_array_equal(environment.render(), [1] + [0] * (len(cure) - 1))  # type: ignore


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_render_with_invalid_modes(cure: Sequence[str]):
    modes = ("human",)
    for mode in modes:
        environment = redgreen.RedGreenSeq(cure, render_mode=mode)
        environment.reset()
        with pytest.raises(NotImplementedError):
            environment.render()


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_correct_next_action(cure_sequence: Sequence[int]):
    obs = {
        "cure_sequence": cure_sequence,
        "pos": 0,
        "id": 0,
    }
    output_obs, output_reward = redgreen.apply_action(obs, cure_sequence[0])
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "pos": 1,
        "id": 1,
    }
    assert output_reward == -1.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_wrong_next_action(cure_sequence: Sequence[int]):
    obs = {
        "cure_sequence": cure_sequence,
        "pos": 0,
    }
    wrong_actions = [
        action for action in redgreen.ACTIONS if action != cure_sequence[0]
    ]
    wrong_action = np.random.default_rng().choice(wrong_actions)
    output_obs, output_reward = redgreen.apply_action(obs, wrong_action)
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "pos": 0,
    }
    assert output_reward == -1.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_action_going_into_terminal_state(
    cure_sequence: Sequence[int],
):
    obs = {
        "cure_sequence": cure_sequence,
        "pos": len(cure_sequence) - 1,
        "id": len(cure_sequence) - 1,
    }
    output_obs, output_reward = redgreen.apply_action(obs, cure_sequence[-1])
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "pos": len(cure_sequence),
        "id": len(cure_sequence),
    }
    assert output_reward == -1.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    ),
    action=st.integers(min_value=0, max_value=len(redgreen.ACTIONS)),
)
def test_apply_action_with_env_in_terminal_state(
    cure_sequence: Sequence[int], action: int
):
    obs = {
        "cure_sequence": cure_sequence,
        "pos": len(cure_sequence),
    }
    output_obs, output_reward = redgreen.apply_action(obs, action)
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "pos": len(cure_sequence),
    }
    assert output_reward == 0.0


def test_is_finished():
    assert not redgreen.is_terminal_state({"cure_sequence": (0,), "pos": 0})
    assert redgreen.is_terminal_state({"cure_sequence": (0,), "pos": 1})
    assert not redgreen.is_terminal_state({"cure_sequence": (0, 1, 0), "pos": 0})
    assert not redgreen.is_terminal_state({"cure_sequence": (0, 1, 0), "pos": 1})
    assert redgreen.is_terminal_state({"cure_sequence": (0, 1, 0), "pos": 3})


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_state_observation(cure_sequence: Sequence[int]):
    pos = np.random.default_rng().integers(0, len(cure_sequence) + 1)
    assert redgreen.state_observation(pos=pos, cure_sequence=cure_sequence) == {
        "cure_sequence": cure_sequence,
        "pos": pos,
        "id": pos,
    }


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_state_observation_with_invalid_position(cure_sequence: Sequence[int]):
    pos = np.random.default_rng().choice([-1, len(cure_sequence) + 1])
    with pytest.raises(ValueError):
        redgreen.state_observation(cure_sequence, pos=pos)


def test_state_representation():
    assert redgreen.state_representation({"cure_sequence": (0, 1, 0), "pos": 0}) == [
        0,
        0,
        0,
    ]
    assert redgreen.state_representation({"cure_sequence": (0, 1, 0), "pos": 1}) == [
        1,
        0,
        0,
    ]
    assert redgreen.state_representation({"cure_sequence": (0, 1, 0), "pos": 2}) == [
        1,
        1,
        0,
    ]
    assert redgreen.state_representation({"cure_sequence": (0, 1, 0), "pos": 3}) == [
        1,
        1,
        1,
    ]


def assert_time_step(output: TimeStep, expected: TimeStep) -> None:
    assert_observation(output[0], expected[0])
    assert output[1] == expected[1]
    assert output[2] is expected[2]
    assert output[3] is expected[3]
    assert output[4] == expected[4]


def assert_observation(output: Any, expected: Any) -> None:
    assert len(output) == 3
    assert output["cure_sequence"] == expected["cure_sequence"]
    assert output["pos"] == expected["pos"]
    assert output["id"] == expected["id"]
