import random
from typing import Any, Mapping, Sequence

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest

from rlplg import core
from rlplg.environments import redgreen

VALID_ACTIONS = ["red", "green", "wait"]


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_init(cure: Sequence[str]):
    cure_sequence = [redgreen.ACTION_NAME_MAPPING[step] for step in cure]
    environment = redgreen.RedGreenSeq(cure)
    assert environment.cure_sequence == cure_sequence
    assert environment.action_spec() == action_spec()
    assert environment.observation_spec() == observation_spec(cure_sequence)


def test_redgreen_simple_sequence():
    cure = ["red", "green", "wait"]
    environment = redgreen.RedGreenSeq(cure)
    assert_time_step(
        environment.reset(),
        core.TimeStep(
            step_type=core.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 0,
            },
        ),
    )
    # final treatment step, prematurely
    assert_time_step(
        environment.step(2),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-2.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 0,
            },
        ),
    )
    # first treatment step
    assert_time_step(
        environment.step(0),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 1,
            },
        ),
    )
    # second treatment step
    assert_time_step(
        environment.step(1),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 2,
            },
        ),
    )
    # wrong treatment step
    assert_time_step(
        environment.step(0),
        core.TimeStep(
            step_type=core.StepType.MID,
            reward=-2.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 2,
            },
        ),
    )
    # third and final treatment step
    assert_time_step(
        environment.step(2),
        core.TimeStep(
            step_type=core.StepType.LAST,
            reward=-1.0,
            discount=0.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 3,
            },
        ),
    )

    # another treatment step in the terminal state
    assert_time_step(
        environment.step(0),
        core.TimeStep(
            step_type=core.StepType.LAST,
            reward=0.0,
            discount=0.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 3,
            },
        ),
    )


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_render(cure: Sequence[str]):
    environment = redgreen.RedGreenSeq(cure, render_mode="rgb_array")
    environment.reset()
    # starting point
    np.testing.assert_array_equal(
        environment.render(),
        [0] * len(cure),
    )
    # one move
    environment.step(redgreen.ACTION_NAME_MAPPING[cure[0]])
    np.testing.assert_array_equal(environment.render(), [1] + [0] * (len(cure) - 1))


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
        "position": 0,
    }
    output_obs, output_reward = redgreen.apply_action(obs, cure_sequence[0])
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": 1,
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
        "position": 0,
    }
    wrong_actions = [
        action for action in redgreen.ACTIONS if action != cure_sequence[0]
    ]
    wrong_action = random.sample(population=wrong_actions, k=1)[0]
    output_obs, output_reward = redgreen.apply_action(obs, wrong_action)
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": 0,
    }
    assert output_reward == -2.0


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
        "position": len(cure_sequence) - 1,
    }
    output_obs, output_reward = redgreen.apply_action(obs, cure_sequence[-1])
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence),
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
        "position": len(cure_sequence),
    }
    output_obs, output_reward = redgreen.apply_action(obs, action)
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence),
    }
    assert output_reward == 0.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_beginning_state(cure_sequence: Sequence[int]):
    output = redgreen.beginning_state(cure_sequence)
    assert output == {"cure_sequence": cure_sequence, "position": 0}


def test_is_finished():
    assert not redgreen.is_finished({"cure_sequence": [0], "position": 0})
    assert redgreen.is_finished({"cure_sequence": [0], "position": 1})
    assert not redgreen.is_finished({"cure_sequence": [0, 1, 0], "position": 0})
    assert not redgreen.is_finished({"cure_sequence": [0, 1, 0], "position": 1})
    assert redgreen.is_finished({"cure_sequence": [0, 1, 0], "position": 3})


def test_get_state_id():
    assert redgreen.get_state_id({"position": 0}) == 0
    assert redgreen.get_state_id({"position": 1}) == 1
    assert redgreen.get_state_id({"position": 2}) == 2


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS))))
    ),
    pos=st.integers(),
)
def test_state_observation(cure_sequence: Sequence[int], pos: int):
    assert redgreen.state_observation(state_id=pos, cure_sequence=cure_sequence)


def test_state_representation():
    assert redgreen.state_representation(
        {"cure_sequence": [0, 1, 0], "position": 0}
    ) == [
        0,
        0,
        0,
    ]
    assert redgreen.state_representation(
        {"cure_sequence": [0, 1, 0], "position": 1}
    ) == [
        1,
        0,
        0,
    ]
    assert redgreen.state_representation(
        {"cure_sequence": [0, 1, 0], "position": 2}
    ) == [
        1,
        1,
        0,
    ]
    assert redgreen.state_representation(
        {"cure_sequence": [0, 1, 0], "position": 3}
    ) == [
        1,
        1,
        1,
    ]


def action_spec() -> Any:
    return ()


def observation_spec(
    cure_actions: Sequence[int],
) -> Mapping[str, Any]:
    del cure_actions
    return {
        "cure_sequence": (),
        "position": (),
    }


def assert_time_step(output: core.TimeStep, expected: core.TimeStep) -> None:
    assert output.step_type == expected.step_type
    assert output.reward == expected.reward
    assert output.discount == expected.discount
    assert len(output.observation) == 2
    assert output.observation["cure_sequence"] == expected.observation["cure_sequence"]
    assert output.observation["position"] == expected.observation["position"]
