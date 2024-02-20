from typing import SupportsFloat

from rlplg import core
from rlplg.core import EnvTransition


def assert_transition_mapping(
    transition_mapping: EnvTransition, env_desc: core.EnvDesc
):
    assert len(transition_mapping) == env_desc.num_states
    for state, action_transitions in transition_mapping.items():
        assert 0 <= state < env_desc.num_states
        assert len(action_transitions) == env_desc.num_actions
        for action, transitions in action_transitions.items():
            assert 0 <= action < env_desc.num_actions
            for prob, next_state, reward, done in transitions:
                assert 0 <= prob <= 1.0
                assert 0 <= next_state < env_desc.num_states
                assert isinstance(reward, SupportsFloat)
                assert isinstance(done, bool)
