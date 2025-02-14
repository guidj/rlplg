from typing import SupportsFloat, Tuple

from rlplg.core import EnvTransition


def assert_transition_mapping(
    transition_mapping: EnvTransition, env_dim: Tuple[int, int]
):
    num_states, num_actions = env_dim
    assert len(transition_mapping) == num_states
    for state, action_transitions in transition_mapping.items():
        assert 0 <= state < num_states
        assert len(action_transitions) == num_actions
        for action, transitions in action_transitions.items():
            assert 0 <= action < num_actions
            for prob, next_state, reward, done in transitions:
                assert 0 <= prob <= 1.0
                assert 0 <= next_state < num_states
                assert isinstance(reward, SupportsFloat)
                assert isinstance(done, bool)
