import logging

import numpy as np

from rlplg.learning.tabular import markovdp
from rlplg.learning.tabular import policies


def iterative_policy_evaluation(
    mdp: markovdp.MDP,
    policy: policies.ObservablePolicy,
    gamma: float = 1.0,
    accuracy: float = 1e-8,
):
    """
    Implementation of dynamic programming for state value function.
    V(s)_{pi}
    """
    state_values = np.zeros(shape=mdp.env_desc().num_states)
    while True:
        delta = 0.0
        for state in range(mdp.env_desc().num_states):
            current_value = state_values[state]
            new_value = 0
            for action in range(mdp.env_desc().num_actions):
                action_prob = policy.action_probability(state, action)
                cu_value = 0
                for new_state in range(mdp.env_desc().num_states):
                    transition_prob = mdp.transition_probability(
                        state, action, new_state
                    )
                    value = (
                        mdp.reward(state, action, new_state)
                        + gamma * state_values[new_state]
                    )
                    cu_value += transition_prob * value
                new_value += action_prob * cu_value
            state_values[state] = new_value

            delta = max(delta, np.abs(current_value - state_values[state]))
            if delta < accuracy:
                return state_values


def action_values_from_state_values(
    mdp: markovdp.MDP, state_values: np.ndarray, gamma: float = 1.0
):
    """
    Compute Q(s,a) using V(s)
    """
    qtable = np.zeros(shape=(mdp.env_desc().num_states, mdp.env_desc().num_actions))
    for state in range(mdp.env_desc().num_states):
        for action in range(mdp.env_desc().num_actions):
            cu_value = 0
            for new_state in range(mdp.env_desc().num_states):
                transition_prob = mdp.transition_probability(state, action, new_state)
                reward = mdp.reward(state, action, new_state)
                next_state_value = gamma * state_values[new_state]
                logging.debug(
                    "state: %d, action: %d, next_state: %d (prob: %f), reward: %f, next_state_value: %d",
                    state,
                    action,
                    new_state,
                    transition_prob,
                    reward,
                    next_state_value,
                )
                cu_value += transition_prob * (reward + next_state_value)
            qtable[state, action] = cu_value
    return qtable
