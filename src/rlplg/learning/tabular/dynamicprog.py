import copy
import logging

import numpy as np

from rlplg.learning.tabular import markovdp, policies


def iterative_policy_evaluation(
    mdp: markovdp.MDP,
    policy: policies.ObservablePolicy,
    gamma: float = 1.0,
    accuracy: float = 1e-8,
):

    """
    Implementation of dynamic programming for state value function computation.
    V(s)_{pi}.
    Vectorized implementation.
    """
    pi_action = np.zeros(
        (mdp.env_desc().num_states, mdp.env_desc().num_actions), dtype=np.float
    )
    transition = np.zeros(
        (
            mdp.env_desc().num_states,
            mdp.env_desc().num_actions,
            mdp.env_desc().num_states,
        ),
        dtype=np.float,
    )
    reward = np.zeros(
        (
            mdp.env_desc().num_states,
            mdp.env_desc().num_actions,
            mdp.env_desc().num_states,
        ),
        dtype=np.float,
    )

    for state in range(mdp.env_desc().num_states):
        for action in range(mdp.env_desc().num_actions):
            pi_action[state, action] = policy.action_probability(state, action)
            for new_state in range(mdp.env_desc().num_states):
                transition[state, action, new_state] = mdp.transition_probability(
                    state, action, new_state
                )
                reward[state, action, new_state] = mdp.reward(state, action, new_state)

    m_state_values = np.tile(
        np.zeros(shape=mdp.env_desc().num_states), (mdp.env_desc().num_actions, 1)
    )
    while True:
        delta = np.zeros(shape=mdp.env_desc().num_states)
        current_state_values = copy.deepcopy(m_state_values[0])
        # |S| x |A| x |S'|
        # |S| x |A|
        values = np.sum(transition * (reward + gamma * m_state_values), axis=2)
        new_state_values = np.diag(np.dot(pi_action, np.transpose(values)))
        m_state_values = np.tile(new_state_values, (mdp.env_desc().num_actions, 1))

        delta = np.maximum(
            delta,
            np.abs(current_state_values - new_state_values),
        )
        if np.alltrue(delta < accuracy):
            return m_state_values[0, :]


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
