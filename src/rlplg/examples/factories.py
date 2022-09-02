"""
Factory functions for examples.
"""
import numpy as np


def initialize_action_values(num_states: int, num_actions: int):
    """
    Q(s,a) initialized to zeros.
    """
    qtable = np.zeros(shape=(num_states, num_actions))
    return qtable


def initialize_state_values(num_states: int):
    """
    V(s) initialized to zeros.
    """
    vtable = np.zeros(shape=(num_states,))
    return vtable
