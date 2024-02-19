"""
Functions to visualize data.
"""

import numpy as np


def vis_learned_array(array: np.ndarray, decimals: int = 3):
    """
    Print the table in a friendly format.
    """
    return f"\n{np.around(array, decimals)}"
