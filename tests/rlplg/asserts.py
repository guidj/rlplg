from typing import Any

import numpy as np

from rlplg.core import TimeStep


def assert_time_step(output: TimeStep, expected: TimeStep) -> None:
    assert_complex_type(output, expected=expected)


def assert_observation(output: Any, expected: Any) -> None:
    assert_complex_type(output, expected=expected)


def assert_complex_type(output: Any, expected: Any) -> None:
    np.testing.assert_equal(output, expected)
