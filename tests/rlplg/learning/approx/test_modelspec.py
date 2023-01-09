import copy
from typing import Callable, Mapping, Tuple

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st
from tf_agents.typing.types import NestedArray

from rlplg.learning.approx import modelspec


class StaticValueFn(modelspec.ValueFnModel):
    """
    A static mapping; value is defined by a mapping
    given at instantiation.
    """

    def __init__(self, value_fn: Callable[[NestedArray], float]):
        self._value_fn = value_fn
        self._weights = None

    def predict(self, features: NestedArray) -> float:
        """
        Computes the value for a given context,
        i.e. state or (state, action)
        """
        return self._value_fn(features)

    def gradients(self, features: NestedArray) -> NestedArray:
        """
        Computes the gradients of the output
        with respect to the weights.
        """
        return features

    def predict_and_gradients(self, features: NestedArray) -> Tuple[float, NestedArray]:
        """
        Computes prediction and gradients of the
        weights for the prediction.
        """
        return self.predict(features), self.gradients(features)

    def weights(self) -> NestedArray:
        """
        Returns the current weights of the model.
        """
        return self._weights

    def assign_weights(self, weights: NestedArray) -> None:
        """
        Assigns new values to the model weights.
        """
        self._weights = copy.deepcopy(weights)


@hypothesis.given(features=st.integers(min_value=-100, max_value=100))
def test_approx_fn_predict(features: int, state_values: Mapping[NestedArray, float]):
    approx_fn = modelspec.ApproxFn(model=StaticValueFn(state_values))
    assert approx_fn.predict(np.array(features)) == features**2


@hypothesis.given(features=st.integers(min_value=-100, max_value=100))
def test_approx_fn_gradients(features: int, state_values: Mapping[NestedArray, float]):
    approx_fn = modelspec.ApproxFn(model=StaticValueFn(state_values))
    assert approx_fn.gradients(features) == features


def test_approx_fn_weights(state_values: Mapping[NestedArray, float]):
    approx_fn = modelspec.ApproxFn(model=StaticValueFn(state_values))
    assert approx_fn.weights() is None
    approx_fn.assign_weights(0)
    assert approx_fn.weights() == 0
    approx_fn.assign_weights(100.0)
    assert approx_fn.weights() == 100.0


@pytest.fixture(scope="session")
def state_values() -> Mapping[NestedArray, float]:
    """
    Static values for a hypothetical environment.
    """

    def value_fn(features: NestedArray):
        return features**2.0

    return value_fn
