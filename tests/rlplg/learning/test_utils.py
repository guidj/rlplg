from typing import Sequence

import hypothesis
from hypothesis import strategies as st

from rlplg.learning import utils


@hypothesis.given(element=st.integers())
def test_chain_map_integers(element: float):
    def func0(value: float):
        return value**2

    def func1(value: float):
        return value + 1

    assert utils.chain_map(element, [func0, func1]) == func1(func0(element))
    assert utils.chain_map(element, [func1, func0]) == func0(func1(element))


@hypothesis.given(element=st.text())
def test_chain_map_strings(element: str):
    def func0(elements: Sequence[str]):
        return list(reversed(elements))

    def func1(elements: Sequence[str]):
        return [x for x in elements if x not in ("a", "e", "i", "o", "u")]

    assert utils.chain_map(element, [func0, func1]) == func1(func0(element))
    assert utils.chain_map(element, [func1, func0]) == func0(func1(element))
