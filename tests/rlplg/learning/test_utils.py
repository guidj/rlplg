import hypothesis
from hypothesis import strategies as st

from rlplg.learning import utils


@hypothesis.given(element=st.integers())
def test_chain_map_integers(element: float):
    func0 = lambda x: x**2
    func1 = lambda x: x + 1

    assert utils.chain_map(element, [func0, func1]) == func1(func0(element))
    assert utils.chain_map(element, [func1, func0]) == func0(func1(element))


@hypothesis.given(element=st.text())
def test_chain_map_strings(element: str):
    func0 = lambda xs: list(reversed(xs))
    func1 = lambda xs: [x for x in xs if x not in ("a", "e", "i", "o", "u")]

    assert utils.chain_map(element, [func0, func1]) == func1(func0(element))
    assert utils.chain_map(element, [func1, func0]) == func0(func1(element))
