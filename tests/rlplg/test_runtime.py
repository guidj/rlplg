import time

import hypothesis
from hypothesis import strategies as st

from rlplg import runtime


@hypothesis.given(st.integers())
def test_run_id(_: int):
    _id = runtime.run_id()
    parts = _id.split("-")
    assert len(parts) == 2
    uuid4_head, timestamp = parts
    assert len(uuid4_head) == 8
    assert uuid4_head.isalnum()
    assert timestamp.isnumeric()
    assert 0 < int(timestamp) <= time.time()
