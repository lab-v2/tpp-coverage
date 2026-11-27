"""The characterization helpers in `characterize.py` record only what the code under test does."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from characterize import FigureCapture  # noqa: E402


def test_a_figure_left_open_before_the_block_is_not_captured() -> None:
    leftover = plt.figure()
    leftover.add_subplot().plot([0, 1], [5, 5])
    with FigureCapture() as cap:
        plt.figure().add_subplot().plot([0, 1], [1, 2])
        plt.show()
    assert len(cap.figures) == 1
    assert plt.get_fignums() == []
