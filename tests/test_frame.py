# SPDX-License-Identifier: MIT
"""Tests for the :class:`Frame` container (stats, binning, array protocol)."""

import numpy as np
import pytest

from getframes.frame import Frame, FrameTruth


def test_binned_sum_combines_super_pixels():
    data = np.arange(16, dtype=float).reshape(4, 4)
    binned = Frame(data).binned(2)
    assert binned.shape == (2, 2)
    # Top-left 2x2 block is 0, 1, 4, 5 -> 10.
    assert binned.data[0, 0] == pytest.approx(10.0)
    # Summing preserves total signal.
    assert float(np.asarray(binned).sum()) == pytest.approx(float(data.sum()))


def test_binned_mean_averages_super_pixels():
    data = np.full((6, 6), 7.0)
    binned = Frame(data).binned(3, method="mean")
    assert binned.shape == (2, 2)
    assert np.allclose(np.asarray(binned), 7.0)


def test_binned_updates_metadata_and_drops_truth():
    data = np.ones((4, 4))
    truth = FrameTruth(mean_electrons=data, mean_photoelectrons=data, photon_rate=1.0)
    frame = Frame(data, metadata={"binning": 1, "camera": "x"}, truth=truth)
    binned = frame.binned(2)
    assert binned.metadata["binning"] == 2
    assert binned.metadata["camera"] == "x"
    assert binned.truth is None
    # The original frame is untouched.
    assert frame.shape == (4, 4)


def test_binned_rejects_indivisible_shape_and_bad_args():
    frame = Frame(np.ones((5, 4)))
    with pytest.raises(ValueError):
        frame.binned(2)
    with pytest.raises(ValueError):
        frame.binned(0)
    with pytest.raises(ValueError):
        frame.binned(2, method="median")
