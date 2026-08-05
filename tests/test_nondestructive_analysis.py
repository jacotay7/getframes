# SPDX-License-Identifier: MIT
"""Characterisation tests for correlated nondestructive-read stacks."""

import numpy as np
import pytest

from getframes.analysis import (
    infer_reset_indices,
    nondestructive_stack_statistics,
    ramp_photon_transfer,
)


def test_infer_reset_indices_rejects_small_electronic_oscillation():
    ramp = np.tile(np.arange(20, dtype=float), 5)
    np.testing.assert_array_equal(infer_reset_indices(ramp), [19, 39, 59, 79])

    oscillation = np.tile([0.0, 1.0, 2.0, 3.0, -1.0], 20)
    assert infer_reset_indices(oscillation).size == 0


def test_nondestructive_stack_statistics_separates_common_and_pixel_noise():
    rng = np.random.default_rng(4)
    shape = (48, 64)
    n_frames = 200
    ramp_length = 40
    channel_sigma = np.linspace(3.0, 6.0, 8)
    sigma = np.broadcast_to(channel_sigma[np.arange(shape[1]) % 8], shape)
    channel_bias = np.broadcast_to(20.0 * (np.arange(shape[1]) % 8), shape)
    fixed = 1000.0 + channel_bias
    cube = np.empty((n_frames, *shape))
    common = np.empty(n_frames)
    common[0] = rng.normal(0.0, 12.0)
    for index in range(n_frames):
        if index:
            common[index] = -0.5 * common[index - 1] + rng.normal(0.0, 12.0 * np.sqrt(0.75))
        signal = 4.0 * (index % ramp_length)
        cube[index] = fixed + signal + common[index] + rng.normal(0.0, sigma)

    stats = nondestructive_stack_statistics(cube, channel_count=8)
    assert stats.inferred_reads_per_reset == ramp_length
    assert stats.ramp_slope_adu_per_read == pytest.approx(4.0, abs=0.2)
    assert stats.temporal_noise_median_adu == pytest.approx(np.median(sigma), rel=0.12)
    assert stats.channel_bias_spread_adu == pytest.approx(np.std(np.arange(8) * 20.0), rel=0.05)
    assert stats.common_mode_noise_adu == pytest.approx(12.0, rel=0.2)
    assert stats.common_mode_lag1_correlation == pytest.approx(-0.5, abs=0.15)


def test_ramp_photon_transfer_recovers_conversion_gain():
    rng = np.random.default_rng(9)
    gain_e_per_adu = 2.2
    shape = (64, 64)
    ramps = 10
    reads = 30
    rate_e_per_read = 60.0
    bias = rng.normal(1000.0, 30.0, size=shape)
    cube = []
    for _ in range(ramps):
        charge = np.zeros(shape)
        reset = rng.normal(0.0, 4.0, size=shape)
        for _ in range(reads):
            charge += rng.poisson(rate_e_per_read, size=shape)
            cube.append(bias + reset + charge / gain_e_per_adu + rng.normal(0.0, 3.0, size=shape))

    result = ramp_photon_transfer(
        np.asarray(cube),
        reset_after_indices=np.arange(reads - 1, ramps * reads - 1, reads),
    )
    assert result.conversion_gain_e_per_adu == pytest.approx(gain_e_per_adu, rel=0.1)
    assert result.fit_correlation > 0.99
    assert result.signal_adu[0] == 0.0
    assert result.variance_intercept_adu2 > 0
    assert result.n_ramps == ramps
