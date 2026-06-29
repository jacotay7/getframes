# SPDX-License-Identifier: MIT
"""Tests for the phase 1.6 dataset generator."""

import numpy as np
import pytest

import getframes as gf
from getframes import dataset


@pytest.fixture
def camera():
    return gf.Camera.from_preset("generic_cmos", precision="float32").with_config(
        resolution=[40, 40]
    )


def test_random_star_fields_len_and_reiterable():
    scenes = dataset.random_star_fields(n=4, shape=(40, 40), seed=0)
    assert len(scenes) == 4
    first = [s.photon_rate_map().sum() for s in scenes]
    second = [s.photon_rate_map().sum() for s in scenes]  # re-iterate
    assert first == second
    assert all(isinstance(s, gf.Scene) for s in scenes)


def test_random_star_fields_seed_changes_fields():
    a = next(iter(dataset.random_star_fields(n=1, shape=(40, 40), seed=0)))
    b = next(iter(dataset.random_star_fields(n=1, shape=(40, 40), seed=1)))
    assert a.photon_rate_map().sum() != b.photon_rate_map().sum()


def test_random_star_fields_fixed_n_stars():
    scene = next(iter(dataset.random_star_fields(n=1, shape=(40, 40), n_stars=7, seed=0)))
    assert len(scene.sources) == 7


def test_pairs_shapes_and_dtype(camera):
    scenes = dataset.random_star_fields(n=3, shape=(40, 40), seed=0)
    ds = dataset.pairs(camera=camera, scenes=scenes, exposure=10.0, seed=1)
    assert len(ds) == 3
    pair = next(iter(ds))
    assert set(pair) == {"raw", "truth"}
    assert pair["raw"].shape == (40, 40)
    assert pair["raw"].dtype == np.float32
    assert pair["truth"].dtype == np.float32


def test_pairs_reproducible_and_independent(camera):
    def raws():
        scenes = dataset.random_star_fields(n=3, shape=(40, 40), seed=0)
        ds = dataset.pairs(camera=camera, scenes=scenes, exposure=10.0, seed=1)
        return [p["raw"] for p in ds]

    a, b = raws(), raws()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))  # reproducible
    assert not np.array_equal(a[0], a[1])  # independent frames


def test_to_npz_round_trip(camera, tmp_path):
    scenes = dataset.random_star_fields(n=2, shape=(40, 40), seed=0)
    ds = dataset.pairs(camera=camera, scenes=scenes, exposure=10.0, seed=1)
    paths = ds.to_npz(str(tmp_path))
    assert len(paths) == 2
    loaded = np.load(paths[0])
    assert set(loaded.keys()) == {"raw", "truth"}
    assert loaded["raw"].shape == (40, 40)


def test_to_arrays_stacks(camera):
    scenes = dataset.random_star_fields(n=3, shape=(40, 40), seed=0)
    ds = dataset.pairs(camera=camera, scenes=scenes, exposure=10.0, seed=1)
    raw, truth = ds.to_arrays()
    assert raw.shape == (3, 40, 40)
    assert truth.shape == (3, 40, 40)


def test_pairs_dtype_float64(camera):
    scenes = dataset.random_star_fields(n=1, shape=(40, 40), seed=0)
    ds = dataset.pairs(camera=camera, scenes=scenes, exposure=10.0, dtype="float64", seed=1)
    assert next(iter(ds))["raw"].dtype == np.float64
