"""Derived display caches are private, bounded, content-specific and always disposable."""

import io
import os
import stat
import zipfile
from copy import deepcopy

import numpy as np
import pytest

from bimax import surface_cache as cache
from bimax.model import Geometry
from bimax.surfaces import Mesh


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def volume():
    labels = np.ones((4, 5, 6), dtype=np.uint8)
    geometry = Geometry(labels.shape, (0.4, 0.7, 1.2), (1.0, 2.0, 3.0), tuple(np.eye(3).ravel()))
    return labels, geometry


@pytest.fixture
def extraction(monkeypatch):
    calls = []
    mesh = Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
        np.tile(np.array([0, 0, 1], dtype=np.float32), (3, 1)),
    )

    def make(labels, geometry, progress=None):
        calls.append(labels)
        return {1: mesh}

    monkeypatch.setattr(cache.surfaces, "make_surfaces", make)
    return calls, mesh


def only_entry():
    return next(cache._directory().glob("*.npz"))


def test_cache_hit_skips_extraction_and_round_trips_numeric_arrays(volume, extraction):
    calls, expected = extraction
    messages = []
    first = cache.cached_surfaces(*volume)
    result = cache.cached_surfaces(*volume, messages.append)
    assert len(calls) == 1
    assert first[1] is expected
    for name in ("vertices", "faces", "normals"):
        np.testing.assert_array_equal(getattr(result[1], name), getattr(expected, name))
    assert "Loaded local display cache" in messages
    assert stat.S_IMODE(cache._directory().stat().st_mode) == 0o700
    assert stat.S_IMODE(only_entry().stat().st_mode) == 0o600
    with np.load(only_entry(), allow_pickle=False) as arrays:
        assert len(arrays.files) == 3
        assert all(arrays[name].dtype.kind in "fi" for name in arrays.files)


@pytest.mark.parametrize("change", ["labels", "geometry", "version", "triangles", "error", "vtk"])
def test_content_geometry_algorithm_and_display_changes_miss_cache(volume, extraction, monkeypatch, change):
    calls, _ = extraction
    labels, geometry = volume
    cache.cached_surfaces(labels, geometry)
    if change == "labels":
        labels = labels.copy()
        labels[1, 2, 3] = 2
    elif change == "geometry":
        geometry = deepcopy(geometry)
        geometry.origin = (2.0, 2.0, 3.0)
    elif change == "version":
        monkeypatch.setattr(cache, "ALGORITHM_VERSION", cache.ALGORITHM_VERSION + 1)
    elif change == "triangles":
        monkeypatch.setattr(cache.surfaces, "DISPLAY_TRIANGLE_TARGET", 12)
    elif change == "error":
        monkeypatch.setattr(cache.surfaces, "DISPLAY_ERROR_VOXEL_FRACTION", 0.1)
    else:
        class Version:
            @staticmethod
            def GetVTKVersion():
                return "different"

        monkeypatch.setattr(cache, "vtkVersion", Version)
    cache.cached_surfaces(labels, geometry)
    assert len(calls) == 2


def test_hash_matches_c_order_content_without_copying_noncontiguous_volume(volume):
    labels, geometry = volume
    labels[:, :, 1] = 2
    view = labels[:, :, ::-1]
    assert cache._key(view, geometry) == cache._key(np.ascontiguousarray(view), geometry)


@pytest.mark.parametrize("corruption", ["truncated", "nan", "index", "dtype", "header", "object"])
def test_corrupt_cache_is_ignored_and_rebuilt(volume, extraction, corruption):
    calls, _ = extraction
    cache.cached_surfaces(*volume)
    path = only_entry()
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if corruption == "truncated":
        path.write_bytes(b"broken display cache")
    elif corruption == "header":
        payload = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            payload, {"shape": (2**40, 3), "fortran_order": False, "descr": "<f4"}
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("1_vertices.npy", payload.getvalue())
    else:
        if corruption == "nan":
            arrays["1_normals"][0, 0] = np.nan
        elif corruption == "index":
            arrays["1_faces"][0, 0] = 1_000_000
        elif corruption == "dtype":
            arrays["1_vertices"] = arrays["1_vertices"].astype(np.float64)
        elif corruption == "object":
            arrays["1_vertices"] = np.array([[object()]], dtype=object)
        np.savez(path, **arrays)
    assert cache.cached_surfaces(*volume)
    assert len(calls) == 2
    assert cache._read(path)


def test_cache_evicts_least_recently_used_entries(volume, extraction, monkeypatch):
    labels, geometry = volume
    cache.cached_surfaces(labels, geometry)
    first = only_entry()
    entry_bytes = first.stat().st_size
    monkeypatch.setattr(cache, "CACHE_BYTES", entry_bytes * 2 + 1)
    second_labels = labels.copy()
    second_labels[0, 0, 0] = 2
    cache.cached_surfaces(second_labels, geometry)
    second = cache._directory() / f"{cache._key(second_labels, geometry)}.npz"
    os.utime(first, ns=(1_000_000_000, 1_000_000_000))
    os.utime(second, ns=(2_000_000_000, 2_000_000_000))
    cache.cached_surfaces(labels, geometry)  # Touch first: second is now the LRU.
    third_labels = labels.copy()
    third_labels[0, 0, 0] = 3
    cache.cached_surfaces(third_labels, geometry)
    assert first.exists()
    assert not second.exists()
    entries = list(cache._directory().glob("*.npz"))
    assert len(entries) == 2
    assert sum(path.stat().st_size for path in entries) <= cache.CACHE_BYTES


def test_cache_write_failure_does_not_fail_surface_generation(volume, extraction, monkeypatch):
    calls, expected = extraction

    def fail(*args, **kwargs):
        raise OSError("read-only cache")

    monkeypatch.setattr(cache.os, "replace", fail)
    assert cache.cached_surfaces(*volume)[1] is expected
    assert len(calls) == 1
    assert list(cache._directory().glob("*.npz")) == []
    assert list(cache._directory().glob("*.tmp")) == []


def test_unavailable_cache_directory_does_not_fail_surface_generation(volume, extraction, monkeypatch):
    def fail():
        raise PermissionError("cache is unavailable")

    monkeypatch.setattr(cache, "_directory", fail)
    assert cache.cached_surfaces(*volume)[1] is extraction[1]


def test_interrupted_write_is_removed_under_writer_lock(volume, extraction):
    orphan = cache._directory() / f".{64 * 'a'}.unfinished.tmp"
    orphan.write_bytes(b"interrupted derived cache")
    cache.cached_surfaces(*volume)
    assert not orphan.exists()
