"""Persistent shared masks, nested upper fragments, and computation-free selection."""

from copy import deepcopy

import numpy as np
import pytest
from factories import confirm, document, upper_geometry

from bimax.engine import (
    build_fragments,
    fragment_transforms,
    motion_matrix,
    required_fragment_labels,
)
from bimax.model import new_scenario
from bimax.prepared import load_geometry, load_options, prepare_geometry
from bimax.storage import load_project, save_project


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def nested_document():
    doc = document()
    for level, height in ((1, 11), (2, 14), (3, 17)):
        doc.cuts[f"lefort{level}"].geometry = upper_geometry(height)
    return doc


@pytest.mark.parametrize("ids", [["lefort1", "lefort3"], ["lefort3", "lefort2", "lefort1"]])
def test_nested_cuts_preserve_partition_and_apply_parent_relative_motion(ids):
    doc = nested_document()
    doc.scenario.cut_ids = [*ids, "bsso", "genio"]
    result = prepare_geometry(doc)
    expected = build_fragments(doc, validate=False)
    np.testing.assert_array_equal(result["fragments"], expected)
    np.testing.assert_array_equal(doc.parent_labels[expected], doc.anatomy)
    assert all(np.count_nonzero(expected == label) for label in required_fragment_labels(doc))
    confirm(doc, expected)
    doc.motions["linked"].translation = [0, 10, 0]
    doc.motions["lefort3"].translation = [0, 2, 0]
    doc.motions["lefort3"].angles = [12, 0, 0]
    doc.motions["lefort3"].pivot_set = True
    doc.motions["lefort1"].translation = [0, 3, 0]
    doc.motions["lefort2"].translation = [1, 0, 0]
    transforms = fragment_transforms(doc)
    inherited = motion_matrix(doc.motions["linked"])
    for key in reversed(doc.upper_cuts):
        inherited = inherited @ motion_matrix(doc.motions[key])
        label = doc.fragment_labels[f"upper:{key}"]
        np.testing.assert_allclose(transforms[label], np.linalg.inv(doc.frame) @ inherited @ doc.frame)
    for label in (1, 3, 4):
        np.testing.assert_array_equal(transforms[label], np.eye(4))


def test_crossing_same_level_and_empty_nested_combinations_fail():
    doc = nested_document()
    doc.scenario.cut_ids = ["lefort1", "lefort3"]
    doc.cuts["lefort3"].geometry = upper_geometry(10)
    with pytest.raises(ValueError, match="cuts cross"):
        prepare_geometry(doc)
    doc.scenario.cut_ids = ["lefort2", "modified-lefort2"]
    with pytest.raises(ValueError, match="one variant per Le Fort level"):
        prepare_geometry(doc)
    doc.scenario.cut_ids = ["lefort1", "lefort3"]
    doc.cuts["lefort3"].geometry = deepcopy(doc.cuts["lefort1"].geometry)
    with pytest.raises(ValueError, match="empty"):
        prepare_geometry(doc)


def test_more_than_two_scenarios_survive_save_reopen_without_cutting_or_meshing(tmp_path, monkeypatch):
    doc = nested_document()
    doc.scenarios.clear()
    for level in (1, 2, 3):
        for chin in (False, True):
            name = f"lf{level}-chin{chin}"
            doc.scenarios[name] = new_scenario([f"lefort{level}", "bsso", *(["genio"] if chin else [])])
            doc.active_scenario = name
            result = prepare_geometry(doc)
            confirm(doc, result["fragments"])
    path = tmp_path / "shared.bimax"
    save_project(doc, path)
    # Prove portability independently of the original local cache.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "fresh-cache"))
    def no_computation(*args, **kwargs):
        pytest.fail("Prepared selections must never cut volumes or extract meshes")
    monkeypatch.setattr("bimax.prepared.cut_membership", no_computation)
    monkeypatch.setattr("bimax.engine.cut_membership", no_computation)
    monkeypatch.setattr("bimax.engine.build_fragments", no_computation)
    monkeypatch.setattr("bimax.surfaces.make_surfaces", no_computation)
    restored = load_project(path)
    assert len(restored.cuts) == 7
    for name in restored.scenarios:
        restored.active_scenario = name
        result = load_geometry(**load_options(restored))
        np.testing.assert_array_equal(restored.parent_labels[result["fragments"]], restored.anatomy)
        assert restored.cuts_accepted


def test_upper_selection_and_motion_reuse_lower_geometry(monkeypatch):
    doc = nested_document()
    doc.scenario.cut_ids = ["lefort1", "bsso", "genio"]
    first = prepare_geometry(doc)
    lower_key = doc.subsystem_key("lower")
    calls = []
    original = __import__("bimax.prepared", fromlist=["cut_membership"]).cut_membership
    def record(points, cut):
        calls.append(cut.kind)
        return original(points, cut)
    monkeypatch.setattr("bimax.prepared.cut_membership", record)
    doc.scenario.cut_ids = ["lefort3", "bsso", "genio"]
    second = prepare_geometry(doc)
    assert doc.subsystem_key("lower") == lower_key
    assert set(calls) == {"lefort"}
    for label in (3, 4, 5, 6):
        for field in ("vertices", "faces", "normals"):
            np.testing.assert_array_equal(getattr(first["surfaces"][label], field), getattr(second["surfaces"][label], field))
    calls.clear()
    key = doc.configuration_key
    doc.motions["lefort3"].translation = [0, 5, 0]
    prepare_geometry(doc)
    assert doc.configuration_key == key
    assert calls == []


def test_cut_edit_invalidates_only_its_dependencies_and_frame_invalidates_all():
    doc = nested_document()
    doc.scenario.cut_ids = ["lefort1", "bsso", "genio"]
    prepare_geometry(doc)
    keys = {key: doc.cut_key(key) for key in doc.cuts}
    lower = doc.subsystem_key("lower")
    upper = doc.subsystem_key("upper")
    doc.cuts["lefort1"].geometry["planes"]["upper"]["origin"][2] += 0.5
    assert doc.subsystem_key("upper") != upper
    assert doc.subsystem_key("lower") == lower
    for key in doc.cuts:
        assert (doc.cut_key(key) == keys[key]) == (key != "lefort1")
    keys = {key: doc.cut_key(key) for key in doc.cuts}
    doc.frame[:3, 3] += 1
    assert all(doc.cut_key(key) != keys[key] for key in doc.cuts)


def test_corrupt_artifact_regenerates_once_and_missing_artifact_cannot_be_loaded():
    from pathlib import Path
    doc = nested_document()
    doc.scenario.cut_ids = ["lefort1", "bsso"]
    first = prepare_geometry(doc)
    ref = doc.artifacts[doc.subsystem_key("upper")]
    Path(ref.path).write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        load_geometry(**load_options(doc))
    restored = prepare_geometry(doc)
    np.testing.assert_array_equal(restored["fragments"], first["fragments"])
    del doc.artifacts[doc.configuration_key]
    with pytest.raises(KeyError):
        load_geometry(**load_options(doc))
