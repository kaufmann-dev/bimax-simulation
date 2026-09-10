"""Accept a trusted worker preview using cached counts and isolated voxel samples."""

from copy import deepcopy

import numpy as np
import pytest

from bimax.engine import build_fragments, label_counts, validate_fragment_preview
from bimax.model import Document, Geometry


@pytest.fixture
def preview():
    source = np.ones((6, 4, 4), dtype=np.uint8)
    anatomy = np.where(np.indices(source.shape)[0] >= 3, 2, 3).astype(np.uint8)
    doc = Document(
        Geometry(source.shape, (0.4, 0.7, 1.2), (1.0, 2.0, 3.0), tuple(np.eye(3).ravel())),
        source,
        anatomy,
        np.zeros_like(source),
        np.zeros_like(source),
        frame=np.eye(4),
    )
    doc.cuts = {
        "lefort": {
            "enabled": True,
            "plane": {"origin": [0, 0, 8], "normal": [0, 0, -1]},
            "region": {"min": [-100, -100, -100], "max": [100, 100, 100]},
        },
        "bsso": {
            "enabled": True,
            "midline": 1.6,
            "left": {
                "front": {"origin": [0, 3, 0], "normal": [0, -1, 0]},
                "upper": {"origin": [0, 0, 0], "normal": [0, 0, 1]},
                "sagittal": {"origin": [0, 0, 0], "normal": [-1, 0, 0]},
            },
            "right": {
                "front": {"origin": [0, 3, 0], "normal": [0, -1, 0]},
                "upper": {"origin": [0, 0, 0], "normal": [0, 0, 1]},
                "sagittal": {"origin": [0, 0, 0], "normal": [1, 0, 0]},
            },
        },
    }
    fragments = build_fragments(doc, validate=False)
    for label in (2, 3, 4, 5):
        xyz = np.argwhere(fragments == label)[0][::-1]
        doc.confirmations[str(label)] = doc.world_to_plan(doc.geometry.index_to_world(xyz)).tolist()
    return doc, fragments, label_counts(fragments, 8), label_counts(anatomy, 4)


def test_acceptance_checks_only_cached_counts_and_confirmation_samples(preview, monkeypatch):
    class SamplesOnly(np.ndarray):
        def __getitem__(self, key):
            assert isinstance(key, tuple) and len(key) == 3
            assert all(isinstance(index, (int, np.integer)) for index in key)
            return super().__getitem__(key)

        def __array_ufunc__(self, *args, **kwargs):
            raise AssertionError("Acceptance must not scan a full volume.")

        def __array_function__(self, *args, **kwargs):
            raise AssertionError("Acceptance must not scan a full volume.")

        def reshape(self, *args, **kwargs):
            raise AssertionError("Acceptance must not scan a full volume.")

        def copy(self, *args, **kwargs):
            raise AssertionError("Acceptance must not copy a full volume.")

    doc, fragments, fragments_count, anatomy_count = preview
    doc.source = doc.source.view(SamplesOnly)
    doc.anatomy = doc.anatomy.view(SamplesOnly)
    fragments = fragments.view(SamplesOnly)

    def no_counting(*args, **kwargs):
        raise AssertionError("Acceptance must use cached counts.")

    monkeypatch.setattr("bimax.engine.label_counts", no_counting)
    validate_fragment_preview(doc, fragments, fragments_count, anatomy_count)


def test_acceptance_reports_missing_confirmation(preview):
    doc, *args = preview
    del doc.confirmations["2"]
    with pytest.raises(ValueError, match="Place a confirmation point inside Maxilla"):
        validate_fragment_preview(doc, *args)


@pytest.mark.parametrize("point", [[1, 2, 3], [-100, 0, 0], [1e30, 0, 0]])
def test_acceptance_reports_wrong_or_outside_confirmation(preview, point):
    doc, *args = preview
    doc.confirmations["2"] = point
    with pytest.raises(ValueError, match="confirmation point for Maxilla is outside"):
        validate_fragment_preview(doc, *args)


@pytest.mark.parametrize("point", [[float("nan"), 0, 0], [0, float("inf"), 0], [1, 2]])
def test_acceptance_rejects_nonfinite_or_invalid_confirmation(preview, point):
    doc, *args = preview
    doc.confirmations["2"] = point
    with pytest.raises(ValueError, match="Fragment confirmation must contain three finite"):
        validate_fragment_preview(doc, *args)


def test_acceptance_reports_empty_enabled_fragment(preview):
    doc, fragments, counts, anatomy_count = preview
    counts[1] += counts[2]
    counts[2] = 0
    with pytest.raises(ValueError, match="cuts leave Maxilla empty"):
        validate_fragment_preview(doc, fragments, counts, anatomy_count)


def test_acceptance_rejects_unassigned_anatomy(preview):
    doc, fragments, counts, anatomy_count = preview
    anatomy_count[1], anatomy_count[2] = 1, anatomy_count[2] - 1
    with pytest.raises(ValueError, match="Assign all bone to anatomy"):
        validate_fragment_preview(doc, fragments, counts, anatomy_count)


def test_acceptance_rejects_wrong_preview_shape_or_dtype(preview):
    doc, fragments, counts, anatomy_count = preview
    for invalid in (fragments[:-1], fragments.astype(np.float32)):
        with pytest.raises(ValueError, match="uint8 volume matching the source"):
            validate_fragment_preview(doc, invalid, counts, anatomy_count)


@pytest.mark.parametrize("kind", ["shape", "negative", "fraction", "total", "ownership"])
def test_acceptance_rejects_invalid_cached_counts(preview, kind):
    doc, fragments, counts, anatomy_count = preview
    if kind == "shape":
        counts = counts[:-1]
    elif kind == "negative":
        counts[0] = -1
    elif kind == "fraction":
        counts = counts.astype(float)
    elif kind == "total":
        counts[0] += 1
    elif kind == "ownership":
        counts[1] -= 1
        counts[5] += 1
    with pytest.raises(ValueError, match="counts are invalid|ownership does not match"):
        validate_fragment_preview(doc, fragments, counts, anatomy_count)


@pytest.mark.parametrize("invalid", ["frame", "spacing", "direction"])
def test_acceptance_rejects_invalid_coordinate_transforms(preview, invalid):
    doc, *args = preview
    if invalid == "frame":
        doc.frame[0, 0] = float("nan")
    elif invalid == "spacing":
        doc.geometry.spacing = (0.0, 1.0, 1.0)
    else:
        doc.geometry.direction = tuple(np.zeros(9))
    with pytest.raises(ValueError, match="Planning frame|invalid physical geometry"):
        validate_fragment_preview(doc, *args)


def test_cached_acceptance_matches_full_build_without_mutating_preview(preview):
    doc, fragments, counts, anatomy_count = preview
    before = deepcopy(doc.confirmations)
    validate_fragment_preview(doc, fragments, counts, anatomy_count)
    np.testing.assert_array_equal(build_fragments(doc), fragments)
    assert doc.confirmations == before
