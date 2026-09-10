"""Focused geometry, foreground conservation, and motion regression checks."""

from copy import deepcopy

import numpy as np
import pytest
import SimpleITK as sitk

from bimax import engine
from bimax.model import Document, Geometry, Motion


def document(source=None, *, spacing=(1.0, 1.0, 1.0), frame=None):
    if source is None:
        source = np.ones((9, 9, 9), dtype=np.uint8)
    source = np.asarray(source, dtype=np.uint8)
    result = Document(
        Geometry(source.shape, spacing, (0.0, 0.0, 0.0), tuple(np.eye(3).ravel())),
        source, source.copy(), np.zeros_like(source), np.zeros_like(source, dtype=bool),
    )
    if frame is not None:
        result.frame = np.asarray(frame, dtype=float)
    result.cuts = engine.default_cuts(result)
    return result


def plane(origin, normal):
    return {"origin": origin, "normal": normal}


def bounded(origin, normal, minimum=(-100, -100, -100), maximum=(100, 100, 100)):
    return {"enabled": True, "plane": plane(origin, normal),
            "region": {"min": minimum, "max": maximum}}


def test_load_preserves_anisotropic_oblique_reflected_geometry(tmp_path):
    source = np.zeros((5, 6, 7), dtype=np.uint8)
    source[1:4, 2:5, 1:6] = 1
    image = sitk.GetImageFromArray(source)
    angle = np.deg2rad(27)
    direction = np.array([[-np.cos(angle), -np.sin(angle), 0],
                          [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    image.SetSpacing((0.4, 0.7, 1.2))
    image.SetOrigin((15.0, -7.0, 9.0))
    image.SetDirection(tuple(direction.ravel()))
    path = tmp_path / "bone.nii.gz"
    sitk.WriteImage(image, str(path))
    doc = engine.load_volume(path)
    np.testing.assert_array_equal(doc.source, source)
    assert not doc.source.flags.writeable
    np.testing.assert_allclose(doc.geometry.spacing, image.GetSpacing())
    np.testing.assert_allclose(doc.geometry.direction, direction.ravel(), atol=1e-7)
    point = np.array([2.5, 3.0, 1.2])
    np.testing.assert_allclose(doc.geometry.index_to_world(point), image.TransformContinuousIndexToPhysicalPoint(point))
    np.testing.assert_allclose(doc.geometry.world_to_index(doc.geometry.index_to_world(point)), point)
    np.testing.assert_allclose(doc.plan_to_world(doc.world_to_plan([1, 2, 3])), [1, 2, 3])


@pytest.mark.parametrize("values, message", [
    (np.zeros((3, 4, 5), np.uint8), "empty"),
    (np.full((3, 4, 5), 150, np.int16), "only background 0 and bone 1"),
    (np.full((3, 4, 5), 2, np.uint8), "only background 0 and bone 1"),
    (np.full((3, 4, 5), np.nan, np.float32), "only background 0 and bone 1"),
    (np.ones((3, 4, 5, 6), np.uint8), "three-dimensional"),
])
def test_load_rejects_invalid_inputs(tmp_path, values, message):
    path = tmp_path / "invalid.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(values, isVector=False), str(path))
    with pytest.raises(ValueError, match=message):
        engine.load_volume(path)


def test_seeded_watershed_preserves_bridge_and_uses_physical_spacing(monkeypatch):
    mask = np.zeros((9, 11, 19), dtype=np.uint8)
    mask[2:7, 2:9, 1:8] = 1
    mask[2:7, 2:9, 11:18] = 1
    mask[4, 5, 8:11] = 1
    doc = document(mask, spacing=(0.4, 0.8, 1.5))
    doc.source.flags.writeable = False
    doc.seeds[4, 5, 4] = 2
    doc.seeds[4, 5, 14] = 3
    original = doc.source.copy()
    measured = []
    actual = sitk.SignedMaurerDistanceMap

    def record(image, *args, **kwargs):
        measured.append((image.GetSpacing(), kwargs["useImageSpacing"]))
        return actual(image, *args, **kwargs)

    monkeypatch.setattr(engine.sitk, "SignedMaurerDistanceMap", record)
    proposal = engine.separate(doc)
    assert measured == [((0.4, 0.8, 1.5), True)]
    assert proposal[4, 5, 4] == 2
    assert proposal[4, 5, 14] == 3
    assert set(np.unique(proposal)) == {0, 2, 3}
    np.testing.assert_array_equal(doc.source, original)
    np.testing.assert_array_equal(doc.anatomy, original)
    engine.assert_partition(original, proposal, maximum_label=4)


def test_barriers_and_unseeded_islands_remain_unassigned():
    mask = np.zeros((5, 7, 15), dtype=np.uint8)
    mask[2, 3, 1:12] = 1
    mask[0, 0, 14] = 1
    doc = document(mask)
    doc.seeds[2, 3, 2] = 2
    doc.seeds[2, 3, 10] = 3
    doc.barriers[2, 3, 6] = True
    proposal = engine.separate(doc)
    assert np.all(proposal[2, 3, 1:6] == 2)
    assert np.all(proposal[2, 3, 7:12] == 3)
    assert proposal[2, 3, 6] == 1
    assert proposal[0, 0, 14] == 1
    engine.assert_partition(mask, proposal)


def test_roi_preserves_accepted_assignments_outside_it():
    doc = document()
    doc.anatomy.fill(4)
    doc.seeds[3, 3, 3] = 2
    roi = (slice(2, 5), slice(2, 5), slice(2, 5))
    result = engine.separate(doc, roi)
    expected = doc.anatomy.copy()
    expected[roi] = 2
    np.testing.assert_array_equal(result, expected)


def test_component_selection_is_face_connected_and_returns_global_flat_indices():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1, 1, 1:3] = 1
    mask[2, 2, 2] = 1  # Corner/edge adjacency must not join the first component.
    doc = document(mask)
    expected = np.ravel_multi_index(([1, 1], [1, 1], [1, 2]), mask.shape)
    np.testing.assert_array_equal(engine.component_indices(doc, [1, 1, 1]), expected)
    assert engine.component_indices(doc, [0, 0, 0]).size == 0


def test_memory_budget_fails_before_large_workspace_allocation(monkeypatch):
    doc = document()
    monkeypatch.setattr(engine, "MEMORY_BUDGET", 1)
    with pytest.raises(ValueError, match="smaller region"):
        engine.separate(doc)


def test_lefort_plane_and_box_use_physical_geometry_and_preserve_partition():
    doc = document(frame=np.eye(4), spacing=(0.4, 0.7, 1.2))
    doc.anatomy.fill(2)
    angle = np.deg2rad(30)
    doc.geometry.direction = tuple(np.array([[-np.cos(angle), -np.sin(angle), 0],
                                             [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]).ravel())
    doc.cuts["lefort"] = bounded([0, 0, 3], [0, 0, -1], [-3, -2, -1], [0, 10, 20])
    fragments = engine.build_fragments(doc, validate=False)
    z, y, x = np.indices(doc.source.shape)
    xyz = np.stack((x, y, z), axis=-1)
    physical = doc.geometry.index_to_world(xyz)
    selected = ((physical[..., 2] <= 3) & (physical[..., 0] >= -3)
                & (physical[..., 0] <= 0) & (physical[..., 1] >= -2))
    np.testing.assert_array_equal(fragments, np.where(selected, 2, 1))
    engine.assert_partition(doc.source, fragments)
    with pytest.raises(ValueError, match="confirmation point"):
        engine.build_fragments(doc)
    zyx = np.argwhere(fragments == 2)[0]
    doc.confirmations["2"] = doc.geometry.index_to_world(zyx[::-1]).tolist()
    np.testing.assert_array_equal(engine.build_fragments(doc), fragments)


def test_bsso_and_genioplasty_own_expected_sides_and_require_valid_seeds():
    doc = document(frame=np.eye(4))
    doc.anatomy.fill(3)
    doc.cuts["bsso"] = {
        "enabled": True, "midline": 4,
        "left": {"front": plane([0, 4, 0], [0, -1, 0]),
                 "upper": plane([0, 0, 6], [0, 0, 1]),
                 "sagittal": plane([1, 0, 0], [-1, 0, 0])},
        "right": {"front": plane([0, 4, 0], [0, -1, 0]),
                  "upper": plane([0, 0, 6], [0, 0, 1]),
                  "sagittal": plane([7, 0, 0], [1, 0, 0])},
    }
    doc.cuts["genio"] = bounded([0, 0, 1], [0, 0, -1], [0, 5, 0], [8, 8, 8])
    fragments = engine.build_fragments(doc, validate=False)
    assert fragments[3, 2, 0] == 3
    assert fragments[7, 2, 3] == 3
    assert fragments[3, 2, 8] == 4
    assert fragments[7, 2, 5] == 4
    assert fragments[3, 7, 4] == 5
    assert fragments[0, 7, 4] == 6
    assert fragments[0, 2, 4] == 5
    assert set(np.unique(fragments)) == {3, 4, 5, 6}
    engine.assert_partition(doc.source, fragments)
    doc.confirmations = {"3": [0, 2, 3], "4": [8, 2, 3], "5": [4, 7, 3], "6": [4, 7, 0]}
    np.testing.assert_array_equal(engine.build_fragments(doc), fragments)
    doc.confirmations["4"] = [0, 2, 3]
    with pytest.raises(ValueError, match="Right proximal.*outside"):
        engine.build_fragments(doc)
    # One side changes independently, without touching the opposite fragment.
    changed = deepcopy(doc.cuts)
    changed["bsso"]["left"]["front"]["origin"][1] = 1
    doc.cuts = changed
    result = engine.build_fragments(doc, validate=False)
    np.testing.assert_array_equal(result[:, :, 4:], fragments[:, :, 4:])
    assert result[3, 2, 0] == 5


def test_partition_preserves_unassigned_and_excluded_material():
    doc = document()
    doc.anatomy[0] = 4
    doc.anatomy[1] = 2
    doc.anatomy[2] = 3
    result = engine.build_fragments(doc, validate=False)
    assert set(np.unique(result)) == {1, 5, 7, 8}
    engine.assert_partition(doc.source, result)
    with pytest.raises(ValueError, match="Assign all bone"):
        engine.build_fragments(doc)
    result[3, 3, 3] = 0
    with pytest.raises(ValueError, match="preserve every"):
        engine.assert_partition(doc.source, result)


def test_motion_pitch_sign_pivot_and_rotation_order():
    anterior = np.array([0, 1, 0, 1])
    np.testing.assert_allclose(engine.motion_matrix(Motion(angles=[90, 0, 0], pivot_set=True)) @ anterior,
                               [0, 0, 1, 1], atol=1e-15)
    np.testing.assert_allclose(engine.motion_matrix(Motion(angles=[-90, 0, 0], pivot_set=True)) @ anterior,
                               [0, 0, -1, 1], atol=1e-15)
    motion = Motion(translation=[1, 5, -3], angles=[23, -12, 41], pivot=[2, 9, -4], pivot_set=True)
    transform = engine.motion_matrix(motion)
    np.testing.assert_allclose(transform @ [*motion.pivot, 1],
                               [*(np.array(motion.pivot) + motion.translation), 1])
    rotation = (engine.motion_matrix(Motion(angles=[0, 0, 41], pivot_set=True))
                @ engine.motion_matrix(Motion(angles=[0, -12, 0], pivot_set=True))
                @ engine.motion_matrix(Motion(angles=[23, 0, 0], pivot_set=True)))
    np.testing.assert_allclose(transform[:3, :3], rotation[:3, :3])


def test_rotation_requires_an_explicit_pivot_but_translation_does_not():
    with pytest.raises(ValueError, match="visible rotation pivot"):
        engine.motion_matrix(Motion(angles=[1, 0, 0]))
    np.testing.assert_array_equal(engine.motion_matrix(Motion(translation=[0, 4, -2]))[:3, 3], [0, 4, -2])


def test_fragment_hierarchy_inherits_mandible_once_in_lps():
    doc = document()
    doc.motions["linked"].translation = [0, 10, 0]
    doc.motions["maxilla"].translation = [0, 3, -4]
    doc.motions["mandible"].translation = [0, 5, 0]
    doc.motions["chin"].translation = [0, 2, 0]
    transforms = engine.fragment_transforms(doc)
    np.testing.assert_array_equal(transforms[2][:3, 3], [0, -13, -4])
    np.testing.assert_array_equal(transforms[5][:3, 3], [0, -15, 0])
    np.testing.assert_array_equal(transforms[6][:3, 3], [0, -17, 0])
    for label in (1, 3, 4, 7, 8):
        np.testing.assert_array_equal(transforms[label], np.eye(4))
    # Absolute recomputation must be stable and may never compound actor motion.
    again = engine.fragment_transforms(doc)
    for label in transforms:
        np.testing.assert_array_equal(again[label], transforms[label])
