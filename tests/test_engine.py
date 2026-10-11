"""Focused geometry, foreground conservation, and motion regression checks."""

from copy import deepcopy

import numpy as np
import pytest
import SimpleITK as sitk

from bimax import engine
from bimax.model import Document, Geometry, Motion, new_scenario


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



def configure(doc, key, geometry):
    doc.cuts[key].geometry = geometry
    if key not in doc.scenario.cut_ids:
        doc.scenario.cut_ids.append(key)


def plane(origin, normal):
    return {"origin": origin, "normal": normal}


def bounded(origin, normal, minimum=(-100, -100, -100), maximum=(100, 100, 100)):
    return {"plane": plane(origin, normal),
            "region": {"min": minimum, "max": maximum}}


def upper_cut(origin, normal, minimum=(-100, -100, -100), maximum=(100, 100, 100)):
    return {
        "planes": {"upper": plane(origin, normal)},
        "groups": [["upper"]],
        "region": {"min": minimum, "max": maximum},
    }


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
    configure(doc, "lefort1", upper_cut([0, 0, 3], [0, 0, -1], [-3, -2, -1], [0, 10, 20]))
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
    doc.review.confirmations["upper:lefort1"] = doc.geometry.index_to_world(zyx[::-1]).tolist()
    np.testing.assert_array_equal(engine.build_fragments(doc), fragments)


def test_bsso_and_genioplasty_own_expected_sides_and_require_valid_seeds():
    doc = document(frame=np.eye(4))
    doc.anatomy.fill(3)
    configure(doc, "bsso", {
        "midline": 4,
        "left": {"front": plane([0, 4, 0], [0, -1, 0]),
                 "upper": plane([0, 0, 6], [0, 0, 1]),
                 "sagittal": plane([1, 0, 0], [-1, 0, 0])},
        "right": {"front": plane([0, 4, 0], [0, -1, 0]),
                  "upper": plane([0, 0, 6], [0, 0, 1]),
                  "sagittal": plane([7, 0, 0], [1, 0, 0])},
    })
    configure(doc, "genio", bounded([0, 0, 1], [0, 0, -1], [0, 5, 0], [8, 8, 8]))
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
    doc.review.confirmations = {"left-proximal": [0, 2, 3], "right-proximal": [8, 2, 3], "mandible": [4, 7, 3], "chin": [4, 7, 0]}
    np.testing.assert_array_equal(engine.build_fragments(doc), fragments)
    doc.review.confirmations["right-proximal"] = [0, 2, 3]
    with pytest.raises(ValueError, match="Right proximal.*outside"):
        engine.build_fragments(doc)
    # One side changes independently, without touching the opposite fragment.
    changed = deepcopy(doc.cuts)
    changed["bsso"].geometry["left"]["front"]["origin"][1] = 1
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
    doc.scenario.cut_ids.append("lefort1")
    doc.scenario.cut_ids.append("genio")
    doc.motions["linked"].translation = [0, 10, 0]
    doc.motions["lefort1"].translation = [0, 3, -4]
    doc.motions["mandible"].translation = [0, 5, 0]
    doc.motions["chin"].translation = [0, 2, 0]
    transforms = engine.fragment_transforms(doc)
    np.testing.assert_array_equal(transforms[2][:3, 3], [0, -13, -4])
    np.testing.assert_array_equal(transforms[5][:3, 3], [0, -15, 0])
    np.testing.assert_array_equal(transforms[6][:3, 3], [0, -17, 0])
    for label in (1, 3, 4, 7, 8):
        np.testing.assert_array_equal(transforms.get(label, np.eye(4)), np.eye(4))
    # Absolute recomputation must be stable and may never compound actor motion.
    again = engine.fragment_transforms(doc)
    for label in transforms:
        np.testing.assert_array_equal(again[label], transforms[label])


@pytest.mark.parametrize("key, operation, label", [("chin", "genio", 6), ("lefort1", "lefort1", 2)])
def test_disabled_operations_ignore_retained_invalid_motion(key, operation, label):
    doc = document()
    doc.motions[key].angles = [15, 0, 0]
    doc.motions[key].pivot_set = False
    doc.motions["linked"].translation = [0, 3, 0]
    doc.motions["mandible"].translation = [0, 2, 0]
    transforms = engine.fragment_transforms(doc)
    np.testing.assert_array_equal(transforms.get(label, np.eye(4)), np.eye(4))
    np.testing.assert_array_equal(transforms[5][:3, 3], [0, -5, 0])
    assert doc.motions[key].angles == [15, 0, 0], "Omitting an operation preserves scenario parameters"
    doc.scenario.cut_ids.append(operation)
    with pytest.raises(ValueError, match="visible rotation pivot"):
        engine.fragment_transforms(doc)


def test_shared_cuts_and_scenarios_isolate_motion_and_share_baseline_arrays():
    doc = document()
    baseline = (doc.source, doc.anatomy, doc.seeds, doc.barriers)
    doc.scenarios["LF III"] = new_scenario(["lefort3"])
    doc.motions["linked"].translation[1] = 4
    first_key = doc.configuration_key
    doc.active_scenario = "LF III"
    assert doc.motions["linked"].translation == [0, 0, 0]
    assert doc.fragment_names[2] == "Midface"
    assert doc.configuration_key != first_key
    for actual, original in zip((doc.source, doc.anatomy, doc.seeds, doc.barriers), baseline, strict=True):
        assert actual is original
    key = doc.configuration_key
    doc.motions["lefort3"].translation[1] = 3
    doc.revision += 1
    assert doc.configuration_key == key
    doc.cuts["lefort3"].geometry["planes"]["left_oblique"]["origin"][2] += 1
    assert doc.configuration_key != key


def test_templates_have_distinct_central_and_lateral_envelopes():
    doc = document(np.ones((21, 21, 21), dtype=np.uint8), frame=np.eye(4))
    doc.anatomy.fill(2)
    # Centre, orbital-floor valley, lateral wing at the same height.
    points = [[10, 19, 12], [15, 19, 12], [19, 19, 12]]
    for level, expected in ((1, [1, 1, 1]), (2, [2, 1, 1]), (3, [2, 1, 2])):
        doc.scenario.cut_ids = [f"lefort{level}"]
        np.testing.assert_array_equal(engine.fragment_labels_at_points(doc, points), expected)
    # The LF III valley lies between two selected points: the envelope is
    # deliberately nonconvex, unlike a single plane or one convex intersection.


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("genio", [False, True])
def test_all_operation_variants_conserve_bone_and_match_exact_point_predicates(level, genio):
    doc = document(np.ones((21, 21, 21), dtype=np.uint8))
    doc.anatomy[:9] = 3
    doc.anatomy[9:] = 2
    # Raise the LF I template enough to intersect this synthetic upper jaw.
    doc.scenario.cut_ids = [f"lefort{level}", "bsso"]
    if level == 1:
        doc.cuts[f"lefort{level}"].geometry["planes"]["upper"]["origin"][2] = 11
    if genio:
        doc.scenario.cut_ids.append("genio")
    fragments = engine.build_fragments(doc, validate=False)
    np.testing.assert_array_equal(fragments != 0, doc.source != 0)
    np.testing.assert_array_equal(np.array([0, 2, 2, 3, 3, 3, 3, 4, 1])[fragments], doc.anatomy)
    assert bool(np.any(fragments == 6)) == genio
    required = {2, 3, 4, 5} | ({6} if genio else set())
    assert engine.required_fragment_labels(doc) == required
    for label in required:
        xyz = np.argwhere(fragments == label)[0][::-1]
        doc.review.confirmations[doc.fragment_descriptors[label].id] = doc.world_to_plan(doc.geometry.index_to_world(xyz)).tolist()
    np.testing.assert_array_equal(engine.build_fragments(doc), fragments)
    xyz = np.indices(doc.source.shape).reshape(3, -1)[::-1].T
    points = doc.world_to_plan(doc.geometry.index_to_world(xyz))
    np.testing.assert_array_equal(engine.fragment_labels_at_points(doc, points), fragments.ravel())


def test_point_membership_rounds_to_voxel_centres_and_handles_outside():
    doc = document(frame=np.eye(4), spacing=(0.4, 0.7, 1.2))
    doc.anatomy.fill(2)
    configure(doc, "lefort1", upper_cut([0, 0, 3], [0, 0, -1]))
    points = [[0, 0, 2.99], [0, 0, 3.01], [-100, 0, 0], [1e30, 0, 0]]
    # 2.99 and 3.01 straddle the cut but round to z=2.4 and z=3.6.
    np.testing.assert_array_equal(engine.fragment_labels_at_points(doc, points), [2, 1, 0, 0])
    assert engine.fragment_labels_at_points(doc, np.empty((0, 3))).size == 0
    with pytest.raises(ValueError, match="finite Nx3"):
        engine.fragment_labels_at_points(doc, [[float("nan"), 0, 0]])


@pytest.mark.parametrize("level", [0, 4, True, 1.0, "II"])
def test_default_cuts_reject_invalid_lefort_level(level):
    with pytest.raises(ValueError, match="Le Fort level"):
        engine.default_cut_geometry(document(), level)


@pytest.mark.parametrize("groups", [[], [[]], [["missing"]], [[42]]])
def test_invalid_upper_cut_groups_fail_clearly(groups):
    doc = document()
    doc.anatomy.fill(2)
    doc.scenario.cut_ids.append("lefort1")
    doc.cuts["lefort1"].geometry["groups"] = groups
    with pytest.raises(ValueError, match="plane groups"):
        engine.build_fragments(doc, validate=False)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("layout", ["dense", "sparse", "empty_slabs"])
@pytest.mark.parametrize("oblique", [False, True])
def test_foreground_cutting_matches_dense_grid_reference(level, layout, oblique, monkeypatch):
    source = np.ones((12, 14, 16), dtype=np.uint8)
    if layout == "sparse":
        source.fill(0)
        source[::3, ::4, ::5] = 1
    elif layout == "empty_slabs":
        source[2:8] = 0
    doc = document(source, spacing=(0.4, 0.7, 1.2))
    if oblique:
        doc.geometry.direction = tuple(np.array([
            [0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, -1],
        ]).ravel())
        doc.geometry.origin = (15.0, -23.0, 8.0)
        doc.frame[:3, :3] = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
        doc.frame[:3, 3] = [4, -8, 2]
    doc.anatomy[:6] = source[:6] * 3
    doc.anatomy[6:] = source[6:] * 2
    doc.anatomy[0, 0, 0] = 4
    doc.anatomy[-1, -1, -1] = int(source[-1, -1, -1])
    doc.scenario.cut_ids = [f"lefort{level}", "bsso"]
    doc.scenario.cut_ids.append("genio")
    # A deliberately tiny slab budget exercises empty slabs and slab offsets.
    monkeypatch.setattr(engine, "CUT_SLAB_VOXELS", source.shape[1] * source.shape[2] * 2)
    affine = doc.frame @ doc.geometry.affine
    z, y, x = np.ogrid[:source.shape[0], :source.shape[1], :source.shape[2]]
    points = tuple(
        affine[axis, 0] * x + affine[axis, 1] * y + affine[axis, 2] * z + affine[axis, 3]
        for axis in range(3)
    )
    expected = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[doc.anatomy]
    engine._cut_fragment_labels(expected, points, doc)
    original = doc.source.copy()
    messages, visited = [], []
    cut_labels = engine._cut_fragment_labels

    def only_foreground(labels, coordinates, plan):
        assert labels.ndim == 1
        assert labels.size <= engine.CUT_SLAB_VOXELS
        assert not np.any(labels == 0), "Background must not reach expensive plane predicates"
        assert all(values.shape == labels.shape for values in coordinates)
        visited.append(labels.size)
        cut_labels(labels, coordinates, plan)

    monkeypatch.setattr(engine, "_cut_fragment_labels", only_foreground)
    actual = engine.build_fragments(doc, validate=False, progress=lambda *value: messages.append(value))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(doc.source, original)
    assert sum(visited) == np.count_nonzero(doc.source)
    assert messages[-1] == (1.0, "Fragment preview ready")
    assert len([message for _, message in messages if message == "Partitioning bone fragments"]) == 6


def test_cut_foreground_workspaces_remain_bounded_with_wide_noncontiguous_input(monkeypatch):
    doc = document(np.ones((2, 14, 16), dtype=np.uint8), frame=np.eye(4))
    doc.source = doc.source[:, :, ::-1]
    doc.anatomy = (doc.source * 2)[:, :, ::-1]
    assert not doc.anatomy.flags.c_contiguous
    configure(doc, "lefort1", upper_cut([0, 0, 0.5], [0, 0, -1]))
    # An entire XY plane is much larger than the deliberately tiny budget.
    monkeypatch.setattr(engine, "CUT_SLAB_VOXELS", 31)
    sizes = []
    classify = engine._cut_fragment_labels

    def bounded(labels, points, plan):
        sizes.append(labels.size)
        assert labels.size <= 31
        classify(labels, points, plan)

    monkeypatch.setattr(engine, "_cut_fragment_labels", bounded)
    fragments = engine.build_fragments(doc, validate=False)
    np.testing.assert_array_equal(fragments[0], 2)
    np.testing.assert_array_equal(fragments[1], 1)
    assert sum(sizes) == doc.source.size
