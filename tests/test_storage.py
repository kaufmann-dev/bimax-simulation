import json
import zipfile
from copy import deepcopy

import numpy as np
import pytest
import SimpleITK as sitk

from bimax.engine import build_fragments, default_cuts
from bimax.model import Document, Geometry, new_scenario
from bimax.storage import export_labelmaps, load_project, save_project


def phantom(reflected=False):
    direction = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1.0]])
    if reflected:
        direction[:, 0] *= -1
    geometry = Geometry(
        (12, 14, 18), (0.5, 0.75, 1.25), (12.5, -30.0, 8.0), tuple(direction.ravel())
    )
    expected = np.zeros(geometry.shape, dtype=np.uint8)
    expected[2:5, 3:6, 2:5] = 1
    expected[2:5, 3:6, 9:12] = 2
    expected[6:9, 5:8, 5:8] = 5
    expected[9:11, 5:8, 5:8] = 6
    expected[2, 10, 14] = 7
    anatomy = np.array([0, 2, 2, 3, 3, 3, 3, 4, 1], dtype=np.uint8)[expected]
    doc = Document(
        geometry=geometry,
        source=(expected > 0).astype(np.uint8),
        anatomy=anatomy,
        seeds=np.zeros_like(expected),
        barriers=np.zeros_like(expected),
        source_name="original-bone.nii.gz",
    )
    doc.plan.cuts = default_cuts(doc)
    matrix = doc.frame @ doc.geometry.affine

    def plane(origin, normal):
        return {
            "origin": (matrix[:3, :3] @ origin + matrix[:3, 3]).tolist(),
            "normal": (np.linalg.inv(matrix[:3, :3]).T @ normal).tolist(),
        }

    doc.plan.cuts["lefort"] = {
        "enabled": True,
        "planes": {"upper": plane([7, 0, 0], [1, 0, 0])},
        "groups": [["upper"]],
        "region": {"min": [-100.0] * 3, "max": [100.0] * 3},
    }
    doc.plan.cuts["genio"] = {
        "enabled": True,
        "plane": plane([0, 0, 8.5], [0, 0, 1]),
        "region": {"min": [-100.0] * 3, "max": [100.0] * 3},
    }
    doc.plan.confirmations = {
        str(label): doc.world_to_plan(geometry.index_to_world(xyz)).tolist()
        for label, xyz in ((2, [9, 3, 2]), (5, [5, 5, 6]), (6, [5, 5, 9]))
    }
    doc.plan.accepted = True
    doc.fragments = build_fragments(doc)
    doc.fragments_key = doc.configuration_key
    np.testing.assert_array_equal(doc.fragments, expected)
    return doc


@pytest.mark.parametrize("reflected", [False, True])
def test_project_roundtrip_preserves_shared_anatomy_plans_and_scenarios(tmp_path, reflected):
    doc = phantom(reflected)
    doc.seeds[2, 3, 2] = 2
    doc.barriers[2, 3, 3] = 1
    for level in (2, 3):
        plan_id = f"lf{level}"
        doc.plans[plan_id] = deepcopy(doc.plan)
        doc.plans[plan_id].name = f"Le Fort {level} without chin"
        doc.plans[plan_id].lefort_level = level
        doc.plans[plan_id].cuts["genio"]["enabled"] = False
        doc.plans[plan_id].confirmations.pop("6")
        doc.plans[plan_id].revision = level
        doc.scenarios[f"Le Fort {level}"] = new_scenario(plan_id)
    doc.scenarios["Advance"] = new_scenario("plan1")
    doc.active_scenario = "Advance"
    doc.scenario.landmarks = [{"name": "Maxillary reference", "point": [1.2, 3.4, 5.6], "fragment": 2}]
    doc.motions["linked"].translation = [1.0, 6.0, -2.0]
    doc.motions["mandible"].angles = [12.0, 3.0, -4.0]
    doc.motions["chin"].pivot = [3.0, -4.0, 5.0]
    doc.motions["chin"].pivot_set = True
    doc.revision = 17
    doc.anatomy_revision = 4
    doc.fragments_key = doc.configuration_key
    path = tmp_path / "plan.bimax"
    save_project(doc, path)
    restored = load_project(path)
    assert restored.geometry == doc.geometry
    for name in ("source", "anatomy", "seeds", "barriers", "frame"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(doc, name))
    for name in ("plans", "scenarios", "active_scenario", "anatomy_revision", "source_name", "revision"):
        assert getattr(restored, name) == getattr(doc, name)
    assert restored.fragments is None
    assert restored.fragments_key is None
    assert not restored.source.flags.writeable
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {
            "project.json", "volumes/source.nii.gz", "volumes/anatomy.nii.gz",
            "volumes/seeds.nii.gz", "volumes/barriers.nii.gz",
        }
        assert json.loads(archive.read("project.json"))["version"] == 2
    restored.motions["linked"].translation[0] = 100
    restored.scenario.landmarks[0]["point"][0] = 100
    assert restored.scenarios["Scenario 1"].motions["linked"].translation == [0.0, 0.0, 0.0]
    assert restored.scenarios["Scenario 1"].landmarks == []
    restored.plans["lf2"].cuts["lefort"]["planes"]["upper"]["origin"][0] = 100
    assert restored.plans["lf3"].cuts == doc.plans["lf3"].cuts
    assert doc.motions["linked"].translation == [1.0, 6.0, -2.0]


def test_unfinished_project_can_save_without_fragments(tmp_path):
    doc = phantom()
    doc.fragments = None
    doc.fragments_key = None
    doc.plan.accepted = False
    doc.plan.confirmations.clear()
    doc.anatomy[doc.source != 0] = 1
    save_project(doc, tmp_path / "unfinished.bimax")
    restored = load_project(tmp_path / "unfinished.bimax")
    assert restored.fragments is None
    assert not restored.plan.accepted
    np.testing.assert_array_equal(restored.anatomy, doc.anatomy)


@pytest.mark.parametrize("corruption", ["outside", "missing", "parent", "seed", "nonbinary", "stale", "unreviewed"])
def test_invalid_volume_partition_cannot_replace_saved_project(tmp_path, corruption):
    doc = phantom()
    path = tmp_path / "plan.bimax"
    save_project(doc, path)
    original = path.read_bytes()
    if corruption == "outside":
        doc.anatomy[0, 0, 0] = 2
    elif corruption == "missing":
        doc.fragments[2, 3, 2] = 0
    elif corruption == "parent":
        doc.fragments[2, 3, 2] = 5
    elif corruption == "seed":
        doc.seeds[2, 3, 2] = 1
    elif corruption == "stale":
        doc.plan.revision += 1
    elif corruption == "unreviewed":
        doc.plan.accepted = False
    else:
        doc.source[2, 3, 2] = 2
    with pytest.raises(ValueError):
        save_project(doc, path)
    assert path.read_bytes() == original


def rewrite_archive(path, transform_metadata=None, extra=None):
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if transform_metadata:
        metadata = json.loads(members["project.json"])
        transform_metadata(metadata)
        members["project.json"] = json.dumps(metadata).encode()
    if extra:
        members.update(extra)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


@pytest.mark.parametrize("corruption", ["traversal", "nan", "geometry", "motion", "landmark", "plan_ref", "active", "level", "reviews", "plane", "group", "empty_group", "old_version", "fragments"])
def test_malformed_archive_is_rejected(tmp_path, corruption):
    path = tmp_path / "plan.bimax"
    save_project(phantom(), path)
    if corruption == "traversal":
        rewrite_archive(path, extra={"../outside": b"unsafe"})
    elif corruption == "fragments":
        rewrite_archive(path, extra={"volumes/fragments.nii.gz": b"derived geometry is forbidden"})
    else:
        changes = {
            "nan": lambda m: m["frame"][0].__setitem__(0, float("nan")),
            "geometry": lambda m: m["geometry"]["shape"].__setitem__(0, 99),
            "motion": lambda m: m["scenarios"]["Scenario 1"]["motions"]["linked"].pop("pivot"),
            "landmark": lambda m: m["scenarios"]["Scenario 1"]["landmarks"].append({"name": "Broken"}),
            "plan_ref": lambda m: m["scenarios"]["Scenario 1"].__setitem__("plan_id", "missing"),
            "active": lambda m: m.__setitem__("active_scenario", "missing"),
            "level": lambda m: m["plans"]["plan1"].__setitem__("lefort_level", 4),
            "reviews": lambda m: m["plans"]["plan1"]["confirmations"].pop("2"),
            "plane": lambda m: m["plans"]["plan1"]["cuts"]["lefort"]["planes"]["upper"].__setitem__("normal", [0, 0, 0]),
            "group": lambda m: m["plans"]["plan1"]["cuts"]["lefort"].__setitem__("groups", [["missing"]]),
            "empty_group": lambda m: m["plans"]["plan1"]["cuts"]["lefort"].__setitem__("groups", [[]]),
            "old_version": lambda m: m.__setitem__("version", 1),
        }
        rewrite_archive(path, changes[corruption])
    with pytest.raises(ValueError):
        load_project(path)
    assert not (tmp_path / "outside").exists()


def test_version_one_rejection_explains_new_format(tmp_path):
    path = tmp_path / "old.bimax"
    save_project(phantom(), path)
    rewrite_archive(path, lambda metadata: metadata.__setitem__("version", 1))
    with pytest.raises(ValueError, match="version 2 project"):
        load_project(path)


def test_inactive_accepted_plan_reviews_are_validated_without_building_volumes(tmp_path, monkeypatch):
    import bimax.engine

    doc = phantom()
    doc.plans["inactive"] = deepcopy(doc.plan)
    doc.plans["inactive"].name = "Inactive plan"
    doc.fragments = None
    doc.fragments_key = None
    path = tmp_path / "plan.bimax"

    def fail(*args, **kwargs):
        pytest.fail("Saving or loading must not regenerate full fragment volumes")

    monkeypatch.setattr(bimax.engine, "build_fragments", fail)
    save_project(doc, path)
    restored = load_project(path)
    assert restored.fragments is None
    rewrite_archive(
        path,
        lambda metadata: metadata["plans"]["inactive"]["confirmations"].__setitem__("2", [999, 999, 999]),
    )
    with pytest.raises(ValueError, match="Inactive plan.*outside"):
        load_project(path)


def read_export(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    images = {
        item["id"]: sitk.ReadImage(str(manifest_path.parent / item["file"]))
        for item in manifest["fragments"]
    }
    return manifest, images


def test_identity_export_preserves_masks_and_adds_shared_border(tmp_path):
    doc = phantom(reflected=True)
    source = doc.source.copy()
    manifest, images = read_export(export_labelmaps(doc, tmp_path / "export"))
    assert set(images) == {1, 2, 5, 6, 7}
    assert manifest["plan"] == {
        "id": "plan1", "name": "Le Fort I", "lefort_level": 1,
        "lefort": True, "bsso": False, "genioplasty": True,
        "revision": 0, "anatomy_revision": 0,
    }
    geometry = manifest["geometry"]
    for label, image in images.items():
        np.testing.assert_allclose(image.GetSpacing(), doc.geometry.spacing)
        np.testing.assert_allclose(image.GetDirection(), doc.geometry.direction, atol=1e-6)
        assert image.GetSize() == tuple(geometry["shape"][::-1])
        np.testing.assert_allclose(image.GetOrigin(), geometry["origin"], atol=1e-5)
        array = sitk.GetArrayFromImage(image)
        assert np.count_nonzero(array) == np.count_nonzero(doc.fragments == label)
        assert np.isin(array, [0, 1]).all()
        for axis in range(3):
            assert not np.take(array, 0, axis=axis).any()
            assert not np.take(array, -1, axis=axis).any()
    np.testing.assert_array_equal(doc.source, source)


def test_export_preserves_overlap_in_separate_files(tmp_path):
    doc = phantom()
    delta_lps = doc.geometry.affine[:3, :3] @ [-7, 0, 0]
    doc.motions["maxilla"].translation = (doc.frame[:3, :3] @ delta_lps).tolist()
    manifest, images = read_export(export_labelmaps(doc, tmp_path / "overlap"))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(images[1]), sitk.GetArrayFromImage(images[2]))
    moved = next(item for item in manifest["fragments"] if item["id"] == 2)
    np.testing.assert_allclose(np.asarray(moved["transform_lps"])[:3, 3], delta_lps)


def test_rotation_translation_export_has_no_clipping(tmp_path):
    from bimax.engine import fragment_transforms

    doc = phantom()
    doc.motions["linked"].angles = [31, -12, 7]
    doc.motions["linked"].translation = [12, 15, -8]
    doc.motions["linked"].pivot = doc.world_to_plan(doc.geometry.index_to_world([7, 6, 4])).tolist()
    doc.motions["linked"].pivot_set = True
    doc.motions["chin"].angles = [-8, 2, 5]
    doc.motions["chin"].pivot_set = True
    manifest, images = read_export(export_labelmaps(doc, tmp_path / "rotated"))
    geometry = Geometry(**{key: tuple(value) for key, value in manifest["geometry"].items()})
    transforms = fragment_transforms(doc)
    for label, image in images.items():
        array = sitk.GetArrayFromImage(image)
        assert array.any()
        xyz = np.argwhere(doc.fragments == label)[:, ::-1]
        world = doc.geometry.index_to_world(xyz)
        moved = world @ transforms[label][:3, :3].T + transforms[label][:3, 3]
        output_index = geometry.world_to_index(moved)
        assert np.all(output_index >= 0.5)
        assert np.all(output_index <= np.array(geometry.shape[::-1]) - 1.5)
        for axis in range(3):
            assert not np.take(array, 0, axis=axis).any()
            assert not np.take(array, -1, axis=axis).any()


def test_export_rebuilds_only_active_accepted_plan_after_loading(tmp_path):
    doc = phantom()
    doc.plans["lf3"] = deepcopy(doc.plan)
    doc.plans["lf3"].lefort_level = 3
    doc.plans["lf3"].name = "Le Fort III without genioplasty"
    doc.plans["lf3"].cuts["genio"]["enabled"] = False
    doc.plans["lf3"].confirmations.pop("6")
    doc.scenarios["Third"] = new_scenario("lf3")
    doc.active_scenario = "Third"
    doc.fragments = None
    doc.fragments_key = None
    save_project(doc, tmp_path / "plan.bimax")
    restored = load_project(tmp_path / "plan.bimax")
    assert restored.fragments is None
    manifest, images = read_export(export_labelmaps(restored, tmp_path / "export"))
    assert set(images) == {1, 2, 5, 7}
    assert manifest["plan"]["id"] == "lf3"
    assert manifest["plan"]["lefort_level"] == 3
    assert not manifest["plan"]["genioplasty"]
    assert next(item for item in manifest["fragments"] if item["id"] == 2)["name"] == "Midface"
    assert restored.fragments_key == restored.configuration_key
    assert restored.plans["plan1"].cuts["genio"]["enabled"]


def test_export_rejects_unassigned_unreviewed_and_stale_fragments(tmp_path):
    doc = phantom()
    unreviewed = deepcopy(doc)
    unreviewed.plan.accepted = False
    unreviewed.fragments = None
    unreviewed.fragments_key = None
    with pytest.raises(ValueError, match="Build and review"):
        export_labelmaps(unreviewed, tmp_path / "export")
    stale = deepcopy(doc)
    stale.anatomy_revision += 1
    with pytest.raises(ValueError, match="stale"):
        export_labelmaps(stale, tmp_path / "export")
    doc.anatomy[2, 3, 2] = 1
    doc.fragments[2, 3, 2] = 8
    with pytest.raises(ValueError, match="Assign"):
        export_labelmaps(doc, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_refuses_nonempty_directory_and_accepts_empty(tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    protected = output / "scan.nii.gz"
    protected.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="new or empty"):
        export_labelmaps(phantom(), output)
    assert protected.read_bytes() == b"preserve"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert export_labelmaps(phantom(), empty).is_file()


def test_failed_export_never_publishes_partial_output(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("disk failure")

    monkeypatch.setattr(sitk, "WriteImage", fail)
    with pytest.raises(RuntimeError, match="disk failure"):
        export_labelmaps(phantom(), tmp_path / "export")
    assert list(tmp_path.iterdir()) == []
