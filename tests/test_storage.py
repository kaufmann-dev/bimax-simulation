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
    doc.cuts = default_cuts(doc)
    doc.scenario.cut_ids = ["lefort1", "genio"]
    matrix = doc.frame @ doc.geometry.affine

    def plane(origin, normal):
        return {
            "origin": (matrix[:3, :3] @ origin + matrix[:3, 3]).tolist(),
            "normal": (np.linalg.inv(matrix[:3, :3]).T @ normal).tolist(),
        }

    doc.cuts["lefort1"].geometry = {
        "planes": {"upper": plane([7, 0, 0], [1, 0, 0])},
        "groups": [["upper"]],
        "region": {"min": [-100.0] * 3, "max": [100.0] * 3},
    }
    doc.cuts["genio"].geometry = {
        "plane": plane([0, 0, 8.5], [0, 0, 1]),
        "region": {"min": [-100.0] * 3, "max": [100.0] * 3},
    }
    doc.review.confirmations = {
        doc.fragment_descriptors[label].id: doc.world_to_plan(geometry.index_to_world(xyz)).tolist()
        for label, xyz in ((2, [9, 3, 2]), (5, [5, 5, 6]), (6, [5, 5, 9]))
    }
    doc.accept_review()
    doc.fragments = build_fragments(doc)
    doc.fragments_key = doc.configuration_key
    np.testing.assert_array_equal(doc.fragments, expected)
    return doc


@pytest.mark.parametrize("reflected", [False, True])
def test_project_roundtrip_preserves_shared_anatomy_plans_and_scenarios(tmp_path, reflected):
    doc = phantom(reflected)
    doc.view_angles = [7.5, -3.0, 2.0]
    doc.seeds[2, 3, 2] = 2
    doc.barriers[2, 3, 3] = 1
    for level in (2, 3):
        key = f"lefort{level}"
        doc.cuts[key].geometry = deepcopy(doc.cuts["lefort1"].geometry)
        doc.cuts[key].confirmations = {f"upper:{key}": list(doc.cuts["lefort1"].confirmations["upper:lefort1"])}
        doc.cuts[key].accepted = True
        doc.cuts[key].revision = level
        doc.scenarios[f"Le Fort {level}"] = new_scenario([key])
        doc.active_scenario = f"Le Fort {level}"
        doc.review.accepted = True
    doc.scenarios["Advance"] = new_scenario(["lefort1", "genio"])
    doc.active_scenario = "Advance"
    doc.scenario.landmarks = [{"name": "Maxillary reference", "point": [1.2, 3.4, 5.6], "fragment": "upper:lefort1"}]
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
    for name in ("cuts", "scenarios", "active_scenario", "anatomy_revision", "source_name", "revision"):
        assert getattr(restored, name) == getattr(doc, name)
    assert restored.fragments is None
    assert restored.fragments_key is None
    assert not restored.source.flags.writeable
    with zipfile.ZipFile(path) as archive:
        assert {name for name in archive.namelist() if not name.startswith("artifacts/")} == {
            "project.json", "volumes/source.nii.gz", "volumes/anatomy.nii.gz",
            "volumes/seeds.nii.gz", "volumes/barriers.nii.gz",
        }
        assert json.loads(archive.read("project.json"))["version"] == 4
    assert restored.view_angles == [7.5, -3.0, 2.0]
    restored.motions["linked"].translation[0] = 100
    restored.scenario.landmarks[0]["point"][0] = 100
    assert restored.scenarios["Scenario 1"].motions["linked"].translation == [0.0, 0.0, 0.0]
    assert restored.scenarios["Scenario 1"].landmarks == []
    restored.cuts["lefort2"].geometry["planes"]["upper"]["origin"][0] = 100
    assert restored.cuts["lefort3"].geometry == doc.cuts["lefort3"].geometry
    assert doc.motions["linked"].translation == [1.0, 6.0, -2.0]


def test_unfinished_project_can_save_without_fragments(tmp_path):
    doc = phantom()
    doc.fragments = None
    doc.fragments_key = None
    for cut in doc.cuts.values():
        cut.accepted = False
        cut.confirmations.clear()
    doc.reviews.clear()
    doc.anatomy_revision += 1
    doc.anatomy[doc.source != 0] = 1
    save_project(doc, tmp_path / "unfinished.bimax")
    restored = load_project(tmp_path / "unfinished.bimax")
    assert restored.fragments is None
    assert not restored.cuts["lefort1"].accepted
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
        doc.cuts["lefort1"].geometry["planes"]["upper"]["origin"][0] += 1
    elif corruption == "unreviewed":
        doc.cuts["lefort1"].accepted = False
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


@pytest.mark.parametrize("corruption", ["traversal", "nan", "geometry", "motion", "landmark", "cut_ref", "active", "level", "reviews", "plane", "group", "empty_group", "old_version", "fragments"])
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
            "cut_ref": lambda m: m["scenarios"]["Scenario 1"].__setitem__("cut_ids", ["missing"]),
            "active": lambda m: m.__setitem__("active_scenario", "missing"),
            "level": lambda m: m["cuts"]["lefort1"].__setitem__("level", 4),
            "reviews": lambda m: m["cuts"]["lefort1"]["confirmations"].pop("upper:lefort1"),
            "plane": lambda m: m["cuts"]["lefort1"]["geometry"]["planes"]["upper"].__setitem__("normal", [0, 0, 0]),
            "group": lambda m: m["cuts"]["lefort1"]["geometry"].__setitem__("groups", [["missing"]]),
            "empty_group": lambda m: m["cuts"]["lefort1"]["geometry"].__setitem__("groups", [[]]),
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
    with pytest.raises(ValueError, match="version 4 project"):
        load_project(path)


def test_inactive_accepted_plan_reviews_are_validated_without_building_volumes(tmp_path, monkeypatch):
    import bimax.engine

    doc = phantom()
    doc.cuts["lefort3"].name = "Inactive cut"
    doc.cuts["lefort3"].geometry = deepcopy(doc.cuts["lefort1"].geometry)
    doc.cuts["lefort3"].accepted = True
    doc.cuts["lefort3"].confirmations = {"upper:lefort3": list(doc.cuts["lefort1"].confirmations["upper:lefort1"])}
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
        lambda metadata: metadata["cuts"]["lefort3"]["confirmations"].__setitem__("upper:lefort3", [999, 999, 999]),
    )
    with pytest.raises(ValueError, match="Inactive cut.*outside"):
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
    assert [(cut["id"], cut["kind"]) for cut in manifest["cuts"]] == [("lefort1", "lefort"), ("genio", "genio")]
    assert all("identity" in fragment for fragment in manifest["fragments"])
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
    doc.motions["lefort1"].translation = (doc.frame[:3, :3] @ delta_lps).tolist()
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
    doc.cuts["lefort3"].geometry = deepcopy(doc.cuts["lefort1"].geometry)
    doc.cuts["lefort3"].confirmations = {"upper:lefort3": list(doc.cuts["lefort1"].confirmations["upper:lefort1"])}
    doc.cuts["lefort3"].accepted = True
    doc.scenarios["Third"] = new_scenario(["lefort3"])
    doc.active_scenario = "Third"
    doc.review.accepted = True
    doc.fragments = None
    doc.fragments_key = None
    save_project(doc, tmp_path / "plan.bimax")
    restored = load_project(tmp_path / "plan.bimax")
    assert restored.fragments is None
    manifest, images = read_export(export_labelmaps(restored, tmp_path / "export"))
    assert set(images) == {1, 2, 5, 7}
    assert manifest["cuts"][0]["id"] == "lefort3"
    assert manifest["cuts"][0]["level"] == 3
    assert len(manifest["cuts"]) == 1
    assert next(item for item in manifest["fragments"] if item["id"] == 2)["name"] == "Midface"
    assert restored.fragments_key == restored.configuration_key
    assert restored.cuts["genio"].accepted


def test_export_rejects_unassigned_unreviewed_and_stale_fragments(tmp_path):
    doc = phantom()
    unreviewed = deepcopy(doc)
    unreviewed.review.accepted = False
    unreviewed.fragments = None
    unreviewed.fragments_key = None
    with pytest.raises(ValueError, match="Build and review"):
        export_labelmaps(unreviewed, tmp_path / "export")
    stale = deepcopy(doc)
    stale.cuts["lefort1"].geometry["planes"]["upper"]["origin"][0] += 1
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


def test_nested_export_preserves_fragment_identity_and_parent_relative_transforms(tmp_path):
    from factories import confirm, document, upper_geometry

    from bimax.engine import fragment_transforms
    from bimax.prepared import prepare_geometry
    doc = document()
    for level, height in ((1, 11), (2, 14), (3, 17)):
        doc.cuts[f"lefort{level}"].geometry = upper_geometry(height)
    doc.scenario.cut_ids = ["lefort3", "lefort2", "lefort1", "bsso", "genio"]
    result = prepare_geometry(doc)
    confirm(doc, result["fragments"])
    doc.fragments, doc.fragments_key = result["fragments"], doc.configuration_key
    doc.motions["lefort3"].translation = [0, 2, 0]
    doc.motions["lefort1"].translation = [0, 3, 0]
    transforms = fragment_transforms(doc)
    manifest, images = read_export(export_labelmaps(doc, tmp_path / "nested"))
    assert {2, 9, 10} <= set(images)
    identities = {item["identity"] for item in manifest["fragments"]}
    assert {"upper:lefort1", "upper:lefort2", "upper:lefort3"} <= identities
    for item in manifest["fragments"]:
        np.testing.assert_allclose(item["transform_lps"], transforms[item["id"]])
        assert item["anatomy"] == doc.fragment_descriptors[item["id"]].anatomy
