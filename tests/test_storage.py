import json
import zipfile
from copy import deepcopy

import numpy as np
import pytest
import SimpleITK as sitk

from bimax.model import Document, Geometry, new_scenario
from bimax.storage import export_labelmaps, load_project, save_project


def phantom(reflected=False):
    direction = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1.0]])
    if reflected:
        direction[:, 0] *= -1
    geometry = Geometry(
        (12, 14, 18), (0.5, 0.75, 1.25), (12.5, -30.0, 8.0), tuple(direction.ravel())
    )
    fragments = np.zeros(geometry.shape, dtype=np.uint8)
    fragments[2:5, 3:6, 2:5] = 1
    fragments[2:5, 3:6, 9:12] = 2
    fragments[6:9, 5:8, 5:8] = 5
    fragments[9:11, 5:8, 5:8] = 6
    fragments[2, 10, 14] = 7
    anatomy = np.array([0, 2, 2, 3, 3, 3, 3, 4, 1], dtype=np.uint8)[fragments]
    return Document(
        geometry=geometry,
        source=(fragments > 0).astype(np.uint8),
        anatomy=anatomy,
        seeds=np.zeros_like(fragments),
        barriers=np.zeros_like(fragments),
        fragments=fragments,
        source_name="original-bone.nii.gz",
    )


@pytest.mark.parametrize("reflected", [False, True])
def test_project_roundtrip_preserves_geometry_edits_and_scenarios(tmp_path, reflected):
    doc = phantom(reflected)
    doc.seeds[2, 3, 2] = 2
    doc.barriers[2, 3, 3] = 1
    doc.confirmations = {"2": [4.1, 2.2, -3.3]}
    doc.landmarks = [{"name": "Maxillary reference", "point": [1.2, 3.4, 5.6], "fragment": 2}]
    doc.cuts = {
        "lefort": {
            "enabled": True,
            "plane": {"origin": [1.0, 2.0, 3.0], "normal": [0.0, 0.0, 1.0]},
            "region": {"min": [-20.0, -30.0, -40.0], "max": [20.0, 30.0, 40.0]},
        }
    }
    doc.scenarios["Advance"] = new_scenario()
    doc.active_scenario = "Advance"
    doc.motions["linked"].translation = [1.0, 6.0, -2.0]
    doc.motions["mandible"].angles = [12.0, 3.0, -4.0]
    doc.motions["chin"].pivot = [3.0, -4.0, 5.0]
    doc.motions["chin"].pivot_set = True
    doc.revision = 17
    path = tmp_path / "plan.bimax"
    save_project(doc, path)
    restored = load_project(path)
    assert restored.geometry == doc.geometry
    for name in ("source", "anatomy", "seeds", "barriers", "fragments", "frame"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(doc, name))
    for name in (
        "cuts",
        "confirmations",
        "scenarios",
        "active_scenario",
        "landmarks",
        "source_name",
        "revision",
    ):
        assert getattr(restored, name) == getattr(doc, name)
    assert not restored.source.flags.writeable


def test_unfinished_project_can_save_without_fragments(tmp_path):
    doc = phantom()
    doc.fragments = None
    doc.anatomy[doc.source != 0] = 1
    save_project(doc, tmp_path / "unfinished.bimax")
    restored = load_project(tmp_path / "unfinished.bimax")
    assert restored.fragments is None
    np.testing.assert_array_equal(restored.anatomy, doc.anatomy)


@pytest.mark.parametrize("corruption", ["outside", "missing", "parent", "seed", "nonbinary"])
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


@pytest.mark.parametrize("corruption", ["traversal", "nan", "geometry", "motion", "landmark"])
def test_malformed_archive_is_rejected(tmp_path, corruption):
    path = tmp_path / "plan.bimax"
    save_project(phantom(), path)
    if corruption == "traversal":
        rewrite_archive(path, extra={"../outside": b"unsafe"})
    elif corruption == "nan":
        rewrite_archive(path, lambda metadata: metadata["frame"][0].__setitem__(0, float("nan")))
    elif corruption == "geometry":
        rewrite_archive(path, lambda metadata: metadata["geometry"]["shape"].__setitem__(0, 99))
    elif corruption == "motion":
        rewrite_archive(
            path, lambda metadata: metadata["scenarios"]["Scenario 1"]["linked"].pop("pivot")
        )
    else:
        rewrite_archive(path, lambda metadata: metadata["landmarks"].append({"name": "Broken"}))
    with pytest.raises(ValueError):
        load_project(path)
    assert not (tmp_path / "outside").exists()


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
    # The two upper-jaw cubes differ only by seven source X voxels.
    delta_lps = doc.geometry.affine[:3, :3] @ [-7, 0, 0]
    doc.motions["maxilla"].translation = (doc.frame[:3, :3] @ delta_lps).tolist()
    manifest, images = read_export(export_labelmaps(doc, tmp_path / "overlap"))
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(images[1]), sitk.GetArrayFromImage(images[2])
    )
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


def test_export_rejects_unassigned_and_missing_fragments(tmp_path):
    doc = phantom()
    missing = deepcopy(doc)
    missing.fragments = None
    with pytest.raises(ValueError, match="Build and review"):
        export_labelmaps(missing, tmp_path / "export")
    doc.anatomy[2, 3, 2] = 1
    doc.fragments[2, 3, 2] = 8
    with pytest.raises(ValueError, match="Assign or explicitly exclude"):
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
