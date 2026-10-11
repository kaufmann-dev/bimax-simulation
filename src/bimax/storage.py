"""Version 5 portable projects and physical-space fragment labelmap exports."""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from bimax.model import (
    CUT_TYPES,
    PROXIMAL_IDS,
    Artifact,
    Cut,
    Document,
    Geometry,
    Measurement,
    Motion,
    ProximalRotation,
    Review,
    Scenario,
    new_motions,
)

_VERSION = 5
_MAX_METADATA = 8 * 1024 * 1024
_MAX_VOXELS = 2**31 - 1
_VOLUMES = ("source", "anatomy", "seeds", "barriers")
_MAX_CONFIGURATIONS = 256
_HEX = re.compile(r"[0-9a-f]{64}")


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must contain {length} finite numbers.")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers.") from exc
    if array.shape != (length,) or array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite numbers.")
    return array.astype(float)


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 256


def _geometry(value: Any) -> Geometry:
    if not isinstance(value, dict) or set(value) != {"shape", "spacing", "origin", "direction"}:
        raise ValueError("Project geometry is incomplete or unsupported.")
    shape = value["shape"]
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(type(size) is not int or size <= 0 for size in shape)
            or int(np.prod(shape, dtype=object)) > _MAX_VOXELS):
        raise ValueError("Project volume dimensions are invalid or too large.")
    spacing = _vector(value["spacing"], 3, "Spacing")
    origin = _vector(value["origin"], 3, "Origin")
    direction = _vector(value["direction"], 9, "Direction").reshape(3, 3)
    if np.any(spacing <= 0) or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Spacing must be positive and image directions must be orthonormal.")
    return Geometry(tuple(shape), tuple(spacing), tuple(origin), tuple(direction.ravel()))


def _plane(value: Any, name: str) -> None:
    if not isinstance(value, dict) or set(value) != {"origin", "normal"}:
        raise ValueError(f"{name} needs an origin and normal.")
    _vector(value["origin"], 3, name + " origin")
    length = np.linalg.norm(_vector(value["normal"], 3, name + " normal"))
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError(f"{name} normal must have a finite, nonzero length.")


def _validate_cut(cut: Cut) -> None:
    geometry = cut.geometry
    expected = {"lefort": {"planes", "groups", "region"},
                "bsso": {"midline", "left", "right"}, "genio": {"plane", "region"}}[cut.kind]
    if not isinstance(geometry, dict) or set(geometry) != expected:
        raise ValueError(f"Invalid {cut.name} cut definition.")
    if cut.kind == "bsso":
        _vector([geometry["midline"]], 1, "BSSO midline")
        for side in ("left", "right"):
            if not isinstance(geometry[side], dict) or set(geometry[side]) != {"front", "upper", "sagittal"}:
                raise ValueError(f"Invalid BSSO {side} planes.")
            for key, plane in geometry[side].items():
                _plane(plane, f"BSSO {side} {key}")
    else:
        region = geometry["region"]
        if not isinstance(region, dict) or set(region) != {"min", "max"}:
            raise ValueError("A cut region needs minimum and maximum coordinates.")
        if np.any(_vector(region["min"], 3, "Region minimum") > _vector(region["max"], 3, "Region maximum")):
            raise ValueError("Cut region minimum must not exceed maximum.")
        if cut.kind == "genio":
            _plane(geometry["plane"], "Genioplasty")
        else:
            planes, groups = geometry["planes"], geometry["groups"]
            if not isinstance(planes, dict) or not 1 <= len(planes) <= 64 or not all(_name(k) for k in planes):
                raise ValueError("Le Fort cuts need between 1 and 64 named planes.")
            for key, plane in planes.items():
                _plane(plane, f"Le Fort {key}")
            if (not isinstance(groups, list) or not 1 <= len(groups) <= 64
                    or any(not isinstance(group, list) or not group
                           or any(not isinstance(key, str) or key not in planes for key in group)
                           for group in groups)):
                raise ValueError("Le Fort groups must reference nonempty sets of named planes.")


def _confirmations(value: Any, allowed: set[str]) -> None:
    if not isinstance(value, dict) or not set(value) <= allowed:
        raise ValueError("Unknown fragment confirmation identity.")
    for point in value.values():
        _vector(point, 3, "Fragment confirmation")


def _metadata(doc: Document) -> dict[str, Any]:
    return {"format": "bimax", "version": _VERSION, "geometry": asdict(doc.geometry),
            "frame": np.asarray(doc.frame).tolist(),
            "view_angles": list(doc.view_angles), "symmetry_right": doc.symmetry_right,
            "cuts": {key: asdict(cut) for key, cut in doc.cuts.items()},
            "scenarios": {name: asdict(scenario) for name, scenario in doc.scenarios.items()},
            "reviews": {key: asdict(review) for key, review in doc.reviews.items()},
            "artifacts": {key: {"kind": ref.kind, "digest": ref.digest} for key, ref in doc.artifacts.items()},
            "active_scenario": doc.active_scenario, "anatomy_revision": doc.anatomy_revision,
            "source_name": doc.source_name, "revision": doc.revision, "volumes": list(_VOLUMES)}


def _validated_metadata(value: Any) -> tuple[Geometry, dict[str, Cut], dict[str, Scenario], dict[str, Review]]:
    if not isinstance(value, dict) or value.get("format") != "bimax" or value.get("version") != _VERSION:
        raise ValueError("Open a version 5 project with symmetry and proximal rotations; older project formats require conversion.")
    if set(value) != {"format", "version", "geometry", "frame", "view_angles", "symmetry_right", "cuts", "scenarios", "reviews", "artifacts",
                      "active_scenario", "anatomy_revision", "source_name", "revision", "volumes"}:
        raise ValueError("Project metadata is incomplete or unsupported.")
    geometry = _geometry(value["geometry"])
    if value["symmetry_right"] is not None:
        _vector([value["symmetry_right"]], 1, "Symmetry reference Right offset")
    _vector(value["view_angles"], 3, "Whole-skull orientation")
    if any(abs(angle) > 180 for angle in value["view_angles"]):
        raise ValueError("Whole-skull orientation must be between -180 and 180 degrees.")
    frame = np.asarray(value["frame"])
    if (frame.shape != (4, 4) or frame.dtype.kind not in "iuf" or not np.all(np.isfinite(frame))
            or not np.allclose(frame[3], [0, 0, 0, 1])
            or not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(frame[:3, :3]), 1)):
        raise ValueError("Planning frame must be a finite, right-handed rigid transform.")
    if not isinstance(value["cuts"], dict) or set(value["cuts"]) != set(CUT_TYPES):
        raise ValueError("A project needs exactly seven reusable cut definitions.")
    cuts = {}
    ids = {"skull", "left-proximal", "right-proximal", "mandible", "chin", "excluded", "unassigned"}
    ids.update(f"upper:{key}" for key, (_, kind, _) in CUT_TYPES.items() if kind == "lefort")
    for key, data in value["cuts"].items():
        _, kind, level = CUT_TYPES[key]
        if (not isinstance(data, dict)
                or set(data) != {"name", "kind", "level", "geometry", "confirmations", "accepted", "revision"}
                or not _name(data["name"]) or data["kind"] != kind
                or type(data["level"]) is not int or data["level"] != level
                or type(data["accepted"]) is not bool or type(data["revision"]) is not int or data["revision"] < 0):
            raise ValueError("Invalid shared cut identity or review state.")
        cut = Cut(**data)
        _validate_cut(cut)
        required = {f"upper:{key}"} if kind == "lefort" else (
            {"left-proximal", "right-proximal", "mandible"} if kind == "bsso" else {"chin", "mandible"})
        _confirmations(cut.confirmations, required)
        if cut.accepted and not required <= set(cut.confirmations):
            raise ValueError("Accepted cuts need confirmation points for every required fragment.")
        cuts[key] = cut
    scenarios = {}
    if not isinstance(value["scenarios"], dict) or not 1 <= len(value["scenarios"]) <= _MAX_CONFIGURATIONS:
        raise ValueError("A project needs between 1 and 256 movement scenarios.")
    for name, data in value["scenarios"].items():
        if (not _name(name) or not isinstance(data, dict) or set(data) != {"cut_ids", "motions", "landmarks", "proximal_rotations", "measurements"}
                or not isinstance(data["cut_ids"], list) or any(not isinstance(k, str) or k not in cuts for k in data["cut_ids"])
                or len(set(data["cut_ids"])) != len(data["cut_ids"])
                or not isinstance(data["motions"], dict) or set(data["motions"]) != set(new_motions())):
            raise ValueError("Invalid scenario or referenced shared cut.")
        levels = [cuts[key].level for key in data["cut_ids"] if cuts[key].kind == "lefort"]
        if len(set(levels)) != len(levels):
            raise ValueError("Select only one variant per Le Fort level.")
        motions = {}
        for key, motion in data["motions"].items():
            if (not isinstance(motion, dict) or set(motion) != {"translation", "angles", "pivot", "pivot_set"}
                    or type(motion["pivot_set"]) is not bool):
                raise ValueError("Invalid movement parameters.")
            for field in ("translation", "angles", "pivot"):
                _vector(motion[field], 3, f"{name} {key} {field}")
            motions[key] = Motion(**motion)
            from .engine import motion_matrix
            motion_matrix(motions[key])
        landmarks = data["landmarks"]
        active_ids = {"skull", "mandible", "excluded", "unassigned"} | {
            f"upper:{key}" for key in data["cut_ids"] if cuts[key].kind == "lefort"}
        if "bsso" in data["cut_ids"]:
            active_ids.update(PROXIMAL_IDS)
        if "genio" in data["cut_ids"]:
            active_ids.add("chin")
        if not isinstance(landmarks, list):
            raise ValueError("Scenario landmarks must be a list.")
        for landmark in landmarks:
            if (not isinstance(landmark, dict) or set(landmark) != {"name", "point", "fragment"}
                    or not _name(landmark["name"]) or not isinstance(landmark["fragment"], str)
                    or landmark["fragment"] not in active_ids):
                raise ValueError("Invalid scenario landmark.")
            _vector(landmark["point"], 3, "Landmark point")
        if len({item["name"] for item in landmarks}) != len(landmarks):
            raise ValueError("Landmark names must be unique within each scenario.")
        proximal = data["proximal_rotations"]
        if not isinstance(proximal, dict) or set(proximal) != set(PROXIMAL_IDS):
            raise ValueError("A scenario requires left and right proximal rotation controls.")
        proximal_rotations = {}
        for key, rotation in proximal.items():
            if (not isinstance(rotation, dict) or set(rotation) != {"angles", "pivot", "pivot_set"}
                    or type(rotation["pivot_set"]) is not bool):
                raise ValueError("Invalid proximal rotation; translation is unsupported.")
            for field in ("angles", "pivot"):
                _vector(rotation[field], 3, f"{key} {field}")
            control = ProximalRotation(**rotation)
            motion_matrix(Motion(angles=control.angles, pivot=control.pivot, pivot_set=control.pivot_set))
            proximal_rotations[key] = control
        definitions = data["measurements"]
        if not isinstance(definitions, dict) or len(definitions) > 256:
            raise ValueError("Scenario measurement definitions must be a named mapping with at most 256 entries.")
        measurements = {}
        for measurement_name, definition in definitions.items():
            if (not _name(measurement_name) or not isinstance(definition, dict)
                    or set(definition) != {"kind", "landmarks"}
                    or not isinstance(definition["kind"], str) or not isinstance(definition["landmarks"], dict)
                    or any(not _name(key) or not _name(item) for key, item in definition["landmarks"].items())):
                raise ValueError("Invalid named measurement definition.")
            measurements[measurement_name] = Measurement(**definition)
        scenarios[name] = Scenario(data["cut_ids"], motions, landmarks, proximal_rotations, measurements)
        from .symmetry import validate_definition
        empty = np.zeros((1, 1, 1), dtype=np.uint8)
        planning = Document(geometry, empty, empty, empty, empty, cuts=cuts,
                            scenarios={name: scenarios[name]}, active_scenario=name,
                            symmetry_right=value["symmetry_right"])
        for definition in measurements.values():
            validate_definition(planning, definition)
    if not isinstance(value["active_scenario"], str) or value["active_scenario"] not in scenarios:
        raise ValueError("Active scenario is missing from the project.")
    reviews = {}
    if not isinstance(value["reviews"], dict) or len(value["reviews"]) > 1024:
        raise ValueError("Invalid combination reviews.")
    for key, review in value["reviews"].items():
        if (not _HEX.fullmatch(key) or not isinstance(review, dict)
                or set(review) != {"confirmations", "accepted"} or type(review["accepted"]) is not bool):
            raise ValueError("Invalid combination review.")
        _confirmations(review["confirmations"], ids)
        reviews[key] = Review(**review)
    if not isinstance(value["artifacts"], dict) or len(value["artifacts"]) > 1024:
        raise ValueError("Invalid prepared geometry index.")
    for key, ref in value["artifacts"].items():
        if (not _HEX.fullmatch(key) or not isinstance(ref, dict) or set(ref) != {"kind", "digest"}
                or ref["kind"] not in {"basis", "cut", "upper", "lower", "scene"}
                or not isinstance(ref["digest"], str) or not _HEX.fullmatch(ref["digest"])):
            raise ValueError("Invalid prepared geometry reference.")
    if (not isinstance(value["source_name"], str) or type(value["revision"]) is not int or value["revision"] < 0
            or type(value["anatomy_revision"]) is not int or value["anatomy_revision"] < 0
            or value["volumes"] != list(_VOLUMES)):
        raise ValueError("Invalid source identity, revision, or volume list.")
    json.dumps(value, allow_nan=False)
    return geometry, cuts, scenarios, reviews


def _validate_arrays(doc: Document) -> None:
    shape = doc.geometry.shape
    for name in (*_VOLUMES, "fragments"):
        array = getattr(doc, name)
        if array is None and name == "fragments":
            continue
        allowed = (np.dtype("uint8"), np.dtype("bool")) if name in {"source", "barriers"} else (np.dtype("uint8"),)
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype not in allowed:
            raise ValueError(f"{name.title()} must be a uint8 volume matching the source geometry.")
    if doc.fragments is None:
        if doc.fragments_key is not None:
            raise ValueError("Fragment geometry key exists without active geometry.")
    elif not doc.cuts_accepted or doc.fragments_key != doc.configuration_key:
        raise ValueError("Active fragments are stale or unreviewed; rebuild and review the selected cuts.")
    reviewed = any(cut.accepted for cut in doc.cuts.values())
    any_bone = False
    counts = np.zeros(max(doc.fragment_descriptors) + 1, dtype=np.int64)
    anatomy_counts = np.zeros(5, dtype=np.int64)
    depth = max(1, 1_000_000 // (shape[1] * shape[2]))
    parent = doc.parent_labels
    for start in range(0, shape[0], depth):
        slab = slice(start, min(start + depth, shape[0]))
        source = doc.source[slab]
        bone = source != 0
        any_bone |= bool(bone.any())
        anatomy, seeds, barriers = doc.anatomy[slab], doc.seeds[slab], doc.barriers[slab]
        if np.any(source > 1):
            raise ValueError("Source must be a binary bone mask containing only 0 and 1.")
        if np.any(anatomy > 4) or not np.array_equal(anatomy != 0, bone):
            raise ValueError("Anatomical labels must preserve every source foreground voxel.")
        if reviewed and np.any(anatomy == 1):
            raise ValueError("Assign all bone to anatomy or explicitly exclude it before accepting cuts.")
        if np.any(seeds > 4) or np.any(seeds == 1) or np.any((seeds != 0) & ~bone):
            raise ValueError("Seeds must be 0, 2, 3 or 4 and lie inside source bone.")
        if np.any(barriers > 1) or np.any((barriers != 0) & ~bone):
            raise ValueError("Barriers must be binary and lie inside source bone.")
        if doc.fragments is not None:
            labels = doc.fragments[slab]
            if np.any(labels >= len(parent)) or not np.array_equal(labels != 0, bone):
                raise ValueError("Fragment labels must preserve every source foreground voxel.")
            if not np.array_equal(parent[labels], anatomy):
                raise ValueError("Fragment ownership does not match its anatomical parent; rebuild cuts.")
            counts += np.bincount(labels.ravel(), minlength=len(parent))
            anatomy_counts += np.bincount(anatomy.ravel(), minlength=5)
    if not any_bone:
        raise ValueError("Source bone mask is empty.")
    if doc.fragments is not None:
        from bimax.engine import validate_fragment_preview
        validate_fragment_preview(doc, doc.fragments, counts, anatomy_counts)
def _image(array: np.ndarray, geometry: Geometry) -> Any:
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))
    image.SetSpacing(geometry.spacing)
    image.SetOrigin(geometry.origin)
    image.SetDirection(geometry.direction)
    return image


def _read_volume(path: Path, geometry: Geometry) -> np.ndarray:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if (
        reader.GetDimension() != 3
        or tuple(reader.GetSize()) != geometry.shape[::-1]
        or reader.GetNumberOfComponents() != 1
        or reader.GetPixelID() != sitk.sitkUInt8
    ):
        raise ValueError(f"{path.name} has unexpected dimensions or pixel type.")
    for actual, expected in (
        (reader.GetSpacing(), geometry.spacing),
        (reader.GetOrigin(), geometry.origin),
        (reader.GetDirection(), geometry.direction),
    ):
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-4):
            raise ValueError(f"{path.name} geometry does not match project metadata.")
    return sitk.GetArrayFromImage(reader.Execute())



def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key}.")
        result[key] = value
    return result


def file_version(path: str | Path) -> list[int] | None:
    try:
        stat = Path(path).stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


_UNSPECIFIED = object()


def save_project(doc: Document, path: str | Path, *, expected_version: Any = _UNSPECIFIED) -> None:
    """Save source-of-truth state plus current prepared artifacts atomically."""
    from bimax.engine import validate_fragment_preview
    from bimax.prepared import MAX_ARTIFACT_BYTES, prepare_geometry, prune_references, read

    destination = Path(path).absolute()
    if destination.suffix.lower() != ".bimax" or destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError("Choose a regular .bimax project filename.")
    if expected_version is _UNSPECIFIED:
        expected_version = file_version(destination)
    if file_version(destination) != expected_version:
        raise ValueError("External file conflict. Reopen the project or save to another filename.")
    _validated_metadata(_metadata(doc))
    _validate_arrays(doc)
    validate_reviews(doc)
    active = doc.active_scenario
    try:
        for name in doc.scenarios:
            doc.active_scenario = name
            if doc.cuts_accepted:
                result = prepare_geometry(doc)
                validate_fragment_preview(doc, result["fragments"], result["fragment_counts"], result["anatomy_counts"])
    finally:
        doc.active_scenario = active
    prune_references(doc)
    metadata = _metadata(doc)
    _validated_metadata(metadata)
    encoded = json.dumps(metadata, allow_nan=False, indent=2).encode("utf-8")
    if len(encoded) > _MAX_METADATA:
        raise ValueError("Project metadata exceeds the 8 MiB limit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bimax-save-", dir=destination.parent) as temporary:
        directory = Path(temporary)
        archive_path = directory / "project.bimax"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr("project.json", encoded)
            for name in _VOLUMES:
                volume_path = directory / f"{name}.nii.gz"
                sitk.WriteImage(_image(getattr(doc, name), doc.geometry), str(volume_path), True)
                archive.write(volume_path, f"volumes/{name}.nii.gz")
                volume_path.unlink()
            for key, ref in doc.artifacts.items():
                read(ref, ref.kind)
                if Path(ref.path).stat().st_size > MAX_ARTIFACT_BYTES:
                    raise ValueError("Prepared geometry is unexpectedly large.")
                archive.write(ref.path, f"artifacts/{key}.npz")
        with archive_path.open("rb") as stream:
            os.fsync(stream.fileno())
        if file_version(destination) != expected_version:
            raise ValueError("External file conflict. Reopen the project or save to another filename.")
        if expected_version is not None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            backup = destination.with_name(f"{destination.stem}.backup-{stamp}.bimax")
            with destination.open("rb") as original, backup.open("xb") as copy_stream:
                shutil.copyfileobj(original, copy_stream)
                copy_stream.flush()
                os.fsync(copy_stream.fileno())
            if file_version(destination) != expected_version:
                raise ValueError("External file conflict during backup. Save to another filename.")
        os.replace(archive_path, destination)


def load_project(path: str | Path) -> Document:
    """Load version 5 only; restore portable artifacts to the private local cache."""
    from bimax.prepared import MAX_ARTIFACT_BYTES, artifact_path, file_digest

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "project.json" not in names:
                raise ValueError("Project archive contains duplicate or missing metadata.")
            if archive.getinfo("project.json").file_size > _MAX_METADATA:
                raise ValueError("Project metadata exceeds the 8 MiB limit.")
            metadata = json.loads(archive.read("project.json"), parse_constant=_reject_constant,
                                  object_pairs_hook=_unique_object)
            geometry, cuts, scenarios, reviews = _validated_metadata(metadata)
            expected = ({"project.json"} | {f"volumes/{name}.nii.gz" for name in _VOLUMES}
                        | {f"artifacts/{key}.npz" for key in metadata["artifacts"]})
            # Missing disposable artifacts may be prepared again. Unexpected members are rejected.
            required = {"project.json"} | {f"volumes/{name}.nii.gz" for name in _VOLUMES}
            if not required <= set(names) or not set(names) <= expected:
                raise ValueError("Project archive contains missing or unexpected members.")
            arrays: dict[str, np.ndarray] = {}
            refs = {}
            max_member = int(np.prod(geometry.shape)) + 1024 * 1024
            with tempfile.TemporaryDirectory(prefix="bimax-load-") as temporary:
                for name in _VOLUMES:
                    member = f"volumes/{name}.nii.gz"
                    if archive.getinfo(member).file_size > max_member:
                        raise ValueError(f"Project volume {name} is unexpectedly large.")
                    volume_path = Path(temporary) / f"{name}.nii.gz"
                    with archive.open(member) as source, volume_path.open("wb") as volume_stream:
                        shutil.copyfileobj(source, volume_stream, length=1024 * 1024)
                    arrays[name] = _read_volume(volume_path, geometry)
                    volume_path.unlink()
                for key, ref in metadata["artifacts"].items():
                    member = f"artifacts/{key}.npz"
                    if member not in names or archive.getinfo(member).file_size > MAX_ARTIFACT_BYTES:
                        continue
                    destination = artifact_path(key)
                    if destination.is_file() and not destination.is_symlink() and file_digest(destination) == ref["digest"]:
                        refs[key] = Artifact(ref["kind"], ref["digest"], str(destination))
                        continue
                    # Never extract archive-controlled paths or replace a cache file until verified.
                    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".restore-", delete=False) as stream:
                        candidate = Path(stream.name)
                        try:
                            with archive.open(member) as source:
                                shutil.copyfileobj(source, stream, length=1024 * 1024)
                            stream.flush()
                            if file_digest(candidate) != ref["digest"]:
                                continue
                            os.replace(candidate, destination)
                            refs[key] = Artifact(ref["kind"], ref["digest"], str(destination))
                        finally:
                            candidate.unlink(missing_ok=True)
            doc = Document(geometry=geometry, source=arrays["source"], anatomy=arrays["anatomy"],
                           seeds=arrays["seeds"], barriers=arrays["barriers"],
                           frame=np.asarray(metadata["frame"], dtype=float), cuts=cuts, scenarios=scenarios,
                           view_angles=list(metadata["view_angles"]), symmetry_right=metadata["symmetry_right"],
                           reviews=reviews, artifacts=refs, active_scenario=metadata["active_scenario"],
                           anatomy_revision=metadata["anatomy_revision"], source_name=metadata["source_name"],
                           revision=metadata["revision"])
            _validate_arrays(doc)
            validate_reviews(doc)
            doc.source.setflags(write=False)
            return doc
    except (zipfile.BadZipFile, RuntimeError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read BIMAX project: {exc}") from exc


def validate_reviews(doc: Document) -> None:
    """Validate accepted samples using saved labels, evaluating predicates only if artifacts are absent."""
    from bimax.engine import fragment_labels_at_points, required_fragment_labels
    from bimax.prepared import sample_cut_labels, sample_labels

    original_ids = doc.scenario.cut_ids
    try:
        for cut_id, cut in doc.cuts.items():
            if not cut.accepted:
                continue
            doc.scenario.cut_ids = [cut_id]
            labels = sorted(required_fragment_labels(doc))
            points = [cut.confirmations.get(doc.fragment_descriptors[label].id) for label in labels]
            if any(point is None for point in points):
                raise ValueError(f"Accepted cut '{cut.name}' needs confirmation points.")
            try:
                actual = sample_cut_labels(doc, cut_id, points)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                actual = fragment_labels_at_points(doc, points)
            if not np.array_equal(actual, labels):
                raise ValueError(f"Accepted cut '{cut.name}' has confirmation points outside its fragments.")
    finally:
        doc.scenario.cut_ids = original_ids
    active = doc.active_scenario
    try:
        for name in doc.scenarios:
            doc.active_scenario = name
            if not doc.cuts_accepted:
                continue
            required = sorted(required_fragment_labels(doc))
            points = [doc.review.confirmations.get(doc.fragment_descriptors[label].id) for label in required]
            if any(point is None for point in points):
                raise ValueError(f"Accepted scenario '{name}' needs confirmation points for every required fragment.")
            if not points:
                continue
            try:
                actual = sample_labels(doc, points)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                actual = fragment_labels_at_points(doc, points)
            if not np.array_equal(actual, required):
                raise ValueError(f"Accepted scenario '{name}' has confirmation points outside its fragments.")
    finally:
        doc.active_scenario = active
def _export_grid(doc: Document, transforms: dict[int, np.ndarray]) -> tuple[Geometry, list[Any]]:
    boxes = ndimage.find_objects(doc.fragments, max_label=max(doc.fragment_descriptors))
    bounds: list[np.ndarray] = []
    inverse_grid = np.linalg.inv(doc.geometry.affine)
    for label, box in enumerate(boxes, start=1):
        if box is None:
            continue
        transform = transforms[label]
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1])
        ):
            raise ValueError("Fragment transform is invalid.")
        corners = np.array(
            list(itertools.product(*[(axis.start - 0.5, axis.stop - 0.5) for axis in box[::-1]]))
        )
        matrix = inverse_grid @ transform @ doc.geometry.affine
        bounds.append(corners @ matrix[:3, :3].T + matrix[:3, 3])
    all_bounds = np.concatenate(bounds)
    low = np.floor(all_bounds.min(axis=0) + 0.5 + 1e-8) - 1
    high = np.ceil(all_bounds.max(axis=0) - 0.5 - 1e-8) + 1
    size = high - low + 1
    if np.any(size <= 0) or np.any(size > _MAX_VOXELS) or float(np.prod(size)) > _MAX_VOXELS:
        raise ValueError("Transformed export grid is too large; reduce movement or source extent.")
    shape = (int(size[2]), int(size[1]), int(size[0]))
    geometry = Geometry(
        shape, doc.geometry.spacing, tuple(doc.geometry.index_to_world(low)), doc.geometry.direction
    )
    return geometry, boxes


def export_labelmaps(doc: Document, output_dir: str | Path) -> Path:
    """Export separate binary fragments on a shared grid, preserving overlaps."""
    from bimax.engine import fragment_transforms
    from bimax.prepared import prepare_geometry

    _validated_metadata(_metadata(doc))
    _validate_arrays(doc)
    if not doc.cuts_accepted:
        raise ValueError("Build and review the selected cuts before exporting.")
    if doc.fragments is None:
        doc.fragments = prepare_geometry(doc)["fragments"]
        doc.fragments_key = doc.configuration_key
    if np.any(doc.anatomy == 1) or np.any(doc.fragments == 8):
        raise ValueError("Assign or explicitly exclude every foreground voxel before exporting.")
    destination = Path(output_dir).absolute()
    if destination.is_symlink() or (
        destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise ValueError("Export destination must be a new or empty directory.")
    transforms = fragment_transforms(doc)
    geometry, boxes = _export_grid(doc, transforms)
    manifest: dict[str, Any] = {
        "format": "bimax-labelmaps",
        "version": _VERSION,
        "scenario": doc.active_scenario,
        "cuts": [{"id": key, "name": doc.cuts[key].name, "kind": doc.cuts[key].kind,
                  "level": doc.cuts[key].level, "revision": doc.cuts[key].revision}
                 for key in doc.scenario.cut_ids],
        "anatomy_revision": doc.anatomy_revision,
        "source_name": doc.source_name,
        "physical_coordinates": "LPS millimetres",
        "array_order": "ZYX",
        "geometry": asdict(geometry),
        "fragments": [],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bimax-export-", dir=destination.parent))
    try:
        for label, box in enumerate(boxes, start=1):
            if box is None:
                continue
            mask = np.asarray(doc.fragments[box] == label, dtype=np.uint8)
            crop_origin = doc.geometry.index_to_world([axis.start for axis in box[::-1]])
            crop_geometry = Geometry(
                mask.shape, doc.geometry.spacing, tuple(crop_origin), doc.geometry.direction
            )
            image = _image(mask, crop_geometry)
            inverse = np.linalg.inv(transforms[label])
            transform = sitk.AffineTransform(3)
            transform.SetMatrix(inverse[:3, :3].ravel().tolist())
            transform.SetTranslation(inverse[:3, 3].tolist())
            result = sitk.Resample(
                image,
                geometry.shape[::-1],
                transform,
                sitk.sitkNearestNeighbor,
                geometry.origin,
                geometry.spacing,
                geometry.direction,
                0,
                sitk.sitkUInt8,
            )
            filename = f"{label:02d}-{doc.fragment_names[label].lower().replace(' ', '-')}.nii.gz"
            sitk.WriteImage(result, str(temporary / filename), True)
            manifest["fragments"].append(
                {
                    "id": label,
                    "identity": doc.fragment_descriptors[label].id,
                    "anatomy": doc.fragment_descriptors[label].anatomy,
                    "motion": doc.fragment_descriptors[label].motion,
                    "name": doc.fragment_names[label],
                    "file": filename,
                    "transform_lps": transforms[label].tolist(),
                }
            )
            del mask, image, result
        with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "manifest.json"
