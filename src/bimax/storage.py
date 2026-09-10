"""Portable projects and physical-space fragment labelmap exports."""

from __future__ import annotations

import itertools
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from bimax.model import FRAGMENTS, Document, Geometry, Motion

_VERSION = 1
_MAX_METADATA = 8 * 1024 * 1024
_MAX_VOXELS = 2**31 - 1
_VOLUMES = ("source", "anatomy", "seeds", "barriers", "fragments")
_MOTION_KEYS = {"linked", "maxilla", "mandible", "chin"}


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must contain {length} finite numbers.")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers.") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite numbers.")
    return array


def _geometry(value: Any) -> Geometry:
    if not isinstance(value, dict) or set(value) != {"shape", "spacing", "origin", "direction"}:
        raise ValueError("Project geometry is incomplete or unsupported.")
    shape = value["shape"]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(type(size) is not int or size <= 0 for size in shape)
        or int(np.prod(shape, dtype=object)) > _MAX_VOXELS
    ):
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
    if np.linalg.norm(_vector(value["normal"], 3, name + " normal")) < 1e-8:
        raise ValueError(f"{name} normal cannot be zero.")


def _validate_cuts(cuts: Any) -> None:
    if not isinstance(cuts, dict) or set(cuts) - {"lefort", "bsso", "genio"}:
        raise ValueError("Unknown operation in project cut definitions.")
    for name, cut in cuts.items():
        expected = (
            {"enabled", "midline", "left", "right"}
            if name == "bsso"
            else {"enabled", "plane", "region"}
        )
        if not isinstance(cut, dict) or set(cut) != expected or type(cut["enabled"]) is not bool:
            raise ValueError(f"Invalid {name} cut definition.")
        if name == "bsso":
            _vector([cut["midline"]], 1, "BSSO midline")
            for side in ("left", "right"):
                if not isinstance(cut[side], dict) or set(cut[side]) != {
                    "front",
                    "upper",
                    "sagittal",
                }:
                    raise ValueError(f"Invalid BSSO {side} planes.")
                for kind, plane in cut[side].items():
                    _plane(plane, f"BSSO {side} {kind}")
        else:
            _plane(cut["plane"], name)
            region = cut["region"]
            if not isinstance(region, dict) or set(region) != {"min", "max"}:
                raise ValueError(f"Invalid {name} region.")
            low = _vector(region["min"], 3, name + " minimum")
            high = _vector(region["max"], 3, name + " maximum")
            if np.any(low >= high):
                raise ValueError(f"{name} region must have positive extent.")


def _metadata(doc: Document) -> dict[str, Any]:
    return {
        "format": "bimax",
        "version": _VERSION,
        "geometry": asdict(doc.geometry),
        "frame": doc.frame.tolist(),
        "cuts": doc.cuts,
        "confirmations": doc.confirmations,
        "scenarios": {
            name: {key: asdict(motion) for key, motion in motions.items()}
            for name, motions in doc.scenarios.items()
        },
        "active_scenario": doc.active_scenario,
        "landmarks": doc.landmarks,
        "source_name": doc.source_name,
        "revision": doc.revision,
        "volumes": [name for name in _VOLUMES if getattr(doc, name) is not None],
    }


def _validated_metadata(value: Any) -> tuple[Geometry, dict[str, dict[str, Motion]]]:
    fields = {
        "format",
        "version",
        "geometry",
        "frame",
        "cuts",
        "confirmations",
        "scenarios",
        "active_scenario",
        "landmarks",
        "source_name",
        "revision",
        "volumes",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value["format"] != "bimax"
        or type(value["version"]) is not int
        or value["version"] != _VERSION
    ):
        raise ValueError("Unsupported or incomplete BIMAX project metadata.")
    geometry = _geometry(value["geometry"])
    frame = np.asarray(value["frame"], dtype=float)
    if (
        frame.shape != (4, 4)
        or not np.all(np.isfinite(frame))
        or not np.allclose(frame[3], [0, 0, 0, 1])
        or not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(frame[:3, :3]), 1.0)
    ):
        raise ValueError("Planning frame must be a finite, right-handed rigid transform.")
    _validate_cuts(value["cuts"])
    if not isinstance(value["confirmations"], dict):
        raise ValueError("Fragment confirmation points must be an object.")
    for key, point in value["confirmations"].items():
        if key not in {str(label) for label in FRAGMENTS}:
            raise ValueError("Unknown fragment confirmation label.")
        _vector(point, 3, "Fragment confirmation")
    scenarios: dict[str, dict[str, Motion]] = {}
    if not isinstance(value["scenarios"], dict) or not value["scenarios"]:
        raise ValueError("A project needs at least one movement scenario.")
    for name, motions in value["scenarios"].items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(motions, dict)
            or set(motions) != _MOTION_KEYS
        ):
            raise ValueError("Invalid movement scenario.")
        scenarios[name] = {}
        for key, motion in motions.items():
            if (
                not isinstance(motion, dict)
                or set(motion) != {"translation", "angles", "pivot", "pivot_set"}
                or type(motion["pivot_set"]) is not bool
            ):
                raise ValueError("Invalid movement parameters.")
            for field in ("translation", "angles", "pivot"):
                _vector(motion[field], 3, f"{name} {key} {field}")
            scenarios[name][key] = Motion(**motion)
    if not isinstance(value["active_scenario"], str) or value["active_scenario"] not in scenarios:
        raise ValueError("Active scenario is missing from the project.")
    if (
        not isinstance(value["source_name"], str)
        or type(value["revision"]) is not int
        or value["revision"] < 0
    ):
        raise ValueError("Invalid project source name or revision.")
    if not isinstance(value["landmarks"], list) or any(
        not isinstance(landmark, dict) for landmark in value["landmarks"]
    ):
        raise ValueError("Project landmarks must be a list of objects.")
    for landmark in value["landmarks"]:
        if (
            set(landmark) != {"name", "point", "fragment"}
            or not isinstance(landmark["name"], str)
            or not landmark["name"].strip()
            or type(landmark["fragment"]) is not int
            or landmark["fragment"] not in FRAGMENTS
        ):
            raise ValueError("Invalid project landmark.")
        _vector(landmark["point"], 3, "Landmark point")
    volumes = value["volumes"]
    if volumes not in [list(_VOLUMES), list(_VOLUMES[:-1])]:
        raise ValueError("Project volume list is incomplete or unsupported.")
    # This also rejects NaN and infinity anywhere in generic JSON fields.
    json.dumps(value, allow_nan=False)
    return geometry, scenarios


def _validate_arrays(doc: Document) -> None:
    shape = doc.geometry.shape
    for name in _VOLUMES:
        array = getattr(doc, name)
        if array is None and name == "fragments":
            continue
        allowed_types = (
            (np.dtype("uint8"), np.dtype("bool"))
            if name in {"source", "barriers"}
            else (np.dtype("uint8"),)
        )
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype not in allowed_types
        ):
            raise ValueError(f"{name.title()} must be a uint8 volume matching the source geometry.")
    any_bone = False
    # Bound temporary masks even on the full-resolution skull volume.
    depth = max(1, 1_000_000 // (shape[1] * shape[2]))
    parent = np.array([0, 2, 2, 3, 3, 3, 3, 4, 1], dtype=np.uint8)
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
        if np.any(seeds > 4) or np.any(seeds == 1) or np.any((seeds != 0) & ~bone):
            raise ValueError("Seeds must be 0, 2, 3 or 4 and lie inside source bone.")
        if np.any(barriers > 1) or np.any((barriers != 0) & ~bone):
            raise ValueError("Barriers must be binary and lie inside source bone.")
        if doc.fragments is not None:
            fragments = doc.fragments[slab]
            if np.any(fragments > 8) or not np.array_equal(fragments != 0, bone):
                raise ValueError("Fragment labels must preserve every source foreground voxel.")
            if not np.array_equal(parent[fragments], anatomy):
                raise ValueError(
                    "Fragment ownership does not match its anatomical parent; rebuild cuts."
                )
    if not any_bone:
        raise ValueError("Source bone mask is empty.")


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


def save_project(doc: Document, path: str | Path) -> None:
    """Atomically save a complete project, never modifying the source volume."""
    destination = Path(path).absolute()
    if destination.suffix.lower() != ".bimax" or destination.is_symlink():
        raise ValueError("Choose a regular .bimax project filename.")
    metadata = _metadata(doc)
    _validated_metadata(metadata)
    _validate_arrays(doc)
    encoded = json.dumps(metadata, allow_nan=False, indent=2).encode("utf-8")
    if len(encoded) > _MAX_METADATA:
        raise ValueError("Project metadata exceeds the 8 MiB limit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bimax-save-", dir=destination.parent) as temporary:
        directory = Path(temporary)
        archive_path = directory / "project.bimax"
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            archive.writestr("project.json", encoded)
            for name in metadata["volumes"]:
                volume_path = directory / f"{name}.nii.gz"
                sitk.WriteImage(_image(getattr(doc, name), doc.geometry), str(volume_path), True)
                archive.write(volume_path, f"volumes/{name}.nii.gz")
                volume_path.unlink()
        with archive_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(archive_path, destination)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key}.")
        result[key] = value
    return result


def load_project(path: str | Path) -> Document:
    """Load only the explicit project members; archive paths are never extracted."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "project.json" not in names:
                raise ValueError("Project archive contains duplicate or missing metadata.")
            if archive.getinfo("project.json").file_size > _MAX_METADATA:
                raise ValueError("Project metadata exceeds the 8 MiB limit.")
            metadata = json.loads(
                archive.read("project.json"),
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
            geometry, scenarios = _validated_metadata(metadata)
            expected = {"project.json"} | {f"volumes/{name}.nii.gz" for name in metadata["volumes"]}
            if set(names) != expected:
                raise ValueError("Project archive contains missing or unexpected members.")
            arrays: dict[str, np.ndarray] = {}
            max_member = int(np.prod(geometry.shape)) + 1024 * 1024
            with tempfile.TemporaryDirectory(prefix="bimax-load-") as temporary:
                for name in metadata["volumes"]:
                    member = f"volumes/{name}.nii.gz"
                    if archive.getinfo(member).file_size > max_member:
                        raise ValueError(f"Project volume {name} is unexpectedly large.")
                    volume_path = Path(temporary) / f"{name}.nii.gz"
                    with archive.open(member) as source, volume_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    arrays[name] = _read_volume(volume_path, geometry)
                    volume_path.unlink()
            doc = Document(
                geometry=geometry,
                source=arrays["source"],
                anatomy=arrays["anatomy"],
                seeds=arrays["seeds"],
                barriers=arrays["barriers"],
                fragments=arrays.get("fragments"),
                frame=np.asarray(metadata["frame"], dtype=float),
                cuts=metadata["cuts"],
                confirmations=metadata["confirmations"],
                scenarios=scenarios,
                active_scenario=metadata["active_scenario"],
                landmarks=metadata["landmarks"],
                source_name=metadata["source_name"],
                revision=metadata["revision"],
            )
            _validate_arrays(doc)
            doc.source.setflags(write=False)
            return doc
    except (zipfile.BadZipFile, RuntimeError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read BIMAX project: {exc}") from exc


def _export_grid(doc: Document, transforms: dict[int, np.ndarray]) -> tuple[Geometry, list[Any]]:
    boxes = ndimage.find_objects(doc.fragments, max_label=max(FRAGMENTS))
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

    _validated_metadata(_metadata(doc))
    _validate_arrays(doc)
    if doc.fragments is None:
        raise ValueError("Build and review operation fragments before exporting.")
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
            filename = f"{label:02d}-{FRAGMENTS[label].lower().replace(' ', '-')}.nii.gz"
            sitk.WriteImage(result, str(temporary / filename), True)
            manifest["fragments"].append(
                {
                    "id": label,
                    "name": FRAGMENTS[label],
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
