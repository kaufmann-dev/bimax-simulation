"""Immutable compact cut masks and prepared anatomical geometry, stored locally and in projects."""

import hashlib
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from bimax.engine import (
    CUT_SLAB_VOXELS,
    Progress,
    _check_memory,
    _report,
    assert_partition,
    cut_membership,
)
from bimax.model import Artifact, Document, Geometry, fingerprint
from bimax.surface_cache import cached_surfaces
from bimax.surfaces import Mesh

MAX_ARTIFACT_BYTES = 1024**3
_KEY = re.compile(r"[0-9a-f]{64}")


def directory() -> Path:
    configured = Path(os.environ.get("XDG_CACHE_HOME", ""))
    base = configured if configured.is_absolute() else Path.home() / ".cache"
    result = base / "bimax" / "prepared"
    for path in (base / "bimax", result):
        if path.is_symlink():
            raise OSError("The prepared geometry cache must be a private directory.")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.stat().st_uid != os.getuid():
            raise OSError("The prepared geometry cache must belong to the current user.")
        path.chmod(0o700)
    return result


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(key: str) -> Path:
    if not _KEY.fullmatch(key):
        raise ValueError("Invalid prepared geometry identity.")
    return directory() / f"{key}.npz"


def read(ref: Artifact, kind: str, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    path = Path(ref.path)
    if ref.kind != kind or path.is_symlink() or not path.is_file():
        raise ValueError("Prepared geometry is missing or has the wrong type.")
    if path.stat().st_size > MAX_ARTIFACT_BYTES or file_digest(path) != ref.digest:
        raise ValueError("Prepared geometry failed its integrity check.")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) > 64 or len({item.filename for item in members}) != len(members)
                or sum(item.file_size for item in members) > MAX_ARTIFACT_BYTES):
            raise ValueError("Prepared geometry is too large or malformed.")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in (payload.files if names is None else names)}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Prepared geometry cannot contain objects.")
    return arrays


def write(doc: Document, key: str, kind: str, arrays: dict[str, np.ndarray]) -> None:
    if sum(array.nbytes for array in arrays.values()) > MAX_ARTIFACT_BYTES - 1024 * 1024:
        raise ValueError("Prepared geometry exceeds the 1 GiB artifact limit; crop the input volume.")
    path = artifact_path(key)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prepared-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, allow_pickle=False, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("Prepared geometry exceeds the 1 GiB artifact limit.")
        digest = file_digest(temporary)
        os.replace(temporary, path)
        doc.artifacts[key] = Artifact(kind, digest, str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _existing(doc: Document, key: str, kind: str) -> dict[str, np.ndarray] | None:
    ref = doc.artifacts.get(key)
    if ref is None:
        return None
    try:
        return read(ref, kind)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        doc.artifacts.pop(key, None)
        return None


def _mesh_arrays(meshes: dict[int, Mesh]) -> dict[str, np.ndarray]:
    return {f"mesh_{label}_{field}": np.asarray(getattr(mesh, field))
            for label, mesh in meshes.items() for field in ("vertices", "faces", "normals")}


def _meshes(arrays: dict[str, np.ndarray], labels: set[int]) -> dict[int, Mesh]:
    result = {}
    for label in labels:
        names = [f"mesh_{label}_{field}" for field in ("vertices", "faces", "normals")]
        if not any(name in arrays for name in names):
            continue
        if not all(name in arrays for name in names):
            raise ValueError("Prepared mesh is incomplete.")
        vertices, faces, normals = [arrays[name] for name in names]
        if (vertices.dtype != np.float32 or normals.dtype != np.float32 or faces.dtype != np.int32
                or any(a.ndim != 2 or a.shape[1] != 3 for a in (vertices, faces, normals))
                or normals.shape != vertices.shape or not len(vertices) or not len(faces)
                or not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals))
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Prepared mesh contains invalid arrays.")
        result[label] = Mesh(vertices, faces, normals)
    unexpected = {name for name in arrays if name.startswith("mesh_")} - set(_mesh_arrays(result))
    if unexpected:
        raise ValueError("Prepared geometry contains unknown meshes.")
    return result


def basis_key(doc: Document) -> str:
    return fingerprint(["basis", doc.anatomy_key])


def _validate_basis(arrays: dict[str, np.ndarray], geometry: Geometry) -> None:
    indices, parents = arrays["indices"], arrays["parents"]
    if (indices.dtype != np.uint32 or parents.dtype != np.uint8
            or indices.ndim != 1 or parents.shape != indices.shape or not len(indices)
            or int(indices[-1]) >= int(np.prod(geometry.shape))
            or np.any(indices[1:] <= indices[:-1]) or np.any((parents < 1) | (parents > 4))):
        raise ValueError("Prepared bone indexing does not match the volume geometry.")
    _meshes(arrays, {1, 2, 3, 4})


def get_basis(doc: Document, progress: Progress = None) -> dict[str, np.ndarray]:
    key = basis_key(doc)
    arrays = _existing(doc, key, "basis")
    if arrays is not None:
        try:
            _validate_basis(arrays, doc.geometry)
            return arrays
        except (ValueError, KeyError):
            doc.artifacts.pop(key, None)
    assert_partition(doc.source, doc.anatomy, maximum_label=4)
    _check_memory(doc, doc.source.size, bytes_per_voxel=12)
    if doc.anatomy.shape != doc.geometry.shape:
        raise ValueError("Anatomy must match the physical volume geometry.")
    _report(progress, 0, "Indexing shared anatomy")
    indices = np.flatnonzero(doc.anatomy.ravel()).astype(np.uint32)
    arrays = {"indices": indices, "parents": doc.anatomy.flat[indices]}
    meshes = cached_surfaces(doc.anatomy, doc.geometry,
                            progress=lambda message: _report(progress, 0, message))
    arrays.update(_mesh_arrays(meshes))
    _validate_basis(arrays, doc.geometry)
    write(doc, key, "basis", arrays)
    return arrays


def anatomy_display(doc: Document, progress: Progress = None) -> tuple[dict[int, Mesh], np.ndarray]:
    arrays = get_basis(doc, progress)
    counts = np.bincount(arrays["parents"], minlength=5)
    counts[0] = doc.source.size - len(arrays["indices"])
    return _meshes(arrays, {1, 2, 3, 4}), counts


def _unpack(arrays: dict[str, np.ndarray], name: str, count: int) -> np.ndarray:
    values = arrays[name]
    if values.dtype != np.uint8 or values.shape != ((count + 7) // 8,):
        raise ValueError("Prepared cut mask has unexpected dimensions.")
    return np.unpackbits(values, count=count).astype(bool)


def prepare_cut(doc: Document, cut_id: str, basis: dict[str, np.ndarray],
                progress: Progress = None) -> dict[str, np.ndarray]:
    key = doc.cut_key(cut_id)
    cut = doc.cuts[cut_id]
    names = ("left", "right") if cut.kind == "bsso" else ("inside",)
    parent = 2 if cut.kind == "lefort" else 3
    indices = basis["indices"][basis["parents"] == parent]
    arrays = _existing(doc, key, "cut")
    if arrays is not None:
        try:
            return {name: _unpack(arrays, name, len(indices)) for name in names}
        except (ValueError, KeyError):
            doc.artifacts.pop(key, None)
    masks = {name: np.zeros(len(indices), dtype=bool) for name in names}
    affine = doc.frame @ doc.geometry.affine
    for start in range(0, len(indices), CUT_SLAB_VOXELS):
        stop = min(start + CUT_SLAB_VOXELS, len(indices))
        z, y, x = np.unravel_index(indices[start:stop], doc.geometry.shape)
        # Match the exact arithmetic order and epsilon of native-voxel cutting.
        points = tuple(affine[axis, 0] * x + affine[axis, 1] * y
                       + affine[axis, 2] * z + affine[axis, 3] for axis in range(3))
        for name, values in cut_membership(points, cut).items():
            masks[name][start:stop] = values
        _report(progress, stop / max(1, len(indices)), f"Preparing {cut.name}")
    write(doc, key, "cut", {name: np.packbits(mask) for name, mask in masks.items()})
    return masks


def _subsystem(doc: Document, kind: str, basis: dict[str, np.ndarray],
               progress: Progress = None) -> dict[str, np.ndarray]:
    key = doc.subsystem_key(kind)
    parent = 2 if kind == "upper" else 3
    selector = basis["parents"] == parent
    count = int(np.count_nonzero(selector))
    upper_labels = {doc.fragment_labels[f"upper:{cut_id}"] for cut_id in doc.upper_cuts}
    allowed = {1, *upper_labels} if kind == "upper" else {3, 4, 5, 6}
    arrays = _existing(doc, key, kind)
    if arrays is not None:
        try:
            _validate_subsystem(arrays, count, allowed)
            return arrays
        except (ValueError, KeyError):
            doc.artifacts.pop(key, None)
    labels = np.full(count, 1 if kind == "upper" else 5, dtype=np.uint8)
    if kind == "upper":
        previous = None
        masks = []
        for cut_id in doc.upper_cuts:
            inside = prepare_cut(doc, cut_id, basis, progress)["inside"]
            if previous is not None and np.any(previous & ~inside):
                raise ValueError("The selected Le Fort cuts cross: the lower-level cut must lie inside the higher-level cut.")
            previous = inside
            masks.append((doc.fragment_labels[f"upper:{cut_id}"], inside))
        for label, mask in reversed(masks):
            labels[mask] = label
        if len(doc.upper_cuts) > 1 and any(not np.any(labels == label) for label in upper_labels):
            raise ValueError("The selected nested cuts leave an upper fragment empty; adjust the cut geometry.")
        ids = doc.upper_cuts
    else:
        if "bsso" in doc.scenario.cut_ids:
            masks_by_side = prepare_cut(doc, "bsso", basis, progress)
            for side, label in (("left", 3), ("right", 4)):
                labels[masks_by_side[side]] = label
        if "genio" in doc.scenario.cut_ids:
            inside = prepare_cut(doc, "genio", basis, progress)["inside"]
            labels[(labels == 5) & inside] = 6
        ids = [cut_id for cut_id in doc.scenario.cut_ids if doc.cuts[cut_id].kind != "lefort"]
    if ids:
        dense = np.zeros(doc.geometry.shape, dtype=np.uint8)
        dense.flat[basis["indices"][selector]] = labels
        meshes = cached_surfaces(dense, doc.geometry,
                                progress=lambda message: _report(progress, 0, message))
        del dense
    else:
        baseline = _meshes(basis, {1, 2, 3, 4}).get(parent)
        meshes = {1 if kind == "upper" else 5: baseline} if baseline is not None else {}
    arrays = {"labels": labels, **_mesh_arrays(meshes)}
    _validate_subsystem(arrays, count, allowed)
    write(doc, key, kind, arrays)
    return arrays


def _validate_subsystem(arrays: dict[str, np.ndarray], count: int, allowed: set[int]) -> None:
    _validate_labels(arrays["labels"], count, allowed)
    labels = arrays["labels"]
    present = set(map(int, np.unique(labels)))
    if set(_meshes(arrays, allowed)) != present:
        raise ValueError("Prepared fragment meshes do not match their assignments.")


def _validate_labels(labels: np.ndarray, count: int, allowed: set[int]) -> None:
    if labels.dtype != np.uint8 or labels.shape != (count,) or not np.isin(labels, sorted(allowed)).all():
        raise ValueError("Prepared fragment assignments are invalid.")


def _text(arrays: dict[str, np.ndarray], key: str) -> str:
    value = arrays[key]
    if value.shape != () or value.dtype.kind != "U" or not _KEY.fullmatch(str(value)):
        raise ValueError("Prepared geometry dependency is invalid.")
    return str(value)


def load_geometry(refs: dict[str, Artifact], key: str, geometry: Geometry,
                  expected: tuple[str, str, str], upper_labels: set[int], document_id: str) -> dict[str, Any]:
    """A cache-hit operation needs only references and geometry, never source arrays."""
    scene = read(refs[key], "scene")
    dependencies = tuple(_text(scene, name) for name in ("basis", "upper", "lower"))
    if dependencies != expected:
        raise ValueError("Prepared geometry belongs to different cuts or anatomy.")
    basis = read(refs[dependencies[0]], "basis")
    _validate_basis(basis, geometry)
    upper, lower = (read(refs[dependencies[1]], "upper"), read(refs[dependencies[2]], "lower"))
    parents = basis["parents"]
    _validate_subsystem(upper, int(np.count_nonzero(parents == 2)), {1, *upper_labels})
    _validate_subsystem(lower, int(np.count_nonzero(parents == 3)), {3, 4, 5, 6})
    compact = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[parents]
    compact[parents == 2] = upper["labels"]
    compact[parents == 3] = lower["labels"]
    fragments = np.zeros(geometry.shape, dtype=np.uint8)
    fragments.flat[basis["indices"]] = compact
    meshes = {**_meshes(upper, {1, *upper_labels}), **_meshes(lower, {3, 4, 5, 6})}
    anatomy_meshes = _meshes(basis, {1, 2, 3, 4})
    for parent, label in ((1, 8), (4, 7)):
        if parent in anatomy_meshes:
            meshes[label] = anatomy_meshes[parent]
    maximum = max({8, *upper_labels})
    counts = np.bincount(compact, minlength=maximum + 1)
    counts[0] = fragments.size - len(compact)
    anatomy_counts = np.bincount(parents, minlength=5)
    anatomy_counts[0] = fragments.size - len(compact)
    return {"fragments": fragments, "surfaces": meshes, "fragment_counts": counts,
            "anatomy_counts": anatomy_counts, "configuration_key": key, "document_id": document_id}


def load_options(doc: Document) -> dict[str, Any]:
    return {"refs": dict(doc.artifacts), "key": doc.configuration_key, "geometry": doc.geometry,
            "expected": (basis_key(doc), doc.subsystem_key("upper"), doc.subsystem_key("lower")),
            "upper_labels": {doc.fragment_labels[f"upper:{key}"] for key in doc.upper_cuts},
            "document_id": doc._identity}


def prepare_geometry(doc: Document, progress: Progress = None) -> dict[str, Any]:
    options = load_options(doc)
    try:
        result = load_geometry(**options)
        _report(progress, 1, "Loaded prepared fragments")
        result["artifacts"] = dict(doc.artifacts)
        result["anatomy_fingerprint"] = {"revision": doc.anatomy_revision, "digest": doc._anatomy_digest}
        return result
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        pass
    basis = get_basis(doc, progress)
    _subsystem(doc, "upper", basis, progress)
    _subsystem(doc, "lower", basis, progress)
    write(doc, doc.configuration_key, "scene", {
        "basis": np.asarray(basis_key(doc)), "upper": np.asarray(doc.subsystem_key("upper")),
        "lower": np.asarray(doc.subsystem_key("lower")),
    })
    result = load_geometry(**load_options(doc))
    result["artifacts"] = dict(doc.artifacts)
    result["anatomy_fingerprint"] = {"revision": doc.anatomy_revision, "digest": doc._anatomy_digest}
    _report(progress, 1, "Prepared fragments ready")
    return result


def prune_references(doc: Document) -> None:
    """Retain current masks and used combinations, without deleting shared disk artifacts."""
    basis = basis_key(doc)
    keep = {basis, *(doc.cut_key(key) for key in doc.cuts)}
    active = doc.active_scenario
    try:
        for name in doc.scenarios:
            doc.active_scenario = name
            keep.update((doc.configuration_key, doc.subsystem_key("upper"), doc.subsystem_key("lower")))
    finally:
        doc.active_scenario = active
    doc.artifacts = {key: ref for key, ref in doc.artifacts.items() if key in keep}


def sample_labels(doc: Document, points: Any) -> np.ndarray:
    """Check a handful of review points directly in compact saved assignments."""
    options = load_options(doc)
    scene = read(doc.artifacts[doc.configuration_key], "scene")
    dependencies = tuple(_text(scene, name) for name in ("basis", "upper", "lower"))
    if dependencies != options["expected"]:
        raise ValueError("Prepared review belongs to different cuts or anatomy.")
    basis = read(doc.artifacts[dependencies[0]], "basis", ("indices", "parents"))
    _validate_basis(basis, doc.geometry)
    upper = read(doc.artifacts[dependencies[1]], "upper", ("labels",))["labels"]
    lower = read(doc.artifacts[dependencies[2]], "lower", ("labels",))["labels"]
    parents, indices = basis["parents"], basis["indices"]
    _validate_labels(upper, int(np.count_nonzero(parents == 2)), {1, *options["upper_labels"]})
    _validate_labels(lower, int(np.count_nonzero(parents == 3)), {3, 4, 5, 6})
    xyz = np.rint(doc.geometry.world_to_index(doc.plan_to_world(points)))
    inside = np.all((xyz >= 0) & (xyz < np.asarray(doc.geometry.shape[::-1])), axis=1)
    result = np.zeros(len(xyz), dtype=np.uint8)
    for i in np.flatnonzero(inside):
        flat = np.ravel_multi_index(tuple(xyz[i, ::-1].astype(int)), doc.geometry.shape)
        position = int(np.searchsorted(indices, flat))
        if position >= len(indices) or indices[position] != flat:
            continue
        parent = int(parents[position])
        rank = np.count_nonzero(parents[:position] == parent)
        result[i] = upper[rank] if parent == 2 else (lower[rank] if parent == 3 else {1: 8, 4: 7}[parent])
    return result


def sample_cut_labels(doc: Document, cut_id: str, points: Any) -> np.ndarray:
    """Validate shared cut confirmation samples from its saved membership bits."""
    basis = read(doc.artifacts[basis_key(doc)], "basis", ("indices", "parents"))
    _validate_basis(basis, doc.geometry)
    arrays = read(doc.artifacts[doc.cut_key(cut_id)], "cut")
    parent = 2 if doc.cuts[cut_id].kind == "lefort" else 3
    indices = basis["indices"][basis["parents"] == parent]
    names = ("left", "right") if doc.cuts[cut_id].kind == "bsso" else ("inside",)
    masks = {name: _unpack(arrays, name, len(indices)) for name in names}
    xyz = np.rint(doc.geometry.world_to_index(doc.plan_to_world(points)))
    inside = np.all((xyz >= 0) & (xyz < np.asarray(doc.geometry.shape[::-1])), axis=1)
    result = np.zeros(len(xyz), dtype=np.uint8)
    for i in np.flatnonzero(inside):
        voxel = tuple(xyz[i, ::-1].astype(int))
        result[i] = {0: 0, 1: 8, 2: 1, 3: 5, 4: 7}[int(doc.anatomy[voxel])]
        flat = np.ravel_multi_index(voxel, doc.geometry.shape)
        position = int(np.searchsorted(indices, flat))
        if position >= len(indices) or indices[position] != flat:
            continue
        if doc.cuts[cut_id].kind == "bsso":
            if masks["left"][position]:
                result[i] = 3
            if masks["right"][position]:
                result[i] = 4
        elif masks["inside"][position]:
            result[i] = 2 if parent == 2 else 6
    return result
