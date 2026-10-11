"""Private, disposable display meshes; authoritative volume/project data is never cached here."""

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vtkmodules.vtkCommonCore import vtkVersion

from bimax import surfaces
from bimax.model import Geometry
from bimax.surfaces import Mesh

ALGORITHM_VERSION = 1
CACHE_BYTES = 512 * 1024**2
_HASH_CHUNK = 4 * 1024**2
_ENTRY = re.compile(r"[0-9a-f]{64}\.npz")
_TEMP = re.compile(r"\.[0-9a-f]{64}\.[a-z0-9_]+\.tmp")
_MEMBER = re.compile(r"([1-9]|10)_(vertices|faces|normals)\.npy")


def _directory() -> Path:
    configured = Path(os.environ.get("XDG_CACHE_HOME", ""))
    base = configured if configured.is_absolute() else Path.home() / ".cache"
    for directory in (base / "bimax", base / "bimax" / "surfaces"):
        if directory.is_symlink():
            raise OSError("The display cache must be a private directory.")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.stat().st_uid != os.getuid():
            raise OSError("The display cache must belong to the current user.")
        directory.chmod(0o700)
    return base / "bimax" / "surfaces"


def _key(labels: np.ndarray, geometry: Geometry) -> str:
    metadata = {
        "algorithm": ALGORITHM_VERSION,
        "vtk": vtkVersion.GetVTKVersion(),
        "triangle_target": surfaces.DISPLAY_TRIANGLE_TARGET,
        "error_voxel_fraction": surfaces.DISPLAY_ERROR_VOXEL_FRACTION,
        "geometry": asdict(geometry),
        "dtype": labels.dtype.str,
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    # Buffered C-order iteration is bounded even for a non-contiguous input.
    with np.nditer(
        labels, flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"], order="C", buffersize=_HASH_CHUNK // labels.dtype.itemsize,
    ) as chunks:
        for chunk in chunks:
            digest.update(np.asarray(chunk).tobytes())
    return digest.hexdigest()


def _validate_meshes(meshes: dict[int, Mesh]) -> None:
    total = 0
    for label, mesh in meshes.items():
        if type(label) is not int or label not in range(1, 11):
            raise ValueError("Invalid cached display label.")
        vertices, faces, normals = mesh.vertices, mesh.faces, mesh.normals
        for array, dtype in ((vertices, np.float32), (faces, np.int32), (normals, np.float32)):
            if array.dtype != dtype or array.ndim != 2 or array.shape[1] != 3:
                raise ValueError("Invalid cached mesh array.")
            total += array.nbytes
        if (
            total > CACHE_BYTES
            or vertices.shape != normals.shape
            or not len(vertices)
            or not len(faces)
            or not np.all(np.isfinite(vertices))
            or not np.all(np.isfinite(normals))
            or faces.min() < 0
            or faces.max() >= len(vertices)
        ):
            raise ValueError("Invalid or oversized cached display mesh.")


def _read(path: Path) -> dict[int, Mesh]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > CACHE_BYTES:
            raise ValueError("Invalid display cache entry.")
        members = _validated_members(stream)
        # Read the same descriptor we validated, even if another worker replaces
        # this cache entry atomically in the meantime.
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as arrays:
            meshes = {
                label: Mesh(*(arrays[f"{label}_{name}"] for name in ("vertices", "faces", "normals")))
                for label in members
            }
    _validate_meshes(meshes)
    return meshes


def _validated_members(stream) -> dict[int, set[str]]:
    members: dict[int, set[str]] = {}
    total = 0
    # Validate NPY headers against stored sizes before np.load may allocate arrays.
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        if len(infos) > 24 or len({info.filename for info in infos}) != len(infos):
            raise ValueError("Invalid cached display members.")
        for info in infos:
            match = _MEMBER.fullmatch(info.filename)
            if match is None or info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("Invalid cached display member.")
            label, name = int(match[1]), match[2]
            members.setdefault(label, set()).add(name)
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("Unsupported cached array header.")
                expected_dtype = np.dtype(np.int32 if name == "faces" else np.float32)
                if len(shape) != 2 or shape[1] != 3 or shape[0] <= 0 or dtype != expected_dtype:
                    raise ValueError("Invalid cached array dimensions or dtype.")
                size = shape[0] * shape[1] * dtype.itemsize
                total += size
                if fortran or total > CACHE_BYTES or member.tell() + size != info.file_size:
                    raise ValueError("Oversized or truncated cached array.")
        if any(names != {"vertices", "faces", "normals"} for names in members.values()):
            raise ValueError("Incomplete cached display mesh.")
    return members


def _prune(directory: Path) -> None:
    entries = []
    for path in directory.iterdir():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            continue
        if _TEMP.fullmatch(path.name):
            # Called under the writer lock: these can only be interrupted writes.
            path.unlink(missing_ok=True)
        elif _ENTRY.fullmatch(path.name):
            entries.append((info.st_mtime_ns, info.st_size, path))
    total = sum(size for _, size, _ in entries)
    for _, size, path in sorted(entries):
        if total <= CACHE_BYTES:
            break
        path.unlink(missing_ok=True)
        total -= size


def _write(path: Path, meshes: dict[int, Mesh]) -> None:
    _validate_meshes(meshes)
    arrays = {
        f"{label}_{name}": np.ascontiguousarray(getattr(mesh, name))
        for label, mesh in meshes.items()
        for name in ("vertices", "faces", "normals")
    }
    lock_fd = os.open(path.parent / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _prune(path.parent)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez(stream, allow_pickle=False, **arrays)
            if temporary.stat().st_size > CACHE_BYTES:
                return
            os.replace(temporary, path)
            _prune(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def cached_surfaces(
    labels: np.ndarray,
    geometry: Geometry,
    progress: Callable[[str], None] | None = None,
) -> dict[int, Mesh]:
    """Reuse only exact-content, current-algorithm display meshes; cache failures are harmless."""
    path = None
    try:
        if progress:
            progress("Checking local display cache")
        path = _directory() / f"{_key(labels, geometry)}.npz"
        meshes = _read(path)
    except Exception:
        # This regenerable optimization must never prevent opening a valid project.
        pass
    else:
        try:
            os.utime(path, follow_symlinks=False)
        except OSError:
            pass
        if progress:
            progress("Loaded local display cache")
        return meshes
    meshes = surfaces.make_surfaces(labels, geometry, progress=progress)
    if path is not None:
        try:
            _write(path, meshes)
        except Exception:
            pass
    return meshes
