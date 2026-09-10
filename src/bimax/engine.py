"""Volume operations in baseline coordinates; this module never imports Qt."""

import gzip
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from bimax.model import Document, Geometry, Motion

Progress = Callable[[float, str], None] | None
MEMORY_BUDGET = 16 * 1024**3
FACE_CONNECTIVITY = ndi.generate_binary_structure(3, 1)


def _report(progress: Progress, value: float, message: str) -> None:
    if progress is not None:
        progress(value, message)


def _foreground_bounds(mask: np.ndarray) -> tuple[slice, slice, slice]:
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != axis)))
        if not occupied.size:
            raise ValueError("The bone mask is empty.")
        bounds.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
    return tuple(bounds)  # type: ignore[return-value]


def _roi_slices(shape: tuple[int, ...], roi: Any) -> tuple[slice, slice, slice]:
    if len(roi) != 3:
        raise ValueError("A region must contain three ZYX slices.")
    result = []
    for size, interval in zip(shape, roi, strict=True):
        if not isinstance(interval, slice):
            raise ValueError("A region must contain three ZYX slices.")
        start, stop, step = interval.indices(size)
        if step != 1 or stop <= start:
            raise ValueError("Choose a nonempty region with unit voxel steps.")
        result.append(slice(start, stop))
    return tuple(result)  # type: ignore[return-value]


def _check_memory(doc: Document, voxels: int, bytes_per_voxel: int = 44) -> None:
    # Includes document arrays, proposal, labels, distance image, watershed's
    # float64 conversion and padded arrays, and working queues with headroom.
    baseline = sum(array.nbytes for array in (doc.source, doc.anatomy, doc.seeds, doc.barriers))
    if doc.fragments is not None:
        baseline += doc.fragments.nbytes
    estimate = baseline + doc.source.size * 2 + voxels * bytes_per_voxel
    if estimate > MEMORY_BUDGET:
        raise ValueError(
            "This operation exceeds the 16 GiB worker memory budget. "
            "Select a smaller region and process it separately."
        )


def _check_stored_floats(path: Path, count: int) -> None:
    # ITK's NIfTI reader replaces NaN/Inf with zero. Validate floating payloads
    # before reading so invalid intensity data cannot silently become a mask.
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as stream:
        header = stream.read(540)
        if len(header) < 348:
            return  # The image reader will report malformed/truncated headers.
        endian = "<" if struct.unpack_from("<i", header)[0] in (348, 540) else ">"
        size = struct.unpack_from(endian + "i", header)[0]
        if size == 348:
            datatype = struct.unpack_from(endian + "h", header, 70)[0]
            offset = int(struct.unpack_from(endian + "f", header, 108)[0])
            scale = struct.unpack_from(endian + "ff", header, 112)
        elif size == 540 and len(header) >= 540:
            datatype = struct.unpack_from(endian + "h", header, 12)[0]
            offset = struct.unpack_from(endian + "q", header, 168)[0]
            scale = struct.unpack_from(endian + "dd", header, 176)
        else:
            return
        if not np.all(np.isfinite(scale)):
            raise ValueError("The input has nonfinite NIfTI intensity scaling.")
        float_types = {16: "f4", 64: "f8", 1536: "f16"}
        if datatype not in float_types:
            return
        dtype = np.dtype(endian + float_types[datatype])
        stream.seek(offset)
        remaining = count * dtype.itemsize
        while remaining:
            payload = stream.read(min(8 * 1024**2, remaining))
            if not payload or len(payload) % dtype.itemsize:
                raise ValueError("The NIfTI voxel payload is truncated.")
            if not np.all(np.isfinite(np.frombuffer(payload, dtype=dtype))):
                raise ValueError("The input must contain only background 0 and bone 1; nonfinite voxels are invalid.")
            remaining -= len(payload)


def load_volume(path: str | Path, progress: Progress = None) -> Document:
    """Read a scalar, nonempty 0/1 NIfTI without resampling its physical grid."""
    path = Path(path)
    if not (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz")):
        raise ValueError("Import a binary bone mask in .nii or .nii.gz format.")
    _report(progress, 0.05, "Reading binary bone volume")
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1:
        raise ValueError("The input must be one scalar three-dimensional NIfTI volume.")
    count = int(np.prod(reader.GetSize(), dtype=np.int64))
    if count * 12 > MEMORY_BUDGET:
        raise ValueError("This volume exceeds the 16 GiB memory budget. Crop it before import.")
    _check_stored_floats(path, count)
    image = reader.Execute()
    values = sitk.GetArrayViewFromImage(image)
    if values.dtype.kind not in "buif":
        raise ValueError("The input must be a scalar real-valued binary bone mask.")
    has_bone = False
    for slab in values:
        if not np.all((slab == 0) | (slab == 1)):
            raise ValueError(
                "The input must contain only background 0 and bone 1. "
                "Convert the intensity scan or labeled volume to a binary bone mask first."
            )
        has_bone = has_bone or bool(np.any(slab))
    if not has_bone:
        raise ValueError("The bone mask is empty.")
    spacing = tuple(image.GetSpacing())
    origin = tuple(image.GetOrigin())
    direction = tuple(image.GetDirection())
    if not np.all(np.isfinite([*spacing, *origin, *direction])) or min(spacing) <= 0:
        raise ValueError("The input has invalid physical geometry.")
    source = np.array(values, dtype=np.uint8, copy=True)
    geometry = Geometry(tuple(source.shape), spacing, origin, direction)
    source.flags.writeable = False
    doc = Document(
        geometry=geometry,
        source=source,
        anatomy=source.copy(),
        seeds=np.zeros(source.shape, dtype=np.uint8),
        barriers=np.zeros(source.shape, dtype=bool),
        source_name=path.name,
    )
    doc.cuts = default_cuts(doc)
    _report(progress, 1.0, "Binary bone volume ready")
    return doc


def planning_bounds(doc: Document) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned planning bounds of occupied voxel centres."""
    z, y, x = _foreground_bounds(doc.source)
    corners = np.array(
        [[xx, yy, zz] for xx in (x.start, x.stop - 1)
         for yy in (y.start, y.stop - 1) for zz in (z.start, z.stop - 1)], dtype=float
    )
    points = doc.world_to_plan(doc.geometry.index_to_world(corners))
    return points.min(axis=0), points.max(axis=0)


def default_cuts(doc: Document) -> dict[str, Any]:
    """Editable suggestions only; cuts remain disabled until chosen by the user."""
    low, high = planning_bounds(doc)
    extent = high - low
    centre = (low + high) / 2

    def plane(origin: Any, normal: Any) -> dict[str, Any]:
        return {"origin": list(map(float, origin)), "normal": list(map(float, normal))}

    def region(minimum: Any, maximum: Any) -> dict[str, Any]:
        return {"min": list(map(float, minimum)), "max": list(map(float, maximum))}

    sides = {}
    for side, sign in (("left", -1), ("right", 1)):
        sides[side] = {
            "front": plane([centre[0], low[1] + extent[1] * 0.32, centre[2]], [0, -1, 0]),
            "upper": plane([centre[0], centre[1], low[2] + extent[2] * 0.28], [0, 0, 1]),
            "sagittal": plane([centre[0] + sign * extent[0] * 0.23, centre[1], centre[2]], [sign, 0, 0]),
        }
    return {
        "lefort": {
            "enabled": False,
            "plane": plane([centre[0], centre[1], low[2] + extent[2] * 0.43], [0, 0, -1]),
            "region": region([low[0], low[1] + extent[1] * 0.35, low[2]], high),
        },
        "bsso": {"enabled": False, "midline": float(centre[0]), **sides},
        "genio": {
            "enabled": False,
            "plane": plane([centre[0], centre[1], low[2] + extent[2] * 0.13], [0, 0, -1]),
            "region": region([low[0], low[1] + extent[1] * 0.50, low[2]], high),
        },
    }


def separate(doc: Document, roi: Any = None, progress: Progress = None) -> np.ndarray:
    """Preview seeded, face-connected anatomical separation on bone only.

    The ROI is three ZYX slices. Accepted assignments outside it are retained.
    Inside it, seed-free islands and barrier voxels are explicitly unassigned.
    Distances use original millimetre spacing; no resolution reduction occurs.
    """
    roi = _foreground_bounds(doc.source) if roi is None else _roi_slices(doc.source.shape, roi)
    shape = tuple(interval.stop - interval.start for interval in roi)
    _check_memory(doc, int(np.prod(shape)))
    source = doc.source[roi]
    seeds = doc.seeds[roi]
    barriers = doc.barriers[roi]
    if not np.all(np.isin(seeds, [0, 2, 3, 4])):
        raise ValueError("Separation seeds must identify skull, mandible, or excluded material.")
    if np.any((seeds != 0) & (source == 0)):
        raise ValueError("Place separation seeds on bone voxels.")
    _report(progress, 0.05, "Finding connected bone components")
    working = (source != 0) & ~barriers.astype(bool)
    labels, count = ndi.label(working, structure=FACE_CONNECTIVITY)
    del working
    objects = ndi.find_objects(labels, max_label=count)
    proposal = doc.anatomy.copy()
    local = proposal[roi]
    local[...] = source
    for component, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        component_mask = labels[bounds] == component
        markers = np.where(component_mask, seeds[bounds], 0).astype(np.uint8)
        classes = np.unique(markers)
        classes = classes[classes != 0]
        if classes.size == 1:
            local[bounds][component_mask] = classes[0]
        elif classes.size > 1:
            _report(progress, 0.15 + 0.75 * component / max(count, 1), "Separating competing anatomical seeds")
            # Explicit background padding keeps distances well-defined even for
            # bone touching a crop boundary or filling the whole crop.
            padded = np.pad(component_mask, 1)
            image = sitk.GetImageFromArray(padded.astype(np.uint8))
            image.SetSpacing(doc.geometry.spacing)
            distance_image = sitk.SignedMaurerDistanceMap(
                image, insideIsPositive=True, squaredDistance=False, useImageSpacing=True
            )
            distance = sitk.GetArrayFromImage(distance_image)[1:-1, 1:-1, 1:-1].copy()
            del image, distance_image, padded
            distance *= -1
            partition = watershed(
                distance, markers=markers, connectivity=FACE_CONNECTIVITY,
                mask=component_mask, watershed_line=False,
            )
            local[bounds][component_mask] = partition[component_mask]
        if component % 100 == 0:
            _report(progress, 0.15 + 0.75 * component / max(count, 1), "Assigning bone islands")
    assert_partition(doc.source, proposal, maximum_label=4)
    _report(progress, 1.0, "Separation preview ready")
    return proposal


def component_indices(doc: Document, xyz: Any) -> np.ndarray:
    """Return flat C-order indices for the face-connected bone at voxel XYZ."""
    point = np.asarray(xyz, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("Pick one finite XYZ voxel coordinate.")
    index = np.rint(point).astype(int)[::-1]
    if np.any(index < 0) or np.any(index >= doc.source.shape):
        raise ValueError("The selected voxel is outside the volume.")
    at = tuple(index)
    if not doc.source[at]:
        return np.empty(0, dtype=np.int64)
    _check_memory(doc, doc.source.size, bytes_per_voxel=14)
    labels, _ = ndi.label(doc.source, structure=FACE_CONNECTIVITY)
    label = labels[at]
    return np.flatnonzero(labels.ravel() == label)


def assert_partition(source: np.ndarray, labels: np.ndarray, maximum_label: int = 8) -> None:
    """A scalar label image owns every original foreground voxel exactly once."""
    if labels.shape != source.shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Partition labels must be an integer array on the source grid.")
    # Slabs avoid allocating additional full-size Boolean validation volumes.
    for source_slice, label_slice in zip(source, labels, strict=True):
        if np.any(label_slice < 0) or np.any(label_slice > maximum_label):
            raise ValueError("The partition contains an invalid fragment label.")
        if not np.array_equal(source_slice != 0, label_slice != 0):
            raise ValueError("The partition must preserve every source bone voxel and no background.")


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite coordinates.")
    return result


def _plane_mask(points: tuple[np.ndarray, ...], plane: dict[str, Any]) -> np.ndarray:
    origin = _vector(plane["origin"], "Plane origin")
    normal = _vector(plane["normal"], "Plane normal")
    length = np.linalg.norm(normal)
    if length < 1e-12:
        raise ValueError("A cut plane needs a nonzero normal.")
    normal = normal / length
    signed = sum(points[axis] * normal[axis] for axis in range(3)) - normal @ origin
    return signed >= -1e-8


def _bounded_mask(points: tuple[np.ndarray, ...], cut: dict[str, Any]) -> np.ndarray:
    minimum = _vector(cut["region"]["min"], "Region minimum")
    maximum = _vector(cut["region"]["max"], "Region maximum")
    if np.any(maximum < minimum):
        raise ValueError("Region maxima must be greater than or equal to minima.")
    mask = _plane_mask(points, cut["plane"])
    for axis in range(3):
        mask &= (points[axis] >= minimum[axis] - 1e-8) & (points[axis] <= maximum[axis] + 1e-8)
    return mask


def _points_for_slab(doc: Document, start: int, stop: int) -> tuple[np.ndarray, ...]:
    affine = doc.frame @ doc.geometry.affine
    z, y, x = np.ogrid[start:stop, :doc.source.shape[1], :doc.source.shape[2]]
    return tuple(affine[axis, 0] * x + affine[axis, 1] * y + affine[axis, 2] * z + affine[axis, 3]
                 for axis in range(3))


def build_fragments(doc: Document, validate: bool = True, progress: Progress = None) -> np.ndarray:
    """Partition approved anatomy using zero-thickness baseline-space cuts."""
    assert_partition(doc.source, doc.anatomy, maximum_label=4)
    if validate and any(np.any(slab == 1) for slab in doc.anatomy):
        raise ValueError("Assign all bone to anatomy or explicitly exclude it before accepting cuts.")
    _check_memory(doc, doc.source.size, bytes_per_voxel=3)
    fragments = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[doc.anatomy]
    lefort = doc.cuts.get("lefort", {"enabled": False})
    bsso = doc.cuts.get("bsso", {"enabled": False})
    genio = doc.cuts.get("genio", {"enabled": False})
    slab_depth = max(1, 2_000_000 // (doc.source.shape[1] * doc.source.shape[2]))
    if any(cut["enabled"] for cut in (lefort, bsso, genio)):
        for start in range(0, doc.source.shape[0], slab_depth):
            stop = min(start + slab_depth, doc.source.shape[0])
            points = _points_for_slab(doc, start, stop)
            labels = fragments[start:stop]
            if lefort["enabled"]:
                labels[(labels == 1) & _bounded_mask(points, lefort)] = 2
            if bsso["enabled"]:
                midline = float(bsso["midline"])
                if not np.isfinite(midline):
                    raise ValueError("The BSSO midline must be finite.")
                for side, label in (("left", 3), ("right", 4)):
                    planes = bsso[side]
                    side_mask = points[0] < midline if side == "left" else points[0] >= midline
                    proximal = (
                        side_mask & _plane_mask(points, planes["front"])
                        & (_plane_mask(points, planes["upper"]) | _plane_mask(points, planes["sagittal"]))
                    )
                    labels[(labels == 5) & proximal] = label
            if genio["enabled"]:
                labels[(labels == 5) & _bounded_mask(points, genio)] = 6
            _report(progress, stop / doc.source.shape[0] * 0.9, "Partitioning bone fragments")
    assert_partition(doc.source, fragments)
    if validate:
        required = set()
        if lefort["enabled"]:
            required.add(2)
        if bsso["enabled"]:
            required.update((3, 4, 5))
        if genio["enabled"]:
            required.update((5, 6))
        from bimax.model import FRAGMENTS

        for label in sorted(required):
            if not np.any(fragments == label):
                raise ValueError(f"The cuts leave {FRAGMENTS[label]} empty. Adjust the cut geometry.")
            confirmation = doc.confirmations.get(str(label))
            if confirmation is None:
                raise ValueError(f"Place a confirmation point inside {FRAGMENTS[label]} before accepting cuts.")
            point = _vector(confirmation, "Fragment confirmation")
            xyz = np.rint(doc.geometry.world_to_index(doc.plan_to_world(point))).astype(int)
            zyx = xyz[::-1]
            if (np.any(zyx < 0) or np.any(zyx >= fragments.shape)
                    or fragments[tuple(zyx)] != label):
                raise ValueError(f"The confirmation point for {FRAGMENTS[label]} is outside its fragment.")
    _report(progress, 1.0, "Fragment preview ready")
    return fragments


def motion_matrix(motion: Motion) -> np.ndarray:
    """Absolute planning transform T(t) T(p) Rz(yaw) Ry(roll) Rx(pitch) T(-p)."""
    translation = _vector(motion.translation, "Translation")
    pivot = _vector(motion.pivot, "Pivot")
    angles = _vector(motion.angles, "Rotation")
    if np.any(angles != 0) and not motion.pivot_set:
        raise ValueError("Place a visible rotation pivot before applying rotation.")
    pitch, roll, yaw = np.deg2rad(angles)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(roll), np.sin(roll)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = translation + pivot - result[:3, :3] @ pivot
    return result


def fragment_transforms(doc: Document) -> dict[int, np.ndarray]:
    """Baseline LPS-to-moved-LPS matrices; proximal fragments remain fixed."""
    linked = motion_matrix(doc.motions["linked"])
    mandible = linked @ motion_matrix(doc.motions["mandible"])
    planning = {
        2: linked @ motion_matrix(doc.motions["maxilla"]),
        5: mandible,
        6: mandible @ motion_matrix(doc.motions["chin"]),
    }
    inverse = np.linalg.inv(doc.frame)
    result = {label: np.eye(4) for label in range(1, 9)}
    for label, transform in planning.items():
        result[label] = inverse @ transform @ doc.frame
    return result
