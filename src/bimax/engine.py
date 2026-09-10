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

from bimax.model import CutPlan, Document, Geometry, Motion

Progress = Callable[[float, str], None] | None
MEMORY_BUDGET = 16 * 1024**3
FACE_CONNECTIVITY = ndi.generate_binary_structure(3, 1)
CUT_SLAB_VOXELS = 2_000_000


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
    doc.plan.cuts = default_cuts(doc)
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


def default_cuts(doc: Document, lefort_level: int = 1) -> dict[str, Any]:
    """Approximate editable envelopes, disabled until deliberately selected.

    Higher-level templates are piecewise planar regions, not single raised
    Le Fort I planes. They are geometric suggestions, not anatomical detection
    or clinically valid osteotomy trajectories.
    """
    if type(lefort_level) is not int or lefort_level not in (1, 2, 3):
        raise ValueError("Le Fort level must be I, II or III.")
    low, high = planning_bounds(doc)
    extent = high - low
    centre = (low + high) / 2

    def plane(origin: Any, normal: Any) -> dict[str, Any]:
        return {"origin": list(map(float, origin)), "normal": list(map(float, normal))}

    def region(minimum: Any, maximum: Any) -> dict[str, Any]:
        return {"min": list(map(float, minimum)), "max": list(map(float, maximum))}

    if lefort_level == 1:
        upper_planes = {
            "upper": plane([centre[0], centre[1], low[2] + extent[2] * 0.43], [0, 0, -1])
        }
        groups = [["upper"]]
    else:
        apex = low[2] + extent[2] * (0.63 if lefort_level == 2 else 0.67)
        valley = low[2] + extent[2] * (0.48 if lefort_level == 2 else 0.51)
        slope = (apex - valley) / max(extent[0] * 0.22, 1e-6)
        posterior_slope = extent[2] * 0.24 / max(extent[1] * 0.28, 1e-6)
        upper_planes = {
            "left_oblique": plane([centre[0], centre[1], apex], [slope, 0, -1]),
            "right_oblique": plane([centre[0], centre[1], apex], [-slope, 0, -1]),
            "posterior": plane(
                [centre[0], low[1] + extent[1] * 0.35, low[2] + extent[2] * 0.43],
                [0, posterior_slope, -1],
            ),
        }
        groups = [["left_oblique", "right_oblique", "posterior"]]
        if lefort_level == 3:
            # The union adds lateral wings while retaining two lower valleys
            # beside the central peak; one convex clipping volume cannot do so.
            wing_slope = extent[2] * 0.16 / max(extent[0] * 0.22, 1e-6)
            upper_planes.update({
                "left_wing": plane(
                    [centre[0] - extent[0] * 0.22, centre[1], valley], [-wing_slope, 0, -1]
                ),
                "right_wing": plane(
                    [centre[0] + extent[0] * 0.22, centre[1], valley], [wing_slope, 0, -1]
                ),
                "wing_ceiling": plane(
                    [centre[0], centre[1], low[2] + extent[2] * 0.67], [0, 0, -1]
                ),
            })
            groups.extend([
                ["left_wing", "wing_ceiling", "posterior"],
                ["right_wing", "wing_ceiling", "posterior"],
            ])

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
            "planes": upper_planes,
            "groups": groups,
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
    return _within_region(points, cut["region"], _plane_mask(points, cut["plane"]))


def _within_region(
    points: tuple[np.ndarray, ...], region: dict[str, Any], mask: np.ndarray
) -> np.ndarray:
    minimum = _vector(region["min"], "Region minimum")
    maximum = _vector(region["max"], "Region maximum")
    if np.any(maximum < minimum):
        raise ValueError("Region maxima must be greater than or equal to minima.")
    for axis in range(3):
        mask &= (points[axis] >= minimum[axis] - 1e-8) & (points[axis] <= maximum[axis] + 1e-8)
    return mask


def _lefort_mask(points: tuple[np.ndarray, ...], cut: dict[str, Any]) -> np.ndarray:
    """Union of convex half-space groups, clipped by a shared physical region."""
    planes, groups = cut["planes"], cut["groups"]
    if not isinstance(planes, dict) or not planes or not isinstance(groups, list) or not groups:
        raise ValueError("A Le Fort cut needs named planes and nonempty plane groups.")
    combined = None
    for group in groups:
        if (
            not isinstance(group, list) or not group
            or any(not isinstance(name, str) or name not in planes for name in group)
        ):
            raise ValueError("Le Fort plane groups must reference existing named planes.")
        intersection = _plane_mask(points, planes[group[0]])
        for name in group[1:]:
            intersection &= _plane_mask(points, planes[name])
        if combined is None:
            combined = intersection
        else:
            combined |= intersection
    assert combined is not None
    return _within_region(points, cut["region"], combined)


def label_counts(labels: np.ndarray, maximum_label: int) -> np.ndarray:
    """Count a validated label volume with bounded temporary allocations."""
    counts = np.zeros(maximum_label + 1, dtype=np.int64)
    flat = labels.reshape(-1)
    for start in range(0, flat.size, 1_000_000):
        counts += np.bincount(flat[start : start + 1_000_000], minlength=maximum_label + 1)
    return counts


def required_fragment_labels(plan: CutPlan) -> set[int]:
    required = set()
    if plan.cuts.get("lefort", {}).get("enabled", False):
        required.add(2)
    if plan.cuts.get("bsso", {}).get("enabled", False):
        required.update((3, 4, 5))
    if plan.cuts.get("genio", {}).get("enabled", False):
        required.update((5, 6))
    return required


def _cut_fragment_labels(
    labels: np.ndarray, points: tuple[np.ndarray, ...], plan: CutPlan
) -> None:
    lefort = plan.cuts.get("lefort", {"enabled": False})
    bsso = plan.cuts.get("bsso", {"enabled": False})
    genio = plan.cuts.get("genio", {"enabled": False})
    if lefort["enabled"]:
        labels[(labels == 1) & _lefort_mask(points, lefort)] = 2
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


def fragment_labels_at_points(
    doc: Document, points: Any, plan: CutPlan | None = None
) -> np.ndarray:
    """Evaluate exact cut ownership at nearest voxel centres, outside as zero.

    Points are Nx3 baseline planning coordinates. This shared predicate path
    validates inactive plans' confirmation samples without allocating volumes.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("Fragment samples must be finite Nx3 planning coordinates.")
    xyz = np.rint(doc.geometry.world_to_index(doc.plan_to_world(values)))
    inside = np.all((xyz >= 0) & (xyz < np.asarray(doc.geometry.shape[::-1])), axis=1)
    result = np.zeros(len(values), dtype=np.uint8)
    indices = xyz[inside].astype(np.intp)
    if not len(indices):
        return result
    centres = doc.world_to_plan(doc.geometry.index_to_world(indices))
    labels = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[
        doc.anatomy[indices[:, 2], indices[:, 1], indices[:, 0]]
    ]
    _cut_fragment_labels(labels, tuple(centres[:, axis] for axis in range(3)), plan or doc.plan)
    result[inside] = labels
    return result


def validate_fragment_preview(
    doc: Document,
    fragments: np.ndarray,
    fragment_counts: np.ndarray,
    anatomy_counts: np.ndarray,
) -> None:
    """Validate acceptance without rescanning or copying a previously built preview.

    The caller must retain the counts returned with the worker's exact preview and
    discard that preview whenever anatomy, frame, or cut geometry changes. Source
    partition conservation was already checked by ``build_fragments``; only cached
    counts and required confirmation voxels are examined here.
    """
    if (
        not isinstance(fragments, np.ndarray)
        or fragments.shape != doc.geometry.shape
        or fragments.shape != doc.source.shape
        or fragments.dtype != np.uint8
    ):
        raise ValueError("Fragment preview must be a uint8 volume matching the source geometry.")
    for counts, length in ((fragment_counts, 9), (anatomy_counts, 5)):
        if (
            not isinstance(counts, np.ndarray)
            or counts.shape != (length,)
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
            or np.any(counts > fragments.size)
            or int(counts.sum()) != fragments.size
        ):
            raise ValueError("Fragment preview counts are invalid. Preview fragments again.")
    if anatomy_counts[1]:
        raise ValueError("Assign all bone to anatomy or explicitly exclude it before accepting cuts.")
    ownership = np.array(
        [fragment_counts[0], fragment_counts[8], fragment_counts[1:3].sum(),
         fragment_counts[3:7].sum(), fragment_counts[7]],
        dtype=np.int64,
    )
    if not np.array_equal(ownership, anatomy_counts):
        raise ValueError("Fragment ownership does not match its anatomical parent; rebuild cuts.")
    frame = np.asarray(doc.frame)
    if (
        frame.shape != (4, 4)
        or not np.all(np.isfinite(frame))
        or not np.allclose(frame[3], [0, 0, 0, 1])
        or not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(frame[:3, :3]), 1.0)
    ):
        raise ValueError("Planning frame must be a finite, right-handed rigid transform.")
    geometry = doc.geometry
    direction = np.asarray(geometry.direction).reshape(3, 3)
    if (
        not np.all(np.isfinite([*geometry.spacing, *geometry.origin, *geometry.direction]))
        or min(geometry.spacing) <= 0
        or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5)
    ):
        raise ValueError("The input has invalid physical geometry.")
    for label in sorted(required_fragment_labels(doc.plan)):
        if not fragment_counts[label]:
            raise ValueError(f"The cuts leave {doc.fragment_names[label]} empty. Adjust the cut geometry.")
        confirmation = doc.plan.confirmations.get(str(label))
        if confirmation is None:
            raise ValueError(f"Place a confirmation point inside {doc.fragment_names[label]} before accepting cuts.")
        point = _vector(confirmation, "Fragment confirmation")
        xyz = np.rint(doc.geometry.world_to_index(doc.plan_to_world(point)))
        zyx = xyz[::-1]
        if (not np.all(np.isfinite(zyx)) or np.any(zyx < 0) or np.any(zyx >= fragments.shape)
                or fragments[tuple(zyx.astype(int))] != label):
            raise ValueError(f"The confirmation point for {doc.fragment_names[label]} is outside its fragment.")


def build_fragments(doc: Document, validate: bool = True, progress: Progress = None) -> np.ndarray:
    """Partition approved anatomy using zero-thickness baseline-space cuts."""
    assert_partition(doc.source, doc.anatomy, maximum_label=4)
    counts = label_counts(doc.anatomy, 4) if validate else None
    if counts is not None and counts[1]:
        raise ValueError("Assign all bone to anatomy or explicitly exclude it before accepting cuts.")
    _check_memory(doc, doc.source.size, bytes_per_voxel=3)
    fragments = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[doc.anatomy]
    if any(cut["enabled"] for cut in doc.plan.cuts.values()):
        affine = doc.frame @ doc.geometry.affine
        for start in range(0, doc.source.size, CUT_SLAB_VOXELS):
            stop = min(start + CUT_SLAB_VOXELS, doc.source.size)
            # Flat slabs also bound unusually wide images and non-contiguous
            # input arrays; no full-volume flattening copy is necessary.
            indices = np.flatnonzero(doc.anatomy.flat[start:stop]) + start
            if indices.size:
                # Background dominates scans. Evaluate the identical physical
                # half-space predicates only at occupied native voxel centres.
                # Retain the original arithmetic order at cut boundaries.
                z, y, x = np.unravel_index(indices, doc.source.shape)
                points = tuple(
                    affine[axis, 0] * x + affine[axis, 1] * y
                    + affine[axis, 2] * z + affine[axis, 3]
                    for axis in range(3)
                )
                occupied = fragments.flat[indices]
                _cut_fragment_labels(occupied, points, doc.plan)
                fragments.flat[indices] = occupied
            _report(progress, stop / doc.source.size * 0.9, "Partitioning bone fragments")
    assert_partition(doc.source, fragments)
    if validate:
        assert counts is not None
        validate_fragment_preview(doc, fragments, label_counts(fragments, 8), counts)
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
    planning = {5: mandible}
    # A scenario may retain motion parameters while its associated plan omits
    # that operation. Hidden motions must neither transform nor invalidate it.
    if doc.plan.cuts.get("lefort", {}).get("enabled", False):
        planning[2] = linked @ motion_matrix(doc.motions["maxilla"])
    if doc.plan.cuts.get("genio", {}).get("enabled", False):
        planning[6] = mandible @ motion_matrix(doc.motions["chin"])
    inverse = np.linalg.inv(doc.frame)
    result = {label: np.eye(4) for label in range(1, 9)}
    for label, transform in planning.items():
        result[label] = inverse @ transform @ doc.frame
    return result
