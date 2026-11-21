"""Planned jaw movements in the aligners surgery format.

The aligners app exports ``out/surgery/landmarks.json``: named dental landmarks of the maxilla
and mandible, each with the point its displacement is measured from (``reference_lps``) and the
same point on the bone in the CT image (``ct_lps``), plus the head-frame axes; all in DICOM LPS
millimetres. Every landmark moves with the fragment that owns the bone at ``ct_lps``. Its
displacement from the reference along the head axes becomes a row of the ``[surgery.maxilla]`` /
``[surgery.mandible]`` tables of the aligners ``plan.toml``. The bone volume must come from the
same CT as the aligners case.
"""

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from .engine import fragment_transforms
from .model import Document

FORMAT = "aligners-surgery-landmarks"
FORMAT_VERSION = 1
LANDMARKS = ("incisor_midline", "canine_right", "canine_left", "molar_right", "molar_left")
JAWS = {"maxilla": (2, "Le Fort I"), "mandible": (3, "BSSO distal segment")}  # anatomy, comment
DISTAL_MANDIBLE = 5
SEARCH_MM = 3.0  # a landmark on a tooth surface belongs to the nearest bone voxel this close
AXES = ("x_patient_right", "y_anterior", "z_superior")


@dataclass
class AlignerMovements:
    tables: dict[str, dict[str, dict[str, float]]]  # jaw -> landmark -> forward/up/left mm
    fragments: dict[str, str]  # jaw -> name of the fragment its landmarks move with
    head_pitch_deg: float


def _vector(value: Any, what: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"landmarks.json: {what} must be three finite numbers.")
    return array


def read_landmarks(path: str | Path) -> dict[str, Any]:
    """Load and validate an aligners ``landmarks.json``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{path} is not an aligners landmarks.json (out/surgery/landmarks.json).")
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"{path} has landmarks format version {data.get('version')}; this BIMAX reads "
            f"version {FORMAT_VERSION}."
        )
    frame = data.get("head_frame", {})
    axes = np.array([_vector(frame.get(axis), f"head_frame.{axis}") for axis in AXES])
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-6) or np.linalg.det(axes) < 0:
        raise ValueError("landmarks.json: the head-frame axes are not right-handed and orthonormal.")
    jaws = data.get("jaws", {})
    if not isinstance(jaws, dict) or not set(jaws) <= set(JAWS) or not jaws:
        raise ValueError("landmarks.json: 'jaws' must list maxilla and/or mandible landmarks.")
    for jaw, points in jaws.items():
        if not isinstance(points, dict) or not set(points) <= set(LANDMARKS) or not points:
            raise ValueError(f"landmarks.json: unknown or missing {jaw} landmarks.")
        for name, entry in points.items():
            for key in ("reference_lps", "ct_lps"):
                _vector(entry.get(key) if isinstance(entry, dict) else None, f"{jaw}.{name}.{key}")
    return data


def owning_labels(doc: Document, points: np.ndarray, anatomy: int) -> list[int | None]:
    """Fragment of the nearest voxel of the given anatomy within SEARCH_MM of each LPS point."""
    assert doc.fragments is not None
    geometry = doc.geometry
    parents = doc.parent_labels
    spacing = np.asarray(geometry.spacing, dtype=float)
    reach = np.ceil(SEARCH_MM / spacing).astype(int)
    upper = np.asarray(geometry.shape[::-1]) - 1
    result: list[int | None] = []
    for point in points:
        centre = np.rint(geometry.world_to_index(point)).astype(int)
        lo, hi = np.maximum(centre - reach, 0), np.minimum(centre + reach, upper)
        if np.any(lo > hi):
            result.append(None)
            continue
        block = doc.fragments[lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        zyx = np.argwhere((block > 0) & (parents[np.minimum(block, len(parents) - 1)] == anatomy))
        if not len(zyx):
            result.append(None)
            continue
        xyz = zyx[:, ::-1] + lo
        distance = np.linalg.norm(geometry.index_to_world(xyz) - point, axis=1)
        best = int(np.argmin(distance))
        result.append(int(block[tuple(zyx[best])]) if distance[best] <= SEARCH_MM else None)
    return result


def aligner_movements(doc: Document, landmarks: dict[str, Any]) -> AlignerMovements:
    """Landmark displacement tables of the active scenario (accepted cuts required)."""
    if not doc.cuts_accepted:
        raise ValueError("Review and accept the active cut combination before exporting aligner movements.")
    if doc.fragments is None:
        from .prepared import prepare_geometry

        doc.fragments = prepare_geometry(doc)["fragments"]
        doc.fragments_key = doc.configuration_key
    frame = landmarks["head_frame"]
    axes = np.array([np.asarray(frame[axis], dtype=float) for axis in AXES])
    transforms = fragment_transforms(doc)
    names = doc.fragment_names
    tables: dict[str, dict[str, dict[str, float]]] = {}
    used: dict[str, str] = {}
    for jaw, points in landmarks["jaws"].items():
        anatomy = JAWS[jaw][0]
        ordered = [name for name in LANDMARKS if name in points]
        ct = np.array([np.asarray(points[name]["ct_lps"], dtype=float) for name in ordered])
        labels = owning_labels(doc, ct, anatomy)
        missing = [name for name, label in zip(ordered, labels, strict=True) if label is None]
        if missing:
            raise ValueError(
                f"The {jaw} landmark(s) {', '.join(missing)} are more than {SEARCH_MM:g} mm from "
                f"the {jaw} bone of this volume: landmarks.json must come from the same CT."
            )
        found = sorted({label for label in labels if label is not None})
        if len(found) > 1:
            parts = ", ".join(
                f"{name} in {names[label]}" for name, label in zip(ordered, labels, strict=True)
                if label is not None
            )
            raise ValueError(f"The {jaw} landmarks lie on different fragments ({parts}): a cut divides the dentition.")
        label = found[0]
        if jaw == "mandible" and label != DISTAL_MANDIBLE:
            raise ValueError(
                f"The mandible landmarks lie in the {names[label]}: the BSSO and genioplasty cuts "
                "must leave the lower dentition on the distal mandible."
            )
        T = transforms.get(label, np.eye(4))
        moved = ct @ T[:3, :3].T + T[:3, 3]
        tables[jaw] = {}
        for name, point in zip(ordered, moved, strict=True):
            d = point - np.asarray(points[name]["reference_lps"], dtype=float)
            right, anterior, superior = axes @ d
            tables[jaw][name] = {
                "forward_mm": round(float(anterior), 1) + 0.0,
                "up_mm": round(float(superior), 1) + 0.0,
                "left_mm": round(float(-right), 1) + 0.0,
            }
        used[jaw] = names[label]
    return AlignerMovements(tables, used, float(landmarks.get("head_pitch_deg", 0.0)))


def movements_toml(doc: Document, movements: AlignerMovements) -> str:
    """The tables in the aligners ``plan.toml`` input format."""
    moved_with = ", ".join(f"{jaw} with {name}" for jaw, name in movements.fragments.items())
    lines = [
        f"# Planned jaw movements from BIMAX {version('bimax-simulation')}, scenario "
        f"{json.dumps(doc.active_scenario)}, for aligners.",
        "# Replace [surgery.maxilla] and [surgery.mandible] in plan.toml with these and set",
        '# [surgery] mode = "manual" (head frame of landmarks.json, mm: + forward, + up,',
        "# + patient-left).",
        f"# head_pitch_deg = {movements.head_pitch_deg}",
        f"# Landmarks moved {moved_with}.",
    ]
    for jaw, (_, label) in JAWS.items():
        if jaw not in movements.tables:
            continue
        lines += ["", f"[surgery.{jaw}]  # {label}"]
        for name, m in movements.tables[jaw].items():
            lines.append(
                f"{name} = {{ forward_mm = {m['forward_mm']:.1f}, up_mm = {m['up_mm']:.1f}, "
                f"left_mm = {m['left_mm']:.1f} }}"
            )
    return "\n".join(lines) + "\n"


def export_aligner_movements(doc: Document, landmarks_path: str | Path, output_path: str | Path) -> Path:
    """Write ``movements.toml`` for the aligners case whose ``landmarks.json`` is given."""
    movements = aligner_movements(doc, read_landmarks(landmarks_path))
    destination = Path(output_path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(movements_toml(doc, movements), encoding="utf-8")
    return destination
