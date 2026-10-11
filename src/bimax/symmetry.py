"""Camera-independent symmetry in the explicit planning Right/Anterior/Superior frame."""

from dataclasses import asdict
from typing import Any

import numpy as np

from .engine import fragment_transforms
from .model import Document, Measurement

ROLES = {
    "midline": ("point",),
    "paired": ("left", "right"),
    "ramus": ("left_condyle", "left_angle", "right_condyle", "right_angle"),
}
SIGNS = {
    "midline": "mm: signed = Right − reference Right (positive right); distance = absolute signed offset",
    "paired": "mm: left − right for Superior height, Anterior position, and absolute lateral distance from the reference",
    "ramus": "degrees: condyle minus jaw angle; frontal atan2(outward lateral, Superior), sagittal atan2(Anterior, Superior); differences left − right",
}


def active_fragment_ids(doc: Document) -> set[str]:
    return {
        fragment.id
        for fragment in doc.fragment_descriptors.values()
        if fragment.cut_id is None or fragment.cut_id in doc.scenario.cut_ids
    }


def validate_definition(doc: Document, definition: Measurement) -> None:
    if definition.kind not in ROLES or set(definition.landmarks) != set(ROLES[definition.kind]):
        raise ValueError(
            "Measurement landmarks must match the named roles for midline, paired or ramus."
        )
    if doc.symmetry_right is None:
        raise ValueError(
            "Set an explicit project symmetry reference plane before defining measurements."
        )
    named = {item["name"]: item for item in doc.scenario.landmarks}
    values = {}
    allowed = active_fragment_ids(doc) - {"unassigned", "excluded"}
    for role, name in definition.landmarks.items():
        if name not in named:
            raise ValueError(
                f"Missing measurement landmark {name!r}. Update or remove the definition explicitly."
            )
        item = named[name]
        if item["fragment"] not in allowed:
            raise ValueError(
                f"Landmark {name!r} references an unavailable or unsuitable fragment. Update it explicitly."
            )
        if definition.kind == "ramus":
            side = role.split("_")[0]
            if item["fragment"] not in {"mandible", f"{side}-proximal"}:
                raise ValueError(
                    f"Ramus landmark {name!r} must be attached to the {side} proximal segment or distal mandible."
                )
        values[role] = np.asarray(item["point"], dtype=float)
    if definition.kind == "paired" and np.linalg.norm(values["left"] - values["right"]) < 1e-8:
        raise ValueError("Paired landmarks must be distinct; the points coincide.")
    if definition.kind == "ramus":
        for side in ("left", "right"):
            _inclinations(values[f"{side}_condyle"] - values[f"{side}_angle"], side)


def landmark_results(doc: Document) -> list[dict[str, Any]]:
    transforms = fragment_transforms(doc)
    active = active_fragment_ids(doc)
    result = []
    for item in doc.scenario.landmarks:
        entry = {**item, "baseline": list(item["point"])}
        if item["fragment"] not in active:
            entry.update(
                status="invalid", error="Attached fragment is unavailable in this scenario."
            )
        else:
            matrix = transforms[doc.fragment_labels[item["fragment"]]]
            baseline = doc.plan_to_world(item["point"])
            moved = doc.world_to_plan(matrix[:3, :3] @ baseline + matrix[:3, 3])
            delta = moved - item["point"]
            entry.update(
                status="valid",
                moved=moved.tolist(),
                displacement=delta.tolist(),
                distance_mm=float(np.linalg.norm(delta)),
            )
        result.append(entry)
    return result


def _inclinations(vector: np.ndarray, side: str) -> dict[str, float]:
    if np.linalg.norm(vector) < 1e-8:
        raise ValueError(f"The {side} condyle and jaw-angle points coincide.")
    if np.hypot(vector[0], vector[2]) < 1e-8 or np.hypot(vector[1], vector[2]) < 1e-8:
        raise ValueError(f"The {side} ramus has an undefined frontal or sagittal projection.")
    return {
        "frontal_deg": float(
            np.rad2deg(np.arctan2((-1 if side == "left" else 1) * vector[0], vector[2]))
        ),
        "sagittal_deg": float(np.rad2deg(np.arctan2(vector[1], vector[2]))),
    }


def _evaluate(kind: str, points: dict[str, np.ndarray], right: float) -> dict[str, float]:
    if kind == "midline":
        signed = float(points["point"][0] - right)
        return {"signed_lateral_mm": signed, "distance_mm": abs(signed)}
    if kind == "paired":
        left, other = points["left"], points["right"]
        if np.linalg.norm(left - other) < 1e-8:
            raise ValueError("The moved paired landmarks coincide.")
        return {
            "height_difference_mm": float(left[2] - other[2]),
            "anterior_difference_mm": float(left[1] - other[1]),
            "lateral_distance_difference_mm": float(abs(left[0] - right) - abs(other[0] - right)),
        }
    left_angles = _inclinations(points["left_condyle"] - points["left_angle"], "left")
    right_angles = _inclinations(points["right_condyle"] - points["right_angle"], "right")
    return {
        "left_frontal_deg": left_angles["frontal_deg"],
        "right_frontal_deg": right_angles["frontal_deg"],
        "frontal_difference_deg": left_angles["frontal_deg"] - right_angles["frontal_deg"],
        "left_sagittal_deg": left_angles["sagittal_deg"],
        "right_sagittal_deg": right_angles["sagittal_deg"],
        "sagittal_difference_deg": left_angles["sagittal_deg"] - right_angles["sagittal_deg"],
    }


def measurement_results(doc: Document) -> dict[str, dict[str, Any]]:
    landmarks = {item["name"]: item for item in landmark_results(doc)}
    result = {}
    for name, definition in doc.scenario.measurements.items():
        entry = {
            "definition": asdict(definition),
            "sign_conventions": SIGNS.get(definition.kind, ""),
        }
        try:
            validate_definition(doc, definition)
            assert doc.symmetry_right is not None
            baseline = {
                role: np.asarray(landmarks[key]["baseline"])
                for role, key in definition.landmarks.items()
            }
            moved = {
                role: np.asarray(landmarks[key]["moved"])
                for role, key in definition.landmarks.items()
            }
            entry.update(
                status="valid",
                baseline=_evaluate(definition.kind, baseline, doc.symmetry_right),
                moved=_evaluate(definition.kind, moved, doc.symmetry_right),
            )
        except (ValueError, KeyError) as error:
            entry.update(status="invalid", error=str(error))
        result[name] = entry
    return result
