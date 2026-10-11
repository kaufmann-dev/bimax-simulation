"""Validated, atomic planning commands shared by desktop and local MCP control."""

from copy import copy, deepcopy
from dataclasses import asdict
from typing import Any

from .engine import motion_matrix, validate_fragment_preview
from .history import Edit, History
from .model import PROXIMAL_IDS, Document, Measurement, Motion, new_scenario
from .storage import _name, _validate_cut, _vector
from .symmetry import (
    active_fragment_ids,
    landmark_results,
    measurement_results,
    validate_definition,
)

CONVENTIONS = {
    "coordinates": "planning Right/Anterior/Superior, millimetres; baseline points",
    "angles": "pitch, roll, yaw in degrees; Rz(yaw) Ry(roll) Rx(pitch)",
    "inheritance": "linked → outer to inner upper cuts; linked → mandible → chin; parent-relative. Proximal segments rotate independently around fixed baseline pivots, without translation or inheritance",
}
FIELDS = ("scenarios", "active_scenario", "cuts", "reviews", "view_angles", "symmetry_right")


class Commands:
    def __init__(self, doc: Document, history: History):
        self.doc, self.history = doc, history

    def execute(
        self,
        operation: str,
        scenario: str | None = None,
        *,
        record_history: bool = True,
        action_label: str | None = None,
        **args: Any,
    ) -> dict:
        doc = self.doc
        candidate = copy(doc)
        before = {key: deepcopy(getattr(doc, key)) for key in FIELDS}
        for key, value in before.items():
            setattr(candidate, key, deepcopy(value))
        if scenario is not None:
            if scenario not in candidate.scenarios:
                raise ValueError(
                    f"Unknown scenario {scenario!r}. Inspect state for scenario names."
                )
            candidate.active_scenario = scenario
        affected: list[str] = []
        if operation == "create_scenario":
            name = args["name"]
            if not _name(name) or name in candidate.scenarios:
                raise ValueError("Use a unique scenario name of 1–256 characters.")
            source = args.get("source")
            if source is not None and source not in candidate.scenarios:
                raise ValueError("Unknown source scenario.")
            candidate.scenarios[name] = (
                deepcopy(candidate.scenarios[source]) if source else new_scenario()
            )
            candidate.active_scenario = name
        elif operation == "select_scenario":
            if scenario is None:
                raise ValueError("A scenario name is required.")
        elif operation == "select_cuts":
            ids = args["cut_ids"]
            if (
                not isinstance(ids, list)
                or len(set(ids)) != len(ids)
                or any(key not in doc.cuts for key in ids)
            ):
                raise ValueError("Provide unique IDs from the seven shared cuts.")
            candidate.scenario.cut_ids = list(ids)
            _ = candidate.upper_cuts  # Validate mutually exclusive variants.
            if candidate.scenario.cut_ids != doc.scenarios[candidate.active_scenario].cut_ids:
                referenced = {
                    name
                    for definition in candidate.scenario.measurements.values()
                    for name in definition.landmarks.values()
                }
                candidate.scenario.landmarks = [
                    item for item in candidate.scenario.landmarks if item["name"] in referenced
                ]
                try:
                    for definition in candidate.scenario.measurements.values():
                        validate_definition(candidate, definition)
                except ValueError as error:
                    raise ValueError(
                        f"Update or remove measurement definitions explicitly before changing selected cuts: {error}"
                    ) from error
        elif operation == "replace_cut":
            key = args["cut_id"]
            if key not in candidate.cuts:
                raise ValueError("Unknown shared cut ID.")
            cut = candidate.cuts[key]
            cut.geometry = deepcopy(args["geometry"])
            _validate_cut(cut)
            affected = [name for name, item in candidate.scenarios.items() if key in item.cut_ids]
            if cut.geometry != doc.cuts[key].geometry:
                cut.accepted, cut.confirmations = False, {}
                cut.revision += 1
                for name in affected:
                    referenced = {
                        landmark
                        for definition in candidate.scenarios[name].measurements.values()
                        for landmark in definition.landmarks.values()
                    }
                    candidate.scenarios[name].landmarks = [
                        item
                        for item in candidate.scenarios[name].landmarks
                        if item["name"] in referenced
                    ]
        elif operation == "set_movements":
            targets = args["targets"]
            if not isinstance(targets, dict) or not targets:
                raise ValueError("Provide at least one movement target.")
            for key, values in targets.items():
                if (
                    key not in candidate.motions
                    or not isinstance(values, dict)
                    or set(values) - {"translation", "rotation", "pivot"}
                ):
                    raise ValueError("Unknown movement target or field.")
                motion = candidate.motions[key]
                for field in ("translation", "pivot"):
                    if field in values and values[field] is not None:
                        setattr(motion, field, _vector(values[field], 3, field).tolist())
                        if field == "pivot":
                            motion.pivot_set = True
                rotation = values.get("rotation")
                if rotation is not None:
                    if not isinstance(rotation, dict) or set(rotation) - {"pitch", "roll", "yaw"}:
                        raise ValueError("Rotation uses named pitch, roll, yaw fields.")
                    motion.angles = _vector(
                        [
                            rotation.get(axis, motion.angles[i])
                            for i, axis in enumerate(("pitch", "roll", "yaw"))
                        ],
                        3,
                        "Rotation",
                    ).tolist()
                motion_matrix(motion)
        elif operation == "set_proximal_rotations":
            if "bsso" not in candidate.scenario.cut_ids:
                raise ValueError("Select BSSO for this scenario before rotating proximal segments.")
            targets = args["targets"]
            if not isinstance(targets, dict) or not targets:
                raise ValueError("Provide one or both proximal targets.")
            for key, values in targets.items():
                if (
                    key not in PROXIMAL_IDS
                    or not isinstance(values, dict)
                    or set(values) - {"rotation", "pivot"}
                ):
                    raise ValueError(
                        "Proximal targets support rotation and pivot only; translation is unavailable."
                    )
                control = candidate.scenario.proximal_rotations[key]
                if "pivot" in values:
                    control.pivot = _vector(values["pivot"], 3, "Condyle pivot").tolist()
                    control.pivot_set = True
                if "rotation" in values:
                    rotation = values["rotation"]
                    if not isinstance(rotation, dict) or set(rotation) - {"pitch", "roll", "yaw"}:
                        raise ValueError("Rotation uses named pitch, roll and yaw fields.")
                    control.angles = _vector(
                        [
                            rotation.get(axis, control.angles[i])
                            for i, axis in enumerate(("pitch", "roll", "yaw"))
                        ],
                        3,
                        "Proximal rotation",
                    ).tolist()
                motion_matrix(
                    Motion(angles=control.angles, pivot=control.pivot, pivot_set=control.pivot_set)
                )
        elif operation == "reset_proximal_rotations":
            targets = args["targets"]
            if (
                not targets
                or len(set(targets)) != len(targets)
                or any(key not in PROXIMAL_IDS for key in targets)
            ):
                raise ValueError("Choose one or both proximal segments to reset.")
            for key in targets:
                control = candidate.scenario.proximal_rotations[key]
                control.angles = [0.0] * 3
                if args.get("clear_pivots", False):
                    control.pivot, control.pivot_set = [0.0] * 3, False
        elif operation == "set_symmetry_reference":
            if (args.get("right") is None) == (args.get("point") is None):
                raise ValueError(
                    "Provide either a Right offset or a baseline planning point for the symmetry reference."
                )
            candidate.symmetry_right = (
                float(_vector([args["right"]], 1, "Symmetry Right offset")[0])
                if args.get("right") is not None
                else float(_vector(args["point"], 3, "Symmetry reference point")[0])
            )
        elif operation == "set_measurement":
            name = args["name"]
            if not _name(name):
                raise ValueError("Measurement name must contain 1–256 characters.")
            if args.get("remove", False):
                if name not in candidate.scenario.measurements:
                    raise ValueError("Unknown measurement definition.")
                del candidate.scenario.measurements[name]
            else:
                if args.get("kind") not in ("midline", "paired", "ramus") or not isinstance(
                    args.get("landmarks"), dict
                ):
                    raise ValueError("Provide a measurement kind and its named landmark roles.")
                definition = Measurement(args["kind"], deepcopy(args["landmarks"]))
                validate_definition(candidate, definition)
                candidate.scenario.measurements[name] = definition
        elif operation == "reset_pivot":
            key = args["target"]
            if key not in candidate.motions:
                raise ValueError("Unknown movement target.")
            motion = candidate.motions[key]
            motion.pivot, motion.angles, motion.pivot_set = [0.0] * 3, [0.0] * 3, False
        elif operation == "set_orientation":
            rotation = args["rotation"]
            if set(rotation) != {"pitch", "roll", "yaw"}:
                raise ValueError("Orientation requires pitch, roll and yaw.")
            candidate.view_angles = _vector(
                [rotation[key] for key in ("pitch", "roll", "yaw")], 3, "Orientation"
            ).tolist()
            if any(abs(angle) > 180 for angle in candidate.view_angles):
                raise ValueError("Orientation angles must be between -180 and 180 degrees.")
        elif operation == "set_landmark":
            name = args["name"]
            if not _name(name):
                raise ValueError("Landmark name must contain 1–256 characters.")
            if args.get("remove", False) and any(
                name in definition.landmarks.values()
                for definition in candidate.scenario.measurements.values()
            ):
                raise ValueError(
                    "Landmark is referenced by a measurement. Remove or update that definition explicitly first."
                )
            landmarks = candidate.scenario.landmarks
            landmarks[:] = [item for item in landmarks if item["name"] != name]
            if not args.get("remove", False):
                fragment = args.get("fragment")
                if fragment not in active_fragment_ids(candidate):
                    raise ValueError("Unknown fragment identity for this scenario.")
                landmarks.append(
                    {
                        "name": name,
                        "fragment": fragment,
                        "point": _vector(args.get("point"), 3, "Landmark").tolist(),
                    }
                )
        elif operation == "confirm_fragment":
            fragment = args["fragment"]
            if fragment not in candidate.fragment_labels:
                raise ValueError("Unknown fragment identity.")
            candidate.review.confirmations[fragment] = _vector(
                args["point"], 3, "Confirmation"
            ).tolist()
            candidate.review.accepted = False
        elif operation == "accept_combination":
            validate_fragment_preview(
                candidate, args["fragments"], args["fragment_counts"], args["anatomy_counts"]
            )
            candidate.accept_review()
        else:
            raise ValueError(f"Unknown planning command {operation}.")
        if (
            operation
            in {
                "set_proximal_rotations",
                "reset_proximal_rotations",
                "set_measurement",
                "select_cuts",
                "set_movements",
                "reset_pivot",
                "set_landmark",
                "confirm_fragment",
                "accept_combination",
            }
            and scenario is None
        ):
            raise ValueError("Scenario edits require an explicit scenario name.")
        if operation == "set_landmark":
            for definition in candidate.scenario.measurements.values():
                validate_definition(candidate, definition)
        if len(candidate.scenarios) > 256 or any(
            len(item.measurements) > 256 for item in candidate.scenarios.values()
        ):
            raise ValueError(
                "A project supports at most 256 scenarios and 256 measurements per scenario."
            )
        old_key = doc.configuration_key
        after = {key: deepcopy(getattr(candidate, key)) for key in FIELDS}
        changed = [key for key in FIELDS if before[key] != after[key]]
        for key in changed:
            setattr(doc, key, after[key])
        if changed:
            if record_history:
                self.history.push(
                    Edit(
                        action_label or operation.replace("_", " ").capitalize(),
                        [],
                        {key: before[key] for key in changed},
                        {key: deepcopy(after[key]) for key in changed},
                    )
                )
            doc.revision += 1
        if old_key != doc.configuration_key:
            doc.fragments = None
            doc.fragments_key = None
        return {"changed": bool(changed), "affected_scenarios": affected, "revision": doc.revision}


def inspect_document(doc: Document) -> dict:
    active = doc.active_scenario
    scenarios = {}
    try:
        for name, scenario in doc.scenarios.items():
            doc.active_scenario = name
            landmarks = landmark_results(doc)
            scenarios[name] = {
                "cut_ids": scenario.cut_ids,
                "movements": {
                    key: {
                        "translation": motion.translation,
                        "rotation": dict(zip(("pitch", "roll", "yaw"), motion.angles, strict=True)),
                        "pivot": motion.pivot if motion.pivot_set else None,
                    }
                    for key, motion in scenario.motions.items()
                },
                "landmarks": landmarks,
                "proximal_rotations": {
                    key: {
                        "rotation": dict(
                            zip(("pitch", "roll", "yaw"), control.angles, strict=True)
                        ),
                        "pivot": list(control.pivot) if control.pivot_set else None,
                    }
                    for key, control in scenario.proximal_rotations.items()
                },
                "measurements": measurement_results(doc),
                "fragments": [asdict(item) for item in doc.fragment_descriptors.values()],
                "accepted": doc.cuts_accepted,
                "confirmations": deepcopy(doc.reviews[doc.configuration_key].confirmations)
                if doc.configuration_key in doc.reviews
                else {},
                "preparation_key": doc.configuration_key,
            }
    finally:
        doc.active_scenario = active
    return {
        "document_id": doc._identity,
        "revision": doc.revision,
        "active_scenario": active,
        "orientation": dict(zip(("pitch", "roll", "yaw"), doc.view_angles, strict=True)),
        "conventions": CONVENTIONS,
        "symmetry_reference": {
            "right_mm": doc.symmetry_right,
            "normal": [1, 0, 0],
            "plane": "parallel to planning Anterior/Superior",
        }
        if doc.symmetry_right is not None
        else None,
        "scenarios": scenarios,
        "cuts": {
            key: {
                **asdict(cut),
                "affected_scenarios": [
                    name for name, item in doc.scenarios.items() if key in item.cut_ids
                ],
            }
            for key, cut in doc.cuts.items()
        },
    }
