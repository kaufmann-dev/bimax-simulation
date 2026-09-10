"""Shared document types. Arrays use ZYX; physical coordinates use LPS millimetres."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ANATOMY = {1: "Unassigned", 2: "Skull and upper jaw", 3: "Mandible and lower teeth", 4: "Excluded"}
FRAGMENTS = {
    1: "Fixed skull",
    2: "Maxilla",
    3: "Left proximal",
    4: "Right proximal",
    5: "Distal mandible",
    6: "Chin",
    7: "Excluded",
    8: "Unassigned",
}
COLORS = {
    1: (0.75, 0.77, 0.80),
    2: (0.28, 0.72, 0.95),
    3: (0.95, 0.63, 0.27),
    4: (0.70, 0.42, 0.90),
    5: (0.30, 0.83, 0.63),
    6: (0.98, 0.43, 0.55),
    7: (0.38, 0.38, 0.38),
    8: (0.95, 0.86, 0.28),
}
ANATOMY_COLORS = {1: COLORS[8], 2: COLORS[2], 3: COLORS[5], 4: COLORS[7]}


@dataclass
class Geometry:
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]

    @property
    def affine(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = np.asarray(self.direction).reshape(3, 3) @ np.diag(self.spacing)
        result[:3, 3] = self.origin
        return result

    def index_to_world(self, xyz: Any) -> np.ndarray:
        points = np.asarray(xyz, dtype=float)
        return points @ self.affine[:3, :3].T + self.affine[:3, 3]

    def world_to_index(self, lps: Any) -> np.ndarray:
        inverse = np.linalg.inv(self.affine)
        return np.asarray(lps, dtype=float) @ inverse[:3, :3].T + inverse[:3, 3]


@dataclass
class Motion:
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angles: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pivot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pivot_set: bool = False


def new_motions() -> dict[str, Motion]:
    return {key: Motion() for key in ("linked", "maxilla", "mandible", "chin")}


@dataclass
class CutPlan:
    name: str
    lefort_level: int = 1
    # All cut coordinates and confirmation points are in baseline planning mm.
    cuts: dict[str, Any] = field(default_factory=dict)
    confirmations: dict[str, list[float]] = field(default_factory=dict)
    accepted: bool = False
    revision: int = 0


@dataclass
class Scenario:
    plan_id: str = "plan1"
    motions: dict[str, Motion] = field(default_factory=new_motions)
    landmarks: list[dict[str, Any]] = field(default_factory=list)


def new_scenario(plan_id: str = "plan1") -> Scenario:
    return Scenario(plan_id=plan_id)


@dataclass
class Document:
    geometry: Geometry
    source: np.ndarray
    anatomy: np.ndarray
    seeds: np.ndarray
    barriers: np.ndarray
    # Derived active-plan geometry only; not a second persisted source of truth.
    fragments: np.ndarray | None = None
    fragments_key: tuple[str, int, int] | None = None
    # Maps LPS world points into the planning frame (Right, Anterior, Superior).
    frame: np.ndarray = field(default_factory=lambda: np.diag([-1.0, -1.0, 1.0, 1.0]))
    plans: dict[str, CutPlan] = field(default_factory=lambda: {"plan1": CutPlan("Le Fort I")})
    scenarios: dict[str, Scenario] = field(
        default_factory=lambda: {"Scenario 1": new_scenario()}
    )
    active_scenario: str = "Scenario 1"
    anatomy_revision: int = 0
    source_name: str = ""
    revision: int = 0

    def world_to_plan(self, points: Any) -> np.ndarray:
        return np.asarray(points) @ self.frame[:3, :3].T + self.frame[:3, 3]

    def plan_to_world(self, points: Any) -> np.ndarray:
        inverse = np.linalg.inv(self.frame)
        return np.asarray(points) @ inverse[:3, :3].T + inverse[:3, 3]

    @property
    def motions(self) -> dict[str, Motion]:
        return self.scenario.motions

    @property
    def scenario(self) -> Scenario:
        return self.scenarios[self.active_scenario]

    @property
    def plan(self) -> CutPlan:
        return self.plans[self.scenario.plan_id]

    @property
    def configuration_key(self) -> tuple[str, int, int]:
        return self.scenario.plan_id, self.plan.revision, self.anatomy_revision

    @property
    def fragment_names(self) -> dict[int, str]:
        names = dict(FRAGMENTS)
        names[2] = {1: "Maxilla", 2: "Central midface", 3: "Midface"}[self.plan.lefort_level]
        return names
