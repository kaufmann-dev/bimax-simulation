"""Shared anatomy, reusable cuts, and scenario-owned movement in baseline LPS mm."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

ANATOMY = {1: "Unassigned", 2: "Skull and upper jaw", 3: "Mandible and lower teeth", 4: "Excluded"}
CUT_TYPES = {
    "lefort1": ("Le Fort I", "lefort", 1),
    "lefort2": ("Le Fort II", "lefort", 2),
    "lefort3": ("Le Fort III", "lefort", 3),
    "modified-lefort2": ("Modified Le Fort II", "lefort", 2),
    "modified-lefort3": ("Modified Le Fort III", "lefort", 3),
    "bsso": ("BSSO", "bsso", 0),
    "genio": ("Genioplasty", "genio", 0),
}
COLORS = {
    1: (0.75, 0.77, 0.80), 2: (0.28, 0.72, 0.95),
    3: (0.95, 0.63, 0.27), 4: (0.70, 0.42, 0.90),
    5: (0.30, 0.83, 0.63), 6: (0.98, 0.43, 0.55),
    7: (0.38, 0.38, 0.38), 8: (0.95, 0.86, 0.28),
    9: (0.37, 0.55, 0.95), 10: (0.30, 0.80, 0.85),
}
ANATOMY_COLORS = {1: COLORS[8], 2: COLORS[2], 3: COLORS[5], 4: COLORS[7]}
# Change whenever cut predicates, fragment composition, or display geometry change.
PREPARATION_VERSION = 1


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


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
    return {key: Motion() for key in ("linked", "mandible", "chin", *(
        key for key, (_, kind, _) in CUT_TYPES.items() if kind == "lefort"
    ))}


@dataclass
class Cut:
    name: str
    kind: str
    level: int
    geometry: dict[str, Any] = field(default_factory=dict)
    confirmations: dict[str, list[float]] = field(default_factory=dict)
    accepted: bool = False
    revision: int = 0

    @property
    def signature(self) -> str:
        return fingerprint([self.kind, self.level, self.geometry])


@dataclass
class Review:
    confirmations: dict[str, list[float]] = field(default_factory=dict)
    accepted: bool = False


@dataclass(frozen=True)
class Fragment:
    id: str
    label: int
    name: str
    anatomy: int
    motion: str | None = None
    cut_id: str | None = None


@dataclass
class Artifact:
    kind: str
    digest: str
    path: str


@dataclass
class Scenario:
    cut_ids: list[str] = field(default_factory=list)
    motions: dict[str, Motion] = field(default_factory=new_motions)
    landmarks: list[dict[str, Any]] = field(default_factory=list)


def new_scenario(cut_ids: list[str] | None = None) -> Scenario:
    return Scenario(cut_ids=list(cut_ids or []))


@dataclass
class Document:
    geometry: Geometry
    source: np.ndarray
    anatomy: np.ndarray
    seeds: np.ndarray
    barriers: np.ndarray
    fragments: np.ndarray | None = None
    fragments_key: str | None = None
    frame: np.ndarray = field(default_factory=lambda: np.diag([-1.0, -1.0, 1.0, 1.0]))
    cuts: dict[str, Cut] = field(default_factory=dict)
    scenarios: dict[str, Scenario] = field(default_factory=lambda: {"Scenario 1": new_scenario()})
    active_scenario: str = "Scenario 1"
    reviews: dict[str, Review] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    anatomy_revision: int = 0
    source_name: str = ""
    revision: int = 0
    _identity: str = field(default_factory=lambda: uuid4().hex, init=False, repr=False)
    _anatomy_digest: str = field(default="", repr=False)
    _digest_revision: int = field(default=-1, repr=False)

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
    def upper_cuts(self) -> list[str]:
        ids = sorted((key for key in self.scenario.cut_ids if self.cuts[key].kind == "lefort"),
                     key=lambda key: self.cuts[key].level)
        if len({self.cuts[key].level for key in ids}) != len(ids):
            raise ValueError("Select only one variant per Le Fort level.")
        return ids

    @property
    def anatomy_key(self) -> str:
        if self._digest_revision != self.anatomy_revision:
            digest = hashlib.sha256()
            with np.nditer(self.anatomy, flags=["external_loop", "buffered", "zerosize_ok"],
                           op_flags=["readonly"], order="C", buffersize=4 * 1024**2) as chunks:
                for chunk in chunks:
                    digest.update(np.asarray(chunk).tobytes())
            self._anatomy_digest = digest.hexdigest()
            self._digest_revision = self.anatomy_revision
        return fingerprint([PREPARATION_VERSION, self._anatomy_digest,
                            asdict(self.geometry), np.asarray(self.frame).tolist()])

    def cut_key(self, key: str) -> str:
        return fingerprint(["cut", self.anatomy_key, key, self.cuts[key].signature])

    def subsystem_key(self, kind: str) -> str:
        ids = self.upper_cuts if kind == "upper" else sorted(
            key for key in self.scenario.cut_ids if self.cuts[key].kind != "lefort"
        )
        return fingerprint([kind, self.anatomy_key, [self.cut_key(key) for key in ids]])

    @property
    def configuration_key(self) -> str:
        return fingerprint(["scene", self.anatomy_key,
                            self.subsystem_key("upper"), self.subsystem_key("lower")])

    @property
    def review(self) -> Review:
        key = self.configuration_key
        if key not in self.reviews:
            points = {}
            for cut_id in self.scenario.cut_ids:
                points.update(deepcopy(self.cuts[cut_id].confirmations))
            self.reviews[key] = Review(points)
        return self.reviews[key]

    @property
    def cuts_accepted(self) -> bool:
        if not self.reviews or any(not self.cuts[key].accepted for key in self.scenario.cut_ids):
            return False
        review = self.reviews.get(self.configuration_key)
        return review is not None and review.accepted

    @property
    def fragment_descriptors(self) -> dict[int, Fragment]:
        result = {
            1: Fragment("skull", 1, "Fixed skull", 2),
            3: Fragment("left-proximal", 3, "Left proximal", 3, cut_id="bsso"),
            4: Fragment("right-proximal", 4, "Right proximal", 3, cut_id="bsso"),
            5: Fragment("mandible", 5, "Distal mandible", 3, "mandible"),
            6: Fragment("chin", 6, "Chin", 3, "chin", "genio"),
            7: Fragment("excluded", 7, "Excluded", 4),
            8: Fragment("unassigned", 8, "Unassigned", 1),
        }
        for key, label in zip(self.upper_cuts, (2, 9, 10), strict=False):
            cut = self.cuts[key]
            name = {1: "Maxilla", 2: "Central midface", 3: "Midface"}[cut.level]
            if key.startswith("modified-"):
                name = "Modified " + name.lower()
            result[label] = Fragment(f"upper:{key}", label, name, 2, key, key)
        return result

    @property
    def fragment_names(self) -> dict[int, str]:
        return {label: fragment.name for label, fragment in self.fragment_descriptors.items()}

    @property
    def fragment_labels(self) -> dict[str, int]:
        return {fragment.id: label for label, fragment in self.fragment_descriptors.items()}

    @property
    def parent_labels(self) -> np.ndarray:
        result = np.zeros(max(self.fragment_descriptors) + 1, dtype=np.uint8)
        for label, fragment in self.fragment_descriptors.items():
            result[label] = fragment.anatomy
        return result

    def accept_review(self) -> None:
        self.review.accepted = True
        for key in self.scenario.cut_ids:
            cut = self.cuts[key]
            cut.accepted = True
            ids = {f.id for f in self.fragment_descriptors.values() if f.cut_id == key}
            if cut.kind in ("bsso", "genio"):
                ids.add("mandible")
            cut.confirmations.update({fid: list(point) for fid, point in self.review.confirmations.items()
                                      if fid in ids})
