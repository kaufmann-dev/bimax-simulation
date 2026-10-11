"""Private local protocol and typed MCP inputs; no persistent session metadata."""

import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_MESSAGE = 24 * 1024**2


def runtime_directory() -> Path:
    root = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    info = root.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("BIMAX requires an owner-only user runtime directory.")
    directory = root / "bimax"
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise PermissionError("BIMAX runtime directory must be owned by you with mode 0700.")
    return directory


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Empty(Input):
    pass


class Connect(Input):
    instance_id: str | None = None


class Begin(Input):
    scenario: str | None = None


class Session(Input):
    session_id: str


class Mutation(Session):
    document_id: str | None
    revision: int = Field(ge=-1)


class ScenarioMutation(Mutation):
    scenario: str


class File(Mutation):
    path: str


class CreateScenario(Mutation):
    name: str
    source: str | None = None


class SelectScenario(ScenarioMutation):
    pass


class SelectCuts(ScenarioMutation):
    cut_ids: list[str]


class Plane(Input):
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]


class Region(Input):
    min: tuple[float, float, float]
    max: tuple[float, float, float]


class LeFortGeometry(Input):
    planes: dict[str, Plane]
    groups: list[list[str]]
    region: Region


class BssoSide(Input):
    front: Plane
    upper: Plane
    sagittal: Plane


class BssoGeometry(Input):
    midline: float
    left: BssoSide
    right: BssoSide


class GenioGeometry(Input):
    plane: Plane
    region: Region


class ReplaceCut(Mutation):
    cut_id: str
    geometry: LeFortGeometry | BssoGeometry | GenioGeometry


class Rotation(Input):
    pitch: float | None = None
    roll: float | None = None
    yaw: float | None = None


class Movement(Input):
    translation: tuple[float, float, float] | None = None
    rotation: Rotation | None = None
    pivot: tuple[float, float, float] | None = None


class Movements(ScenarioMutation):
    targets: dict[str, Movement]


class ProximalTarget(Input):
    rotation: Rotation | None = None
    pivot: tuple[float, float, float] | None = None


class ProximalRotations(ScenarioMutation):
    targets: dict[Literal["left-proximal", "right-proximal"], ProximalTarget]


class ResetProximal(ScenarioMutation):
    targets: list[Literal["left-proximal", "right-proximal"]]
    clear_pivots: bool = False


class SymmetryReference(Mutation):
    right: float | None = None
    point: tuple[float, float, float] | None = None


class MeasurementDefinition(ScenarioMutation):
    name: str
    kind: Literal["midline", "paired", "ramus"] | None = None
    landmarks: dict[str, str] | None = None
    remove: bool = False


class Measurements(Input):
    scenario: str


class OrientationAngles(Input):
    pitch: float
    roll: float
    yaw: float


class Orientation(Mutation):
    rotation: OrientationAngles


class Landmark(ScenarioMutation):
    name: str
    fragment: str | None = None
    point: tuple[float, float, float] | None = None
    remove: bool = False


class Confirm(ScenarioMutation):
    fragment: str
    point: tuple[float, float, float]


class Preview(ScenarioMutation):
    view: Literal[
        "contact-sheet", "right", "left", "front", "birds-eye", "axial", "coronal", "sagittal"
    ] = "contact-sheet"
    position: float | None = None


class Job(Input):
    job_id: str


class Cancel(Session, Job):
    pass


class Pick(Input):
    preview_id: str
    view: Literal["right", "left", "front", "birds-eye", "axial", "coronal", "sagittal"]
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


TOOLS: dict[str, tuple[type[Input], str]] = {
    "discover_instances": (Empty, "List local BIMAX instance IDs and open project paths."),
    "connect_instance": (
        Connect,
        "Attach to one BIMAX instance. Launch the installed app if none exist; choose an ID if several exist.",
    ),
    "begin_session": (
        Begin,
        "Begin exclusive AI editing. The user can revoke it with Take control. Returns session and document identity.",
    ),
    "end_session": (
        Session,
        "Release AI editing without saving. Cancel unfinished work; completed edits and undo remain.",
    ),
    "inspect_state": (
        Empty,
        "Inspect document identity, revision, scenarios, seven shared cuts, fragments, movements, pivots, landmarks, acceptance and jobs.",
    ),
    "open_project": (
        File,
        "Open an absolute .bimax file in the worker. Refuse replacement with unsaved changes; returns a job ID.",
    ),
    "save_project": (
        File,
        "Explicitly save to an absolute .bimax filename. Current filename overwrites with timestamped backup; another is Save As. Reject external file conflicts; returns job ID.",
    ),
    "create_scenario": (
        CreateScenario,
        "Create a uniquely named scenario, optionally duplicated from source, and display it.",
    ),
    "select_scenario": (SelectScenario, "Make the named scenario active and visible."),
    "select_cuts": (
        SelectCuts,
        "Replace the named scenario's selected shared cut IDs; at most one variant per Le Fort level.",
    ),
    "replace_cut": (
        ReplaceCut,
        "Replace a shared cut's complete validated geometry. Inspect state for representation and affected scenarios.",
    ),
    "set_movements": (
        Movements,
        "Atomically set one or several targets for the named scenario. Omitted fields retain values. Nonzero rotation requires an explicit or existing pivot.",
    ),
    "set_proximal_rotations": (
        ProximalRotations,
        "Atomically rotate one or both BSSO proximal segments about fixed baseline condyle pivots. No translation or inherited movement. Each pivot approximates the joint center geometrically.",
    ),
    "reset_proximal_rotations": (
        ResetProximal,
        "Reset selected proximal rotations to zero, optionally clearing their pivots.",
    ),
    "set_symmetry_reference": (
        SymmetryReference,
        "Set the project midline plane parallel to planning Anterior/Superior using either a Right offset or a baseline point. No anatomical midline is assumed.",
    ),
    "set_measurement": (
        MeasurementDefinition,
        "Create/update a named definition or explicitly remove it. Midline roles: point. Paired roles: left,right. Ramus roles: left_condyle,left_angle,right_condyle,right_angle. Roles refer to existing named landmarks; no automatic retargeting.",
    ),
    "measurements": (
        Measurements,
        "Inspect reference, named landmarks, and baseline/moved results for an explicit scenario. Midline signed offset is Right minus reference. Paired differences are left minus right in Superior, Anterior and absolute lateral distance (mm). Ramus inclination uses condyle minus angle: frontal atan2(outward,Superior), sagittal atan2(Anterior,Superior), in degrees; differences left minus right. Camera and display orientation do not affect measurements.",
    ),
    "set_orientation": (
        Orientation,
        "Set whole-project view orientation with explicit pitch, roll and yaw; retain prepared geometry.",
    ),
    "set_landmark": (
        Landmark,
        "Create/update a named baseline measurement landmark on a fragment, or remove it. State returns moved position and displacement.",
    ),
    "confirm_fragment": (
        Confirm,
        "Supply a baseline point inside a fragment of the named scenario; does not accept the combination.",
    ),
    "accept_combination": (
        ScenarioMutation,
        "Explicitly accept the named scenario's prepared combination through geometric and confirmation checks. Preview it first.",
    ),
    "undo": (Mutation, "Undo one document edit in this editing session."),
    "redo": (Mutation, "Redo one document edit in this editing session."),
    "preview": (
        Preview,
        "Prepare and display the named scenario with movements, including provisional unreviewed cuts. Returns job ID immediately. Poll job_status for PNG. Default contact sheet is 1024 pixels; individual views 1024. Slice position is millimetres along the oriented planning normal.",
    ),
    "job_status": (
        Job,
        "Inspect preparation/save/open/capture progress; completed previews return PNG image content and metadata.",
    ),
    "cancel_job": (Cancel, "Cancel an active job belonging to the editing session."),
    "pick_image_point": (
        Pick,
        "Pick a fragment and baseline planning point from a preview's named view with normalized top-left pixel coordinates. Contact sheet picking uses coordinates within its individual view. Reject older revisions.",
    ),
}
