"""bimax-cli: headless planning commands for AI agents and scripts.

Every command works on a .bimax project file and prints one JSON document on stdout; failures
print {"error": "..."} and exit 1. Edits are validated like the desktop app's, then saved in
place: atomically, without backups, and refused if the file changed since it was read. Previews
and exports render offscreen, without a display. Points are baseline planning
Right/Anterior/Superior millimetres; angles are pitch, roll, yaw in degrees.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROTATION_AXES = ("pitch", "roll", "yaw")
_qt_application: Any = None  # kept alive while rendering commands run


# -- project IO --------------------------------------------------------------------------------
def _path(value: str) -> Path:
    path = Path(value).absolute()
    if path.suffix.lower() != ".bimax" or not path.is_file():
        raise ValueError(f"{value} is not an existing .bimax project file.")
    return path


def _load(value: str) -> tuple[Any, Path, Any]:
    from .engine import default_cuts
    from .storage import file_version, load_project

    path = _path(value)
    version = file_version(path)
    doc = load_project(path)
    if not doc.cuts:
        doc.cuts = default_cuts(doc)
    return doc, path, version


def _save(doc: Any, path: Path, version: Any) -> None:
    from .storage import save_project

    save_project(doc, path, expected_version=version, backup=False)


def _activate(doc: Any, scenario: str) -> None:
    if scenario not in doc.scenarios:
        raise ValueError(f"Unknown scenario {scenario!r}; scenarios: {', '.join(doc.scenarios)}.")
    doc.active_scenario = scenario


def _prepared(doc: Any, scenario: str) -> dict[str, Any]:
    """Fragments and display surfaces of a scenario's cut combination (cached on disk)."""
    from .prepared import prepare_geometry

    _activate(doc, scenario)
    result = prepare_geometry(doc)
    doc.fragments, doc.fragments_key = result["fragments"], doc.configuration_key
    return result


def _hidden(doc: Any, identities: list[str]) -> set[int]:
    labels = doc.fragment_labels
    unknown = [identity for identity in identities if identity not in labels]
    if unknown:
        raise ValueError(f"Unknown fragment(s) {', '.join(unknown)}; fragments: {', '.join(labels)}.")
    return {labels[identity] for identity in identities}


def _graphics() -> None:
    """Offscreen Qt (text, PNG, PDF) for rendering commands."""
    global _qt_application
    from PySide6.QtGui import QGuiApplication

    _qt_application = QGuiApplication.instance() or QGuiApplication([])


def _edit(project: str, operation: str, scenario: str | None = None, **arguments: Any) -> dict:
    from .commands import Commands
    from .history import History

    doc, path, version = _load(project)
    result = Commands(doc, History()).execute(operation, scenario, **arguments)
    if result["changed"]:
        _save(doc, path, version)
    return {**result, "saved": result["changed"], "project": str(path)}


def _rotation(args: argparse.Namespace) -> dict[str, float] | None:
    values = {axis: getattr(args, axis) for axis in ROTATION_AXES if getattr(args, axis) is not None}
    return values or None


def _json_argument(value: str) -> Any:
    text = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    try:
        return json.loads(text)
    except ValueError as error:
        raise ValueError(f"Invalid JSON: {error}") from error


# -- commands ----------------------------------------------------------------------------------
def inspect(args: argparse.Namespace) -> dict:
    from .commands import inspect_document

    doc, path, _ = _load(args.project)
    return {"project": str(path), **inspect_document(doc)}


def measurements(args: argparse.Namespace) -> dict:
    from .commands import inspect_document

    doc, _, _ = _load(args.project)
    _activate(doc, args.scenario)
    state = inspect_document(doc)
    return {"symmetry_reference": state["symmetry_reference"], "scenario": args.scenario,
            **state["scenarios"][args.scenario]}


def create_scenario(args: argparse.Namespace) -> dict:
    return _edit(args.project, "create_scenario", name=args.name, source=args.source)


def select_scenario(args: argparse.Namespace) -> dict:
    return _edit(args.project, "select_scenario", args.name)


def select_cuts(args: argparse.Namespace) -> dict:
    return _edit(args.project, "select_cuts", args.scenario, cut_ids=list(args.cut_ids))


def replace_cut(args: argparse.Namespace) -> dict:
    return _edit(args.project, "replace_cut", cut_id=args.cut_id, geometry=_json_argument(args.geometry))


def set_movement(args: argparse.Namespace) -> dict:
    values: dict[str, Any] = {}
    if args.translation is not None:
        values["translation"] = args.translation
    if args.pivot is not None:
        values["pivot"] = args.pivot
    if (rotation := _rotation(args)) is not None:
        values["rotation"] = rotation
    if not values:
        raise ValueError("Give --translation, --pivot and/or --pitch/--roll/--yaw.")
    return _edit(args.project, "set_movements", args.scenario, targets={args.target: values})


def reset_pivot(args: argparse.Namespace) -> dict:
    return _edit(args.project, "reset_pivot", args.scenario, target=args.target)


def set_proximal(args: argparse.Namespace) -> dict:
    values: dict[str, Any] = {}
    if args.pivot is not None:
        values["pivot"] = args.pivot
    if (rotation := _rotation(args)) is not None:
        values["rotation"] = rotation
    if not values:
        raise ValueError("Give --pivot and/or --pitch/--roll/--yaw.")
    return _edit(args.project, "set_proximal_rotations", args.scenario,
                 targets={f"{args.side}-proximal": values})


def reset_proximal(args: argparse.Namespace) -> dict:
    return _edit(args.project, "reset_proximal_rotations", args.scenario,
                 targets=[f"{side}-proximal" for side in args.sides], clear_pivots=args.clear_pivots)


def set_symmetry(args: argparse.Namespace) -> dict:
    return _edit(args.project, "set_symmetry_reference", right=args.right, point=args.point)


def set_measurement(args: argparse.Namespace) -> dict:
    roles = {}
    for item in args.role:
        role, separator, landmark = item.partition("=")
        if not separator or not role or not landmark:
            raise ValueError(f"--role takes ROLE=LANDMARK, got {item!r}.")
        roles[role] = landmark
    return _edit(args.project, "set_measurement", args.scenario, name=args.name, kind=args.kind,
                 landmarks=roles)


def remove_measurement(args: argparse.Namespace) -> dict:
    return _edit(args.project, "set_measurement", args.scenario, name=args.name, remove=True)


def set_orientation(args: argparse.Namespace) -> dict:
    return _edit(args.project, "set_orientation",
                 rotation={axis: getattr(args, axis) for axis in ROTATION_AXES})


def set_landmark(args: argparse.Namespace) -> dict:
    return _edit(args.project, "set_landmark", args.scenario, name=args.name, fragment=args.fragment,
                 point=args.point)


def remove_landmark(args: argparse.Namespace) -> dict:
    return _edit(args.project, "set_landmark", args.scenario, name=args.name, remove=True)


def confirm_fragment(args: argparse.Namespace) -> dict:
    return _edit(args.project, "confirm_fragment", args.scenario, fragment=args.fragment, point=args.point)


def _accept(doc: Any, commands: Any, scenario: str) -> dict:
    result = _prepared(doc, scenario)
    return commands.execute("accept_combination", scenario, fragments=result["fragments"],
                            fragment_counts=result["fragment_counts"],
                            anatomy_counts=result["anatomy_counts"])


def accept(args: argparse.Namespace) -> dict:
    from .commands import Commands
    from .history import History

    doc, path, version = _load(args.project)
    result = _accept(doc, Commands(doc, History()), args.scenario)
    _save(doc, path, version)
    return {**result, "saved": True, "project": str(path)}


def prepare(args: argparse.Namespace) -> dict:
    from .engine import required_fragment_labels

    doc, _, _ = _load(args.project)
    result = _prepared(doc, args.scenario)
    counts = result["fragment_counts"]
    required = required_fragment_labels(doc)
    confirmations = doc.review.confirmations
    fragments = [
        {
            "id": fragment.id,
            "name": fragment.name,
            "voxels": int(counts[label]) if label < len(counts) else 0,
            "required": label in required,
            "confirmation": confirmations.get(fragment.id),
        }
        for label, fragment in sorted(doc.fragment_descriptors.items())
    ]
    return {
        "scenario": args.scenario,
        "accepted": doc.cuts_accepted,
        "fragments": fragments,
        "empty_required": [f["id"] for f in fragments if f["required"] and not f["voxels"]],
        "missing_confirmations": [f["id"] for f in fragments if f["required"] and f["confirmation"] is None],
        "preparation_key": doc.configuration_key,
    }


def _scene(args: argparse.Namespace) -> tuple[Any, Any]:
    from .scene import Scene

    doc, _, _ = _load(args.project)
    result = _prepared(doc, args.scenario)
    _graphics()
    return doc, Scene(doc, result["surfaces"], hidden=_hidden(doc, args.hide))


def preview(args: argparse.Namespace) -> dict:
    from .commands import CONVENTIONS
    from .model import COLORS
    from .scene import EXPORT_VIEWS, save_png

    doc, scene = _scene(args)
    with scene:
        image = scene.preview(args.view, args.size, args.position)
        visible = [label for label in scene.actors if scene.is_visible(label)]
    out = Path(args.out).absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_png(image, out)
    names = doc.fragment_descriptors
    return {
        "image": str(out),
        "scenario": args.scenario,
        "view": args.view,
        "tiles": [name for name, *_ in EXPORT_VIEWS] if args.view == "contact-sheet" else [args.view],
        "size": [image.width(), image.height()],
        "position": args.position,
        "review_state": "accepted" if doc.cuts_accepted else "unreviewed",
        "orientation": dict(zip(ROTATION_AXES, doc.view_angles, strict=True)),
        "fragment_colours": {names[label].id: list(COLORS[label]) for label in visible},
        "conventions": CONVENTIONS,
    }


def pick(args: argparse.Namespace) -> dict:
    _, scene = _scene(args)
    with scene:
        return scene.pick(args.view, args.x, args.y, args.size, args.position)


def export_labelmaps(args: argparse.Namespace) -> dict:
    from .storage import export_labelmaps as export

    doc, _, _ = _load(args.project)
    _prepared(doc, args.scenario)
    return {"manifest": str(export(doc, args.out))}


def export_images(args: argparse.Namespace) -> dict:
    _, scene = _scene(args)
    with scene:
        return {"images": [str(path) for path in scene.export_scenario_images(args.out)]}


def export_report(args: argparse.Namespace) -> dict:
    from .report import CaseInfo
    from .report import export_report as export

    doc, scene = _scene(args)
    case = CaseInfo(patient=args.patient, birth_date=args.birth_date, case_id=args.case_id,
                    surgeon=args.surgeon, notes=args.notes)
    path = args.out if args.out.lower().endswith(".pdf") else args.out + ".pdf"
    with scene:
        exported = export(scene, doc, path, project_path=str(_path(args.project)), dirty=False, case=case)
    return {"report": str(exported)}


def export_aligners(args: argparse.Namespace) -> dict:
    from .aligners import aligner_movements, movements_toml, read_landmarks

    doc, _, _ = _load(args.project)
    _prepared(doc, args.scenario)
    movements = aligner_movements(doc, read_landmarks(args.landmarks))
    out = Path(args.out).absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(movements_toml(doc, movements), encoding="utf-8")
    return {"movements": str(out), "tables": movements.tables, "moved_with": movements.fragments}


def batch(args: argparse.Namespace) -> dict:
    from .commands import Commands
    from .history import History

    source = sys.stdin.read() if args.operations == "-" else Path(args.operations).read_text(encoding="utf-8")
    try:
        operations = json.loads(source)
    except ValueError as error:
        raise ValueError(f"Invalid JSON operations: {error}") from error
    if not isinstance(operations, list) or not all(isinstance(item, dict) for item in operations):
        raise ValueError('Operations must be a JSON list of {"operation": ..., "scenario": ..., ...}.')
    doc, path, version = _load(args.project)
    commands = Commands(doc, History())
    results = []
    for index, item in enumerate(operations):
        item = dict(item)
        operation, scenario = item.pop("operation", None), item.pop("scenario", None)
        try:
            if operation == "accept_combination":
                if item:
                    raise ValueError("accept_combination takes only a scenario.")
                results.append(_accept(doc, commands, scenario))
            else:
                results.append(commands.execute(str(operation), scenario, **item))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Operation {index} ({operation}): {error}. Nothing was saved.") from error
    changed = any(result["changed"] for result in results)
    if changed:
        _save(doc, path, version)
    return {"results": results, "saved": changed, "project": str(path)}


# -- parser ------------------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    from .model import CUT_TYPES, new_motions
    from .scene import PREVIEW_VIEWS, VIEW_NAMES

    parser = argparse.ArgumentParser(
        prog="bimax-cli",
        description=__doc__.split("\n\n", 1)[1] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, handler: Any, help_text: str, scenario: bool = False) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("project", help=".bimax project file")
        if scenario:
            sub.add_argument("--scenario", required=True, help="scenario name (see inspect)")
        sub.set_defaults(handler=handler)
        return sub

    def point(sub: argparse.ArgumentParser, flag: str, help_text: str, required: bool = False) -> None:
        sub.add_argument(flag, nargs=3, type=float, metavar=("R", "A", "S"), required=required,
                         help=help_text)

    def rotation(sub: argparse.ArgumentParser, required: bool = False) -> None:
        signs = {"pitch": "+ counterclockwise in the right profile", "roll": "+ right side down",
                 "yaw": "+ front to the left"}
        for axis in ROTATION_AXES:
            sub.add_argument(f"--{axis}", type=float, required=required, help=f"degrees, {signs[axis]}")

    def render_options(sub: argparse.ArgumentParser, default_size: int) -> None:
        sub.add_argument("--hide", nargs="+", default=[], metavar="FRAGMENT",
                         help="fragment identities to hide (e.g. skull); also changes the framing")
        sub.add_argument("--size", type=int, default=default_size, help="image size in pixels")

    command("inspect", inspect, "Scenarios, the seven shared cuts, movements, pivots, landmarks, "
            "measurements, fragments, confirmations and acceptance.")
    command("measurements", measurements, "Symmetry reference, landmarks and baseline/moved "
            "measurement results of a scenario.", scenario=True)

    sub = command("create-scenario", create_scenario, "Create a scenario (optionally a copy) and make it active.")
    sub.add_argument("name")
    sub.add_argument("--from", dest="source", help="scenario to duplicate")
    sub = command("select-scenario", select_scenario, "Make a scenario the project's active scenario.")
    sub.add_argument("name")
    sub = command("select-cuts", select_cuts, "Replace a scenario's selected shared cuts "
                  "(at most one Le Fort variant per level; none clears them).", scenario=True)
    sub.add_argument("cut_ids", nargs="*", choices=sorted(CUT_TYPES), metavar="CUT",
                     help=f"cut ids: {', '.join(CUT_TYPES)}")
    sub = command("replace-cut", replace_cut, "Replace a shared cut's complete geometry (planning "
                  "R/A/S planes; see inspect for the current representation).")
    sub.add_argument("cut_id", choices=sorted(CUT_TYPES))
    sub.add_argument("--geometry", required=True, help="JSON object, or @file.json")

    targets = sorted(new_motions())
    sub = command("set-movement", set_movement, "Set one movement target; omitted values are kept. "
                  "Nonzero rotation needs a pivot (given now or before).", scenario=True)
    sub.add_argument("--target", required=True, choices=targets,
                     help="linked = both jaws, mandible = distal mandible, chin, or a Le Fort cut id")
    point(sub, "--translation", "mm, planning R/A/S")
    point(sub, "--pivot", "rotation pivot, baseline planning R/A/S mm")
    rotation(sub)
    sub = command("reset-pivot", reset_pivot, "Clear a target's pivot and rotation, keeping its translation.",
                  scenario=True)
    sub.add_argument("--target", required=True, choices=targets)
    sub = command("set-proximal", set_proximal, "Rotate a BSSO proximal segment about a fixed condyle "
                  "pivot (no translation, no inherited movement).", scenario=True)
    sub.add_argument("--side", required=True, choices=("left", "right"))
    point(sub, "--pivot", "condyle pivot, baseline planning R/A/S mm")
    rotation(sub)
    sub = command("reset-proximal", reset_proximal, "Zero proximal rotations.", scenario=True)
    sub.add_argument("--sides", nargs="+", required=True, choices=("left", "right"))
    sub.add_argument("--clear-pivots", action="store_true")

    sub = command("set-symmetry", set_symmetry, "Set the midline reference plane, parallel to planning "
                  "Anterior/Superior, by a Right offset or a baseline point.")
    group = sub.add_mutually_exclusive_group(required=True)
    group.add_argument("--right", type=float, help="Right coordinate of the plane, mm")
    group.add_argument("--point", nargs=3, type=float, metavar=("R", "A", "S"))
    sub = command("set-measurement", set_measurement, "Create or update a named measurement over named "
                  "landmarks. Roles: midline point; paired left,right; ramus "
                  "left_condyle,left_angle,right_condyle,right_angle.", scenario=True)
    sub.add_argument("name")
    sub.add_argument("--kind", required=True, choices=("midline", "paired", "ramus"))
    sub.add_argument("--role", nargs="+", required=True, metavar="ROLE=LANDMARK")
    sub = command("remove-measurement", remove_measurement, "Remove a measurement definition.", scenario=True)
    sub.add_argument("name")
    sub = command("set-orientation", set_orientation, "Whole-skull display orientation for images "
                  "(does not change cuts or movements).")
    rotation(sub, required=True)
    sub = command("set-landmark", set_landmark, "Create or update a named baseline landmark on a fragment.",
                  scenario=True)
    sub.add_argument("name")
    sub.add_argument("--fragment", required=True, help="fragment identity (see inspect)")
    point(sub, "--point", "baseline planning R/A/S mm", required=True)
    sub = command("remove-landmark", remove_landmark, "Remove a landmark not used by a measurement.",
                  scenario=True)
    sub.add_argument("name")
    sub = command("confirm-fragment", confirm_fragment, "Place a baseline confirmation point inside a "
                  "fragment (required before accept).", scenario=True)
    sub.add_argument("--fragment", required=True, help="fragment identity (see prepare)")
    point(sub, "--point", "baseline planning R/A/S mm", required=True)
    command("prepare", prepare, "Build the scenario's fragments; list sizes, empty required fragments "
            "and missing confirmations. Does not change the project.", scenario=True)
    command("accept", accept, "Accept the scenario's cut combination after geometric and confirmation "
            "checks (exports and reports need it).", scenario=True)

    sub = command("preview", preview, "Render the scenario with its movements to a PNG: a contact sheet "
                  "(right, left, front, bird's-eye), one 3D view, or a slice.", scenario=True)
    sub.add_argument("--view", default="contact-sheet", choices=PREVIEW_VIEWS)
    sub.add_argument("--position", type=float, help="slice position along the oriented planning normal, mm")
    sub.add_argument("--out", required=True, help="PNG file")
    render_options(sub, 1024)
    sub = command("pick", pick, "Fragment and baseline planning point at normalized top-left image "
                  "coordinates of a single view (also within a contact-sheet tile). Use the same "
                  "--hide and --position as the preview.", scenario=True)
    sub.add_argument("--view", required=True, choices=VIEW_NAMES)
    sub.add_argument("--x", type=float, required=True, help="0..1 from the left")
    sub.add_argument("--y", type=float, required=True, help="0..1 from the top")
    sub.add_argument("--position", type=float, help="slice position, mm")
    render_options(sub, 1024)

    sub = command("export-labelmaps", export_labelmaps, "Moved fragment labelmaps (NIfTI) and manifest.json "
                  "on a shared grid; the combination must be accepted.", scenario=True)
    sub.add_argument("--out", required=True, help="new or empty folder")
    sub = command("export-images", export_images, "Right, left, front and bird's-eye PNGs plus a contact "
                  "sheet.", scenario=True)
    sub.add_argument("--out", required=True, help="new folder")
    sub.add_argument("--hide", nargs="+", default=[], metavar="FRAGMENT")
    sub = command("export-report", export_report, "Landscape PDF plan summary for surgeons; the "
                  "combination must be accepted. Case details are printed, never saved.", scenario=True)
    sub.add_argument("--out", required=True, help="PDF file")
    sub.add_argument("--hide", nargs="+", default=[], metavar="FRAGMENT")
    for field in ("patient", "birth-date", "case-id", "surgeon", "notes"):
        sub.add_argument(f"--{field}", default="")
    sub = command("export-aligners", export_aligners, "Jaw movements as plan.toml tables for the "
                  "aligners app, from its out/surgery/landmarks.json (same CT).", scenario=True)
    sub.add_argument("--landmarks", required=True, help="aligners landmarks.json")
    sub.add_argument("--out", required=True, help="movements.toml")

    sub = command("batch", batch, 'Apply several edits atomically with one save: a JSON list of '
                  '{"operation": ..., "scenario": ..., ...} using the planning operations '
                  "(create_scenario, select_scenario, select_cuts, replace_cut, set_movements, "
                  "reset_pivot, set_proximal_rotations, reset_proximal_rotations, "
                  "set_symmetry_reference, set_measurement, set_orientation, set_landmark, "
                  "confirm_fragment, accept_combination).")
    sub.add_argument("operations", help="JSON file, or - for stdin")
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    # Headless rendering: offscreen Qt, and EGL for VTK when there is no display server.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    args = _parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        sys.stdout.write(json.dumps({"error": str(error) or type(error).__name__}) + "\n")
        return 1
    sys.stdout.write(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
