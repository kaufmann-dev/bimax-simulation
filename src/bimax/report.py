"""Printable scenario summary: case details, pre-op/planned views, and every planning parameter."""

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .engine import fragment_transforms, label_counts
from .model import ANATOMY, COLORS, PROXIMAL_IDS, Document
from .symmetry import SIGNS, landmark_results, measurement_results

SCOPE = (
    "Geometric exploration of a binary bone volume. Bone interference, occlusion, joint "
    "biomechanics, soft-tissue prediction and clinical feasibility are not modelled, and "
    "fractures are not predicted. Cut shapes are editable geometric approximations, not "
    "detected anatomy or validated osteotomy trajectories."
)
MOTION_FORMULA = "M = T(t) · T(p) · Rz(yaw) · Ry(roll) · Rx(pitch) · T(−p)"


@dataclass
class CaseInfo:
    patient: str = ""
    birth_date: str = ""
    case_id: str = ""
    surgeon: str = ""
    notes: str = ""


def _plan(doc: Document, world: Any) -> list[float]:
    return [float(value) for value in doc.world_to_plan(world)]


def _controls(doc: Document) -> list[dict[str, Any]]:
    """Movement controls in Move-tab order, with their ownership and inheritance."""
    names = doc.fragment_names
    entries: list[dict[str, Any]] = [{"key": "linked", "name": "Linked jaws", "parent": None, "label": None}]
    parent = "Linked jaws"
    for key in reversed(doc.upper_cuts):
        label = doc.fragment_labels[f"upper:{key}"]
        entries.append({"key": key, "name": names[label], "parent": parent, "label": label})
        parent = names[label]
    entries.append({"key": "mandible", "name": "Distal mandible", "parent": "Linked jaws", "label": 5})
    if "genio" in doc.scenario.cut_ids:
        entries.append({"key": "chin", "name": "Chin", "parent": "Distal mandible", "label": 6})
    result: list[dict[str, Any]] = []
    for entry in entries:
        motion = doc.motions[entry["key"]]
        t = [float(value) for value in motion.translation]
        result.append({
            **entry,
            "proximal": False,
            "translation": t,
            "ui": {"right": t[0], "advance": t[1], "downgraft": -t[2]},
            "angles": [float(value) for value in motion.angles],
            "pivot": [float(value) for value in motion.pivot],
            "pivot_set": motion.pivot_set,
        })
    if "bsso" in doc.scenario.cut_ids:
        for key, label in zip(PROXIMAL_IDS, (3, 4), strict=True):
            rotation = doc.scenario.proximal_rotations[key]
            result.append({
                "key": key,
                "name": f"{names[label]} segment",
                "parent": None,
                "label": label,
                "proximal": True,
                "translation": [0.0, 0.0, 0.0],
                "ui": {"right": 0.0, "advance": 0.0, "downgraft": 0.0},
                "angles": [float(value) for value in rotation.angles],
                "pivot": [float(value) for value in rotation.pivot],
                "pivot_set": rotation.pivot_set,
            })
    return result


def _geometry_rows(value: Any, path: str = "") -> list[dict[str, Any]]:
    """Flatten a cut definition into planes, regions, groups and scalars."""
    if isinstance(value, dict) and {"origin", "normal"} <= set(value):
        return [{"name": path, "type": "plane", "origin": value["origin"], "normal": value["normal"]}]
    if isinstance(value, dict) and {"min", "max"} <= set(value):
        return [{"name": path, "type": "region", "min": value["min"], "max": value["max"]}]
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            rows.extend(_geometry_rows(item, f"{path}.{key}" if path else key))
        return rows
    if path == "groups":
        return [{"name": path, "type": "groups", "value": value}]
    return [{"name": path, "type": "value", "value": value}]


def _cut_rule(kind: str) -> str:
    return {
        "lefort": "Fragment = bone inside the region and inside the union of plane groups; "
        "each group is the intersection of its planes' positive half-spaces (normals point into the fragment).",
        "bsso": "Per side (left: Right < midline, right: Right ≥ midline): positive side of the front plane "
        "and of the upper or the sagittal plane, applied to the distal mandible.",
        "genio": "Chin = distal-mandible bone inside the region on the positive side of the plane.",
    }[kind]


def report_data(
    doc: Document,
    *,
    project_path: str | None,
    dirty: bool,
    case: CaseInfo,
    hidden: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Collect every scenario parameter without Qt, in planning Right/Anterior/Superior mm."""
    from scipy.spatial.transform import Rotation

    if doc.fragments is None:
        raise ValueError("Review and accept the active cut combination before exporting a report.")
    geometry = doc.geometry
    voxel_mm3 = float(np.prod(geometry.spacing))
    anatomy_counts = label_counts(doc.anatomy, max(ANATOMY))
    descriptors = doc.fragment_descriptors
    counts = label_counts(doc.fragments, max(descriptors))
    present = [label for label in sorted(descriptors) if counts[label]]
    centres = ndimage.center_of_mass(np.ones_like(doc.fragments, dtype=np.uint8), doc.fragments, present)
    transforms = fragment_transforms(doc)
    inverse_frame = np.linalg.inv(doc.frame)
    fragments = []
    for label, centre in zip(present, centres, strict=True):
        world = geometry.index_to_world(np.asarray(centre)[::-1])
        matrix = transforms[label]
        baseline = _plan(doc, world)
        moved = _plan(doc, matrix[:3, :3] @ world + matrix[:3, 3])
        delta = np.subtract(moved, baseline)
        fragment = descriptors[label]
        fragments.append({
            "label": label,
            "name": fragment.name,
            "identity": fragment.id,
            "color": COLORS.get(label, (0.8, 0.8, 0.8)),
            "voxels": int(counts[label]),
            "volume_cm3": counts[label] * voxel_mm3 / 1000,
            "centroid": baseline,
            "moved_centroid": moved,
            "displacement": delta.tolist(),
            "distance_mm": float(np.linalg.norm(delta)),
            "hidden": fragment.name in hidden,
            "transform_planning": (doc.frame @ matrix @ inverse_frame).tolist(),
            "transform_lps": matrix.tolist(),
        })
    cuts = []
    for key in doc.scenario.cut_ids:
        cut = doc.cuts[key]
        cuts.append({
            "id": key,
            "name": cut.name,
            "kind": cut.kind,
            "level": cut.level,
            "revision": cut.revision,
            "accepted": cut.accepted,
            "rule": _cut_rule(cut.kind),
            "geometry": _geometry_rows(cut.geometry),
        })
    measurements = []
    for name, entry in measurement_results(doc).items():
        values = []
        if entry["status"] == "valid":
            for key, moved_value in entry["moved"].items():
                baseline_value = entry["baseline"][key]
                values.append({
                    "key": key,
                    "baseline": baseline_value,
                    "moved": moved_value,
                    "change": moved_value - baseline_value,
                })
        measurements.append({
            "name": name,
            "kind": entry["definition"]["kind"],
            "landmarks": entry["definition"]["landmarks"],
            "status": entry["status"],
            "error": entry.get("error", ""),
            "signs": entry["sign_conventions"],
            "values": values,
        })
    frame_angles = Rotation.from_matrix(doc.frame[:3, :3] @ np.diag([-1.0, -1.0, 1.0])).as_euler(
        "xyz", degrees=True
    )
    confirmations = doc.review.confirmations
    return {
        "case": asdict(case),
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "version": version("bimax-simulation"),
        "project": project_path,
        "dirty": dirty,
        "source": doc.source_name,
        "scenario": doc.active_scenario,
        "accepted": doc.cuts_accepted,
        "hidden": list(hidden),
        "names": dict(doc.fragment_names),
        "identities": {fragment.id: fragment.name for fragment in descriptors.values()},
        "volume": {
            "shape_xyz": list(geometry.shape[::-1]),
            "spacing": list(geometry.spacing),
            "origin": list(geometry.origin),
            "direction": np.asarray(geometry.direction).reshape(3, 3).tolist(),
            "voxel_mm3": voxel_mm3,
        },
        "frame": {
            "angles": [float(value) for value in frame_angles],
            "matrix": np.asarray(doc.frame).tolist(),
        },
        "view_angles": [float(value) for value in doc.view_angles],
        "symmetry_right": doc.symmetry_right,
        "anatomy": [
            {"label": label, "name": name, "voxels": int(anatomy_counts[label]),
             "volume_cm3": anatomy_counts[label] * voxel_mm3 / 1000}
            for label, name in ANATOMY.items()
        ],
        "anatomy_revision": doc.anatomy_revision,
        "seed_voxels": int(np.count_nonzero(doc.seeds)),
        "barrier_voxels": int(np.count_nonzero(doc.barriers)),
        "cuts": cuts,
        "confirmations": [
            {"identity": identity, "point": [float(value) for value in point]}
            for identity, point in sorted(confirmations.items())
        ],
        "fragments": fragments,
        "controls": _controls(doc),
        "landmarks": landmark_results(doc),
        "measurements": measurements,
        "signs": dict(SIGNS),
    }


# --- PDF layout -----------------------------------------------------------

def _n(value: Any, digits: int = 2) -> str:
    text = f"{float(value):.{digits}f}"
    return text.removeprefix("-") if float(text) == 0 else text


def _cells(values: list[Any], digits: int = 2) -> str:
    return "".join(f'<td class="num">{_n(value, digits)}</td>' for value in values)


def _table(headers: list[str] | None, rows: list[str], numeric_from: int = 1) -> str:
    head = "" if headers is None else "<tr>" + "".join(
        f'<th class="{"num" if index >= numeric_from else ""}">{escape(header)}</th>'
        for index, header in enumerate(headers)
    ) + "</tr>"
    body = "".join(f"<tr>{row}</tr>" for row in rows)
    return f'<table width="100%" cellspacing="0" cellpadding="5">{head}{body}</table>'


def _text(value: Any) -> str:
    return f"<td>{escape(str(value))}</td>"


def _rgb(color: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(component * 255):02x}" for component in color)


def _matrix(matrix: list[list[float]], digits: int = 4) -> str:
    rows = "".join(f"<tr>{_cells(row, digits)}</tr>" for row in matrix)
    return f'<table class="matrix" cellspacing="0" cellpadding="3">{rows}</table>'


def _case_rows(data: dict[str, Any]) -> list[str]:
    case = data["case"]
    labels = (
        ("Patient", case["patient"]), ("Date of birth", case["birth_date"]),
        ("Case ID", case["case_id"]), ("Surgeon / planner", case["surgeon"]),
        ("Scenario", data["scenario"]), ("Cuts", " + ".join(c["name"] for c in data["cuts"]) or "None"),
        ("Combination", "Accepted" if data["accepted"] else "NOT ACCEPTED"),
        ("Source volume", data["source"]), ("Project file", data["project"] or "Not saved"),
        ("Generated", data["generated"]), ("BIMAX version", data["version"]),
    )
    rows = [f'<td class="key">{escape(key)}</td>{_text(value or "—")}' for key, value in labels]
    if case["notes"]:
        notes = escape(case["notes"]).replace("\n", "<br>")
        rows.append(f'<td class="key">Notes</td><td>{notes}</td>')
    return rows


def _movement_summary(data: dict[str, Any]) -> str:
    rows = []
    for control in data["controls"]:
        ui = control["ui"]
        translation = ["—"] * 3 if control["proximal"] else [_n(ui[k]) for k in ("right", "advance", "downgraft")]
        rows.append(
            _text(control["name"])
            + "".join(f'<td class="num">{value}</td>' for value in translation)
            + _cells(control["angles"])
        )
    return _table(
        ["Control", "Right (+) mm", "Advance (+) mm", "Downgraft (+) mm", "Pitch °", "Roll °", "Yaw °"], rows
    )


def _image_cell(name: str, caption: str, width: int) -> str:
    return (f'<td align="center"><img src="{name}" width="{width}" height="{width}">'
            f'<p class="caption">{escape(caption)}</p></td>')


def report_html(data: dict[str, Any], views: list[str], image_width: int) -> str:
    """Images are referenced as ``pre:<view>`` and ``post:<view>`` resources."""
    parts = ['<h1>BIMAX scenario report</h1>',
             f'<p class="subtitle">{escape(data["scenario"])} · planned jaw movement summary</p>']
    if data["dirty"]:
        parts.append('<p class="warning">The project has unsaved changes. This report reflects the '
                     'state in the app, not the saved project file.</p>')
    if not data["accepted"]:
        parts.append('<p class="warning">The cut combination is not accepted.</p>')
    parts.append(_table(None, _case_rows(data)))
    parts.append("<h2>Planned movements</h2>")
    parts.append(_movement_summary(data))
    parts.append('<p class="small">Translations in the head reference frame. Right: toward the patient\'s right. '
                 'Advance: anterior. Downgraft: inferior. Positive pitch is counterclockwise in the right '
                 'profile (anterior to screen-right). Proximal segments rotate only.</p>')
    parts.append(f'<p class="small"><b>Scope.</b> {escape(SCOPE)}</p>')
    cover = [view for view in ("Front", "Right") if view in views]
    if cover:
        parts.append("<table width=\"100%\" cellspacing=\"0\"><tr>"
                     + "".join(_image_cell(f"post:{view}", f"Planned · {view}", image_width * 3 // 5) for view in cover)
                     + "</tr></table>")

    for index in range(0, len(views), 2):
        parts.append('<div class="page">' + ("<h2>Pre-op and planned views</h2>" if index == 0 else ""))
        if index == 0:
            parts.append('<p class="small">Moved bone surfaces of the visible structures. Pre-op and '
                         'planned images of each angle share one camera and scale. Soft tissue is not shown.'
                         + (f' Hidden in images: {escape(", ".join(data["hidden"]))}.' if data["hidden"] else "")
                         + "</p>")
        for view in views[index:index + 2]:
            parts.append(f"<h3>{escape(view)}</h3><table width=\"100%\" cellspacing=\"0\"><tr>"
                         + _image_cell(f"pre:{view}", "Pre-op (baseline)", image_width)
                         + _image_cell(f"post:{view}", "Planned", image_width)
                         + "</tr></table>")
        parts.append("</div>")

    parts.append('<div class="page"><h2>Movement parameters</h2>')
    parts.append('<p class="small">All values are in the head reference (planning) frame: Right / Anterior / '
                 f'Superior in millimetres. Each control applies {escape(MOTION_FORMULA)} with stored '
                 'translation t and pivot p, then inherits its parent control.</p>')
    rows = []
    for control in data["controls"]:
        rows.append(
            _text(control["name"]) + _cells(control["translation"]) + _cells(control["angles"])
            + (_cells(control["pivot"]) if control["pivot_set"] else '<td class="num" colspan="3">not set</td>')
        )
    parts.append(_table(["Control", "t R", "t A", "t S", "Pitch °", "Roll °", "Yaw °",
                         "Pivot R", "Pivot A", "Pivot S"], rows))
    parts.append("<h3>Inheritance</h3>")
    parts.append(_table(["Control", "Moves fragment", "Inherits from"], [
        _text(control["name"])
        + _text(data["names"].get(control["label"], "—") if control["label"] else "All jaw fragments")
        + _text("None (rotation only)" if control["proximal"] else control["parent"] or "None (fixed skull)")
        for control in data["controls"]
    ], numeric_from=3))
    parts.append('<p class="small">Inheritance: the linked control carries every upper-jaw fragment and '
                 'the distal mandible; nested Le Fort controls inherit the next higher level; the chin '
                 'inherits the distal mandible. Proximal segments rotate independently about their fixed '
                 'baseline condyle pivots and are not carried by any other control.</p>')

    parts.append("<h2>Fragments</h2>")
    rows = []
    for fragment in data["fragments"]:
        rows.append(
            f'<td style="background-color:{_rgb(fragment["color"])}">&nbsp;&nbsp;&nbsp;</td>'
            + _text(fragment["name"] + (" (hidden in images)" if fragment["hidden"] else ""))
            + f'<td class="num">{fragment["voxels"]:,}</td>' + _cells([fragment["volume_cm3"]])
            + _cells(fragment["centroid"], 1) + _cells(fragment["displacement"]) + _cells([fragment["distance_mm"]])
        )
    parts.append(_table(["", "Fragment", "Voxels", "cm³", "Centroid R", "A", "S",
                         "ΔR mm", "ΔA mm", "ΔS mm", "|Δ| mm"], rows, numeric_from=2))
    parts.append('<p class="small">Centroid: baseline centre of the fragment\'s bone voxels. Δ: centroid '
                 'displacement from baseline to planned position.</p>')
    parts.append("</div>")

    parts.append('<div class="page"><h2>Landmarks</h2>')
    if data["landmarks"]:
        rows = []
        for item in data["landmarks"]:
            moved = (_cells(item["moved"]) + _cells(item["displacement"]) + _cells([item["distance_mm"]])
                     if item["status"] == "valid"
                     else f'<td colspan="7">{escape(item.get("error", "invalid"))}</td>')
            rows.append(_text(item["name"]) + _text(data["identities"].get(item["fragment"], item["fragment"]))
                        + _cells(item["baseline"]) + moved)
        parts.append(_table(["Landmark", "Fragment", "Base R", "Base A", "Base S", "Plan R", "Plan A",
                             "Plan S", "ΔR", "ΔA", "ΔS", "|Δ| mm"], rows, numeric_from=2))
    else:
        parts.append("<p>No landmarks defined.</p>")
    parts.append("<h2>Symmetry measurements</h2>")
    reference = data["symmetry_right"]
    parts.append("<p>Reference plane: " + (
        f"parallel to the planning Anterior/Superior plane at Right = {_n(reference)} mm."
        if reference is not None else "unset.") + "</p>")
    if data["measurements"]:
        for measurement in data["measurements"]:
            roles = ", ".join(f"{role}: {name}" for role, name in measurement["landmarks"].items())
            parts.append(f"<h3>{escape(measurement['name'])} ({escape(measurement['kind'])})</h3>"
                         f'<p class="small">{escape(roles)}. {escape(measurement["signs"])}</p>')
            if measurement["status"] != "valid":
                parts.append(f'<p class="warning">{escape(measurement["error"])}</p>')
                continue
            rows = []
            for value in measurement["values"]:
                key = value["key"]
                unit = "mm" if key.endswith("_mm") else "°" if key.endswith("_deg") else ""
                label = key.removesuffix("_mm").removesuffix("_deg").replace("_", " ").capitalize()
                rows.append(_text(f"{label} ({unit})" if unit else label)
                            + _cells([value["baseline"], value["moved"], value["change"]]))
            parts.append(_table(["Value", "Pre-op", "Planned", "Change"], rows))
    else:
        parts.append("<p>No measurements defined.</p>")
    parts.append("</div>")

    parts.append('<div class="page"><h2>Cut definitions</h2>')
    parts.append('<p class="small">Planes are given by an origin point and a normal in planning coordinates; '
                 'a point is on the positive side when (x − origin) · normal ≥ 0. Regions are axis-aligned '
                 'boxes in planning coordinates.</p>')
    for cut in data["cuts"]:
        parts.append(f"<h3>{escape(cut['name'])} · id {escape(cut['id'])} · revision {cut['revision']}"
                     f" · {'accepted' if cut['accepted'] else 'not accepted'}</h3>"
                     f'<p class="small">{escape(cut["rule"])}</p>')
        rows = []
        for row in cut["geometry"]:
            if row["type"] == "plane":
                rows.append(_text(row["name"]) + _text("plane") + _cells(row["origin"]) + _cells(row["normal"], 4))
            elif row["type"] == "region":
                rows.append(_text(row["name"]) + _text("region min / max")
                            + _cells(row["min"]) + _cells(row["max"]))
            elif row["type"] == "groups":
                text = " ∪ ".join("(" + " ∩ ".join(group) + ")" for group in row["value"])
                rows.append(_text(row["name"]) + _text("plane groups") + f'<td colspan="6">{escape(text)}</td>')
            else:
                rows.append(_text(row["name"]) + _text("value") + f'<td colspan="6">{escape(str(row["value"]))}</td>')
        parts.append(_table(["Element", "Type", "Origin / min R", "A", "S", "Normal / max R", "A", "S"],
                            rows, numeric_from=2))
    if data["confirmations"]:
        parts.append("<h3>Fragment confirmation points</h3>")
        parts.append(_table(["Fragment", "R", "A", "S"],
                            [_text(data["identities"].get(item["identity"], item["identity"])) + _cells(item["point"])
                             for item in data["confirmations"]]))
    parts.append("</div>")

    parts.append('<div class="page"><h2>Volume, reference and segmentation</h2>')
    volume = data["volume"]
    rows = [
        '<td class="key">Shape X × Y × Z</td>' + _text(" × ".join(str(v) for v in volume["shape_xyz"]) + " voxels"),
        '<td class="key">Spacing</td>' + _text(" × ".join(_n(v, 4) for v in volume["spacing"]) + " mm"),
        '<td class="key">Origin (LPS)</td>' + _text(", ".join(_n(v, 3) for v in volume["origin"]) + " mm"),
        '<td class="key">Voxel volume</td>' + _text(_n(volume["voxel_mm3"], 4) + " mm³"),
        '<td class="key">Head reference pitch / roll / yaw</td>'
        + _text(" / ".join(_n(v) for v in data["frame"]["angles"]) + " °"),
        '<td class="key">Whole-skull display orientation pitch / roll / yaw</td>'
        + _text(" / ".join(_n(v) for v in data["view_angles"]) + " ° (display and images only)"),
        '<td class="key">Anatomy revision</td>' + _text(data["anatomy_revision"]),
        '<td class="key">Painted seed / barrier voxels</td>'
        + _text(f'{data["seed_voxels"]:,} / {data["barrier_voxels"]:,} (voxel data, stored in the project file)'),
    ]
    parts.append(_table(None, rows))
    parts.append("<h3>Direction matrix (index to LPS)</h3>" + _matrix(volume["direction"]))
    parts.append("<h3>Head reference matrix (LPS to planning)</h3>" + _matrix(data["frame"]["matrix"]))
    parts.append("<h3>Segmentation</h3>")
    parts.append(_table(["Anatomy", "Voxels", "cm³"], [
        _text(item["name"]) + f'<td class="num">{item["voxels"]:,}</td>' + _cells([item["volume_cm3"]])
        for item in data["anatomy"]
    ]))
    parts.append("</div>")

    parts.append('<div class="page"><h2>Resulting fragment transforms</h2>')
    parts.append('<p class="small">Composite baseline-to-planned 4 × 4 matrices including inheritance, in the '
                 'planning frame (mm) and in scanner LPS (mm).</p>')
    rows = []
    for fragment in data["fragments"]:
        rows.append(_text(fragment["name"]) + f"<td>{_matrix(fragment['transform_planning'])}</td>"
                    + f"<td>{_matrix(fragment['transform_lps'])}</td>")
    parts.append(_table(["Fragment", "Planning frame", "LPS"], rows, numeric_from=3))

    parts.append("<h2>Conventions</h2><ul>"
                 "<li>Planning frame: Right / Anterior / Superior millimetres, defined by the head reference "
                 "matrix applied to scanner LPS. All cuts, pivots, landmarks and translations use it.</li>"
                 f"<li>Movement: {escape(MOTION_FORMULA)}; angles in degrees; pitch about Right is applied first, then roll about "
                 "Anterior, then yaw about Superior (matrix order Rz · Ry · Rx).</li>"
                 "<li>Positive pitch is counterclockwise in the right-profile view, anterior to screen-right.</li>"
                 "<li>UI terms: Right = t R, Advance = t A, Downgraft = −t S.</li>"
                 + "".join(f"<li>{escape(kind.capitalize())} measurement: {escape(text)}.</li>"
                           for kind, text in data["signs"].items())
                 + "</ul>")
    parts.append(f'<p class="small">{escape(SCOPE)}</p></div>')
    return "".join(parts)


STYLE = """
body { font-family: sans-serif; font-size: 9pt; color: #111; }
td, th { font-size: 8pt; }
h1 { font-size: 20pt; margin: 0; }
h2 { font-size: 13pt; margin-top: 14px; margin-bottom: 6px; color: #12406b; }
h3 { font-size: 10pt; margin-top: 10px; margin-bottom: 4px; }
p.subtitle { font-size: 12pt; color: #444; margin-bottom: 10px; }
p.small, p.caption { font-size: 8pt; color: #444; }
p.caption { margin-top: 2px; }
p.warning { color: #a01010; font-weight: bold; }
table { border-collapse: collapse; }
th { background-color: #e6edf4; font-weight: bold; text-align: left; border-bottom: 1px solid #8aa; }
td { border-bottom: 1px solid #dde; vertical-align: top; }
th.num, td.num { text-align: right; }
td.key { font-weight: bold; width: 32%; }
table.matrix td { border-bottom: 0; font-size: 8pt; }
div.page { page-break-before: always; }
"""


def write_report_pdf(path: str | Path, data: dict[str, Any], images: dict[str, tuple[Any, Any]]) -> Path:
    """Paginate the report onto A4 pages with a footer, replacing ``path`` atomically."""
    from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt, QUrl
    from PySide6.QtGui import QFont, QPageLayout, QPageSize, QPainter, QPdfWriter, QTextDocument

    destination = Path(path).absolute()
    if destination.is_symlink() or destination.is_dir():
        raise ValueError("Report destination must be a regular file path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".bimax-report-", suffix=".pdf", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer = QPdfWriter(str(temporary))
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)
        writer.setResolution(300)
        writer.setTitle(f"BIMAX scenario report — {data['scenario']}")
        writer.setCreator(f"BIMAX {data['version']}")
        dots = writer.resolution() / 25.4
        margin, footer = 15 * dots, 8 * dots
        width, height = writer.width(), writer.height()
        content = QSizeF(width - 2 * margin, height - 2 * margin - footer)

        document = QTextDocument()
        document.documentLayout().setPaintDevice(writer)
        document.setDocumentMargin(0)
        document.setDefaultStyleSheet(STYLE)
        document.setPageSize(content)
        for view, (pre, post) in images.items():
            document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(f"pre:{view}"), pre)
            document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(f"post:{view}"), post)
        # HTML pixel lengths are scaled from 96 dpi to the paint device.
        image_width = int(content.width() * 0.48 * 96 / writer.logicalDpiX())
        document.setHtml(report_html(data, list(images), image_width))
        pages = document.pageCount()

        case = data["case"]
        identity = " · ".join(value for value in (case["patient"], case["case_id"]) if value)
        footer_text = " · ".join(value for value in (identity, data["scenario"], data["generated"]) if value)
        font = QFont("sans-serif")
        font.setPointSizeF(7.5)
        painter = QPainter()
        if not painter.begin(writer):
            raise RuntimeError("Could not open the PDF for writing.")
        try:
            for page in range(pages):
                if page:
                    writer.newPage()
                painter.save()
                painter.translate(margin, margin - page * content.height())
                document.drawContents(painter, QRectF(0, page * content.height(), content.width(), content.height()))
                painter.restore()
                painter.setFont(font)
                painter.setPen(Qt.GlobalColor.darkGray)
                line = QRectF(margin, height - margin - footer * 0.6, content.width(), footer * 0.6)
                painter.drawLine(line.topLeft(), line.topRight())
                painter.drawText(line, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, footer_text)
                painter.drawText(line, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                                 f"Page {page + 1} of {pages}")
        finally:
            painter.end()
        del painter, writer
        if temporary.stat().st_size == 0:
            raise RuntimeError("The PDF writer produced an empty file.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def export_report(views: Any, doc: Document, path: str | Path, *, project_path: str | None,
                  dirty: bool, case: CaseInfo) -> Path:
    """Render pre-op and planned views with shared framing and write the PDF report."""
    from .views import REPORT_IMAGE_SIZE, REPORT_VIEWS

    if doc.fragments is None or not views.fragment_mode or not doc.cuts_accepted:
        raise ValueError("Review and accept the active cut combination before exporting a report.")
    hidden = [doc.fragment_names[label] for label, actor in views.actors.items()
              if not actor.GetVisibility() and label in doc.fragment_names]
    data = report_data(doc, project_path=project_path, dirty=dirty, case=case, hidden=hidden)
    bounds = views.comparison_bounds()
    pre = views.render_views(REPORT_VIEWS, REPORT_IMAGE_SIZE, baseline=True, bounds=bounds)
    post = views.render_views(REPORT_VIEWS, REPORT_IMAGE_SIZE, bounds=bounds)
    images = {label: (pre_image, post_image)
              for (label, pre_image), (_, post_image) in zip(pre, post, strict=True)}
    return write_report_pdf(path, data, images)
