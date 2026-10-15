"""Surgeon-facing scenario overview: case details, operation movements, and before/after views."""

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cache
from html import escape
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from .model import PROXIMAL_IDS, Document
from .symmetry import landmark_results, measurement_results

FONTS = Path(__file__).with_name("fonts")
SCOPE = (
    "Geometric simulation of bone segments from a CT-derived bone volume. Occlusion, bone "
    "interference, condylar seating and soft-tissue changes are not modelled. For planning "
    "discussion only."
)
# Labels for the values returned by symmetry measurements.
MEASUREMENT_LABELS = {
    "signed_lateral_mm": ("Offset from midline (+ right)", "mm"),
    "distance_mm": ("Distance from midline", "mm"),
    "height_difference_mm": ("Height difference, left − right", "mm"),
    "anterior_difference_mm": ("Anterior difference, left − right", "mm"),
    "lateral_distance_difference_mm": ("Lateral distance difference, left − right", "mm"),
    "left_frontal_deg": ("Left ramus, frontal inclination", "°"),
    "right_frontal_deg": ("Right ramus, frontal inclination", "°"),
    "frontal_difference_deg": ("Frontal inclination difference, left − right", "°"),
    "left_sagittal_deg": ("Left ramus, sagittal inclination", "°"),
    "right_sagittal_deg": ("Right ramus, sagittal inclination", "°"),
    "sagittal_difference_deg": ("Sagittal inclination difference, left − right", "°"),
}


@dataclass
class CaseInfo:
    patient: str = ""
    birth_date: str = ""
    case_id: str = ""
    surgeon: str = ""
    notes: str = ""


def _directional(value: float, positive: str, negative: str, unit: str) -> str | None:
    if abs(value) < 0.05:
        return None
    amount = f"{abs(value):.1f}°" if unit == "°" else f"{abs(value):.1f} {unit}"
    return f"{amount} {positive if value > 0 else negative}"


def movement_terms(translation: list[float], angles: list[float]) -> list[tuple[str, str]]:
    """Describe nonzero planning-frame movement in surgical terms, from the patient's view."""
    right, anterior, superior = translation
    pitch, roll, yaw = angles
    terms = [
        ("Advancement" if anterior > 0 else "Setback", _directional(anterior, "forward", "back", "mm")),
        ("Downgraft" if superior < 0 else "Impaction", _directional(-superior, "down", "up", "mm")),
        ("Transverse shift", _directional(right, "to the patient's right", "to the patient's left", "mm")),
        # Rx(+) lifts anterior points: counterclockwise in the right profile.
        ("Pitch", _directional(pitch, "counterclockwise (anterior up)", "clockwise (anterior down)", "°")),
        # Ry(+) lowers the patient's right side; Rz(+) turns the front toward the patient's left.
        ("Roll (cant)", _directional(roll, "right side down", "left side down", "°")),
        ("Yaw", _directional(yaw, "front toward the patient's left", "front toward the patient's right", "°")),
    ]
    return [(name, text) for name, text in terms if text is not None]


def _movements(doc: Document) -> list[dict[str, Any]]:
    """Operation segments in Move-tab order with their surgical description."""
    names = doc.fragment_names
    selected = doc.scenario.cut_ids
    both = "Both jaws (maxillomandibular complex)"
    entries: list[tuple[str, str, str | None]] = [("linked", both, None)]
    parent = both
    for key in reversed(doc.upper_cuts):
        name = f"{names[doc.fragment_labels[f'upper:{key}']]} ({doc.cuts[key].name})"
        entries.append((key, name, parent))
        parent = name
    mandible = "Distal mandible (BSSO)" if "bsso" in selected else "Mandible"
    entries.append(("mandible", mandible, both))
    if "genio" in selected:
        entries.append(("chin", "Chin (genioplasty)", mandible))
    result = []
    for key, name, relative_to in entries:
        motion = doc.motions[key]
        result.append({"key": key, "name": name, "relative_to": relative_to,
                       "terms": movement_terms(list(motion.translation), list(motion.angles))})
    if "bsso" in selected:
        for key, side in zip(PROXIMAL_IDS, ("Left", "Right"), strict=True):
            rotation = doc.scenario.proximal_rotations[key]
            result.append({"key": key, "name": f"{side} proximal segment (BSSO)",
                           "relative_to": "the condyle, rotation only",
                           "terms": movement_terms([0.0, 0.0, 0.0], list(rotation.angles))})
    return result


def report_data(doc: Document, *, project_path: str | None, dirty: bool, case: CaseInfo,
                hidden: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Collect the plan overview without Qt or full-volume computation."""
    identities = {fragment.id: fragment.name for fragment in doc.fragment_descriptors.values()}
    landmarks = []
    for item in landmark_results(doc):
        if item["status"] != "valid":
            continue
        dr, da, ds = item["displacement"]
        landmarks.append({
            "name": item["name"],
            "segment": identities.get(item["fragment"], item["fragment"]),
            "anterior": _directional(da, "forward", "back", "mm") or "—",
            "vertical": _directional(ds, "up", "down", "mm") or "—",
            "transverse": _directional(dr, "right", "left", "mm") or "—",
            "distance_mm": item["distance_mm"],
        })
    measurements = []
    for name, entry in measurement_results(doc).items():
        if entry["status"] != "valid":
            continue
        values = []
        for key, after in entry["moved"].items():
            label, unit = MEASUREMENT_LABELS.get(key, (key.replace("_", " ").capitalize(), ""))
            before = entry["baseline"][key]
            values.append({"label": label, "unit": unit, "before": before, "after": after,
                           "change": after - before})
        measurements.append({"name": name, "values": values})
    upper = list(reversed(doc.upper_cuts))
    lower = [key for key in ("bsso", "genio") if key in doc.scenario.cut_ids]
    return {
        "case": asdict(case),
        "generated": datetime.now().astimezone().strftime("%d %b %Y, %H:%M"),
        "version": version("bimax-simulation"),
        "project": Path(project_path).name if project_path else None,
        "dirty": dirty,
        "scenario": doc.active_scenario,
        "accepted": doc.cuts_accepted,
        "operations": [doc.cuts[key].name for key in upper + lower],
        "hidden": list(hidden),
        "movements": _movements(doc),
        "landmarks": landmarks,
        "measurements": measurements,
    }


# --- PDF layout -----------------------------------------------------------

STYLE = """
body { font-family: 'Inter'; font-size: 9.5pt; color: #ececec; }
h1 { font-size: 22pt; font-weight: 700; margin: 0; color: #ffffff; }
h2 { font-size: 13pt; font-weight: 600; margin-top: 18px; margin-bottom: 8px; color: #ffffff; }
h3 { font-size: 10.5pt; font-weight: 600; margin-top: 12px; margin-bottom: 4px; color: #8ec5ff; }
p { margin-top: 0; margin-bottom: 6px; }
.subtitle { font-size: 12pt; color: #b5b5b5; margin-bottom: 14px; }
.muted { font-size: 8pt; color: #8a8a8a; }
.warning { color: #ff8a80; font-weight: 600; }
.segment { color: #8ec5ff; font-weight: 600; }
.label { color: #8a8a8a; }
.caption { font-size: 9pt; color: #b5b5b5; font-weight: 600; }
table { border-collapse: collapse; }
th { font-size: 8pt; font-weight: 600; color: #8a8a8a; text-align: left; border-bottom: 1px solid #3a3a3a; }
td { color: #ececec; border-bottom: 1px solid #262626; }
td.key { color: #8a8a8a; width: 30%; }
th.num, td.num { text-align: right; }
div.page { page-break-before: always; }
"""


def _table(headers: list[str] | None, rows: list[str], numeric_from: int = 99) -> str:
    # A thead row repeats when Qt splits the table across pages.
    head = "" if headers is None else "<thead><tr>" + "".join(
        f'<th class="{"num" if index >= numeric_from else ""}">{escape(header)}</th>'
        for index, header in enumerate(headers)
    ) + "</tr></thead>"
    body = "".join(f"<tr>{row}</tr>" for row in rows)
    return f'<table width="100%" cellspacing="0" cellpadding="6">{head}{body}</table>'


def _td(value: Any, css: str = "") -> str:
    return f'<td class="{css}">{escape(str(value))}</td>'


def _signed(value: float, unit: str) -> str:
    text = f"{value:+.1f}"
    return ("0.0" if float(text) == 0 else text) + (f" {unit}" if unit else "")


def _pair(view: str, size: tuple[int, int]) -> str:
    cell = ('<td align="center" width="50%"><img src="{name}" width="{w}" height="{h}">'
            '<p class="caption">{label}</p></td>')
    return ('<table width="100%" cellspacing="0" cellpadding="0"><tr>'
            + cell.format(name=f"pre:{view}", w=size[0], h=size[1], label="Before")
            + cell.format(name=f"post:{view}", w=size[0], h=size[1], label="After")
            + "</tr></table>")


def crop_pair(before: Any, after: Any, padding: float = 0.04) -> tuple[Any, Any]:
    """Crop both images to their combined non-black content, keeping them comparable."""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QImage

    box = None
    for image in (before, after):
        rgb = image.convertToFormat(QImage.Format.Format_RGB888)
        pixels = np.frombuffer(rgb.constBits(), dtype=np.uint8).reshape(rgb.height(), rgb.bytesPerLine())
        content = pixels[:, :rgb.width() * 3].reshape(rgb.height(), rgb.width(), 3).max(axis=2) > 8
        rows, columns = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
        if rows.size:
            found = np.array([columns[0], rows[0], columns[-1], rows[-1]])
            box = found if box is None else np.r_[np.minimum(box[:2], found[:2]), np.maximum(box[2:], found[2:])]
    if box is None:
        return before, after
    pad = round(max(before.width(), before.height()) * padding)
    left, top = max(0, box[0] - pad), max(0, box[1] - pad)
    right, bottom = min(before.width() - 1, box[2] + pad), min(before.height() - 1, box[3] + pad)
    rect = QRect(int(left), int(top), int(right - left + 1), int(bottom - top + 1))
    return before.copy(rect), after.copy(rect)


def report_html(data: dict[str, Any], sizes: dict[str, tuple[int, int]]) -> str:
    """Images are referenced as ``pre:<view>`` and ``post:<view>`` resources sized by view."""
    views = list(sizes)
    case = data["case"]
    parts = ["<h1>Surgical plan summary</h1>",
             f'<p class="subtitle">{escape(data["scenario"])}</p>']
    rows = [f'<td class="key">{escape(key)}</td>{_td(value)}' for key, value in (
        ("Patient", case["patient"]), ("Date of birth", case["birth_date"]), ("Case ID", case["case_id"]),
        ("Surgeon / planner", case["surgeon"]), ("Report date", data["generated"]),
    ) if value]
    rows.append(f'<td class="key">Operations</td>{_td(" · ".join(data["operations"]) or "None")}')
    if case["notes"]:
        rows.append(f'<td class="key">Notes</td><td>{escape(case["notes"]).replace(chr(10), "<br>")}</td>')
    parts.append(_table(None, rows))
    if not data["accepted"]:
        parts.append('<p class="warning">The cut combination has not been reviewed and accepted.</p>')
    hero = "Right" if "Right" in views else (views[0] if views else None)
    if hero:
        parts.append(f"<h2>{escape(hero)} profile</h2>" + _pair(hero, sizes[hero]))
    parts.append(f'<p class="muted">{escape(SCOPE)}</p>')

    parts.append('<div class="page"><h2>Planned movements</h2>')
    parts.append('<p class="muted">Directions from the patient\'s perspective. Each segment\'s movement '
                 'is added to the movement of the segment it is listed relative to. Rotations are about '
                 'the planned pivot point.</p>')
    rows = []
    for movement in data["movements"]:
        relative = (f'<br><span class="muted">Relative to {escape(movement["relative_to"])}</span>'
                    if movement["relative_to"] else "")
        terms = "<br>".join(f'<span class="label">{escape(name)}</span>&nbsp;&nbsp;{escape(text)}'
                            for name, text in movement["terms"]) or '<span class="label">No movement</span>'
        rows.append(f'<td width="42%"><span class="segment">{escape(movement["name"])}</span>{relative}</td>'
                    f"<td>{terms}</td>")
    parts.append(_table(["Segment", "Movement"], rows))
    if data["landmarks"]:
        parts.append("<h2>Landmark movement</h2>")
        parts.append(_table(["Landmark", "Segment", "Anterior / posterior", "Vertical", "Transverse", "Total"], [
            _td(item["name"]) + _td(item["segment"]) + _td(item["anterior"]) + _td(item["vertical"])
            + _td(item["transverse"]) + _td(f"{item['distance_mm']:.1f} mm", "num")
            for item in data["landmarks"]
        ], numeric_from=5))
    if data["measurements"]:
        parts.append("<h2>Symmetry</h2>")
        rows = []
        for measurement in data["measurements"]:
            rows.append(f'<td colspan="4"><span class="segment">{escape(measurement["name"])}</span></td>')
            rows.extend(
                _td(value["label"]) + _td(_signed(value["before"], value["unit"]), "num")
                + _td(_signed(value["after"], value["unit"]), "num")
                + _td(_signed(value["change"], value["unit"]), "num")
                for value in measurement["values"]
            )
        parts.append(_table(["Measurement", "Before", "After", "Change"], rows, numeric_from=1))
    parts.append("</div>")

    for index in range(0, len(views), 2):
        parts.append('<div class="page">')
        if index == 0:
            parts.append("<h2>Before and after</h2>")
            if data["hidden"]:
                parts.append(f'<p class="muted">Not shown: {escape(", ".join(data["hidden"]))}.</p>')
        for view in views[index:index + 2]:
            parts.append(f"<h3>{escape(view)}</h3>" + _pair(view, sizes[view]))
        parts.append("</div>")
    return "".join(parts)


@cache
def _register_inter() -> None:
    from PySide6.QtGui import QFontDatabase

    for name in ("Inter-Regular.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf"):
        if QFontDatabase.addApplicationFont(str(FONTS / name)) < 0:
            raise RuntimeError(f"Could not load the bundled font {name}.")


def write_report_pdf(path: str | Path, data: dict[str, Any], images: dict[str, tuple[Any, Any]]) -> Path:
    """Paginate the report onto black A4 pages with a footer, replacing ``path`` atomically."""
    from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt, QUrl
    from PySide6.QtGui import (
        QColor,
        QFont,
        QPageLayout,
        QPageSize,
        QPainter,
        QPdfWriter,
        QTextDocument,
    )

    _register_inter()
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
        writer.setTitle(f"Surgical plan summary — {data['scenario']}")
        writer.setCreator(f"BIMAX {data['version']}")
        dots = writer.resolution() / 25.4
        margin, footer = 15 * dots, 8 * dots
        width, height = writer.width(), writer.height()
        content = QSizeF(width - 2 * margin, height - 2 * margin - footer)

        document = QTextDocument()
        document.documentLayout().setPaintDevice(writer)
        document.setDocumentMargin(0)
        document.setDefaultFont(QFont("Inter"))
        document.setDefaultStyleSheet(STYLE)
        document.setPageSize(content)
        for view, (pre, post) in images.items():
            document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(f"pre:{view}"), pre)
            document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(f"post:{view}"), post)
        # HTML pixel lengths are scaled from 96 dpi to the paint device; two
        # images share a row and two rows share a page.
        scale = 96 / writer.logicalDpiX()
        width_limit, height_limit = content.width() * 0.49 * scale, content.height() * 0.36 * scale
        sizes = {}
        for view, (pre, _) in images.items():
            factor = min(width_limit / pre.width(), height_limit / pre.height())
            sizes[view] = (int(pre.width() * factor), int(pre.height() * factor))
        document.setHtml(report_html(data, sizes))
        pages = document.pageCount()

        case = data["case"]
        footer_text = " · ".join(value for value in (
            case["patient"], case["case_id"], data["scenario"], data["generated"],
            f"BIMAX {data['version']}" + (" · unsaved changes" if data["dirty"] else ""),
        ) if value)
        font = QFont("Inter")
        font.setPointSizeF(7)
        painter = QPainter()
        if not painter.begin(writer):
            raise RuntimeError("Could not open the PDF for writing.")
        try:
            for page in range(pages):
                if page:
                    writer.newPage()
                painter.fillRect(QRectF(0, 0, width, height), QColor("#000000"))
                painter.save()
                painter.translate(margin, margin - page * content.height())
                document.drawContents(painter, QRectF(0, page * content.height(), content.width(), content.height()))
                painter.restore()
                painter.setFont(font)
                painter.setPen(QColor("#7a7a7a"))
                line = QRectF(margin, height - margin - footer * 0.6, content.width(), footer * 0.6)
                painter.drawText(line, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, footer_text)
                painter.drawText(line, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                                 f"{page + 1} / {pages}")
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
    """Render before/after views with shared framing and write the PDF report."""
    from .views import REPORT_IMAGE_SIZE, REPORT_VIEWS

    if doc.fragments is None or not views.fragment_mode or not doc.cuts_accepted:
        raise ValueError("Review and accept the active cut combination before exporting a report.")
    hidden = [doc.fragment_names[label] for label, actor in views.actors.items()
              if not actor.GetVisibility() and label in doc.fragment_names]
    data = report_data(doc, project_path=project_path, dirty=dirty, case=case, hidden=hidden)
    bounds = views.comparison_bounds()
    pre = views.render_views(REPORT_VIEWS, REPORT_IMAGE_SIZE, baseline=True, bounds=bounds)
    post = views.render_views(REPORT_VIEWS, REPORT_IMAGE_SIZE, bounds=bounds)
    images = {label: crop_pair(pre_image, post_image)
              for (label, pre_image), (_, post_image) in zip(pre, post, strict=True)}
    return write_report_pdf(path, data, images)
