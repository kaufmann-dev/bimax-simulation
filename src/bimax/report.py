"""Surgeon-facing plan summary on fixed landscape pages: movements, measurements, before/after."""

import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from .model import PROXIMAL_IDS, Document
from .symmetry import landmark_results, measurement_results

FONTS = Path(__file__).with_name("fonts")
SCOPE = (
    "Geometric bone simulation from a CT-derived bone volume. Occlusion, interferences, condylar "
    "seating and soft tissue are not modelled. For planning discussion only."
)
# Labels for the values returned by symmetry measurements.
MEASUREMENT_LABELS = {
    "signed_lateral_mm": ("Offset from midline, + right", "mm"),
    "height_difference_mm": ("Height difference, L − R", "mm"),
    "anterior_difference_mm": ("Anterior difference, L − R", "mm"),
    "lateral_distance_difference_mm": ("Lateral distance difference, L − R", "mm"),
    "left_frontal_deg": ("Left ramus, frontal inclination", "°"),
    "right_frontal_deg": ("Right ramus, frontal inclination", "°"),
    "frontal_difference_deg": ("Frontal inclination difference, L − R", "°"),
    "left_sagittal_deg": ("Left ramus, sagittal inclination", "°"),
    "right_sagittal_deg": ("Right ramus, sagittal inclination", "°"),
    "sagittal_difference_deg": ("Sagittal inclination difference, L − R", "°"),
}
OVERVIEW_VIEWS = ("Right", "Front")
VIEWS_PER_PAGE = 3


@dataclass
class CaseInfo:
    patient: str = ""
    birth_date: str = ""
    case_id: str = ""
    surgeon: str = ""
    notes: str = ""


def clinical_values(translation: list[float], angles: list[float]) -> list[float]:
    """Planning R/A/S movement as AP (+ forward), vertical (+ down), transverse (+ right), angles."""
    right, anterior, superior = (float(value) for value in translation)
    return [anterior, -superior, right, *(float(value) for value in angles)]


def _movements(doc: Document) -> list[dict[str, Any]]:
    """Segments in inheritance order; ``level`` is the depth below the linked jaws."""
    names = doc.fragment_names
    selected = doc.scenario.cut_ids
    entries = [("linked", "Both jaws", "Maxillomandibular complex", 0)]
    for level, key in enumerate(reversed(doc.upper_cuts), start=1):
        entries.append((key, names[doc.fragment_labels[f"upper:{key}"]], doc.cuts[key].name, level))
    bsso = "bsso" in selected
    entries.append(("mandible", "Distal mandible" if bsso else "Mandible", "BSSO" if bsso else "", 1))
    if "genio" in selected:
        entries.append(("chin", "Chin", "Genioplasty", 2))
    result = []
    for key, name, detail, level in entries:
        motion = doc.motions[key]
        result.append({"key": key, "name": name, "detail": detail, "level": level,
                       "values": clinical_values(motion.translation, motion.angles)})
    if bsso:
        for key, side in zip(PROXIMAL_IDS, ("Left", "Right"), strict=True):
            angles = doc.scenario.proximal_rotations[key].angles
            values: list[float | None] = [None, None, None, *(float(value) for value in angles)]
            result.append({"key": key, "name": f"{side} proximal", "detail": "BSSO · condyle pivot",
                           "level": 0, "values": values})
    return result


def report_data(doc: Document, *, project_path: str | None, dirty: bool, case: CaseInfo,
                hidden: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Collect the plan overview without Qt or full-volume computation."""
    identities = {fragment.id: fragment.name for fragment in doc.fragment_descriptors.values()}
    landmarks = []
    for item in landmark_results(doc):
        if item["status"] == "valid":
            dr, da, ds = item["displacement"]
            landmarks.append({"name": item["name"], "segment": identities.get(item["fragment"], item["fragment"]),
                              "values": [da, -ds, dr], "distance_mm": item["distance_mm"]})
    measurements = []
    for name, entry in measurement_results(doc).items():
        if entry["status"] != "valid":
            continue
        values = []
        for key, after in entry["moved"].items():
            if key == "distance_mm":
                continue  # The absolute value of the signed midline offset.
            label, unit = MEASUREMENT_LABELS.get(key, (key.replace("_", " ").capitalize(), ""))
            before = entry["baseline"][key]
            values.append({"label": label, "unit": unit, "before": before, "after": after,
                           "change": after - before})
        measurements.append({"name": name, "values": values})
    upper = list(reversed(doc.upper_cuts))
    lower = [key for key in ("bsso", "genio") if key in doc.scenario.cut_ids]
    return {
        "case": asdict(case),
        "generated": datetime.now().astimezone().strftime("%d %b %Y"),
        "version": version("bimax-simulation"),
        "project": Path(project_path).name if project_path else None,
        "dirty": dirty,
        "scenario": doc.active_scenario,
        "operations": [doc.cuts[key].name for key in upper + lower],
        "hidden": list(hidden),
        "movements": _movements(doc),
        "landmarks": landmarks,
        "measurements": measurements,
    }


def crop_pair(before: Any, after: Any, padding: float = 0.03) -> tuple[Any, Any]:
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


# --- Page layout (millimetres on landscape A4) ------------------------------

PAGE_W, PAGE_H = 297.0, 210.0
MARGIN = 14.0
BODY_TOP, BODY_BOTTOM = 34.0, 194.0
GUTTER = 9.0
INK = {
    "page": "#000000", "title": "#f4f4f6", "text": "#e3e3e7", "muted": "#9b9ba3",
    "faint": "#606067", "rule": "#2b2b30", "line": "#1c1c20", "accent": "#8ab4ff",
}


@dataclass
class Column:
    heading: str
    legend: str
    width: float  # Fraction of the table width.
    align: str = "right"


@dataclass
class Table:
    title: str
    columns: list[Column]
    rows: list[list[Any]]
    row_height: float = 4.9
    note: str = ""
    header: bool = True


@dataclass
class Placement:
    table: Table
    x: float
    y: float
    width: float
    rows: range
    continued: bool


@dataclass
class Region:
    page: int
    x: float
    top: float
    width: float
    bottom: float
    y: float = field(init=False)

    def __post_init__(self) -> None:
        self.y = self.top


def _signed(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    if abs(value) < 0.5 * 10 ** -digits:
        return "–"
    return f"{value:+.{digits}f}".replace("-", "−")


def _wrap(canvas: "Canvas", text: str, font: Any, width: float) -> list[str]:
    from PySide6.QtGui import QFontMetricsF

    metrics, limit, lines = QFontMetricsF(font, canvas.writer), width * canvas.k, []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and metrics.horizontalAdvance(candidate) > limit:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def _tables(canvas: "Canvas", data: dict[str, Any], width: float) -> list[Table]:
    tables = []
    if data["case"]["notes"]:
        lines = _wrap(canvas, data["case"]["notes"], canvas.font(7.4), width - 2.4)
        tables.append(Table("Notes", [Column("", "", 1.0, "left")], [[line] for line in lines],
                            row_height=4.4, header=False))
    movement_columns = [
        Column("Segment", "", 0.37, "left"),
        Column("AP mm", "+ forward", 0.105), Column("Vert mm", "+ down", 0.105),
        Column("Trans mm", "+ right", 0.105), Column("Pitch °", "+ CCW", 0.105),
        Column("Roll °", "+ R down", 0.105), Column("Yaw °", "+ front L", 0.105),
    ]
    rows = [[{"name": item["name"], "detail": item["detail"], "level": item["level"]},
             *(_signed(value) for value in item["values"])] for item in data["movements"]]
    tables.append(Table(
        "Planned movements", movement_columns, rows, row_height=6.6,
        note="Patient perspective. Indented segments move relative to the segment above them; "
             "proximal segments rotate only, about the condyle. Rotations are about the planned pivot.",
    ))
    if data["landmarks"]:
        columns = [Column("Landmark", "", 0.27, "left"), Column("Segment", "", 0.22, "left"),
                   Column("AP mm", "+ forward", 0.1275), Column("Vert mm", "+ down", 0.1275),
                   Column("Trans mm", "+ right", 0.1275), Column("Total mm", "", 0.1275)]
        rows = [[item["name"], item["segment"], *(_signed(value) for value in item["values"]),
                 f"{item['distance_mm']:.1f}"] for item in data["landmarks"]]
        tables.append(Table("Landmark movement", columns, rows))
    if data["measurements"]:
        columns = [Column("Measurement", "", 0.52, "left"), Column("Before", "", 0.16),
                   Column("After", "", 0.16), Column("Change", "", 0.16)]
        rows = []
        for measurement in data["measurements"]:
            rows.append([{"group": measurement["name"]}])
            for value in measurement["values"]:
                unit = f" {value['unit']}" if value["unit"] == "mm" else value["unit"]
                rows.append([{"label": value["label"]},
                             *(_signed(value[key]) + ("" if _signed(value[key]) == "–" else unit)
                               for key in ("before", "after", "change"))])
        tables.append(Table("Symmetry", columns, rows))
    return tables


class Canvas:
    """Millimetre drawing helpers over a QPdfWriter at its native resolution."""

    def __init__(self, writer: Any):
        from PySide6.QtGui import QPainter

        self.writer = writer
        self.k = writer.resolution() / 25.4
        self.painter = QPainter()
        if not self.painter.begin(writer):
            raise RuntimeError("Could not open the PDF for writing.")
        hints = QPainter.RenderHint
        self.painter.setRenderHints(hints.Antialiasing | hints.TextAntialiasing | hints.SmoothPixmapTransform)

    def rect(self, x: float, y: float, w: float, h: float) -> Any:
        from PySide6.QtCore import QRectF

        return QRectF(x * self.k, y * self.k, w * self.k, h * self.k)

    @staticmethod
    def font(size: float, weight: str = "regular", caps: bool = False) -> Any:
        from PySide6.QtGui import QFont

        font = QFont("Inter")
        font.setPointSizeF(size)
        font.setWeight({"regular": QFont.Weight.Normal, "semibold": QFont.Weight.DemiBold,
                        "bold": QFont.Weight.Bold}[weight])
        # Alternate glyphs (tabular figures, contextual hyphens) lose their text
        # mapping in Qt's PDF subsets, so copied values would drop characters.
        font.setFeature(QFont.Tag("calt"), 0)
        if caps:
            font.setCapitalization(QFont.Capitalization.AllUppercase)
            font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
        return font

    def text(self, x: float, y: float, w: float, h: float, text: str, font: Any, color: str,
             align: str = "left", wrap: bool = False) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QFontMetricsF

        horizontal = {"left": Qt.AlignmentFlag.AlignLeft, "right": Qt.AlignmentFlag.AlignRight,
                      "center": Qt.AlignmentFlag.AlignHCenter}[align]
        if wrap:
            flags = int(horizontal | Qt.AlignmentFlag.AlignTop) | int(Qt.TextFlag.TextWordWrap)
        else:
            flags = int(horizontal | Qt.AlignmentFlag.AlignVCenter)
            text = QFontMetricsF(font, self.writer).elidedText(text, Qt.TextElideMode.ElideRight, w * self.k)
        self.painter.setFont(font)
        self.painter.setPen(QColor(color))
        self.painter.drawText(self.rect(x, y, w, h), flags, text)

    def text_height(self, text: str, font: Any, w: float) -> float:
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QFontMetricsF

        bounds = QFontMetricsF(font, self.writer).boundingRect(
            QRectF(0, 0, w * self.k, 1e6), int(Qt.TextFlag.TextWordWrap), text)
        return bounds.height() / self.k

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 0.2) -> None:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QPen

        self.painter.setPen(QPen(QColor(color), width * self.k))
        self.painter.drawLine(QPointF(x1 * self.k, y1 * self.k), QPointF(x2 * self.k, y2 * self.k))

    def rule(self, x1: float, y: float, x2: float, color: str, width: float = 0.2) -> None:
        self.line(x1, y, x2, y, color, width)

    def image(self, image: Any, x: float, y: float, w: float, h: float) -> None:
        scale = min(w / image.width(), h / image.height())
        dw, dh = image.width() * scale, image.height() * scale
        self.painter.drawImage(self.rect(x + (w - dw) / 2, y + (h - dh) / 2, dw, dh), image)

    def background(self) -> None:
        from PySide6.QtGui import QColor

        self.painter.fillRect(self.rect(0, 0, PAGE_W, PAGE_H), QColor(INK["page"]))


def _section(canvas: Canvas, x: float, y: float, w: float, title: str) -> float:
    canvas.text(x, y, w, 4, title, canvas.font(6.5, "semibold", caps=True), INK["accent"])
    return y + SECTION_HEIGHT


TABLE_HEADER = 7.0
SECTION_HEIGHT = 5.0
BLOCK_GAP = 5.5


def _note_height(canvas: Canvas, table: Table, width: float) -> float:
    return canvas.text_height(table.note, canvas.font(6.2), width) + 2.0 if table.note else 0.0


def _draw_table(canvas: Canvas, placement: Placement) -> None:
    table, x, w = placement.table, placement.x, placement.width
    y = _section(canvas, x, placement.y, w, table.title + (" (continued)" if placement.continued else ""))
    if table.header:
        head, legend = canvas.font(6.2, "semibold"), canvas.font(5.6)
        position = x
        for column in table.columns:
            width = column.width * w
            inset = 1.2 if column.align == "left" else 0
            canvas.text(position + inset, y, width - 1.2, 3.4, column.heading, head, INK["muted"], column.align)
            if column.legend:
                canvas.text(position + inset, y + 3.2, width - 1.2, 3, column.legend, legend, INK["faint"],
                            column.align)
            position += width
        y += TABLE_HEADER - 0.6
        canvas.rule(x, y, x + w, INK["rule"], 0.25)
    body, strong, small = canvas.font(7.4), canvas.font(7.4, "semibold"), canvas.font(6.0)
    for index in placement.rows:
        row, h = table.rows[index], table.row_height
        first = row[0]
        if isinstance(first, dict) and "group" in first:
            canvas.text(x + 1.2, y, w - 2.4, h, first["group"], strong, INK["title"])
        else:
            position = x
            for column, cell in zip(table.columns, row, strict=True):
                width = column.width * w
                if isinstance(cell, dict) and "name" in cell:
                    indent = 1.2 + 3.6 * cell["level"]
                    if cell["level"]:
                        elbow = position + indent - 2.4
                        canvas.line(elbow, y + 0.4, elbow, y + h * 0.34, INK["faint"], 0.18)
                        canvas.line(elbow, y + h * 0.34, elbow + 1.5, y + h * 0.34, INK["faint"], 0.18)
                    canvas.text(position + indent, y + h * 0.16, width - indent, h * 0.4, cell["name"], strong, INK["title"])
                    canvas.text(position + indent, y + h * 0.52, width - indent, h * 0.34, cell["detail"], small, INK["muted"])
                elif isinstance(cell, dict):
                    canvas.text(position + 4.0, y, width - 4.0, h, cell["label"], body, INK["muted"])
                else:
                    color = INK["faint"] if cell == "–" else INK["text"]
                    inset = 1.2 if column.align == "left" else 0
                    canvas.text(position + inset, y, width - 1.2 - inset, h, str(cell), body, color, column.align)
                position += width
        y += h
        if table.header:
            canvas.rule(x, y, x + w, INK["line"], 0.18)
    if table.note and placement.rows.stop == len(table.rows):
        canvas.text(x, y + 1.6, w, 20, table.note, canvas.font(6.2), INK["faint"], wrap=True)


def _flow(canvas: Canvas, tables: list[Table], first: Region) -> list[list[Placement]]:
    """Place tables into the overview column, then two-column continuation pages."""
    pages: list[list[Placement]] = [[]]
    queue = [first]

    def next_region() -> Region:
        queue.pop(0)
        if not queue:
            pages.append([])
            width = (PAGE_W - 2 * MARGIN - GUTTER) / 2
            queue.extend(Region(len(pages) - 1, MARGIN + column * (width + GUTTER), BODY_TOP, width, BODY_BOTTOM)
                         for column in range(2))
        return queue[0]

    for table in tables:
        start, continued = 0, False
        fixed = SECTION_HEIGHT + (TABLE_HEADER if table.header else 0)
        while start < len(table.rows):
            region = queue[0]
            remaining = len(table.rows) - start
            whole = fixed + remaining * table.row_height + _note_height(canvas, table, region.width)
            if region.y + whole <= region.bottom:
                pages[region.page].append(Placement(table, region.x, region.y, region.width,
                                                    range(start, len(table.rows)), continued))
                region.y += whole + BLOCK_GAP
                break
            # Move a table whole when a fresh column holds it; split only longer tables.
            available = int((region.bottom - region.y - fixed) // table.row_height)
            if region.y > region.top and (whole <= region.bottom - region.top or available < 3):
                next_region()
                continue
            count = max(1, min(available, remaining - 1))
            pages[region.page].append(Placement(table, region.x, region.y, region.width,
                                                range(start, start + count), continued))
            start += count
            next_region()
            continued = True
    return pages


def _header(canvas: Canvas, data: dict[str, Any], label: str, title: str, subtitle: str) -> None:
    left_width = PAGE_W - 2 * MARGIN - 95
    canvas.text(MARGIN, 11.5, left_width, 4, label, canvas.font(6.5, "semibold", caps=True), INK["accent"])
    canvas.text(MARGIN, 15.6, left_width, 7, title, canvas.font(13.5, "semibold"), INK["title"])
    canvas.text(MARGIN, 23.0, left_width, 4, subtitle, canvas.font(7.2), INK["muted"])
    case = data["case"]
    details = " · ".join(value for value in (
        f"Born {case['birth_date']}" if case["birth_date"] else "",
        f"Case {case['case_id']}" if case["case_id"] else "",
        case["surgeon"], data["generated"],
    ) if value)
    right = PAGE_W - MARGIN - 95
    if case["patient"]:
        canvas.text(right, 15.6, 95, 7, case["patient"], canvas.font(10, "semibold"), INK["title"], "right")
    canvas.text(right, 23.0, 95, 4, details, canvas.font(7.2), INK["muted"], "right")
    canvas.rule(MARGIN, 28.6, PAGE_W - MARGIN, INK["rule"], 0.25)


def _footer(canvas: Canvas, data: dict[str, Any], page: int, pages: int) -> None:
    canvas.rule(MARGIN, 198.6, PAGE_W - MARGIN, INK["rule"], 0.25)
    small = canvas.font(5.8)
    canvas.text(MARGIN, 199.6, 200, 4, SCOPE, small, INK["faint"])
    status = "Unsaved project changes · " if data["dirty"] else ""
    canvas.text(PAGE_W - MARGIN - 70, 199.6, 70, 4,
                f"{status}BIMAX {data['version']} · Page {page} of {pages}", small, INK["faint"], "right")


def _comparison(canvas: Canvas, images: dict[str, tuple[Any, Any]], views: list[str],
                x: float, w: float, gap: float) -> None:
    """Columns of views; the top row is before and the bottom row after, sharing each crop."""
    width = (w - gap * (len(views) - 1)) / len(views)
    image_top, row_gap = BODY_TOP + 5.2, 3.0
    height = (BODY_BOTTOM - image_top - row_gap) / 2
    caption = canvas.font(5.8, "semibold", caps=True)
    for index, view in enumerate(views):
        left = x + index * (width + gap)
        canvas.text(left, BODY_TOP, width, 4, view, canvas.font(7.6, "semibold"), INK["title"])
        for row, (label, image) in enumerate(zip(("Before", "After"), images[view], strict=True)):
            top = image_top + row * (height + row_gap)
            canvas.image(image, left, top, width, height)
            canvas.text(left, top, width, 3.5, label, caption, INK["muted"])
        canvas.rule(left, image_top + height + row_gap / 2, left + width, INK["line"], 0.18)


@cache
def _register_inter() -> None:
    from PySide6.QtGui import QFontDatabase

    for name in ("Inter-Regular.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf"):
        if QFontDatabase.addApplicationFont(str(FONTS / name)) < 0:
            raise RuntimeError(f"Could not load the bundled font {name}.")


def write_report_pdf(path: str | Path, data: dict[str, Any], images: dict[str, tuple[Any, Any]]) -> Path:
    """Draw landscape A4 pages and replace ``path`` atomically."""
    from PySide6.QtCore import QMarginsF
    from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter

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
        writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Landscape,
                                         QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter))
        writer.setResolution(300)
        writer.setTitle(f"Surgical plan summary — {data['scenario']}")
        writer.setCreator(f"BIMAX {data['version']}")
        canvas = Canvas(writer)
        try:
            overview = [view for view in OVERVIEW_VIEWS if view in images]
            grid_width = 140.0 if overview else 0.0
            column_x = MARGIN + grid_width + (GUTTER if overview else 0)
            column_width = PAGE_W - MARGIN - column_x
            flow = _flow(canvas, _tables(canvas, data, column_width),
                         Region(0, column_x, BODY_TOP, column_width, BODY_BOTTOM))
            views = list(images)
            view_pages = [views[index:index + VIEWS_PER_PAGE] for index in range(0, len(views), VIEWS_PER_PAGE)]
            pages = len(flow) + len(view_pages)
            operations = " · ".join(data["operations"]) or "No osteotomies selected"
            hidden = f" Not shown: {', '.join(data['hidden'])}." if data["hidden"] else ""
            for number in range(1, pages + 1):
                if number > 1:
                    writer.newPage()
                canvas.background()
                if number == 1:
                    _header(canvas, data, "Surgical plan summary", data["scenario"], operations)
                    if overview:
                        _comparison(canvas, images, overview, MARGIN, grid_width, 5.0)
                elif number <= len(flow):
                    _header(canvas, data, "Surgical plan summary", "Measurements", operations)
                else:
                    group = view_pages[number - len(flow) - 1]
                    _header(canvas, data, "Before and after", " · ".join(group),
                            "Same camera, scale and crop before and after. Bone surfaces only." + hidden)
                    _comparison(canvas, images, group, MARGIN, PAGE_W - 2 * MARGIN, 7.0)
                for placement in flow[number - 1] if number <= len(flow) else []:
                    _draw_table(canvas, placement)
                _footer(canvas, data, number, pages)
        finally:
            canvas.painter.end()
        del canvas, writer
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

