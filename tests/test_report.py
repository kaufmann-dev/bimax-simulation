"""Surgeon-facing scenario report content and paginated PDF output."""

import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from factories import confirm, document
from PySide6.QtGui import QColor, QImage, QPdfWriter
from PySide6.QtWidgets import QApplication

from bimax.commands import Commands
from bimax.engine import motion_matrix
from bimax.history import History
from bimax.model import Motion
from bimax.report import (
    BODY_BOTTOM,
    BODY_TOP,
    Canvas,
    CaseInfo,
    Column,
    Region,
    Table,
    _flow,
    _tables,
    clinical_values,
    crop_pair,
    export_report,
    report_data,
    write_report_pdf,
)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def planned_document():
    doc = document()
    commands = Commands(doc, History())
    commands.execute("select_cuts", "Scenario 1", cut_ids=["lefort1", "bsso", "genio"])
    doc.fragments = confirm(doc)
    doc.fragments_key = doc.configuration_key
    commands.execute("set_symmetry_reference", point=[10, -4, 7])
    commands.execute("set_movements", "Scenario 1", targets={
        "linked": {"translation": [1, 3, -2], "rotation": {"pitch": 4}, "pivot": [10, 10, 10]},
        "lefort1": {"translation": [0, 2, 0]},
        "mandible": {"translation": [0, 4, 0]},
        "chin": {"translation": [0, 2, 1]},
    })
    commands.execute("set_proximal_rotations", "Scenario 1", targets={
        "left-proximal": {"rotation": {"roll": 3}, "pivot": [4, 2, 18]},
    })
    commands.execute("set_landmark", "Scenario 1", name="Pogonion", fragment="chin", point=[10, 4, 2])
    commands.execute("set_measurement", "Scenario 1", name="Chin midline", kind="midline",
                     landmarks={"point": "Pogonion"})
    return doc


def test_clinical_values_and_legends_match_engine_rotation_signs(qapp, tmp_path):
    pivot = [0.0, 0.0, 0.0]
    anterior, right = np.array([0, 10, 0, 1.0]), np.array([10, 0, 0, 1.0])
    assert (motion_matrix(Motion(angles=[5, 0, 0], pivot=pivot, pivot_set=True)) @ anterior)[2] > 0
    assert (motion_matrix(Motion(angles=[0, 5, 0], pivot=pivot, pivot_set=True)) @ right)[2] < 0
    assert (motion_matrix(Motion(angles=[0, 0, 5], pivot=pivot, pivot_set=True)) @ anterior)[0] < 0
    assert clinical_values([1, 2, 3], [4, 5, 6]) == [2, -3, 1, 4, 5, 6]

    canvas = Canvas(QPdfWriter(str(tmp_path / "legend.pdf")))
    try:
        data = report_data(planned_document(), project_path=None, dirty=False, case=CaseInfo())
        movement = _tables(canvas, data, 120)[0]
    finally:
        canvas.painter.end()
    assert [column.legend for column in movement.columns[1:]] == [
        "+ forward", "+ down", "+ right", "+ CCW", "+ R down", "+ front L"
    ]


def test_report_data_lists_each_operation_segment_in_inheritance_order():
    doc = planned_document()
    data = report_data(doc, project_path="/cases/a.bimax", dirty=False,
                       case=CaseInfo(patient="Jane Doe", case_id="C-1"), hidden=["Fixed skull"])

    movements = {movement["key"]: movement for movement in data["movements"]}
    assert list(movements) == ["linked", "lefort1", "mandible", "chin", "left-proximal", "right-proximal"]
    assert [movements[key]["level"] for key in movements] == [0, 1, 1, 2, 0, 0]
    assert movements["linked"]["values"] == [3, 2, 1, 4, 0, 0]
    assert movements["lefort1"]["name"] == "Maxilla" and movements["lefort1"]["detail"] == "Le Fort I"
    assert movements["chin"]["values"][:3] == [2, -1, 0]
    assert movements["left-proximal"]["values"] == [None, None, None, 0, 3, 0]
    assert data["operations"] == ["Le Fort I", "BSSO", "Genioplasty"]

    landmark = data["landmarks"][0]
    assert landmark["name"] == "Pogonion" and landmark["segment"] == "Chin"
    assert landmark["values"][0] > 0
    values = {value["label"]: value for value in data["measurements"][0]["values"]}
    assert list(values) == ["Offset from midline, + right"]
    offset = values["Offset from midline, + right"]
    assert offset["change"] == pytest.approx(offset["after"] - offset["before"])
    assert data["hidden"] == ["Fixed skull"] and data["project"] == "a.bimax"


def test_flow_moves_short_tables_whole_and_splits_long_ones(qapp, tmp_path):
    canvas = Canvas(QPdfWriter(str(tmp_path / "flow.pdf")))
    columns = [Column("Name", "", 1.0, "left")]
    try:
        filler = Table("Filler", columns, [["x"]] * 26)
        short = Table("Short", columns, [["y"]] * 6)
        long = Table("Long", columns, [["z"]] * 80)
        pages = _flow(canvas, [filler, short, long], Region(0, 150, BODY_TOP, 130, BODY_BOTTOM))
    finally:
        canvas.painter.end()

    short_parts = [placement for page in pages for placement in page if placement.table is short]
    assert len(short_parts) == 1 and short_parts[0].rows == range(6)
    long_parts = [placement for page in pages for placement in page if placement.table is long]
    assert len(long_parts) > 1 and long_parts[1].continued
    assert sum(len(placement.rows) for placement in long_parts) == 80
    assert [len(placement.rows) for placement in long_parts if not placement.continued] == [long_parts[0].rows.stop]


def test_crop_pair_uses_one_shared_content_box(qapp):
    before = QImage(100, 80, QImage.Format.Format_RGB32)
    before.fill(QColor("black"))
    after = QImage(before)
    before.setPixelColor(20, 30, QColor("white"))
    after.setPixelColor(70, 50, QColor("white"))

    cropped_before, cropped_after = crop_pair(before, after, padding=0)

    assert cropped_before.size() == cropped_after.size()
    assert (cropped_before.width(), cropped_before.height()) == (51, 21)


def test_export_report_requires_an_accepted_combination(tmp_path):
    doc = planned_document()
    doc.reviews.clear()

    class Views:
        fragment_mode = True
        actors: dict = {}

    with pytest.raises(ValueError, match="accept the active cut combination"):
        export_report(Views(), doc, tmp_path / "report.pdf", project_path=None, dirty=True,
                      case=CaseInfo())
    assert not (tmp_path / "report.pdf").exists()


def test_write_report_pdf_paginates_and_replaces_atomically(qapp, tmp_path):
    doc = planned_document()
    data = report_data(doc, project_path=None, dirty=True, case=CaseInfo(notes="First\nSecond"))
    image = QImage(64, 64, QImage.Format.Format_RGB32)
    image.fill(QColor("#4080c0"))
    images = {view: (image, image) for view in ("Right", "Front", "Left", "Front-right oblique", "Inferior")}
    target = tmp_path / "report.pdf"
    target.write_bytes(b"old")

    assert write_report_pdf(target, data, images) == target

    content = target.read_bytes()
    assert content.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page(?!s)", content)) == 3  # Overview and two view pages.
    assert b"Inter-Regular" in content and b"Inter-SemiBold" in content
    assert [path.name for path in tmp_path.iterdir()] == ["report.pdf"]
