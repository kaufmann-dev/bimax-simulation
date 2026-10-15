"""Surgeon-facing scenario report content and paginated PDF output."""

import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from factories import confirm, document
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from bimax.commands import Commands
from bimax.engine import motion_matrix
from bimax.history import History
from bimax.model import Motion
from bimax.report import (
    CaseInfo,
    crop_pair,
    export_report,
    movement_terms,
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


def test_movement_terms_match_engine_rotation_signs():
    pivot = [0.0, 0.0, 0.0]
    anterior, right = np.array([0, 10, 0, 1.0]), np.array([10, 0, 0, 1.0])
    pitch = motion_matrix(Motion(angles=[5, 0, 0], pivot=pivot, pivot_set=True))
    roll = motion_matrix(Motion(angles=[0, 5, 0], pivot=pivot, pivot_set=True))
    yaw = motion_matrix(Motion(angles=[0, 0, 5], pivot=pivot, pivot_set=True))
    assert (pitch @ anterior)[2] > 0
    assert (roll @ right)[2] < 0
    assert (yaw @ anterior)[0] < 0
    assert movement_terms([0, 0, 0], [5, 5, 5]) == [
        ("Pitch", "5.0° counterclockwise (anterior up)"),
        ("Roll (cant)", "5.0° right side down"),
        ("Yaw", "5.0° front toward the patient's left"),
    ]
    assert movement_terms([-1, -2, 3], [0, 0, 0]) == [
        ("Setback", "2.0 mm back"),
        ("Impaction", "3.0 mm up"),
        ("Transverse shift", "1.0 mm to the patient's left"),
    ]


def test_report_data_describes_each_operation_segment_for_a_surgeon():
    doc = planned_document()
    data = report_data(doc, project_path="/cases/a.bimax", dirty=False,
                       case=CaseInfo(patient="Jane Doe", case_id="C-1"), hidden=["Fixed skull"])

    movements = {movement["key"]: movement for movement in data["movements"]}
    assert list(movements) == ["linked", "lefort1", "mandible", "chin", "left-proximal", "right-proximal"]
    assert movements["linked"]["terms"] == [
        ("Advancement", "3.0 mm forward"),
        ("Downgraft", "2.0 mm down"),
        ("Transverse shift", "1.0 mm to the patient's right"),
        ("Pitch", "4.0° counterclockwise (anterior up)"),
    ]
    assert movements["lefort1"]["name"] == "Maxilla (Le Fort I)"
    assert movements["lefort1"]["relative_to"] == "Both jaws (maxillomandibular complex)"
    assert movements["chin"]["relative_to"] == "Distal mandible (BSSO)"
    assert movements["chin"]["terms"] == [("Advancement", "2.0 mm forward"), ("Impaction", "1.0 mm up")]
    assert movements["left-proximal"]["terms"] == [("Roll (cant)", "3.0° right side down")]
    assert movements["right-proximal"]["terms"] == []
    assert data["operations"] == ["Le Fort I", "BSSO", "Genioplasty"]

    landmark = data["landmarks"][0]
    assert landmark["name"] == "Pogonion" and landmark["segment"] == "Chin"
    assert landmark["anterior"].endswith("forward")
    values = {value["label"]: value for value in data["measurements"][0]["values"]}
    offset = values["Offset from midline (+ right)"]
    assert offset["change"] == pytest.approx(offset["after"] - offset["before"])
    assert data["hidden"] == ["Fixed skull"] and data["project"] == "a.bimax"


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
    images = {view: (image, image) for view in ("Right", "Left", "Front", "Bird's-eye", "Inferior")}
    target = tmp_path / "report.pdf"
    target.write_bytes(b"old")

    assert write_report_pdf(target, data, images) == target

    content = target.read_bytes()
    assert content.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page(?!s)", content)) > 3
    assert b"Inter-Regular" in content and b"Inter-SemiBold" in content
    assert [path.name for path in tmp_path.iterdir()] == ["report.pdf"]
