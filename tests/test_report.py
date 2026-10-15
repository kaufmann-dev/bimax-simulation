"""Scenario report data completeness and paginated PDF output."""

import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from factories import confirm, document
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from bimax.commands import Commands
from bimax.engine import fragment_transforms
from bimax.history import History
from bimax.report import CaseInfo, export_report, report_data, write_report_pdf


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


def test_report_data_contains_every_parameter_needed_to_rebuild_the_scenario():
    doc = planned_document()
    data = report_data(doc, project_path="/cases/a.bimax", dirty=False,
                       case=CaseInfo(patient="Jane Doe", case_id="C-1"), hidden=["Fixed skull"])

    controls = {control["key"]: control for control in data["controls"]}
    assert list(controls) == ["linked", "lefort1", "mandible", "chin", "left-proximal", "right-proximal"]
    assert controls["linked"]["translation"] == [1, 3, -2]
    assert controls["linked"]["ui"] == {"right": 1, "advance": 3, "downgraft": 2}
    assert controls["linked"]["angles"] == [4, 0, 0]
    assert controls["linked"]["pivot"] == [10, 10, 10] and controls["linked"]["pivot_set"]
    assert controls["lefort1"]["parent"] == "Linked jaws"
    assert controls["chin"]["parent"] == "Distal mandible"
    assert controls["left-proximal"]["proximal"] and controls["left-proximal"]["angles"] == [0, 3, 0]
    assert controls["left-proximal"]["pivot"] == [4, 2, 18]
    assert not controls["right-proximal"]["pivot_set"]

    cuts = {cut["id"]: cut for cut in data["cuts"]}
    assert list(cuts) == ["lefort1", "bsso", "genio"]
    planes = {row["name"]: row for row in cuts["lefort1"]["geometry"]}
    assert planes["planes.upper"]["origin"] == doc.cuts["lefort1"].geometry["planes"]["upper"]["origin"]
    assert planes["groups"]["value"] == [["upper"]]
    bsso = {row["name"] for row in cuts["bsso"]["geometry"]}
    assert {"midline", "left.front", "left.upper", "left.sagittal", "right.front"} <= bsso
    assert {item["identity"] for item in data["confirmations"]} >= {"chin", "left-proximal", "upper:lefort1"}

    fragments = {fragment["label"]: fragment for fragment in data["fragments"]}
    np.testing.assert_allclose(fragments[6]["transform_lps"], fragment_transforms(doc)[6])
    assert fragments[1]["hidden"] and fragments[1]["distance_mm"] == 0
    assert sum(fragment["voxels"] for fragment in data["fragments"]) == doc.source.size

    landmark = data["landmarks"][0]
    assert landmark["name"] == "Pogonion" and landmark["status"] == "valid"
    measurement = data["measurements"][0]
    values = {value["key"]: value for value in measurement["values"]}
    assert values["signed_lateral_mm"]["change"] == pytest.approx(
        values["signed_lateral_mm"]["moved"] - values["signed_lateral_mm"]["baseline"]
    )
    assert data["symmetry_right"] == 10
    assert data["case"]["patient"] == "Jane Doe" and data["project"] == "/cases/a.bimax"


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
    assert [path.name for path in tmp_path.iterdir()] == ["report.pdf"]
