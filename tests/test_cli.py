"""bimax-cli: JSON in, validated edits saved in place, offscreen previews and exports."""

import json
import os

import pytest
from factories import document
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from test_aligners import landmarks

import bimax.scene as scene_module
from bimax import cli
from bimax.storage import file_version, load_project, save_project

SCENARIO = "Scenario 1"


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "case.bimax"
    save_project(document(), path)
    return path


@pytest.fixture
def run(capsys):
    def invoke(*args):
        code = cli.main([str(arg) for arg in args])
        return code, json.loads(capsys.readouterr().out)

    return invoke


def test_edits_are_validated_and_saved_in_place_without_backups(project, run):
    code, state = run("inspect", project)
    assert code == 0 and list(state["scenarios"]) == [SCENARIO] and len(state["cuts"]) == 7
    assert run("select-cuts", project, "--scenario", SCENARIO, "lefort1")[1]["saved"]
    code, result = run("set-movement", project, "--scenario", SCENARIO, "--target", "lefort1",
                       "--translation", 0, 3, 2)
    assert code == 0 and result["saved"]
    assert run("create-scenario", project, "Copy", "--from", SCENARIO)[1]["saved"]
    doc = load_project(project)
    assert doc.scenarios["Copy"].motions["lefort1"].translation == [0, 3, 2]
    assert doc.active_scenario == "Copy"
    assert run("select-scenario", project, SCENARIO)[1]["saved"]
    assert load_project(project).active_scenario == SCENARIO
    assert not list(project.parent.glob("*.backup-*"))
    version = file_version(project)
    code, error = run("set-movement", project, "--scenario", SCENARIO, "--target", "linked", "--pitch", 5)
    assert code == 1 and "pivot" in error["error"]
    code, error = run("set-movement", project, "--scenario", "Nope", "--target", "linked", "--translation", 1, 0, 0)
    assert code == 1 and "Unknown scenario" in error["error"]
    assert file_version(project) == version, "failed edits leave the file untouched"


def test_pick_confirm_accept_and_export(qapp, project, run, tmp_path, monkeypatch):
    monkeypatch.setattr(scene_module, "EXPORT_IMAGE_SIZE", 64)
    monkeypatch.setattr(scene_module, "REPORT_IMAGE_SIZE", 64)
    run("select-cuts", project, "--scenario", SCENARIO, "lefort1")
    code, prepared = run("prepare", project, "--scenario", SCENARIO)
    assert code == 0 and not prepared["accepted"] and not prepared["empty_required"]
    assert prepared["missing_confirmations"]
    heights = {"upper:lefort1": 10, "mandible": 4, "skull": 15}
    for fragment in prepared["missing_confirmations"]:
        code, picked = run("pick", project, "--scenario", SCENARIO, "--view", "axial",
                           "--position", heights[fragment], "--x", 0.5, "--y", 0.5, "--size", 128)
        assert code == 0 and picked["fragment"] == fragment
        assert run("confirm-fragment", project, "--scenario", SCENARIO, "--fragment", fragment,
                   "--point", *picked["point"])[1]["saved"]
    code, accepted = run("accept", project, "--scenario", SCENARIO)
    assert code == 0 and accepted["saved"]
    assert run("inspect", project)[1]["scenarios"][SCENARIO]["accepted"]
    run("set-movement", project, "--scenario", SCENARIO, "--target", "lefort1", "--translation", 0, 3, 2)

    code, preview = run("preview", project, "--scenario", SCENARIO, "--out", tmp_path / "p.png", "--size", 128)
    assert code == 0 and preview["review_state"] == "accepted" and QImage(preview["image"]).width() == 128
    code, error = run("preview", project, "--scenario", SCENARIO, "--view", "axial", "--out", tmp_path / "a.png")
    assert code == 1 and "position" in error["error"]
    code, images = run("export-images", project, "--scenario", SCENARIO, "--out", tmp_path / "images",
                       "--hide", "skull")
    assert code == 0 and len(images["images"]) == 5
    code, report = run("export-report", project, "--scenario", SCENARIO, "--out", tmp_path / "plan",
                       "--case-id", "C-1")
    assert code == 0 and open(report["report"], "rb").read(4) == b"%PDF"
    code, labelmaps = run("export-labelmaps", project, "--scenario", SCENARIO, "--out", tmp_path / "labels")
    assert code == 0 and os.path.isfile(labelmaps["manifest"])
    source = tmp_path / "landmarks.json"
    source.write_text(json.dumps(landmarks()))
    code, movements = run("export-aligners", project, "--scenario", SCENARIO, "--landmarks", source,
                          "--out", tmp_path / "movements.toml")
    assert code == 0
    assert movements["tables"]["maxilla"]["incisor_midline"] == {"forward_mm": 3.0, "up_mm": 2.0, "left_mm": 0.0}


def test_batch_applies_all_edits_with_one_save_or_none(project, run, tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps([
        {"operation": "select_cuts", "scenario": SCENARIO, "cut_ids": ["lefort1", "genio"]},
        {"operation": "set_movements", "scenario": SCENARIO,
         "targets": {"mandible": {"translation": [0, -4, 0]}, "chin": {"translation": [0, 2, 0]}}},
        {"operation": "set_symmetry_reference", "right": 10.0},
    ]))
    code, result = run("batch", project, good)
    assert code == 0 and result["saved"] and len(result["results"]) == 3
    doc = load_project(project)
    assert doc.scenarios[SCENARIO].motions["chin"].translation == [0, 2, 0]
    assert doc.symmetry_right == 10.0
    version = file_version(project)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"operation": "set_symmetry_reference", "right": 3.0},
        {"operation": "set_movements", "scenario": SCENARIO, "targets": {"nose": {}}},
    ]))
    code, error = run("batch", project, bad)
    assert code == 1 and "Operation 1" in error["error"] and "Nothing was saved" in error["error"]
    assert file_version(project) == version
