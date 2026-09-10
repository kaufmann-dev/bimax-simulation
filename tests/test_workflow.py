"""Targeted native Qt/VTK workflow and subprocess lifecycle checks."""

import os
import time
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import pytest
import SimpleITK as sitk
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

from bimax.app import MainWindow
from bimax.engine import default_cuts
from bimax.history import Edit, History
from bimax.jobs import Jobs
from bimax.model import Document, Geometry, new_scenario
from bimax.storage import load_project, save_project
from bimax.surfaces import make_surfaces


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        pytest.skip("Native workflow verification needs a Linux DISPLAY.")
    yield QApplication.instance() or QApplication([])


def wait_until(qapp, predicate, timeout=20):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        qapp.processEvents()
        QTest.qWait(10)
    qapp.processEvents()
    assert predicate(), "Native workflow did not complete before timeout"


@pytest.fixture
def window(qapp, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: errors.append(text))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    result = MainWindow()
    result._test_errors = errors
    result.show()
    result.views.initialize()
    qapp.processEvents()
    yield result
    result.jobs.close()
    wait_until(qapp, lambda: not result.jobs.busy)
    result.dirty = False
    result.views.close()
    result.close()
    qapp.processEvents()


def synthetic_document(assigned=False, fragments=False):
    geometry = Geometry((26, 26, 26), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), tuple(np.eye(3).ravel()))
    source = np.zeros(geometry.shape, dtype=np.uint8)
    source[3:12, 3:21, 3:21] = 1
    source[14:23, 3:21, 3:21] = 1
    anatomy = source.copy()
    if assigned:
        anatomy[3:12, 3:21, 3:21] = 3
        anatomy[14:23, 3:21, 3:21] = 2
    doc = Document(
        geometry,
        source,
        anatomy,
        np.zeros_like(source),
        np.zeros_like(source),
        source_name="synthetic-bone.nii.gz",
    )
    doc.cuts = default_cuts(doc)
    if fragments:
        doc.fragments = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[anatomy]
    return doc


def install_document(window, doc):
    labels = doc.fragments if doc.fragments is not None else doc.anatomy
    window._job_finished(
        "open",
        -1,
        {
            "document": doc,
            "surfaces": make_surfaces(labels, doc.geometry),
            "fragment_mode": doc.fragments is not None,
        },
    )


def write_input(doc, path):
    image = sitk.GetImageFromArray(doc.source)
    image.SetSpacing(doc.geometry.spacing)
    image.SetOrigin(doc.geometry.origin)
    image.SetDirection(doc.geometry.direction)
    sitk.WriteImage(image, str(path))


def test_native_import_separate_cut_move_save_reopen_export(window, qapp, tmp_path, monkeypatch):
    source_doc = synthetic_document()
    source = tmp_path / "bone.nii.gz"
    write_input(source_doc, source)
    window.load_path(source)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window._test_errors == []
    assert window.tabs.currentIndex() == 1
    assert set(window.views.actors) == {1}
    np.testing.assert_array_equal(window.doc.source, source_doc.source)
    assert not window.doc.source.flags.writeable

    window.tool_combo.setCurrentIndex(window.tool_combo.findData("seed"))
    for label, xyz in ((2, [10, 10, 18]), (3, [10, 10, 7])):
        window.label_combo.setCurrentIndex(window.label_combo.findData(label))
        point = window.doc.geometry.index_to_world(xyz)
        window.views.set_crosshair(point)
        qapp.processEvents()
        view = window.views.slices[0]
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=view.world_to_screen(point).toPoint())
        assert np.any(window.doc.seeds == label)
    window.preview_separation_button.click()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window.pending_separation is not None
    assert window.accept_separation_button.isEnabled()
    window.accept_separation_button.click()
    assert not np.any(window.doc.anatomy == 1)
    assert window.doc.fragments is None

    window.tabs.setCurrentIndex(2)
    for operation in ("lefort", "bsso", "genio"):
        window.operation_combo.setCurrentIndex(window.operation_combo.findData(operation))
        window.cut_enabled.setChecked(True)
        if operation == "lefort":
            window.plane_origin[2].setValue(18)
            window.plane_origin[2].editingFinished.emit()
    window.cut_preview_button.click()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window._test_errors == []
    assert set(np.unique(window.pending_fragments)) == set(range(7))
    assert window.doc.fragments is None
    assert not window.export_button.isEnabled()
    # Acceptance runs validation in the actual computation process.
    window.accept_cuts_button.click()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window.doc.fragments is None
    assert "confirm" in window._test_errors[-1].lower()
    window._test_errors.clear()
    for label in (2, 3, 4, 5, 6):
        positions = np.argwhere(window.pending_fragments == label)
        xyz = positions[len(positions) // 2][::-1]
        point = window.doc.geometry.index_to_world(xyz)
        window.confirm_combo.setCurrentIndex(window.confirm_combo.findData(label))
        window._arm_pick("confirm")
        window.views.pointPicked.emit(point.tolist())
        assert str(label) in window.doc.confirmations
    window.accept_cuts_button.click()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window._test_errors == []
    assert window.tabs.currentIndex() == 3
    assert window.export_button.isEnabled()
    assert not window.motion_spins[3].isEnabled()

    window._arm_pick("pivot")
    window.views.pointPicked.emit([12.0, 12.0, 12.0])
    assert window.motion_spins[3].isEnabled()
    window.motion_spins[1].setValue(6)
    window.motion_spins[2].setValue(2)
    window.motion_spins[3].setValue(8)
    window.motion_spins[3].editingFinished.emit()
    assert window.doc.motions["linked"].translation == [0, 6, -2]
    assert window.doc.motions["linked"].angles == [8, 0, 0]
    assert window.views.actors[2].GetUserMatrix().GetElement(1, 3) != 0

    project = tmp_path / "simulation.bimax"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(project), ""))
    window.save_project()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window._test_errors == []
    assert not window.dirty
    assert window.project_path == str(project)
    expected = deepcopy(window.doc.scenarios)
    window.load_path(project)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window.doc.scenarios == expected
    assert not window.doc.source.flags.writeable
    assert window.tabs.currentIndex() == 3
    assert window.doc.fragments is not None
    assert window.motion_spins[3].value() == 8
    assert load_project(project).scenarios == expected
    np.testing.assert_array_equal(window.doc.source, source_doc.source)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("labelmaps", True))
    window._export()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window._test_errors == []
    assert (tmp_path / "labelmaps" / "manifest.json").is_file()
    assert len(list((tmp_path / "labelmaps").glob("*.nii.gz"))) == 6


def test_movement_undo_redo_retains_accepted_fragments(window, qapp):
    doc = synthetic_document(assigned=True, fragments=True)
    install_document(window, doc)
    accepted = doc.fragments.copy()
    window.motion_spins[1].setValue(7)
    window.motion_spins[1].editingFinished.emit()
    window._undo(False)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert doc.motions["linked"].translation == [0, 0, 0]
    np.testing.assert_array_equal(doc.fragments, accepted)
    assert window.export_button.isEnabled()
    window._undo(True)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert doc.motions["linked"].translation == [0, 7, 0]
    np.testing.assert_array_equal(doc.fragments, accepted)
    assert window._test_errors == []


def test_reset_pivot_preserves_translation_other_controls_and_saved_state(window, qapp, tmp_path):
    doc = synthetic_document(assigned=True, fragments=True)
    selected = doc.motions["mandible"]
    selected.translation = [2.0, 7.0, -3.0]
    selected.angles = [12.0, 4.0, -2.0]
    selected.pivot, selected.pivot_set = [-12.0, -10.0, 9.0], True
    other = doc.motions["maxilla"]
    other.translation, other.angles = [1.0, 4.0, 0.0], [3.0, 0.0, 0.0]
    other.pivot, other.pivot_set = [-9.0, -8.0, 15.0], True
    doc.scenarios["Alternative"] = deepcopy(doc.motions)
    before = deepcopy(doc.scenarios)
    accepted = doc.fragments.copy()
    install_document(window, doc)
    window.motion_combo.setCurrentIndex(window.motion_combo.findData("mandible"))
    window._update_transforms()
    assert window.views.pivot_actor.GetVisibility()
    assert window.reset_pivot_button.isEnabled()

    window.reset_pivot_button.click()
    selected = doc.motions["mandible"]
    assert selected.translation == [2.0, 7.0, -3.0]
    assert selected.angles == selected.pivot == [0.0, 0.0, 0.0]
    assert not selected.pivot_set
    assert doc.scenarios["Alternative"] == before["Alternative"]
    for key in ("linked", "maxilla", "chin"):
        assert doc.motions[key] == before["Scenario 1"][key]
    np.testing.assert_array_equal(doc.fragments, accepted)
    assert window.dirty
    assert [edit.name for edit in window.history.undo_stack] == ["Reset pivot"]
    assert not window.reset_pivot_button.isEnabled()
    assert not window.views.pivot_actor.GetVisibility()
    assert all(view._pivot is None for view in window.views.slices)
    assert not any(spin.isEnabled() for spin in window.motion_spins[3:])
    assert [spin.value() for spin in window.motion_spins] == [2, 7, 3, 0, 0, 0]
    matrix = window.views.actors[5].GetUserMatrix()
    for row in range(3):
        assert matrix.GetElement(row, 3) == [-2.0, -7.0, -3.0][row]
        for column in range(3):
            assert matrix.GetElement(row, column) == float(row == column)
    after = deepcopy(doc.scenarios)

    project = tmp_path / "reset-pivot.bimax"
    save_project(doc, project)
    saved = load_project(project)
    assert saved.scenarios == after
    np.testing.assert_array_equal(saved.fragments, accepted)

    window._undo(False)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert doc.scenarios == before
    assert window.views.pivot_actor.GetVisibility()
    assert all(spin.isEnabled() for spin in window.motion_spins[3:])
    assert window.reset_pivot_button.isEnabled()
    window._undo(True)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert doc.scenarios == after
    assert not window.views.pivot_actor.GetVisibility()
    assert window._test_errors == []


def test_reset_pivot_finishes_numeric_edit_and_cancels_pending_pick(window, qapp):
    doc = synthetic_document(assigned=True, fragments=True)
    motion = doc.motions["linked"]
    motion.pivot, motion.pivot_set = [-12.0, -12.0, 12.0], True
    install_document(window, doc)
    window.motion_spins[1].setValue(8.5)
    window.motion_spins[3].setValue(11.0)
    assert window._motion_before is not None
    edited = deepcopy(doc.scenarios)
    window._arm_pick("pivot")
    window.reset_pivot_button.click()
    assert window._motion_before is None
    assert [edit.name for edit in window.history.undo_stack] == ["Move fragment", "Reset pivot"]
    assert window.pick_mode is None
    assert window.views.tool == "navigate"
    assert all(view.tool == "navigate" for view in window.views.slices)
    window.views.pointPicked.emit([10.0, 11.0, 12.0])
    assert not doc.motions["linked"].pivot_set
    assert doc.motions["linked"].translation == [0.0, 8.5, 0.0]

    window._undo(False)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert doc.scenarios == edited
    window._undo(False)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert doc.motions["linked"].translation == [0.0, 0.0, 0.0]
    assert doc.motions["linked"].angles == [0.0, 0.0, 0.0]
    assert doc.motions["linked"].pivot_set
    assert window._test_errors == []


def test_reset_pivot_without_pivot_is_noop_but_can_cancel_armed_pick(window):
    assert not window.reset_pivot_button.isEnabled()
    window._reset_pivot()
    assert not window.history.undo_stack
    assert not window.dirty

    doc = synthetic_document(assigned=True, fragments=True)
    install_document(window, doc)
    revision = doc.revision
    assert not window.reset_pivot_button.isEnabled()
    window._reset_pivot()
    window._arm_pick("pivot")
    assert window.reset_pivot_button.isEnabled()
    window.reset_pivot_button.click()
    assert window.pick_mode is None
    assert window.views.tool == "navigate"
    assert not window.reset_pivot_button.isEnabled()
    assert not window.history.undo_stack
    assert not window.dirty
    assert doc.revision == revision


def test_stale_worker_result_does_not_change_document_or_preview(window):
    doc = synthetic_document(assigned=True)
    install_document(window, doc)
    before = doc.anatomy.copy()
    stale_revision = doc.revision
    doc.revision += 1
    proposal = doc.source.copy()
    window.jobs.finished.emit("separate", stale_revision, {"proposal": proposal, "surfaces": {}})
    assert window.pending_separation is None
    np.testing.assert_array_equal(doc.anatomy, before)
    assert "outdated" in window.statusBar().currentMessage()


@pytest.mark.parametrize("stage", [1, 2])
def test_editing_stages_display_baseline_geometry_after_movement(window, stage):
    doc = synthetic_document(assigned=True, fragments=True)
    install_document(window, doc)
    window.motion_spins[1].setValue(7)
    window.motion_spins[1].editingFinished.emit()
    assert window.views.actors[5].GetUserMatrix().GetElement(1, 3) == -7
    window.tabs.setCurrentIndex(stage)
    if stage == 1:
        assert all(not view.fragment_mode for view in window.views.slices)
        for view in window.views.slices:
            np.testing.assert_array_equal(view.labels, doc.anatomy)
    for actor in window.views.actors.values():
        for row in range(4):
            for column in range(4):
                assert actor.GetUserMatrix().GetElement(row, column) == float(row == column)
    assert doc.motions["linked"].translation == [0, 7, 0]
    window.tabs.setCurrentIndex(3)
    assert window.views.actors[5].GetUserMatrix().GetElement(1, 3) == -7


def test_array_undo_invalidates_fragments_but_preserves_source(window, qapp):
    doc = synthetic_document(assigned=True, fragments=True)
    install_document(window, doc)
    source = doc.source.copy()
    indices = np.array([np.ravel_multi_index((5, 8, 8), doc.source.shape)])
    window._apply_array_edit("Exclude selected bone", "anatomy", indices, 4)
    assert doc.fragments is None
    assert doc.anatomy[5, 8, 8] == 4
    window._undo(False)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert doc.anatomy[5, 8, 8] == 3
    assert doc.fragments is None
    np.testing.assert_array_equal(doc.source, source)


@pytest.mark.parametrize("surface_pick", [False, True])
def test_boundary_confirmation_snaps_only_for_3d_surface_pick(window, surface_pick):
    doc = synthetic_document(assigned=True, fragments=True)
    install_document(window, doc)
    window.tabs.setCurrentIndex(2)
    window.pending_fragments = doc.fragments.copy()
    window.confirm_combo.setCurrentIndex(window.confirm_combo.findData(5))
    window._arm_pick("confirm")
    # Surface picking may return a point just outside the x=2.5 bone boundary.
    point = np.array([2.49, 8.0, 5.0])
    if surface_pick:
        window.views.last_pick_fragment = 5
        window.views.pointPicked.emit(point.tolist())
        assert "5" in doc.confirmations
        voxel_center = doc.plan_to_world(doc.confirmations["5"])
        np.testing.assert_allclose(voxel_center, [3, 8, 5])
    else:
        window.views._point_picked(point.tolist())
        assert doc.confirmations == {}


@pytest.mark.parametrize("surface_pick", [False, True])
def test_boundary_landmark_preserves_surface_point_without_assigning_background(
    window, surface_pick
):
    doc = synthetic_document(assigned=True, fragments=True)
    install_document(window, doc)
    window._arm_pick("landmark")
    point = np.array([2.49, 8.0, 5.0])
    if surface_pick:
        # Actor provenance takes precedence over a previously selected fragment.
        window.selected_fragment = 1
        window.views.last_pick_fragment = 5
        window.views.pointPicked.emit(point.tolist())
        assert len(doc.landmarks) == 1
        assert doc.landmarks[0]["fragment"] == 5
        np.testing.assert_allclose(doc.plan_to_world(doc.landmarks[0]["point"]), point)
    else:
        window.views._point_picked(point.tolist())
        assert doc.landmarks == []


def test_surface_confirmation_does_not_snap_to_a_different_fragment(window):
    doc = synthetic_document(assigned=True, fragments=True)
    install_document(window, doc)
    window.tabs.setCurrentIndex(2)
    window.pending_fragments = doc.fragments.copy()
    window.confirm_combo.setCurrentIndex(window.confirm_combo.findData(5))
    window._arm_pick("confirm")
    window.views.last_pick_fragment = 1
    window.views.pointPicked.emit([2.49, 8.0, 5.0])
    assert doc.confirmations == {}


def test_frame_undo_and_duplicate_scenario_remain_valid(window, qapp, monkeypatch):
    doc = synthetic_document(assigned=True, fragments=True)
    install_document(window, doc)
    original_frame = doc.frame.copy()
    window.frame_spins[0].setValue(12)
    window.frame_spins[0].editingFinished.emit()
    assert not np.array_equal(doc.frame, original_frame)
    window._undo(False)
    wait_until(qapp, lambda: not window.jobs.busy)
    np.testing.assert_array_equal(doc.frame, original_frame)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("Alternative", True))
    window._duplicate_scenario()
    assert doc.active_scenario == "Alternative"
    window._undo(False)
    wait_until(qapp, lambda: not window.jobs.busy)
    assert doc.active_scenario in doc.scenarios
    assert list(doc.scenarios) == ["Scenario 1"]
    assert window._test_errors == []


def test_history_scenario_snapshot_with_array_frame():
    doc = synthetic_document(assigned=True)
    history = History()
    before = {
        "frame": doc.frame.copy(),
        "scenarios": deepcopy(doc.scenarios),
        "active_scenario": doc.active_scenario,
    }
    doc.frame = np.eye(4)
    doc.scenarios["Alternative"] = new_scenario()
    doc.active_scenario = "Alternative"
    after = {
        "frame": doc.frame.copy(),
        "scenarios": deepcopy(doc.scenarios),
        "active_scenario": doc.active_scenario,
    }
    history.push(Edit("Change frame", [], before, after))
    assert history.apply(doc, False)
    np.testing.assert_array_equal(doc.frame, before["frame"])
    assert doc.active_scenario == "Scenario 1"
    assert history.apply(doc, True)
    assert doc.active_scenario == "Alternative"


def test_worker_cancel_running_import_emits_no_result(qapp, tmp_path):
    source = tmp_path / "input.nii.gz"
    write_input(synthetic_document(), source)
    jobs = Jobs()
    results: list[tuple] = []
    failures: list[str] = []
    statuses: list[str] = []
    jobs.finished.connect(lambda *args: results.append(args))
    jobs.failed.connect(failures.append)
    jobs.progress.connect(statuses.append)
    jobs.submit("import", path=str(source))
    with pytest.raises(RuntimeError, match="already running"):
        jobs.submit("import", path=str(source))
    wait_until(qapp, lambda: jobs._process is not None)
    process = jobs._process
    jobs.cancel()
    wait_until(qapp, lambda: not jobs.busy)
    assert results == []
    assert failures == []
    assert any("cancelled" in message.lower() for message in statuses)
    assert process.poll() is not None
    jobs.close()


def test_cancelling_close_save_does_not_close_on_later_save(window, qapp, tmp_path):
    doc = synthetic_document(assigned=True)
    install_document(window, doc)
    window.project_path = str(tmp_path / "working.bimax")
    closed = []
    window._after_save = lambda: closed.append(True)
    window.save_project()
    # Cancel preparation synchronously: a tiny save may finish before a GUI
    # event-loop poll can observe its short-lived subprocess.
    window.cancel_button.click()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window._after_save is None
    assert closed == []
    window.save_project()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window._test_errors == []
    assert closed == []
    assert not window.dirty


def test_failed_open_preserves_previous_project_save_target(window, qapp, tmp_path):
    doc = synthetic_document(assigned=True)
    install_document(window, doc)
    previous = str(tmp_path / "working.bimax")
    window.project_path = previous
    window.load_path(tmp_path / "missing.bimax")
    wait_until(qapp, lambda: not window.jobs.busy)
    assert window._test_errors
    assert window.doc is doc
    assert window.project_path == previous
