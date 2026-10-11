"""Native Qt sessions and SDK-driven stdio access to the real local bridge."""

import asyncio
import base64
import json
import os
import sys
import tempfile
import threading
import time

import numpy as np
import pytest
from factories import upper_geometry
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QMessageBox
from test_workflow import install_document, synthetic_document, wait_until

from bimax.control import ControlError
from bimax.mcp_server import Connection
from bimax.protocol import runtime_directory
from bimax.storage import load_project


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        pytest.skip("Native MCP preview checks require DISPLAY.")
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from bimax.app import MainWindow

    runtime = tempfile.TemporaryDirectory(prefix="bm-")
    monkeypatch.setenv("XDG_RUNTIME_DIR", runtime.name)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: errors.append(text))
    result = MainWindow()
    result._test_errors = errors
    result.show()
    result.views.initialize()
    qapp.processEvents()
    install_document(result, synthetic_document(assigned=True, fragments=True))
    yield result
    result.ai_control.release()
    result.jobs.close()
    wait_until(qapp, lambda: not result.jobs.busy)
    result.dirty = False
    result.views.close()
    result.close()
    qapp.processEvents()
    runtime.cleanup()


def guard(control):
    state = control.state()
    return {key: state[key] for key in ("session_id", "document_id", "revision")}


def dispatch(window, peer, tool, **arguments):
    return window.ai_control.dispatch(peer, tool, {**guard(window.ai_control), **arguments})


def test_session_locks_mutations_but_keeps_navigation_and_take_control(window, qapp):
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    assert not window.tabs.isEnabled()
    assert not window.scenario_combo.isEnabled()
    assert window.structure_box.isEnabled()
    assert all(view.isEnabled() for view in window.views.slices)
    assert window.toolbar_actions["Right profile"].isEnabled()
    assert not window.toolbar_actions["Save"].isEnabled()
    revision = window.doc.revision
    window._arm_pick("pivot")
    window._undo(False)
    assert window.pick_mode is None
    assert window.doc.revision == revision
    dispatch(
        window,
        peer,
        "set_movements",
        scenario="Scenario 1",
        targets={"linked": {"translation": [0, 5, 0]}},
    )
    assert window.views.actors[5].GetUserMatrix().GetElement(1, 3) == -5
    assert "Scenario 1" in window.ai_banner.text()
    old_guard = guard(control)
    window.take_control_button.click()
    assert control.session is None
    assert window.tabs.isEnabled()
    assert window.doc.motions["linked"].translation == [0, 5, 0]
    with pytest.raises(ControlError, match="revoked"):
        control.dispatch(peer, "undo", old_guard)
    assert window.history.apply(window.doc, False)
    assert window.doc.motions["linked"].translation == [0, 0, 0]


def test_stale_revision_unsaved_open_and_inactivity(window, tmp_path):
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    old = guard(control)
    dispatch(window, peer, "set_orientation", rotation={"pitch": 3, "roll": 0, "yaw": 0})
    with pytest.raises(ControlError, match="revision changed"):
        control.dispatch(peer, "undo", old)
    with pytest.raises(ControlError, match="Save the current"):
        dispatch(window, peer, "open_project", path=str(tmp_path / "other.bimax"))
    control.last_request = time.monotonic() - 301
    control.expire()
    assert control.session is None
    assert window.dirty
    assert window.doc.view_angles == [3, 0, 0]


def test_preview_png_picking_and_reuse_for_movement_edit(window, qapp, monkeypatch):
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    fragments = window.doc.fragments

    def unexpected(*args, **kwargs):
        pytest.fail("Movement-only preview must reuse prepared geometry")

    monkeypatch.setattr(window.jobs, "submit", unexpected)
    result = dispatch(window, peer, "preview", scenario="Scenario 1")
    job_id = result["job_id"]
    assert control.jobs[job_id]["status"] == "running"
    with pytest.raises(ControlError, match="job is active"):
        dispatch(window, peer, "undo")
    wait_until(qapp, lambda: control.active_job is None)
    job = control.jobs[job_id]
    assert job["status"] == "completed", job.get("error", job)
    image = QImage.fromData(base64.b64decode(job["image"]))
    assert (image.width(), image.height()) == (1024, 1024)
    preview_id = job["metadata"]["preview_id"]
    capture = control.previews[preview_id][1]
    picked = None
    for x in np.linspace(0.3, 0.7, 9):
        for y in np.linspace(0.3, 0.7, 9):
            try:
                picked = capture.pick("right", float(x), float(y))
                break
            except ValueError:
                pass
        if picked:
            break
    assert picked and len(picked["point"]) == 3
    assert picked["fragment"] in window.doc.fragment_labels
    dispatch(
        window,
        peer,
        "set_movements",
        scenario="Scenario 1",
        targets={"mandible": {"translation": [0, 4, 0]}},
    )
    assert window.doc.fragments is fragments
    with pytest.raises(ControlError, match="older"):
        control.dispatch(
            peer,
            "pick_image_point",
            {"preview_id": preview_id, "view": "right", "x": 0.5, "y": 0.5},
        )
    result = dispatch(window, peer, "preview", scenario="Scenario 1", view="axial", position=5)
    wait_until(qapp, lambda: control.active_job is None)
    job = control.jobs[result["job_id"]]
    assert job["status"] == "completed", job.get("error", job)
    capture = control.previews[job["metadata"]["preview_id"]][1]
    labels, points = capture.slices["axial"]
    row, col = np.argwhere(labels == 5)[0]
    picked = capture.pick("axial", (col + 0.5) / 1024, (row + 0.5) / 1024)
    np.testing.assert_allclose(picked["point"], window.doc.world_to_plan(points[row, col]))
    assert picked["fragment"] == "mandible"
    assert window._test_errors == []


def test_provisional_preview_acceptance_and_cancel_discard_late_result(window, qapp):
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    dispatch(window, peer, "replace_cut", cut_id="lefort1", geometry=upper_geometry(18))
    dispatch(window, peer, "select_cuts", scenario="Scenario 1", cut_ids=["lefort1"])
    dispatch(
        window,
        peer,
        "set_movements",
        scenario="Scenario 1",
        targets={"lefort1": {"translation": [0, 3, 0]}},
    )
    result = dispatch(window, peer, "preview", scenario="Scenario 1", view="right")
    wait_until(qapp, lambda: control.active_job is None)
    job = control.jobs[result["job_id"]]
    assert job["status"] == "completed", job.get("error", job)
    assert job["metadata"]["review_state"] == "unreviewed"
    assert not window.doc.cuts_accepted
    assert window.doc.fragments is not None
    with pytest.raises(ValueError, match="confirmation"):
        dispatch(window, peer, "accept_combination", scenario="Scenario 1")
    label = window.doc.fragment_labels["upper:lefort1"]
    voxel = np.argwhere(window.doc.fragments == label)[0][::-1]
    point = window.doc.world_to_plan(window.doc.geometry.index_to_world(voxel)).tolist()
    dispatch(
        window,
        peer,
        "confirm_fragment",
        scenario="Scenario 1",
        fragment="upper:lefort1",
        point=point,
    )
    dispatch(window, peer, "accept_combination", scenario="Scenario 1")
    assert window.doc.cuts_accepted
    dispatch(window, peer, "select_cuts", scenario="Scenario 1", cut_ids=["bsso"])
    result = dispatch(window, peer, "preview", scenario="Scenario 1", view="right")
    active = control.jobs[result["job_id"]]
    old_doc, old_session = window.doc, control.session
    control.release()
    wait_until(qapp, lambda: not window.jobs.busy)
    assert active["status"] == "cancelled"
    window._job_finished(
        "cuts_preview",
        old_doc.revision,
        {
            "_control_session": old_session,
            "_control_job": result["job_id"],
            "fragments": np.ones_like(old_doc.source),
        },
    )
    assert window.doc.fragments is None
    assert "revoked" in window.statusBar().currentMessage()


def test_sdk_stdio_typed_tools_png_save_reopen_and_disconnect(window, qapp, tmp_path):
    path = tmp_path / "sdk.bimax"
    instance = window.ai_control.instance_id
    results, failures = {}, []

    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "bimax.mcp_server"], env=dict(os.environ)
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools = {tool.name: tool for tool in listed.tools}
                assert "preview" in tools and "set_movements" in tools
                assert tools["set_movements"].input_schema["required"] == [
                    "session_id",
                    "document_id",
                    "revision",
                    "scenario",
                    "targets",
                ]
                connected = await session.call_tool("connect_instance", {"instance_id": instance})
                assert not connected.is_error
                begin = await session.call_tool("begin_session", {})
                state = begin.structured_content
                g = {key: state[key] for key in ("session_id", "document_id", "revision")}
                edited = await session.call_tool(
                    "set_movements",
                    {
                        **g,
                        "scenario": "Scenario 1",
                        "targets": {"linked": {"translation": [0, 6, 0]}},
                    },
                )
                assert not edited.is_error, edited
                stale = await session.call_tool("undo", g)
                assert (
                    stale.is_error and stale.structured_content["error"]["code"] == "stale_document"
                )
                state = edited.structured_content["state"]
                g = {key: state[key] for key in ("session_id", "document_id", "revision")}

                async def completed(job):
                    deadline = time.monotonic() + 20
                    while time.monotonic() < deadline:
                        result = await session.call_tool("job_status", {"job_id": job})
                        if result.structured_content["status"] != "running":
                            assert result.structured_content["status"] == "completed", (
                                result.structured_content.get("error", result)
                            )
                            return result
                        await asyncio.sleep(0.02)
                    raise AssertionError("SDK job timed out")

                preview = await session.call_tool(
                    "preview", {**g, "scenario": "Scenario 1", "view": "front"}
                )
                png_result = await completed(preview.structured_content["job_id"])
                images = [item for item in png_result.content if item.type == "image"]
                assert len(images) == 1 and images[0].mime_type == "image/png"
                assert base64.b64decode(images[0].data).startswith(b"\x89PNG\r\n\x1a\n")
                saved = await session.call_tool("save_project", {**g, "path": str(path)})
                assert not saved.is_error, saved
                saved = await completed(saved.structured_content["job_id"])
                assert not saved.structured_content["state"]["dirty"]
                json.dumps(saved.structured_content)
                pending = await session.call_tool("open_project", {**g, "path": str(path)})
                cancelled = await session.call_tool(
                    "cancel_job",
                    {"session_id": g["session_id"], "job_id": pending.structured_content["job_id"]},
                )
                assert (
                    not cancelled.is_error and cancelled.structured_content["status"] == "cancelled"
                )
                deadline = time.monotonic() + 10
                while (await session.call_tool("inspect_state", {})).structured_content[
                    "worker_busy"
                ]:
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.02)
                opened = await session.call_tool("open_project", {**g, "path": str(path)})
                opened = await completed(opened.structured_content["job_id"])
                assert opened.structured_content["state"]["document_id"] != g["document_id"]
                results["ok"] = True

    def client():
        try:
            asyncio.run(run())
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    wait_until(qapp, lambda: not thread.is_alive(), timeout=40)
    assert failures == []
    assert results["ok"]
    wait_until(qapp, lambda: window.ai_control.session is None)
    restored = load_project(path)
    assert restored.motions["linked"].translation == [0, 6, 0]
    assert window._test_errors == []


def test_runtime_directory_permissions_and_discovery(window, qapp):
    assert runtime_directory().stat().st_mode & 0o777 == 0o700
    assert window.ai_control.path.stat().st_mode & 0o777 == 0o600
    results = []
    thread = threading.Thread(target=lambda: results.extend(Connection.discover()))
    thread.start()
    wait_until(qapp, lambda: not thread.is_alive())
    assert results == [{"instance_id": window.ai_control.instance_id, "project_path": None}]


def proximal_document(window):
    from factories import confirm, document

    doc = document()
    doc.scenario.cut_ids = ["bsso", "genio"]
    doc.fragments = confirm(doc)
    doc.fragments_key = doc.configuration_key
    install_document(window, doc)
    return doc


def test_gui_proximal_rotation_only_pivot_visible_measurement_editor_and_lock(
    window, qapp, monkeypatch
):
    from PySide6.QtWidgets import QPushButton

    doc = proximal_document(window)
    window.tabs.setCurrentIndex(3)
    window.motion_combo.setCurrentIndex(window.motion_combo.findData("left-proximal"))
    assert all(not spin.isVisible() and not spin.isEnabled() for spin in window.motion_spins[:3])
    assert all(not spin.isEnabled() for spin in window.motion_spins[3:])
    assert "geometric approximation" in window.proximal_note.text()
    assert not window.share_pivot_button.isEnabled()
    fragments = doc.fragments
    voxel = np.argwhere(doc.fragments == 3)[0][::-1]
    point = doc.geometry.index_to_world(voxel)
    window._arm_pick("pivot")
    window.views.pointPicked.emit(point.tolist())
    assert doc.scenario.proximal_rotations["left-proximal"].pivot_set
    assert window.views.pivot_actor.GetVisibility()
    assert all(spin.isEnabled() for spin in window.motion_spins[3:])

    def unexpected(*args, **kwargs):
        pytest.fail("Proximal rotation must reuse geometry")

    monkeypatch.setattr(window.jobs, "submit", unexpected)
    window.motion_spins[3].setValue(15)
    window.motion_spins[3].editingFinished.emit()
    assert doc.scenario.proximal_rotations["left-proximal"].angles == [15, 0, 0]
    assert doc.scenario.proximal_rotations["right-proximal"].angles == [0, 0, 0]
    assert doc.fragments is fragments
    matrix = window.views.actors[3].GetUserMatrix()
    transform = np.array([[matrix.GetElement(row, col) for col in range(4)] for row in range(4)])
    np.testing.assert_allclose(transform[:3, :3] @ point + transform[:3, 3], point, atol=1e-12)
    window._undo(False)
    assert doc.scenario.proximal_rotations["left-proximal"].angles == [0, 0, 0]
    assert doc.fragments is fragments
    window._undo(True)
    assert doc.scenario.proximal_rotations["left-proximal"].angles == [15, 0, 0]
    assert doc.fragments is fragments
    window._reset_motion()
    assert doc.scenario.proximal_rotations["left-proximal"].angles == [0, 0, 0]
    assert doc.scenario.proximal_rotations["left-proximal"].pivot_set
    # Edit the same named landmark and measurement controls used by GUI users.
    window._new_landmark()
    window.landmark_name.setText("Condyle")
    window.landmark_fragment_combo.setCurrentIndex(
        window.landmark_fragment_combo.findData("left-proximal")
    )
    plan = doc.world_to_plan(point)
    for spin, value in zip(window.landmark_point_spins, plan, strict=True):
        spin.setValue(float(value))
    window._save_landmark()
    assert doc.scenario.landmarks[0]["name"] == "Condyle"
    window.symmetry_right_spin.setValue(-10)
    reference_button = next(
        widget
        for widget in window.findChildren(QPushButton)
        if widget.text() == "Set reference offset"
    )
    reference_button.click()
    window.measurement_name.setText("Condyle midline")
    window.measurement_kind_combo.setCurrentIndex(0)
    window.measurement_role_combos[0].setCurrentText("Condyle")
    window._save_measurement()
    assert "Condyle midline" in doc.scenario.measurements
    assert "baseline" in window.measurement_results_label.text()
    window.measurement_name.setText("Condyle midline")
    window._delete_measurement()
    assert not doc.scenario.measurements
    window.landmark_name.setText("Condyle")
    window._delete_landmark()
    assert not doc.scenario.landmarks
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    assert not window.landmark_name.isEnabled()
    assert not window.measurement_kind_combo.isEnabled()
    assert window.views.isEnabled()
    dispatch(
        window,
        peer,
        "set_proximal_rotations",
        scenario="Scenario 1",
        targets={"right-proximal": {"pivot": [0, 0, 0], "rotation": {"yaw": 8}}},
    )
    assert doc.scenario.proximal_rotations["right-proximal"].angles == [0, 0, 8]
    window.take_control_button.click()
    assert window.measurement_kind_combo.isEnabled()
    assert window._test_errors == []


def test_sdk_proximal_symmetry_atomic_tools_previews_and_save(window, qapp, tmp_path):
    doc = proximal_document(window)
    instance, failures, results = window.ai_control.instance_id, [], {}
    path = tmp_path / "proximal.bimax"

    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "bimax.mcp_server"], env=dict(os.environ)
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                assert "set_proximal_rotations" in tools and "set_measurement" in tools
                target_schema = tools["set_proximal_rotations"].input_schema["$defs"][
                    "ProximalTarget"
                ]["properties"]
                assert "translation" not in target_schema
                await session.call_tool("connect_instance", {"instance_id": instance})
                unauthorized = await session.call_tool(
                    "set_proximal_rotations",
                    {
                        "session_id": "none",
                        "document_id": doc._identity,
                        "revision": doc.revision,
                        "scenario": "Scenario 1",
                        "targets": {"left-proximal": {"rotation": {"pitch": 4}}},
                    },
                )
                assert (
                    unauthorized.is_error
                    and unauthorized.structured_content["error"]["code"] == "session_revoked"
                )
                state = (await session.call_tool("begin_session", {})).structured_content

                async def mutate(tool_name, **arguments):
                    nonlocal state
                    guard = {key: state[key] for key in ("session_id", "document_id", "revision")}
                    result = await session.call_tool(tool_name, {**guard, **arguments})
                    assert not result.is_error, result
                    state = result.structured_content["state"]
                    return result

                await mutate(
                    "set_proximal_rotations",
                    scenario="Scenario 1",
                    targets={
                        "left-proximal": {"pivot": [4, 2, 20], "rotation": {"pitch": 12}},
                        "right-proximal": {"pivot": [17, 1, 22], "rotation": {"yaw": 5}},
                    },
                )
                await mutate("set_symmetry_reference", right=10)
                await mutate(
                    "set_landmark",
                    scenario="Scenario 1",
                    name="Left",
                    fragment="left-proximal",
                    point=[4, 2, 20],
                )
                await mutate(
                    "set_landmark",
                    scenario="Scenario 1",
                    name="Right",
                    fragment="right-proximal",
                    point=[17, 1, 22],
                )
                await mutate(
                    "set_measurement",
                    scenario="Scenario 1",
                    name="Pair",
                    kind="paired",
                    landmarks={"left": "Left", "right": "Right"},
                )
                readout = await session.call_tool("measurements", {"scenario": "Scenario 1"})
                assert readout.structured_content["symmetry_reference"]["right_mm"] == 10
                pair = readout.structured_content["measurements"]["Pair"]
                assert pair["baseline"]["height_difference_mm"] == -2
                assert pair["moved"]["height_difference_mm"] == pytest.approx(-2)
                await mutate(
                    "reset_proximal_rotations", scenario="Scenario 1", targets=["right-proximal"]
                )
                assert (
                    state["scenarios"]["Scenario 1"]["proximal_rotations"]["right-proximal"][
                        "rotation"
                    ]["yaw"]
                    == 0
                )

                async def wait_job(job):
                    deadline = time.monotonic() + 20
                    while time.monotonic() < deadline:
                        result = await session.call_tool("job_status", {"job_id": job})
                        if result.structured_content["status"] != "running":
                            assert result.structured_content["status"] == "completed", (
                                result.structured_content.get("error")
                            )
                            return result
                        await asyncio.sleep(0.02)
                    raise AssertionError("SDK proximal job timed out")

                g = {key: state[key] for key in ("session_id", "document_id", "revision")}
                preview = await session.call_tool(
                    "preview", {**g, "scenario": "Scenario 1", "view": "right"}
                )
                image = await wait_job(preview.structured_content["job_id"])
                assert any(item.type == "image" for item in image.content)
                saved = await session.call_tool("save_project", {**g, "path": str(path)})
                saved = await wait_job(saved.structured_content["job_id"])
                assert not saved.structured_content["state"]["dirty"]
                opened = await session.call_tool("open_project", {**g, "path": str(path)})
                opened = await wait_job(opened.structured_content["job_id"])
                state = opened.structured_content["state"]
                assert state["symmetry_reference"]["right_mm"] == 10
                assert (
                    state["scenarios"]["Scenario 1"]["proximal_rotations"]["left-proximal"][
                        "rotation"
                    ]["pitch"]
                    == 12
                )
                assert "Pair" in state["scenarios"]["Scenario 1"]["measurements"]
                await mutate("set_measurement", scenario="Scenario 1", name="Pair", remove=True)
                await mutate("set_landmark", scenario="Scenario 1", name="Left", remove=True)
                results["ok"] = True

    def client():
        try:
            asyncio.run(run())
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    wait_until(qapp, lambda: not thread.is_alive(), timeout=40)
    assert failures == []
    assert results["ok"]
    wait_until(qapp, lambda: window.ai_control.session is None)
    restored = load_project(path)
    assert restored.scenario.proximal_rotations["left-proximal"].angles == [12, 0, 0]
    assert "Pair" in restored.scenario.measurements
    assert window._test_errors == []


def test_mcp_save_as_overwrite_conflict_and_provisional_saving(window, qapp, tmp_path):
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    first, second = tmp_path / "first.bimax", tmp_path / "second.bimax"

    def save(path):
        result = dispatch(window, peer, "save_project", path=str(path))
        wait_until(qapp, lambda: control.active_job is None)
        return control.jobs[result["job_id"]]

    assert save(first)["status"] == "completed"
    assert window.project_path == str(first)
    original = first.read_bytes()
    dispatch(window, peer, "set_orientation", rotation={"pitch": 3, "roll": 0, "yaw": 0})
    assert save(second)["status"] == "completed"
    assert window.project_path == str(second) and first.read_bytes() == original
    previous = second.read_bytes()
    assert save(second)["status"] == "completed"
    assert next(tmp_path.glob("second.backup-*.bimax")).read_bytes() == previous
    second.write_bytes(b"external change")
    dispatch(window, peer, "set_orientation", rotation={"pitch": 5, "roll": 0, "yaw": 0})
    failed = save(second)
    assert failed["status"] == "failed" and "External file conflict" in failed["error"]["message"]
    assert window.dirty and second.read_bytes() == b"external change"
    dispatch(window, peer, "replace_cut", cut_id="lefort1", geometry=upper_geometry(18))
    dispatch(window, peer, "select_cuts", scenario="Scenario 1", cut_ids=["lefort1"])
    result = dispatch(window, peer, "preview", scenario="Scenario 1", view="front")
    wait_until(qapp, lambda: control.active_job is None)
    assert control.jobs[result["job_id"]]["status"] == "completed"
    assert not window.doc.cuts_accepted
    third = tmp_path / "provisional.bimax"
    assert save(third)["status"] == "completed"
    restored = load_project(third)
    assert restored.scenario.cut_ids == ["lefort1"]
    assert not restored.cuts_accepted and restored.fragments is None
    assert window.doc.fragments is not None
    assert window._test_errors == []


@pytest.mark.parametrize("view,position", [("axial", 5), ("coronal", -10), ("sagittal", -10)])
def test_oriented_slice_previews_and_baseline_picking(window, qapp, view, position):
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    dispatch(window, peer, "set_orientation", rotation={"pitch": 10, "roll": 7, "yaw": 5})
    dispatch(
        window,
        peer,
        "set_movements",
        scenario="Scenario 1",
        targets={"mandible": {"translation": [0, 2, 0]}},
    )
    result = dispatch(window, peer, "preview", scenario="Scenario 1", view=view, position=position)
    wait_until(qapp, lambda: control.active_job is None)
    job = control.jobs[result["job_id"]]
    assert job["status"] == "completed", job.get("error")
    capture = control.previews[job["metadata"]["preview_id"]][1]
    labels, baseline = capture.slices[view]
    row, col = np.argwhere(labels > 0)[0]
    picked = control.dispatch(
        peer,
        "pick_image_point",
        {
            "preview_id": job["metadata"]["preview_id"],
            "view": view,
            "x": (col + 0.5) / 1024,
            "y": (row + 0.5) / 1024,
        },
    )
    assert picked["fragment"] == window.doc.fragment_descriptors[int(labels[row, col])].id
    np.testing.assert_allclose(picked["point"], window.doc.world_to_plan(baseline[row, col]))


def test_empty_geometry_job_error_and_inactivity_during_active_work(window, qapp):
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    dispatch(window, peer, "replace_cut", cut_id="lefort1", geometry=upper_geometry(-1000))
    dispatch(window, peer, "select_cuts", scenario="Scenario 1", cut_ids=["lefort1"])
    result = dispatch(window, peer, "preview", scenario="Scenario 1", view="right")
    control.last_request = time.monotonic() - 1000
    control.expire()
    assert control.session is not None
    wait_until(qapp, lambda: control.active_job is None)
    job = control.jobs[result["job_id"]]
    assert job["status"] == "failed" and "empty" in job["error"]["message"]
    assert window.doc.fragments is None and not window.doc.cuts_accepted
    control.expire()
    assert control.session is not None
    assert window._test_errors == []


def test_cancel_capture_yields_before_remaining_views_and_never_saves(
    window, qapp, tmp_path, monkeypatch
):
    control, peer = window.ai_control, object()
    control.dispatch(peer, "begin_session", {})
    from bimax.preview import Capture

    rendered = []
    original = Capture.render

    def cancel_after_first(capture, name):
        rendered.append(name)
        original(capture, name)
        control.dispatch(
            peer, "cancel_job", {"session_id": control.session, "job_id": control.active_job}
        )

    monkeypatch.setattr(Capture, "render", cancel_after_first)
    result = dispatch(window, peer, "preview", scenario="Scenario 1")
    qapp.processEvents()
    qapp.processEvents()
    assert rendered == ["right"]
    assert control.jobs[result["job_id"]]["status"] == "cancelled"
    assert not list(tmp_path.glob("*.bimax"))
    assert control.session is not None


def test_multiple_instances_require_selection_and_attach_by_id(window, qapp):
    from bimax.app import MainWindow
    from bimax.mcp_server import RemoteError

    second = MainWindow()
    connection = Connection()
    results, failures = {}, []

    def connect():
        try:
            connection.connect(None)
        except RemoteError as error:
            results["choice"] = error.error
        except Exception as error:
            failures.append(error)
        try:
            results["selected"] = connection.connect(window.ai_control.instance_id)
        except Exception as error:
            failures.append(error)
        finally:
            connection.close()

    thread = threading.Thread(target=connect, daemon=True)
    try:
        thread.start()
        wait_until(qapp, lambda: not thread.is_alive())
        assert failures == []
        assert results["choice"]["code"] == "select_instance"
        assert {item["instance_id"] for item in results["choice"]["instances"]} == {
            window.ai_control.instance_id,
            second.ai_control.instance_id,
        }
        assert results["selected"]["document_id"] == window.doc._identity
    finally:
        second.jobs.close()
        second.views.close()
        second.close()
        qapp.processEvents()
