"""Worker handoff remains cancellable and never waits on the GUI thread."""

import os
import threading
import time

import numpy as np
import pytest
from factories import confirm
from factories import document as assigned_document
from PySide6.QtWidgets import QApplication

from bimax.engine import build_fragments, default_cuts
from bimax.jobs import Jobs
from bimax.model import Document, Geometry, new_scenario
from bimax.prepared import load_options, prepare_geometry
from bimax.worker import anatomy_counts, execute


@pytest.fixture(autouse=True)
def isolated_display_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    yield QApplication.instance() or QApplication([])


def pump_until(qapp, condition):
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.001)
    assert condition()


def test_cancel_after_worker_completion_discards_queued_result(qapp):
    jobs = Jobs()
    jobs.busy = True
    results, failures, states = [], [], []
    jobs.finished.connect(lambda *args: results.append(args))
    jobs.failed.connect(failures.append)
    jobs.busyChanged.connect(states.append)
    worker = threading.Thread(target=lambda: jobs._completed.emit("open", -1, {"ok": True}, ""))
    worker.start()
    worker.join()
    assert jobs.busy
    jobs.cancel()
    pump_until(qapp, lambda: not jobs.busy)
    assert not jobs.busy
    assert results == failures == []
    assert states == [False]


def test_finished_callback_can_submit_next_job(qapp, monkeypatch):
    jobs = Jobs()
    seen = []

    def run(operation, revision, doc, options):
        jobs._completed.emit(operation, revision, {"ok": True}, "")

    def finished(operation, revision, result):
        seen.append((operation, jobs.busy))
        if operation == "first":
            jobs.submit("second")

    monkeypatch.setattr(jobs, "_run", run)
    jobs.finished.connect(finished)
    jobs.submit("first")
    pump_until(qapp, lambda: len(seen) == 2)
    assert seen == [("first", False), ("second", False)]
    assert not jobs.busy


def test_cancelled_completion_stays_cancelled_when_idle_callback_starts_new_job(qapp, monkeypatch):
    jobs = Jobs()
    jobs.busy = True
    jobs.cancel()
    results, launched = [], []

    def run(operation, revision, doc, options):
        jobs._completed.emit(operation, revision, {"ok": True}, "")

    def busy_changed(busy):
        if not busy and not launched:
            launched.append(True)
            jobs.submit("next")

    monkeypatch.setattr(jobs, "_run", run)
    jobs.busyChanged.connect(busy_changed)
    jobs.finished.connect(lambda operation, *args: results.append(operation))
    jobs._on_completed("cancelled", -1, {"old": True}, "")
    pump_until(qapp, lambda: results == ["next"])
    assert not jobs.busy


def test_failure_callback_observes_idle_state(qapp):
    jobs = Jobs()
    jobs.busy = True
    failures = []
    jobs.failed.connect(lambda text: failures.append((text, jobs.busy)))
    jobs._on_completed("open", -1, None, "failed to load")
    assert failures == [("failed to load", False)]


def test_close_does_not_join_slow_transport_and_prevents_process_spawn(qapp, monkeypatch):
    jobs = Jobs()
    copying = threading.Event()
    release = threading.Event()
    spawned = []

    def slow_dump(*args, **kwargs):
        copying.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr("bimax.jobs.pickle.dump", slow_dump)
    monkeypatch.setattr("bimax.jobs.subprocess.Popen", lambda *args, **kwargs: spawned.append(args))
    jobs.submit("save")
    assert copying.wait(timeout=2)
    started = time.monotonic()
    jobs.close()
    elapsed = time.monotonic() - started
    release.set()
    pump_until(qapp, lambda: not jobs.busy)
    assert elapsed < 0.1
    assert spawned == []


def document():
    labels = np.array([0, 2, 2, 3, 4, 1, 3, 0], dtype=np.uint8).reshape(2, 2, 2)
    doc = Document(
        Geometry(labels.shape, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), tuple(np.eye(3).ravel())),
        (labels != 0).astype(np.uint8),
        labels,
        np.zeros_like(labels),
        np.zeros_like(labels),
    )
    doc.cuts = default_cuts(doc)
    return doc


def test_anatomy_counts_bounds_bincount_temporary_memory(monkeypatch):
    counts = np.bincount
    sizes = []

    def bounded(values, *args, **kwargs):
        sizes.append(values.size)
        return counts(values, *args, **kwargs)

    monkeypatch.setattr("bimax.engine.np.bincount", bounded)
    labels = np.tile(np.arange(5, dtype=np.uint8), 250_001)
    np.testing.assert_array_equal(anatomy_counts(labels), np.full(5, 250_001))
    assert max(sizes) <= 1_000_000


@pytest.mark.parametrize("operation", ["import", "open", "separate", "surfaces"])
def test_worker_supplies_anatomy_counts_for_ui(operation, monkeypatch):
    doc = document()
    proposal = doc.anatomy.copy()
    proposal[proposal == 1] = 3
    monkeypatch.setattr("bimax.engine.load_volume", lambda *args, **kwargs: doc)
    monkeypatch.setattr("bimax.storage.load_project", lambda *args, **kwargs: doc)
    monkeypatch.setattr("bimax.engine.separate", lambda *args, **kwargs: proposal)
    result = execute(operation, doc, {"path": "unused"})
    labels = proposal if operation == "separate" else doc.anatomy
    np.testing.assert_array_equal(result["anatomy_counts"], np.bincount(labels.ravel(), minlength=5))


@pytest.mark.parametrize("accepted", [False, True])
def test_open_prepares_only_active_reviewed_fragments(accepted, monkeypatch):
    doc = assigned_document()
    doc.scenario.cut_ids = ["lefort1", "bsso", "genio"]
    if accepted:
        confirm(doc)
    monkeypatch.setattr("bimax.storage.load_project", lambda *args, **kwargs: doc)
    result = execute("open", None, {"path": "unused"})
    assert set(result["surface_sets"]) == ({False, True} if accepted else {False})
    assert result["fragment_mode"] == accepted
    if accepted:
        np.testing.assert_array_equal(doc.fragments, build_fragments(doc))
        assert doc.fragments_key == doc.configuration_key
    else:
        assert doc.fragments is None


def test_cut_worker_returns_cached_counts_and_artifacts_without_accepting():
    doc = document()
    result = execute("cuts_preview", doc, {})
    np.testing.assert_array_equal(result["fragment_counts"], np.bincount(result["fragments"].ravel(), minlength=9))
    assert result["configuration_key"] == doc.configuration_key
    assert result["artifacts"]
    assert not doc.cuts_accepted
    assert doc.fragments is None


def six_scenario_document():
    doc = assigned_document()
    doc.scenarios.clear()
    for level in (1, 2, 3):
        for genio in (False, True):
            name = f"lf{level}-genio{int(genio)}"
            doc.scenarios[name] = new_scenario([f"lefort{level}", "bsso", *(["genio"] if genio else [])])
            doc.active_scenario = name
            confirm(doc)
    doc.active_scenario = "lf2-genio1"
    return doc


def test_open_six_scenarios_prepares_only_active_configuration(monkeypatch):
    doc = six_scenario_document()
    keys = []
    def record(document, *args, **kwargs):
        keys.append(document.configuration_key)
        return prepare_geometry(document, *args, **kwargs)
    monkeypatch.setattr("bimax.storage.load_project", lambda *args, **kwargs: doc)
    monkeypatch.setattr("bimax.prepared.prepare_geometry", record)
    result = execute("open", None, {"path": "unused"})
    assert keys == [doc.configuration_key]
    assert result["fragment_mode"]
    assert len(doc.cuts) == 7
    assert len(doc.scenarios) == 6
    assert doc.cut_key("lefort2") in doc.artifacts
    assert doc.cut_key("lefort1") not in doc.artifacts
    assert doc.cut_key("lefort3") not in doc.artifacts


def test_prepared_cache_hit_worker_needs_no_document_arrays(monkeypatch):
    doc = six_scenario_document()
    first = execute("prepare_scene", doc, {})
    def unexpected(*args, **kwargs):
        pytest.fail("A cache hit must not evaluate cuts or rebuild surfaces")
    monkeypatch.setattr("bimax.prepared.cut_membership", unexpected)
    monkeypatch.setattr("bimax.surfaces.make_surfaces", unexpected)
    warm = execute("load_prepared", None, load_options(doc))
    np.testing.assert_array_equal(warm["fragments"], first["fragments"])
    assert warm["configuration_key"] == doc.configuration_key


def test_missing_cache_hit_requests_preparation():
    doc = six_scenario_document()
    result = execute("load_prepared", None, load_options(doc))
    assert result == {"needs_preparation": True, "configuration_key": doc.configuration_key, "document_id": doc._identity}


def test_prepare_scene_rejects_unreviewed_geometry():
    with pytest.raises(ValueError, match="Preview and accept"):
        execute("prepare_scene", document(), {})
