"""Worker handoff remains cancellable and never waits on the GUI thread."""

import os
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from bimax.jobs import Jobs
from bimax.model import Document, Geometry
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
    return Document(
        Geometry(labels.shape, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), tuple(np.eye(3).ravel())),
        (labels != 0).astype(np.uint8),
        labels,
        np.zeros_like(labels),
        np.zeros_like(labels),
        cuts={"not_empty": True},
    )


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
    monkeypatch.setattr("bimax.surfaces.make_surfaces", lambda *args, **kwargs: {})
    result = execute(operation, doc, {"path": "unused"})
    labels = proposal if operation == "separate" else doc.anatomy
    np.testing.assert_array_equal(result["anatomy_counts"], np.bincount(labels.ravel(), minlength=5))


@pytest.mark.parametrize("has_fragments", [False, True])
def test_open_prepares_exact_anatomy_and_fragment_surface_sets(has_fragments, monkeypatch, capsys):
    doc = document()
    if has_fragments:
        doc.fragments = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[doc.anatomy]
    labels_seen = []

    def surfaces(labels, geometry, **kwargs):
        assert geometry is doc.geometry
        labels_seen.append(labels)
        return {"unique-result": len(labels_seen)}

    monkeypatch.setattr("bimax.storage.load_project", lambda *args, **kwargs: doc)
    monkeypatch.setattr("bimax.surfaces.make_surfaces", surfaces)
    result = execute("open", None, {"path": "unused"})
    assert "surfaces" not in result
    assert labels_seen[0] is doc.anatomy
    assert result["surface_sets"][False] == {"unique-result": 1}
    progress = capsys.readouterr().out
    assert "Building anatomy display" in progress
    if has_fragments:
        assert len(labels_seen) == 2
        assert labels_seen[1] is doc.fragments
        assert result["surface_sets"][True] == {"unique-result": 2}
        assert "Building fragment display" in progress
    else:
        assert len(labels_seen) == 1
        assert set(result["surface_sets"]) == {False}


def test_cut_worker_returns_cached_fragment_counts(monkeypatch):
    doc = document()
    fragments = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[doc.anatomy]
    validation = []

    def build(document, *, validate, **kwargs):
        assert document is doc
        validation.append(validate)
        return fragments

    monkeypatch.setattr("bimax.engine.build_fragments", build)
    monkeypatch.setattr("bimax.surfaces.make_surfaces", lambda *args, **kwargs: {})
    result = execute("cuts_preview", doc, {})
    np.testing.assert_array_equal(result["fragment_counts"], np.bincount(fragments.ravel(), minlength=9))
    assert validation == [False]
