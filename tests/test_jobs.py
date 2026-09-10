"""Worker handoff remains cancellable and never waits on the GUI thread."""

import os
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from bimax.engine import build_fragments, default_cuts, required_fragment_labels
from bimax.jobs import Jobs
from bimax.model import CutPlan, Document, Geometry, new_scenario
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
    doc.plan.cuts = default_cuts(doc)
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
    monkeypatch.setattr("bimax.surfaces.make_surfaces", lambda *args, **kwargs: {})
    result = execute(operation, doc, {"path": "unused"})
    labels = proposal if operation == "separate" else doc.anatomy
    np.testing.assert_array_equal(result["anatomy_counts"], np.bincount(labels.ravel(), minlength=5))


@pytest.mark.parametrize("accepted", [False, True])
def test_open_prepares_exact_anatomy_and_active_reviewed_fragment_sets(accepted, monkeypatch, capsys):
    doc = document()
    doc.plan.accepted = accepted
    fragments = np.array([0, 8, 1, 5, 7], dtype=np.uint8)[doc.anatomy]
    builds = []

    def build(document, *, validate, **kwargs):
        builds.append((document.configuration_key, validate))
        return fragments

    labels_seen = []

    def surfaces(labels, geometry, **kwargs):
        assert geometry is doc.geometry
        labels_seen.append(labels)
        return {"unique-result": len(labels_seen)}

    monkeypatch.setattr("bimax.storage.load_project", lambda *args, **kwargs: doc)
    monkeypatch.setattr("bimax.engine.build_fragments", build)
    monkeypatch.setattr("bimax.surface_cache.cached_surfaces", surfaces)
    result = execute("open", None, {"path": "unused"})
    assert "surfaces" not in result
    assert labels_seen[0] is doc.anatomy
    assert result["surface_sets"][False] == {"unique-result": 1}
    progress = capsys.readouterr().out
    assert "Building anatomy display" in progress
    if accepted:
        assert builds == [(doc.configuration_key, True)]
        assert doc.fragments is fragments
        assert doc.fragments_key == doc.configuration_key
        assert len(labels_seen) == 2
        assert labels_seen[1] is doc.fragments
        assert result["surface_sets"][True] == {"unique-result": 2}
        assert "Building fragment display" in progress
    else:
        assert builds == []
        assert doc.fragments is None
        assert doc.fragments_key is None
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
    assert result["configuration_key"] == doc.configuration_key
    assert not doc.plan.accepted
    assert doc.fragments is None


def six_plan_document():
    source = np.ones((21, 21, 21), dtype=np.uint8)
    anatomy = np.where(np.indices(source.shape)[0] < 9, 3, 2).astype(np.uint8)
    doc = Document(
        Geometry(source.shape, (1.0,) * 3, (0.0,) * 3, tuple(np.eye(3).ravel())),
        source, anatomy, np.zeros_like(source), np.zeros_like(source),
    )
    doc.plans.clear()
    doc.scenarios.clear()
    for level in (1, 2, 3):
        for genio in (False, True):
            name = f"lf{level}-genio{int(genio)}"
            cuts = default_cuts(doc, level)
            if level == 1:
                cuts["lefort"]["planes"]["upper"]["origin"][2] = 11
            cuts["lefort"]["enabled"] = cuts["bsso"]["enabled"] = True
            cuts["genio"]["enabled"] = genio
            doc.plans[name] = CutPlan(name, level, cuts)
            doc.scenarios[name] = new_scenario(name)
            doc.active_scenario = name
            labels = build_fragments(doc, validate=False)
            for label in required_fragment_labels(doc.plan):
                xyz = np.argwhere(labels == label)[0][::-1]
                doc.plan.confirmations[str(label)] = doc.world_to_plan(doc.geometry.index_to_world(xyz)).tolist()
            doc.plan.accepted = True
    doc.active_scenario = "lf2-genio1"
    return doc


def test_open_six_plans_builds_only_active_configuration(monkeypatch):
    doc = six_plan_document()
    expected = build_fragments(doc)
    built, meshed = [], []

    def build(document, *, validate, **kwargs):
        built.append((document.scenario.plan_id, validate))
        return expected

    def surfaces(labels, *args, **kwargs):
        meshed.append(labels)
        return {}

    monkeypatch.setattr("bimax.storage.load_project", lambda *args, **kwargs: doc)
    monkeypatch.setattr("bimax.engine.build_fragments", build)
    monkeypatch.setattr("bimax.surface_cache.cached_surfaces", surfaces)
    result = execute("open", None, {"path": "six-plans.bimax"})
    assert built == [("lf2-genio1", True)]
    assert len(meshed) == 2
    assert meshed[0] is doc.anatomy
    assert meshed[1] is expected
    assert result["document"].fragments_key == ("lf2-genio1", 0, 0)
    assert result["document"].fragments is expected
    assert result["fragment_mode"] is True
    assert len(doc.plans) == len(doc.scenarios) == 6
    assert all(not any(isinstance(value, np.ndarray) for value in vars(plan).values()) for plan in doc.plans.values())


def test_prepare_plan_reuses_exact_mesh_cache_and_returns_configuration_key(monkeypatch):
    from bimax.surfaces import make_surfaces

    doc = six_plan_document()
    doc.active_scenario = "lf1-genio0"
    meshed = []

    def surfaces(labels, geometry, **kwargs):
        meshed.append(labels.copy())
        return make_surfaces(labels, geometry, **kwargs)

    monkeypatch.setattr("bimax.surfaces.make_surfaces", surfaces)
    cold = execute("prepare_plan", doc, {})
    warm = execute("prepare_plan", doc, {})
    assert len(meshed) == 1
    assert cold["configuration_key"] == warm["configuration_key"] == doc.configuration_key
    np.testing.assert_array_equal(cold["fragments"], warm["fragments"])
    for label, mesh in cold["surfaces"].items():
        for field in ("vertices", "faces", "normals"):
            np.testing.assert_array_equal(getattr(mesh, field), getattr(warm["surfaces"][label], field))
    assert doc.fragments is None, "Workers return derived geometry; the GUI owns acceptance"
    # Changing geometry requires different labels and a different cache entry,
    # even though this is still the same named cut plan.
    doc.plan.cuts["lefort"]["planes"]["upper"]["origin"][2] += 1
    doc.plan.revision += 1
    changed = execute("prepare_plan", doc, {})
    assert len(meshed) == 2
    assert changed["configuration_key"] != cold["configuration_key"]
    assert changed["configuration_key"] == doc.configuration_key
    assert np.any(changed["fragments"] != cold["fragments"])


def test_prepare_plan_rejects_unreviewed_geometry_before_building(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Unreviewed plans must not prepare movement geometry.")

    monkeypatch.setattr("bimax.engine.build_fragments", unexpected)
    with pytest.raises(ValueError, match="Preview and accept"):
        execute("prepare_plan", document(), {})
