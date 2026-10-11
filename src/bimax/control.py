"""Owner-only Unix socket, Qt main-thread commands, and revocable editing sessions."""

import json
import time
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QObject, QTimer
from PySide6.QtNetwork import QLocalServer

from .commands import CONVENTIONS, inspect_document
from .engine import required_fragment_labels
from .model import COLORS
from .preview import Capture
from .protocol import MAX_MESSAGE, TOOLS, runtime_directory
from .storage import file_version


class ControlError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class Control(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.instance_id = uuid4().hex
        self.session = None
        self.owner = None
        self.last_request = time.monotonic()
        self.jobs = OrderedDict()
        self.previews = OrderedDict()
        self.active_job = None
        self.buffers = {}
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self.path = runtime_directory() / f"{self.instance_id}.sock"
        if not self.server.listen(str(self.path)):
            raise RuntimeError(self.server.errorString())
        self.path.chmod(0o600)
        self.server.newConnection.connect(self._connect)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.expire)
        self.timer.start()
        window.jobs.finished.connect(self._finished)
        window.jobs.failed.connect(self._failed)
        window.jobs.progress.connect(self._progress)
        window.jobs.busyChanged.connect(self._busy)

    def _connect(self):
        while self.server.hasPendingConnections():
            peer = self.server.nextPendingConnection()
            self.buffers[peer] = bytearray()
            peer.readyRead.connect(lambda peer=peer: self._read(peer))
            peer.disconnected.connect(lambda peer=peer: self._disconnect(peer))

    def _disconnect(self, peer):
        if peer == self.owner:
            self.release()
        self.buffers.pop(peer, None)
        peer.deleteLater()

    def _read(self, peer):
        buffer = self.buffers[peer]
        buffer.extend(peer.readAll().data())
        if len(buffer) > MAX_MESSAGE:
            peer.abort()
            return
        while b"\n" in buffer:
            line, _, remaining = buffer.partition(b"\n")
            buffer[:] = remaining
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or set(request) != {"tool", "arguments"}:
                    raise ValueError("Expected tool and arguments.")
                result = self.dispatch(peer, request["tool"], request["arguments"])
                response = {"result": result}
            except Exception as error:
                response = {
                    "error": {
                        "code": getattr(error, "code", "invalid_request"),
                        "message": str(error),
                    }
                }
            encoded = json.dumps(response, allow_nan=False).encode() + b"\n"
            if len(encoded) > MAX_MESSAGE:
                encoded = b'{"error":{"code":"message_too_large","message":"Result exceeds local message limit."}}\n'
            peer.write(encoded)

    def state(self):
        w = self.window
        result = (
            inspect_document(w.doc)
            if w.doc
            else {"document_id": None, "revision": -1, "conventions": CONVENTIONS}
        )
        result.update(
            instance_id=self.instance_id,
            project_path=w.project_path,
            dirty=w.dirty,
            session_id=self.session,
            active_jobs=[
                {
                    key: value
                    for key, value in self.jobs[self.active_job].items()
                    if key not in ("state", "image", "metadata")
                }
            ]
            if self.active_job
            else [],
            worker_busy=w.jobs.busy,
        )
        return result

    def dispatch(self, peer, tool, arguments):
        if tool not in TOOLS:
            raise ValueError("Unknown tool.")
        args = TOOLS[tool][0].model_validate(arguments).model_dump(mode="json", exclude_none=True)
        self.expire()
        if self.session and peer == self.owner:
            self.last_request = time.monotonic()
        w = self.window
        if tool in ("inspect_state", "connect_instance"):
            return self.state()
        if tool == "measurements":
            state = self.state()
            scenario = args["scenario"]
            if w.doc is None or scenario not in w.doc.scenarios:
                raise ValueError("Unknown scenario.")
            return {
                "symmetry_reference": state["symmetry_reference"],
                "scenario": scenario,
                **state["scenarios"][scenario],
            }
        if tool == "begin_session":
            if self.session or w.jobs.busy or self.active_job:
                raise ControlError(
                    "busy",
                    "BIMAX is already editing or computing. Finish or cancel that work first.",
                )
            if args.get("scenario") and (w.doc is None or args["scenario"] not in w.doc.scenarios):
                raise ValueError("Unknown scenario.")
            w._finish_motion_edit()
            w._remesh_timer.stop()
            w.pick_mode = None
            w.views.set_tool("navigate")
            w.views.set_plane([0, 0, 0], [0, 0, 1], False)
            for view in w.views.slices:
                view._stroke.clear()
                view._polygon.clear()
                view._dragging = False
            self.session, self.owner = uuid4().hex, peer
            self.last_request = time.monotonic()
            if args.get("scenario"):
                w._command("select_scenario", args["scenario"])
            self._banner()
            w._sync_enabled()
            return self.state()
        if tool == "job_status":
            if args["job_id"] not in self.jobs:
                raise ValueError("Unknown job ID.")
            return self.jobs[args["job_id"]]
        if tool == "pick_image_point":
            item = self.previews.get(args["preview_id"])
            if item is None or w.doc is None or item[0] != (w.doc._identity, w.doc.revision):
                raise ControlError(
                    "stale_preview",
                    "Preview is unavailable or older than the document revision. Request a fresh preview.",
                )
            return item[1].pick(args["view"], args["x"], args["y"])
        if not self.session or peer != self.owner or args.pop("session_id", None) != self.session:
            raise ControlError(
                "session_revoked",
                "Editing session is absent or revoked. Begin a new editing session.",
            )
        if tool == "end_session":
            self.release()
            return self.state()
        if tool == "cancel_job":
            if args["job_id"] != self.active_job:
                raise ValueError("That job is not active in this session.")
            self._cancel()
            return self.jobs[args["job_id"]]
        if self.active_job or w.jobs.busy:
            raise ControlError(
                "busy", "A job is active. Poll job_status or cancel_job before further mutations."
            )
        identity = args.pop("document_id", None)
        revision = args.pop("revision")
        if (identity, revision) != ((w.doc._identity, w.doc.revision) if w.doc else (None, -1)):
            raise ControlError(
                "stale_document",
                "Document identity or revision changed. Inspect state and retry with fresh values.",
            )
        if tool == "open_project":
            if w.dirty:
                raise ControlError(
                    "unsaved_changes",
                    "Save the current project to an explicit path before opening another project.",
                )
            path = self._path(args["path"])
            if not path.is_file():
                raise ValueError("Project file does not exist.")
            return self._start("open", path=str(path))
        if w.doc is None:
            raise ControlError("no_project", "Open an absolute .bimax project first.")
        if tool == "save_project":
            path = self._path(args["path"])
            version = w.project_version if str(path) == w.project_path else file_version(path)
            return self._start("save", document=w.doc, path=str(path), expected_version=version)
        if tool in ("undo", "redo"):
            changed = w.history.apply(w.doc, tool == "redo")
            if changed:
                w.dirty = True
                w.setWindowTitle(f"BIMAX — {w.doc.source_name} *")
                w._refresh_command(tool)
            self._banner()
            return {"changed": changed, **self.state()}
        if tool == "preview":
            if args["view"] in ("axial", "coronal", "sagittal") and "position" not in args:
                raise ValueError("Provide a physical slice position in millimetres.")
            w._command("select_scenario", args["scenario"])
            record = self._record("preview")
            record["view"], record["position"] = args["view"], args.get("position")
            if w.doc.fragments is not None and w.doc.fragments_key == w.doc.configuration_key:
                self._capture()
            else:
                from .prepared import load_options

                if w.doc.configuration_key in w.doc.artifacts:
                    w.jobs.submit("load_prepared", None, **load_options(w.doc), **self._context())
                else:
                    w.jobs.submit("cuts_preview", w.doc, **self._context())
            return {"job_id": self.active_job, "state": self.state()}
        if tool == "accept_combination":
            if (
                args["scenario"] != w.doc.active_scenario
                or w.doc.fragments is None
                or w._fragment_counts is None
                or w.doc.fragments_key != w.doc.configuration_key
            ):
                raise ValueError("Preview this scenario before accepting it.")
            args.update(
                fragments=w.doc.fragments,
                fragment_counts=w._fragment_counts,
                anatomy_counts=w._anatomy_counts,
            )
        result = w._command(tool, **args)
        self._banner()
        return {**result, "state": self.state()}

    @staticmethod
    def _path(value):
        path = Path(value)
        if not path.is_absolute() or path.suffix.lower() != ".bimax" or path.is_symlink():
            raise ValueError("Provide an absolute, regular .bimax filename.")
        return path.resolve()

    def _record(self, operation):
        w = self.window
        job_id = uuid4().hex
        record = {
            "job_id": job_id,
            "operation": operation,
            "status": "running",
            "progress": "Starting…",
            "session_id": self.session,
            "document_id": w.doc._identity if w.doc else None,
            "revision": w.doc.revision if w.doc else -1,
        }
        self.jobs[job_id] = record
        self.active_job = job_id
        while len(self.jobs) > 32:
            self.jobs.popitem(last=False)
        return record

    def _context(self):
        return {"_control_session": self.session, "_control_job": self.active_job}

    def _start(self, operation, document=None, **kwargs):
        record = self._record(operation)
        try:
            self.window.jobs.submit(operation, document, **kwargs, **self._context())
        except Exception:
            self.active_job = None
            record["status"] = "failed"
            raise
        return {"job_id": record["job_id"]}

    def owns_result(self, operation, revision, result):
        if not self.active_job:
            return False
        record = self.jobs[self.active_job]
        if record["session_id"] != self.session:
            return False
        if operation == "open":
            return record["operation"] == "open"
        doc = self.window.doc
        return doc is not None and (record["document_id"], record["revision"]) == (
            doc._identity,
            doc.revision,
        )

    def _finished(self, operation, revision, result):
        if not self.active_job:
            return
        w = self.window
        record = self.jobs[self.active_job]
        if record["session_id"] != self.session:
            return
        if result.get("_control_job") != self.active_job or not self.owns_result(
            operation, revision, result
        ):
            self._failed("Discarded stale computation; inspect state and retry.")
            return
        if operation in ("cuts_preview", "load_prepared") and record["operation"] == "preview":
            if result.get("needs_preparation"):
                w.jobs.submit("cuts_preview", w.doc, **self._context())
                return
            digest = result.get("anatomy_fingerprint")
            if digest and digest["revision"] == w.doc.anatomy_revision:
                w.doc._anatomy_digest, w.doc._digest_revision = digest["digest"], digest["revision"]
            if (
                result.get("document_id") != w.doc._identity
                or (record["document_id"], record["revision"]) != (w.doc._identity, w.doc.revision)
                or result["configuration_key"] != w.doc.configuration_key
            ):
                self._failed("Discarded stale geometry; request another preview.")
                return
            counts = result["fragment_counts"]
            for label in required_fragment_labels(w.doc):
                if not counts[label]:
                    self._failed(
                        f"The cuts leave {w.doc.fragment_names[label]} empty. Adjust cut geometry."
                    )
                    return
            w.doc.artifacts.update(result.get("artifacts", {}))
            w.doc.fragments, w.doc.fragments_key = result["fragments"], result["configuration_key"]
            w._loaded_meshes[True] = result["surfaces"]
            w._anatomy_counts = result["anatomy_counts"]
            w._fragment_counts = result["fragment_counts"]
            w.pending_fragments = result["fragments"]
            w._pending_key = result["configuration_key"]
            w._fragment_meshes = result["surfaces"]
            self._capture()
        elif operation in ("open", "save"):
            self.active_job = None
            self.last_request = time.monotonic()
            record.update(status="completed", progress="Completed", state=self.state())
            if operation == "open":
                self.show_scene()
            self._banner()

    def show_scene(self):
        w = self.window
        if w.doc is None:
            return
        w.tabs.blockSignals(True)
        w.tabs.setCurrentIndex(3)
        w.tabs.blockSignals(False)
        mode = w.doc.fragments is not None
        meshes = w._loaded_meshes.get(mode, {})
        w._display(meshes, mode, w.doc.fragments if mode else w.doc.anatomy)
        w.views.set_tool("navigate")
        w.views.set_plane([0, 0, 0], [0, 0, 1], False)
        w._update_transforms()

    def _capture(self):
        self.show_scene()
        record = self.jobs[self.active_job]
        names = (
            ["right", "left", "front", "birds-eye"]
            if record["view"] == "contact-sheet"
            else [record["view"]]
        )
        capture = Capture(
            self.window.doc,
            self.window.views,
            names,
            record["position"],
            512 if len(names) == 4 else 1024,
        )
        job_id = self.active_job
        QTimer.singleShot(0, lambda: self._capture_next(job_id, capture, 0))

    def _capture_next(self, job_id, capture, index):
        if self.active_job != job_id or not self.session:
            capture.close()
            return
        record = self.jobs[job_id]
        try:
            record["progress"] = (
                f"Capturing {capture.names[index]} ({index + 1}/{len(capture.names)})"
            )
            capture.render(capture.names[index])
            if self.active_job != job_id or not self.session:
                capture.close()
                return
            if index + 1 < len(capture.names):
                QTimer.singleShot(0, lambda: self._capture_next(job_id, capture, index + 1))
                return
            doc = self.window.doc
            preview_id = uuid4().hex
            self.previews[preview_id] = ((doc._identity, doc.revision), capture)
            while len(self.previews) > 4:
                _, old = self.previews.popitem(last=False)
                old[1].close()
            record.update(
                status="completed",
                progress="Completed",
                image=capture.result(),
                metadata={
                    "preview_id": preview_id,
                    "scenario": doc.active_scenario,
                    "document_id": doc._identity,
                    "revision": doc.revision,
                    "preparation_key": doc.configuration_key,
                    "orientation": dict(
                        zip(("pitch", "roll", "yaw"), doc.view_angles, strict=True)
                    ),
                    "width": 1024,
                    "height": 1024,
                    "view_size": capture.size,
                    "fragment_colours": {
                        str(label): list(COLORS[label]) for label in self.window.views.actors
                    },
                    "review_state": "accepted" if doc.cuts_accepted else "unreviewed",
                    "views": capture.names,
                    "position": record["position"],
                    "conventions": CONVENTIONS,
                },
            )
            self.active_job = None
            self.last_request = time.monotonic()
        except Exception as error:
            capture.close()
            self._failed(str(error))

    def _failed(self, error):
        if self.active_job:
            self.jobs[self.active_job].update(
                status="failed", error={"code": "job_failed", "message": error}
            )
            self.active_job = None
            self.last_request = time.monotonic()

    def _progress(self, message):
        if self.active_job:
            self.jobs[self.active_job]["progress"] = message

    def _busy(self, busy):
        if not busy and self.window.jobs._cancel.is_set():
            self._cancel()

    def _cancel(self):
        if self.active_job:
            self.jobs[self.active_job].update(status="cancelled", progress="Cancelled")
            self.active_job = None
            self.last_request = time.monotonic()
        if self.window.jobs.busy:
            self.window.jobs.cancel()

    def _banner(self):
        self.window.ai_banner.setText(
            f"AI editing · {self.window.doc.active_scenario if self.window.doc else 'No project'}"
            if self.session
            else ""
        )
        self.window.take_control_button.setVisible(bool(self.session))

    def release(self):
        if not self.session:
            return
        self._cancel()
        self.session, self.owner = None, None
        self._banner()
        w = self.window
        if w.doc and not w.doc.cuts_accepted:
            w.doc.fragments = None
            w.doc.fragments_key = None
            w._loaded_meshes.pop(True, None)
            w.views.invalidate_surfaces(True)
        w._sync_enabled()
        if w.doc and not w.jobs.busy:
            w._stage_changed(w.tabs.currentIndex())

    def expire(self):
        if (
            self.session
            and not self.active_job
            and not self.window.jobs.busy
            and time.monotonic() - self.last_request >= 300
        ):
            self.release()

    def close(self):
        self.release()
        self.timer.stop()
        for _, capture in self.previews.values():
            capture.close()
        self.previews.clear()
        for peer in list(self.buffers):
            peer.abort()
        self.server.close()
        self.path.unlink(missing_ok=True)
