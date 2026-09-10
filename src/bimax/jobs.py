"""One cancellable computation process; disk transport keeps the UI responsive."""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psutil
from PySide6.QtCore import QObject, Signal


class Jobs(QObject):
    progress = Signal(str)
    finished = Signal(str, int, object)
    failed = Signal(str)
    busyChanged = Signal(bool)
    _done = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.busy = False
        self._cancel = threading.Event()
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._done.connect(self._on_done)

    def submit(self, operation: str, document: Any = None, **kwargs: Any) -> None:
        if self.busy:
            raise RuntimeError("A computation is already running.")
        self.busy = True
        self._cancel.clear()
        self.busyChanged.emit(True)
        revision = document.revision if document is not None else -1
        self._thread = threading.Thread(
            target=self._run, args=(operation, revision, document, kwargs), daemon=True
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def close(self) -> None:
        self.cancel()
        if self._thread:
            self._thread.join(timeout=5)
        if self._process and self._process.poll() is None:
            self._process.kill()

    def _on_done(self) -> None:
        self.busy = False
        self.busyChanged.emit(False)

    def _run(self, operation: str, revision: int, doc: Any, kwargs: dict) -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="bimax-job-") as directory:
                path = Path(directory)
                self.progress.emit("Preparing computation…")
                with (path / "input.pickle").open("wb") as stream:
                    pickle.dump((operation, doc, kwargs), stream, protocol=5)
                if self._cancel.is_set():
                    return
                environment = dict(os.environ)
                environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="6")
                with (path / "log.jsonl").open("w") as log:
                    self._process = subprocess.Popen(
                        [sys.executable, "-m", "bimax.worker", directory],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                    )
                    process = self._process
                    monitor = psutil.Process(process.pid)
                    log_position = 0
                    while process.poll() is None:
                        if self._cancel.is_set():
                            process.terminate()
                            try:
                                process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            return
                        try:
                            memory = monitor.memory_info().rss
                            if memory > 16 * 1024**3:
                                process.kill()
                                process.wait()
                                raise MemoryError(
                                    "Computation exceeded 16 GiB. Select a smaller editing region."
                                )
                        except psutil.NoSuchProcess:
                            pass
                        with (path / "log.jsonl").open() as progress_log:
                            progress_log.seek(log_position)
                            for line in progress_log:
                                if line.startswith("BIMAX_PROGRESS "):
                                    try:
                                        self.progress.emit(json.loads(line[15:])["message"])
                                    except (ValueError, KeyError):
                                        pass
                            log_position = progress_log.tell()
                        time.sleep(0.1)
                if self._cancel.is_set():
                    return
                result_path = path / "result.pickle"
                if not result_path.exists():
                    lines = (path / "log.jsonl").read_text(errors="replace").strip()
                    raise RuntimeError(
                        lines[-1800:] or "The computation process stopped unexpectedly."
                    )
                with result_path.open("rb") as stream:
                    result = pickle.load(stream)  # private, locally generated worker result
                if "error" in result:
                    raise RuntimeError(result["error"])
                self.finished.emit(operation, revision, result)
        except Exception as error:
            if not self._cancel.is_set():
                self.failed.emit(str(error))
        finally:
            self._process = None
            if self._cancel.is_set():
                self.progress.emit("Computation cancelled.")
            self._done.emit()
