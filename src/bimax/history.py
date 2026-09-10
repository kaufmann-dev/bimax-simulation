"""Bounded sparse document edits and small metadata snapshots."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Edit:
    name: str
    patches: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]
    before: dict[str, Any]
    after: dict[str, Any]

    @property
    def size(self):
        return sum(i.nbytes + b.nbytes + a.nbytes for _, i, b, a in self.patches)


class History:
    def __init__(self):
        self.undo_stack: list[Edit] = []
        self.redo_stack: list[Edit] = []

    def push(self, edit):
        if not edit.patches and not edit.before and not edit.after:
            return
        self.undo_stack.append(edit)
        self.redo_stack.clear()
        while len(self.undo_stack) > 1 and sum(e.size for e in self.undo_stack) > 512 * 1024**2:
            self.undo_stack.pop(0)

    def apply(self, doc, forward):
        source = self.redo_stack if forward else self.undo_stack
        destination = self.undo_stack if forward else self.redo_stack
        if not source:
            return False
        edit = source.pop()
        for field, indices, before, after in edit.patches:
            getattr(doc, field).reshape(-1)[indices] = after if forward else before
        for field, value in (edit.after if forward else edit.before).items():
            setattr(doc, field, deepcopy(value))
        if edit.patches or "cuts" in edit.before or "frame" in edit.before:
            doc.fragments = None
            doc.confirmations = {}
        if doc.active_scenario not in doc.scenarios:
            doc.active_scenario = next(iter(doc.scenarios))
        doc.revision += 1
        destination.append(edit)
        return True
