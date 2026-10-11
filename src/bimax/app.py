"""Native desktop workflow for binary-volume jaw simulation."""

import argparse
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCursor, QIcon, QKeySequence, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .history import Edit, History
from .jobs import Jobs
from .model import ANATOMY, FRAGMENTS, Document, new_scenario

STYLE = """
QWidget { background: #000000; color: #e6e6e6; font-size: 13px; }
QPushButton, QComboBox, QDoubleSpinBox, QSpinBox { background: #202020; padding: 5px; min-height: 24px; border: 1px solid #454545; }
QPushButton:hover { background: #303030; }
QPushButton:checked { background: #16445b; }
QWidget:disabled { color: #757575; }
QPushButton:focus, QComboBox:focus, QAbstractSpinBox:focus { border: 1px solid #71ceff; }
QTabBar::tab { padding: 10px 14px; background: #181818; }
QTabBar::tab:selected { background: #303030; }
QTabWidget::pane { border: 0; }
QGroupBox { border: 1px solid #303030; margin-top: 16px; padding-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QToolBar { spacing: 6px; border-bottom: 1px solid #303030; }
QLabel { background: transparent; }
QToolTip { color: #ffffff; background: #252525; border: 1px solid #777777; }
"""


def button(text, action, layout):
    result = QPushButton(text)
    result.clicked.connect(action)
    layout.addWidget(result)
    return result


def note(text, layout):
    result = QLabel(text)
    result.setWordWrap(True)
    result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(result)
    return result


def form_layout(parent=None):
    form = QFormLayout(parent)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


def vector_inputs(form, label, callback, minimum=-10000, maximum=10000, decimals=2):
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    widgets = []
    for axis in ("R", "A", "S"):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(1 if decimals == 2 else 0.1)
        # QDoubleSpinBox sizes itself for the widest value in its entire range.
        # Let the three coordinate editors share the sidebar width instead.
        spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        spin.setMinimumWidth(72)
        spin.setAccessibleName(f"{label}, {axis}")
        spin.setToolTip(f"{label}: {axis} coordinate")
        spin.editingFinished.connect(callback)
        layout.addWidget(spin)
        widgets.append(spin)
    heading = QLabel(label)
    heading.setWordWrap(True)
    form.addRow(heading)
    form.addRow(row)
    return widgets


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from .views import FourViews

        self.setWindowTitle("BIMAX")
        self.resize(1480, 960)
        self.doc: Document | None = None
        self.project_path = None
        self.dirty = False
        self.history = History()
        self.jobs = Jobs(self)
        self.jobs.finished.connect(self._job_finished)
        self.jobs.failed.connect(self._error)
        self.jobs.progress.connect(self._status)
        self.jobs.busyChanged.connect(self._busy_changed)
        self.loading = False
        self.pending_separation = None
        self.pending_fragments = None
        self._separation_meshes = None
        self._separation_counts = None
        self._fragment_meshes = None
        self._fragment_counts = None
        self._pending_key = None
        self._plan_cache = OrderedDict()
        self.pick_mode = None
        self.last_point = None
        self.selected_fragment = 1
        self._after_save = None
        self._close_after_cancel = False
        self._loaded_meshes = {}
        self._anatomy_counts = None
        self._remesh_timer = QTimer(self)
        self._remesh_timer.setSingleShot(True)
        self._remesh_timer.timeout.connect(self._refresh_surfaces)
        self.views = FourViews()
        self.views.pointPicked.connect(self._point_picked)
        self.views.paintRequested.connect(self._paint)
        self.views.selectedFragment.connect(self._fragment_selected)
        self.views.planeChanged.connect(self._plane_dragged)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(350)
        self.tabs.setMaximumWidth(460)
        self._build_import()
        self._build_separate()
        self._build_cut()
        self._build_move()
        for combo in (self.cut_plan_combo, self.move_plan_combo, self.cut_scenario_combo, self.scenario_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tabs.currentChanged.connect(self._stage_changed)
        split = QSplitter()
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(self.tabs, 1)
        self.structure_box = QGroupBox("Visible structures")
        self.structure_layout = QGridLayout(self.structure_box)
        sidebar_layout.addWidget(self.structure_box)
        sidebar.setMaximumWidth(460)
        split.addWidget(sidebar)
        split.addWidget(self.views)
        split.setSizes([375, 1100])
        self.setCentralWidget(split)
        self._toolbar()
        self._status("Open a binary bone volume to begin.")
        self._sync_enabled()

    def _tab(self, name):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, name)
        return layout

    def _toolbar(self):
        toolbar = self.addToolBar("Project")
        toolbar.setMovable(False)
        self.toolbar_actions = {}
        for name, shortcut, callback in (
            ("Import volume", "Ctrl+I", self.import_volume),
            ("Open project", "Ctrl+O", self.open_project),
            ("Save", "Ctrl+S", self.save_project),
            ("Save as", "Ctrl+Shift+S", lambda: self.save_project(True)),
            ("Undo", "Ctrl+Z", lambda: self._undo(False)),
            ("Redo", "Ctrl+Shift+Z", lambda: self._undo(True)),
            ("Right profile", "1", self.views.profile_view),
        ):
            action = QAction(name, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            toolbar.addAction(action)
            self.toolbar_actions[name] = action
        self.expand_3d_action = QAction("Expand 3D", self)
        self.expand_3d_action.setCheckable(True)
        self.expand_3d_action.setShortcut(QKeySequence("F11"))
        self.expand_3d_action.setToolTip(
            "Expand the 3D workspace (F11). Keep the sidebar available."
        )
        self.expand_3d_action.toggled.connect(self._expand_3d)
        toolbar.addAction(self.expand_3d_action)
        self.restore_views_action = QAction("Restore four views", self)
        self.restore_views_action.setShortcut(QKeySequence("Esc"))
        self.restore_views_action.setEnabled(False)
        self.restore_views_action.triggered.connect(lambda: self.expand_3d_action.setChecked(False))
        self.addAction(self.restore_views_action)
        self.cancel_button = QPushButton("Cancel computation")
        self.cancel_button.clicked.connect(self.jobs.cancel)
        toolbar.addWidget(self.cancel_button)
        self.cancel_button.hide()

    def _expand_3d(self, expanded):
        self.views.set_3d_expanded(expanded)
        self.expand_3d_action.setText("Show four views" if expanded else "Expand 3D")
        self.expand_3d_action.setToolTip(
            "Restore all four views (F11 or Escape)."
            if expanded
            else "Expand the 3D workspace (F11). Keep the sidebar available."
        )
        self.restore_views_action.setEnabled(expanded)

    def _build_import(self):
        layout = self._tab("Import")
        note("BIMAX", layout).setStyleSheet("font-size: 24px; font-weight: 600;")
        note("Explore bone movement using an unsegmented binary bone volume.", layout)
        button("Import binary NIfTI…", self.import_volume, layout)
        button("Open BIMAX project…", self.open_project, layout)
        self.info_label = note(
            "Input: one 3D .nii or .nii.gz file with values 0 and 1. Prepare the bone volume with medsurface before import.",
            layout,
        )
        group = QGroupBox("Head reference frame")
        form = form_layout(group)
        self.frame_spins = vector_inputs(
            form, "Pitch / roll / yaw °", self._frame_changed, -180, 180
        )
        layout.addWidget(group)
        note(
            "The reference starts in patient Right / Anterior / Superior coordinates. Adjust it before placing cuts and movement pivots.",
            layout,
        )
        button("Continue to separation", lambda: self.tabs.setCurrentIndex(1), layout)
        layout.addStretch()
        note(
            "Personal geometric simulation. Cut shapes and movement do not predict fracture, bite or soft-tissue behavior.",
            layout,
        )

    def _build_separate(self):
        layout = self._tab("Separate")
        note("Separate the anatomy", layout).setStyleSheet("font-size: 18px;")
        note(
            "Mark the skull with upper teeth, and the mandible with lower teeth. Preview separation, then correct fused contacts in the slice views.",
            layout,
        )
        form = form_layout()
        self.label_combo = QComboBox()
        for value, label in ANATOMY.items():
            self.label_combo.addItem(label, value)
        self.label_combo.setCurrentIndex(1)
        self.label_combo.currentIndexChanged.connect(
            lambda: self.views.set_label(self.label_combo.currentData())
        )
        form.addRow("Assign to", self.label_combo)
        self.tool_combo = QComboBox()
        for label, tool_key in (
            ("Navigate / pick", "navigate"),
            ("Paint seeds", "seed"),
            ("Paint assignments", "paint"),
            ("Erase assignments", "erase"),
            ("Separation barrier", "barrier"),
            ("Remove barrier", "unbarrier"),
            ("Remove seeds", "unseed"),
            ("Polygon assignment", "polygon"),
            ("Assign connected piece", "component"),
        ):
            self.tool_combo.addItem(label, tool_key)
        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        form.addRow("Tool", self.tool_combo)
        self.brush_spin = QDoubleSpinBox()
        self.brush_spin.setRange(0.4, 40)
        self.brush_spin.setValue(3)
        self.brush_spin.setSuffix(" mm")
        self.brush_spin.valueChanged.connect(self.views.set_brush_mm)
        form.addRow("Brush diameter", self.brush_spin)
        layout.addLayout(form)
        note(
            "Paint on slices. For polygons, click vertices and double-click to finish. Seeds guide a proposal; assignment painting changes labels directly.",
            layout,
        )
        group = QGroupBox("Editing region (voxel XYZ, end exclusive)")
        roi_layout = QVBoxLayout(group)
        self.roi_check = QCheckBox("Limit separation to this region")
        roi_layout.addWidget(self.roi_check)
        roi_form = form_layout()
        self.roi_start = vector_inputs(roi_form, "Start XYZ", lambda: None, 0, 100000, 0)
        self.roi_end = vector_inputs(roi_form, "End XYZ", lambda: None, 0, 100000, 0)
        roi_layout.addLayout(roi_form)
        button("Center 80 mm region at last pick", self._center_roi, roi_layout)
        layout.addWidget(group)
        self.preview_separation_button = button("Preview separation", self._separate, layout)
        self.accept_separation_button = button("Accept separation", self._accept_separation, layout)
        self.discard_separation_button = button("Discard proposal", self._discard_proposal, layout)
        button("Refresh 3D surfaces", self._refresh_surfaces, layout)
        self.assignment_label = note("", layout)
        button("Continue to cuts", lambda: self.tabs.setCurrentIndex(2), layout)
        layout.addStretch()

    def _build_cut(self):
        layout = self._tab("Cut")
        note("Define bone fragments", layout).setStyleSheet("font-size: 18px;")
        note(
            "Adjust geometric cuts in the unmoved anatomy. BSSO uses a stepped split; its two proximal pieces remain fixed.",
            layout,
        )
        selectors = form_layout()
        self.cut_scenario_combo = QComboBox()
        self.cut_scenario_combo.currentTextChanged.connect(self._scenario_changed)
        selectors.addRow("Scenario", self.cut_scenario_combo)
        self.cut_plan_combo = QComboBox()
        self.cut_plan_combo.currentIndexChanged.connect(
            lambda: self._plan_selected(self.cut_plan_combo.currentData())
        )
        selectors.addRow("Scenario's cut plan", self.cut_plan_combo)
        layout.addLayout(selectors)
        button("Create cut plan and scenario…", self._create_plan, layout)
        self.plan_note = note(
            "A cut plan is shared by its linked scenarios. Duplicate it to edit an independent alternative.",
            layout,
        )
        form = form_layout()
        self.operation_combo = QComboBox()
        for name, key in (("Le Fort I", "lefort"), ("BSSO", "bsso"), ("Genioplasty", "genio")):
            self.operation_combo.addItem(name, key)
        self.operation_combo.currentIndexChanged.connect(self._load_cut_controls)
        form.addRow("Operation", self.operation_combo)
        self.cut_enabled = QCheckBox("Enable operation")
        self.cut_enabled.toggled.connect(self._cut_toggled)
        form.addRow(self.cut_enabled)
        self.plane_combo = QComboBox()
        self.plane_combo.currentIndexChanged.connect(self._load_plane_controls)
        form.addRow("Cut plane", self.plane_combo)
        self.plane_origin = vector_inputs(form, "Origin R/A/S mm", self._plane_controls_changed)
        self.plane_normal = vector_inputs(
            form, "Normal R/A/S", self._plane_controls_changed, -1, 1, 4
        )
        self.region_min = vector_inputs(form, "Region minimum", self._region_changed)
        self.region_max = vector_inputs(form, "Region maximum", self._region_changed)
        self.midline = QDoubleSpinBox()
        self.midline.setRange(-10000, 10000)
        self.midline.setDecimals(2)
        self.midline.editingFinished.connect(self._region_changed)
        form.addRow("BSSO midline R", self.midline)
        layout.addLayout(form)
        button("Place plane origin at next pick", lambda: self._arm_pick("plane"), layout)
        button("Set region minimum at next pick", lambda: self._arm_pick("region_min"), layout)
        button("Set region maximum at next pick", lambda: self._arm_pick("region_max"), layout)
        self.cut_preview_button = button(
            "Preview fragments", lambda: self._build_cuts(False), layout
        )
        confirm_form = form_layout()
        self.confirm_combo = QComboBox()
        for value in (2, 3, 4, 5, 6):
            self.confirm_combo.addItem(FRAGMENTS[value], value)
        confirm_form.addRow("Confirm piece", self.confirm_combo)
        layout.addLayout(confirm_form)
        button("Identify piece at next pick", lambda: self._arm_pick("confirm"), layout)
        self.confirm_label = note(
            "Confirm the enabled fragments by picking a point inside each one.", layout
        )
        self.accept_cuts_button = button("Accept cuts", lambda: self._build_cuts(True), layout)
        button("Continue to movement", lambda: self.tabs.setCurrentIndex(3), layout)
        layout.addStretch()

    def _build_move(self):
        layout = self._tab("Move")
        note("Move the jaws", layout).setStyleSheet("font-size: 18px;")
        form = form_layout()
        self.scenario_combo = QComboBox()
        self.scenario_combo.currentTextChanged.connect(self._scenario_changed)
        form.addRow("Scenario", self.scenario_combo)
        self.move_plan_combo = QComboBox()
        self.move_plan_combo.currentIndexChanged.connect(
            lambda: self._plan_selected(self.move_plan_combo.currentData())
        )
        form.addRow("Scenario's cut plan", self.move_plan_combo)
        self.move_plan_note = QLabel()
        self.move_plan_note.setWordWrap(True)
        form.addRow(self.move_plan_note)
        self.motion_combo = QComboBox()
        for name, key in (
            ("Linked jaws", "linked"),
            ("Maxilla", "maxilla"),
            ("Distal mandible", "mandible"),
            ("Chin", "chin"),
        ):
            self.motion_combo.addItem(name, key)
        self.motion_combo.currentIndexChanged.connect(self._load_motion)
        form.addRow("Move", self.motion_combo)
        self.motion_spins = []
        for label in (
            "Right (+) mm",
            "Advance (+) mm",
            "Downgraft (+) mm",
            "CCW pitch (+) °",
            "Roll °",
            "Yaw °",
        ):
            spin = QDoubleSpinBox()
            spin.setRange(-180, 180)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setAccessibleName(label)
            spin.valueChanged.connect(self._motion_changed)
            spin.editingFinished.connect(self._finish_motion_edit)
            self.motion_spins.append(spin)
            form.addRow(label, spin)
        layout.addLayout(form)
        self._motion_before = None
        note(
            "Positive pitch is counterclockwise in the right-profile view (anterior to screen-right). Place a pivot before rotating.",
            layout,
        )
        self.pivot_label = note("No pivot selected.", layout)
        button("Place pivot at next pick", lambda: self._arm_pick("pivot"), layout)
        self.reset_pivot_button = button("Reset pivot", self._reset_pivot, layout)
        self.reset_pivot_button.setToolTip(
            "Clear the selected control's pivot and rotation; keep its translation. "
            "Also cancel a pending pivot pick."
        )
        button("Use this pivot for all jaw controls", self._share_pivot, layout)
        button("Reset selected movement", self._reset_motion, layout)
        button("Duplicate scenario…", self._duplicate_scenario, layout)
        self.baseline_check = QCheckBox("Show baseline overlay")
        self.baseline_check.toggled.connect(self.views.show_baseline)
        layout.addWidget(self.baseline_check)
        button("Right-profile view", self.views.profile_view, layout)
        button("Add measurement landmark at next pick", lambda: self._arm_pick("landmark"), layout)
        self.landmark_label = note("Landmarks show displacement from the baseline.", layout)
        button("Save project…", self.save_project, layout)
        self.image_export_button = button("Export scenario images…", self._export_images, layout)
        self.export_button = button("Export fragment labelmaps…", self._export, layout)
        layout.addStretch()

    def _status(self, message):
        self.statusBar().showMessage(message)

    def _error(self, message):
        self._after_save = None
        self._status(message)
        QMessageBox.warning(self, "BIMAX", message)

    def _busy_changed(self, busy):
        self.cancel_button.setVisible(busy)
        self.tabs.setEnabled(not busy)
        if busy:
            self._remesh_timer.stop()
        if not busy and self.jobs._cancel.is_set():
            self._after_save = None
        self._sync_enabled()
        if not busy and self._close_after_cancel:
            self._close_after_cancel = False
            QTimer.singleShot(0, self.close)

    def _display(self, meshes, fragment_mode, labels=None):
        self.views.set_surfaces(meshes, fragment_mode=fragment_mode)
        if labels is not None:
            self.views.refresh_slices(labels)
        while self.structure_layout.count():
            item = self.structure_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        names = self.doc.fragment_names if fragment_mode and self.doc else ANATOMY
        for row, value in enumerate(sorted(meshes)):
            check = QCheckBox(names.get(value, str(value)))
            check.setChecked(self.views.is_visible(value))
            check.toggled.connect(
                lambda visible, label=value: self.views.set_visibility(label, visible)
            )
            self.structure_layout.addWidget(check, row // 2, row % 2)

    def _sync_enabled(self):
        has_doc = self.doc is not None
        busy = self.jobs.busy
        for name, action in (
            self.toolbar_actions.items() if hasattr(self, "toolbar_actions") else []
        ):
            action.setEnabled(not busy and (has_doc or name in ("Import volume", "Open project")))
        for index in range(1, 4):
            self.tabs.setTabEnabled(index, has_doc)
        if hasattr(self, "accept_separation_button"):
            self.accept_separation_button.setEnabled(self.pending_separation is not None)
            self.discard_separation_button.setEnabled(self.pending_separation is not None)
        if hasattr(self, "export_button"):
            can_export = self.doc is not None and self.doc.fragments is not None and not busy
            self.export_button.setEnabled(can_export)
            self.image_export_button.setEnabled(can_export)
            self.accept_cuts_button.setEnabled(self.pending_fragments is not None)
            can_move = self.doc is not None and self.doc.fragments is not None and not busy
            self.motion_combo.setEnabled(can_move)
            if self.doc is not None:
                chin_enabled = self.doc.plan.cuts.get("genio", {}).get("enabled", False)
                motion_model = self.motion_combo.model()
                if isinstance(motion_model, QStandardItemModel):
                    motion_model.item(self.motion_combo.findData("chin")).setEnabled(chin_enabled)
                if not chin_enabled and self.motion_combo.currentData() == "chin":
                    self.motion_combo.setCurrentIndex(0)
            motion = self.doc.motions[self.motion_combo.currentData()] if self.doc else None
            self.reset_pivot_button.setEnabled(
                not busy
                and motion is not None
                and (
                    motion.pivot_set
                    or any(motion.pivot)
                    or any(motion.angles)
                    or self.pick_mode == "pivot"
                )
            )
            for i, spin in enumerate(self.motion_spins):
                spin.setEnabled(
                    can_move
                    and (
                        i < 3
                        or (
                            self.doc is not None
                            and self.doc.motions[self.motion_combo.currentData()].pivot_set
                        )
                    )
                )

    def _mark_dirty(self):
        if self.doc:
            self.doc.revision += 1
        self.dirty = True
        self.setWindowTitle(f"BIMAX — {self.doc.source_name if self.doc else ''} *")

    def _invalidate(self, all_plans=False):
        if self.doc:
            self.doc.fragments = None
            self.doc.fragments_key = None
            affected = self.doc.plans.values() if all_plans else [self.doc.plan]
            for plan in affected:
                plan.confirmations = {}
                plan.accepted = False
                plan.revision += 1
            for scenario in self.doc.scenarios.values():
                if all_plans or scenario.plan_id == self.doc.scenario.plan_id:
                    scenario.landmarks = []
            if all_plans:
                self.doc.anatomy_revision += 1
                self._plan_cache.clear()
            else:
                for key in list(self._plan_cache):
                    if key[0] == self.doc.scenario.plan_id:
                        del self._plan_cache[key]
        self.pending_fragments = None
        self.pending_separation = None
        self._separation_meshes = self._separation_counts = None
        self._fragment_meshes = self._fragment_counts = None
        self._pending_key = None
        self._loaded_meshes.pop(True, None)
        self.views.invalidate_surfaces(True)
        self.views.set_transforms({})
        if self.doc is not None:
            self._display(self._loaded_meshes.get(False, {}), False, self.doc.anatomy)
        self.confirm_label.setText(
            "Confirm the enabled fragments by picking a point inside each one."
        )
        self._sync_enabled()
        self._load_plan_selectors()

    def _cache_active_plan(self):
        """Keep at most two prepared alternatives, within a 768 MiB label/mesh budget."""
        if self.doc is None or self.doc.fragments is None or True not in self._loaded_meshes:
            return
        key = self.doc.configuration_key
        if self.doc.fragments_key != key or not self.doc.plan.accepted:
            return
        self._plan_cache[key] = (self.doc.fragments, self._loaded_meshes[True])
        self._plan_cache.move_to_end(key)

        def size(entry):
            labels, meshes = entry
            return labels.nbytes + sum(
                mesh.vertices.nbytes + mesh.faces.nbytes + mesh.normals.nbytes
                for mesh in meshes.values()
            )

        while self._plan_cache and (
            len(self._plan_cache) > 2
            or sum(size(entry) for entry in self._plan_cache.values()) > 768 * 1024**2
        ):
            self._plan_cache.popitem(last=False)

    def _activate_plan(self):
        """Swap only derived geometry; every scenario keeps its own movements and landmarks."""
        if self.doc is None:
            return
        self.pending_fragments = self.pending_separation = None
        self._fragment_meshes = self._fragment_counts = None
        self._separation_meshes = self._separation_counts = None
        self._pending_key = None
        self.pick_mode = None
        key = self.doc.configuration_key
        if self.doc.fragments_key != key:
            self.doc.fragments = None
            self.doc.fragments_key = None
            self._loaded_meshes.pop(True, None)
            self.views.invalidate_surfaces(True)
        if self.doc.plan.accepted and self.doc.fragments is None and key in self._plan_cache:
            self.doc.fragments, self._loaded_meshes[True] = self._plan_cache[key]
            self.doc.fragments_key = key
            self._plan_cache.move_to_end(key)
        self._load_plan_selectors()
        self._load_cut_controls()
        self._load_motion()
        self._stage_changed(self.tabs.currentIndex())

    def _load_plan_selectors(self):
        if self.doc is None:
            return
        previous, self.loading = self.loading, True
        for combo in (self.scenario_combo, self.cut_scenario_combo):
            combo.clear()
            combo.addItems(list(self.doc.scenarios))
            combo.setCurrentText(self.doc.active_scenario)
            combo.setToolTip(combo.currentText())
        for combo in (self.cut_plan_combo, self.move_plan_combo):
            combo.clear()
            for plan_id, plan in self.doc.plans.items():
                combo.addItem(plan.name, plan_id)
            combo.setCurrentIndex(combo.findData(self.doc.scenario.plan_id))
            combo.setToolTip(combo.currentText())
        plan = self.doc.plan
        level = {1: "I", 2: "II", 3: "III"}[plan.lefort_level]
        operations = [f"Le Fort {level}"]
        if plan.cuts.get("bsso", {}).get("enabled"):
            operations.append("BSSO")
        operations.append(
            "with genioplasty"
            if plan.cuts.get("genio", {}).get("enabled")
            else "without genioplasty"
        )
        summary = " · ".join(operations)
        self.plan_note.setText(
            summary
            + ". Edits affect every scenario linked to this plan. These are geometric approximations, not surgical cut guides."
        )
        self.move_plan_note.setText(
            summary + (" · cuts accepted" if plan.accepted else " · preview and accept cuts first")
        )
        self.operation_combo.setItemText(
            self.operation_combo.findData("lefort"), f"Le Fort {level}"
        )
        self.motion_combo.setItemText(
            self.motion_combo.findData("maxilla"), self.doc.fragment_names[2]
        )
        for i in range(self.confirm_combo.count()):
            self.confirm_combo.setItemText(
                i, self.doc.fragment_names[self.confirm_combo.itemData(i)]
            )
        self.loading = previous

    def _plan_selected(self, plan_id):
        if (
            self.loading
            or self.doc is None
            or self.jobs.busy
            or plan_id not in self.doc.plans
            or plan_id == self.doc.scenario.plan_id
        ):
            return
        self._finish_motion_edit()
        before = deepcopy(self.doc.scenarios)
        self.doc.scenario.plan_id = plan_id
        # A landmark's fragment membership is specific to its cut plan.
        self.doc.scenario.landmarks = []
        self._remember(
            "Select cut plan", {"scenarios": before}, {"scenarios": deepcopy(self.doc.scenarios)}
        )
        self._activate_plan()

    def _create_plan(self):
        if self.doc is None or self.jobs.busy:
            return
        operation, ok = QInputDialog.getItem(
            self,
            "New cut plan",
            "Le Fort operation",
            ["Le Fort I", "Le Fort II", "Le Fort III"],
            self.doc.plan.lefort_level - 1,
            False,
        )
        if not ok:
            return
        name, ok = QInputDialog.getText(
            self, "New cut plan", "New plan and scenario name", text=f"{operation} alternative"
        )
        name = name.strip()
        if not ok or not name:
            return
        if name in self.doc.scenarios or any(plan.name == name for plan in self.doc.plans.values()):
            self._error("Choose a unique plan and scenario name.")
            return
        self._finish_motion_edit()
        self.jobs.submit(
            "create_plan",
            self.doc,
            name=name,
            plan_id=uuid4().hex,
            level=["Le Fort I", "Le Fort II", "Le Fort III"].index(operation) + 1,
        )

    def _remember(self, name, before, after, patches=None):
        self.history.push(Edit(name, patches or [], before, after))
        self._mark_dirty()

    def _confirm_replace(self):
        if not self.dirty:
            return True
        result = QMessageBox.question(
            self,
            "Unsaved project",
            "Discard unsaved changes?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return result == QMessageBox.StandardButton.Discard

    def import_volume(self):
        if self.jobs.busy or not self._confirm_replace():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import binary bone volume", "", "Binary NIfTI (*.nii *.nii.gz)"
        )
        if path:
            self.load_path(path)

    def open_project(self):
        if self.jobs.busy or not self._confirm_replace():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "BIMAX project (*.bimax)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        operation = "open" if str(path).endswith(".bimax") else "import"
        self.jobs.submit(operation, path=str(path))
        self._status("Loading volume and extracting display surfaces…")

    def save_project(self, save_as=False):
        if self.doc is None or self.jobs.busy:
            return
        path = self.project_path
        if not path or save_as:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save project", path or "simulation.bimax", "BIMAX project (*.bimax)"
            )
        if not path:
            self._after_save = None
            return
        if not path.endswith(".bimax"):
            path += ".bimax"
        self.jobs.submit("save", self.doc, path=path)
        self._status("Saving portable project…")

    def _export(self):
        if self.doc is None or self.doc.fragments is None:
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent folder for labelmap export")
        if not parent:
            return
        name, ok = QInputDialog.getText(
            self, "Export folder", "New folder name", text="bimax-labelmaps"
        )
        if not ok or not name:
            return
        if Path(name).name != name or name in (".", ".."):
            self._error("Enter a folder name without path separators.")
            return
        self.jobs.submit("export", self.doc, path=str(Path(parent) / name))
        self._status("Resampling positioned fragments to a shared output grid…")

    def _export_images(self):
        if self.doc is None or self.doc.fragments is None or self.jobs.busy:
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent folder for scenario images")
        if not parent:
            return
        name, ok = QInputDialog.getText(
            self, "Export scenario images", "New folder name", text="bimax-images"
        )
        if not ok or not name:
            return
        if Path(name).name != name or name in (".", ".."):
            self._error("Enter a folder name without path separators.")
            return
        self._status("Rendering standardized scenario views…")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            exported = self.views.export_scenario_images(Path(parent) / name)
        except Exception as error:
            failure = str(error) or "Could not export scenario images."
        else:
            failure = ""
        finally:
            QApplication.restoreOverrideCursor()
        if failure:
            self._error(failure)
        else:
            self._status(f"Exported {len(exported)} scenario images to {exported[0].parent}.")

    def _job_finished(self, operation, revision, result):
        if operation not in ("import", "open") and self.doc is None:
            return
        if operation not in ("import", "open") and self.doc and revision != self.doc.revision:
            self._status("Discarded an outdated computation result.")
            return
        if (
            self.doc
            and "configuration_key" in result
            and tuple(result["configuration_key"]) != self.doc.configuration_key
        ):
            self._status("Discarded a result for a different cut plan.")
            return
        if operation in ("import", "open"):
            self.doc = result["document"]
            assert self.doc is not None
            self.doc.source.setflags(write=False)
            self.project_path = result.get("path") if operation == "open" else None
            self._loaded_meshes = result["surface_sets"]
            self._plan_cache.clear()
            self._cache_active_plan()
            self._anatomy_counts = result["anatomy_counts"]
            self.history = History()
            self.pending_separation = self.pending_fragments = None
            self._separation_meshes = self._separation_counts = None
            self._fragment_meshes = self._fragment_counts = None
            self._pending_key = None
            self.dirty = operation == "import"
            self.views.set_document(self.doc)
            self._display(self._loaded_meshes[result["fragment_mode"]], result["fragment_mode"])
            self.views.set_label(self.label_combo.currentData())
            self.views.set_brush_mm(self.brush_spin.value())
            self._load_document_controls()
            self.tabs.setCurrentIndex(
                1 if operation == "import" else (3 if self.doc.fragments is not None else 1)
            )
            self.setWindowTitle(f"BIMAX — {self.doc.source_name}" + (" *" if self.dirty else ""))
            self._status("Volume ready. Source voxels and physical geometry are preserved.")
        elif operation == "separate":
            self.pending_separation = result["proposal"]
            self._separation_meshes = result["surfaces"]
            self._separation_counts = result["anatomy_counts"]
            self._display(result["surfaces"], False, self.pending_separation)
            self._status("Separation proposal ready. Inspect the slices, then accept or discard.")
        elif operation == "cuts_preview":
            self.pending_fragments = result["fragments"]
            self._fragment_meshes = result["surfaces"]
            self._fragment_counts = result["fragment_counts"]
            self._pending_key = tuple(result["configuration_key"])
            self._display(result["surfaces"], True, self.pending_fragments)
            self.views.set_transforms({})
            self._status(
                "Fragment preview ready. Identify the enabled pieces, then accept the cuts."
            )
        elif operation == "prepare_plan":
            assert self.doc is not None
            self.doc.fragments = result["fragments"]
            self.doc.fragments_key = tuple(result["configuration_key"])
            self._loaded_meshes[True] = result["surfaces"]
            self._cache_active_plan()
            self._stage_changed(self.tabs.currentIndex())
            self._status(f"{self.doc.plan.name} ready.")
        elif operation == "create_plan":
            assert self.doc is not None
            before = {
                key: deepcopy(getattr(self.doc, key))
                for key in ("plans", "scenarios", "active_scenario")
            }
            self.doc.plans[result["plan_id"]] = result["plan"]
            name = result["plan"].name
            self.doc.scenarios[name] = new_scenario(result["plan_id"])
            self.doc.active_scenario = name
            self._remember(
                "Create cut plan", before, {key: deepcopy(getattr(self.doc, key)) for key in before}
            )
            self._activate_plan()
            self._status(
                "Independent cut plan created. Adjust and preview its cuts before movement."
            )
        elif operation == "surfaces":
            self._loaded_meshes[result["fragment_mode"]] = result["surfaces"]
            self._anatomy_counts = result["anatomy_counts"]
            self._assignment_counts()
            self._stage_changed(self.tabs.currentIndex())
            self._status("Display surfaces updated.")
        elif operation == "component":
            self._apply_array_edit(
                "Assign component", "anatomy", result["indices"], result["label"]
            )
        elif operation == "save":
            assert self.doc is not None
            self.project_path = result["path"]
            self.dirty = False
            self.setWindowTitle(f"BIMAX — {self.doc.source_name}")
            self._status(f"Project saved: {self.project_path}")
            if self._after_save:
                action, self._after_save = self._after_save, None
                QTimer.singleShot(0, action)
        elif operation == "export":
            self._status(f"Labelmaps exported: {result['path']}")
            QMessageBox.information(
                self,
                "Export complete",
                f"Separate binary fragments and manifest saved to:\n{Path(result['path']).parent}",
            )
        self._sync_enabled()

    def _load_document_controls(self):
        assert self.doc is not None
        g = self.doc.geometry
        self.info_label.setText(
            f"{self.doc.source_name}\n{g.shape[2]} × {g.shape[1]} × {g.shape[0]} voxels\nSpacing: {g.spacing[0]:.3f} × {g.spacing[1]:.3f} × {g.spacing[2]:.3f} mm\nSource mask is retained unchanged."
        )
        self.loading = True
        for spin, size in zip(self.roi_end, g.shape[::-1], strict=True):
            spin.setValue(size)
        for spin in self.roi_start:
            spin.setValue(0)
        self._load_plan_selectors()
        # Decode the orthonormal planning rotation relative to the initial LPS→RAS basis.
        from scipy.spatial.transform import Rotation

        rotation = self.doc.frame[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
        angles = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        for spin, value in zip(self.frame_spins, angles, strict=True):
            spin.setValue(float(value))
        self.loading = False
        self._load_cut_controls()
        self._load_motion()
        self._assignment_counts()
        self._update_transforms()

    def _assignment_counts(self):
        if self.doc is None:
            return
        if self._anatomy_counts is None:
            self.assignment_label.setText("Updating assignment counts in the background…")
            return
        counts = self._anatomy_counts
        self.assignment_label.setText(
            " · ".join(f"{ANATOMY[i]}: {counts[i]:,}" for i in range(1, 5))
        )

    def _tool_changed(self):
        tool = self.tool_combo.currentData()
        self.pick_mode = "component" if tool == "component" else None
        visual_tool = {"component": "pick", "unbarrier": "barrier", "unseed": "seed"}.get(
            tool, tool
        )
        self.views.set_tool(visual_tool)
        self.views.set_label(self.label_combo.currentData())

    def _paint(self, payload):
        if self.doc is None or self.jobs.busy or self.tabs.currentIndex() != 1:
            return
        indices = np.unique(payload["indices"])
        indices = indices[self.doc.source.reshape(-1)[indices] != 0]
        if not len(indices):
            return
        tool = self.tool_combo.currentData()
        label = self.label_combo.currentData()
        if tool in ("seed", "unseed"):
            if tool == "seed" and label == 1:
                self._status("Choose skull, mandible or excluded material for a seed.")
                return
            self._apply_array_edit(
                "Paint seeds", "seeds", indices, 0 if tool == "unseed" else label
            )
        elif tool in ("barrier", "unbarrier"):
            self._apply_array_edit("Edit barrier", "barriers", indices, int(tool == "barrier"))
        elif tool in ("paint", "erase", "polygon"):
            self._apply_array_edit(
                "Assign bone", "anatomy", indices, 1 if tool == "erase" else label
            )

    def _apply_array_edit(self, name, field, indices, values):
        assert self.doc is not None
        array = getattr(self.doc, field).reshape(-1)
        before = array[indices].copy()
        after = np.broadcast_to(values, before.shape).astype(array.dtype).copy()
        keep = before != after
        indices, before, after = indices[keep], before[keep], after[keep]
        if not len(indices):
            return
        metadata_before = self._plan_edit_snapshot() if field == "anatomy" else {}
        array[indices] = after
        if field == "anatomy" and self._anatomy_counts is not None:
            self._anatomy_counts += np.bincount(after, minlength=5) - np.bincount(
                before, minlength=5
            )
        if field == "anatomy":
            self._loaded_meshes.pop(False, None)
            self._invalidate(all_plans=True)
        else:
            # Editing separation hints does not change the accepted anatomy or
            # any operation plan. Only discard the now-stale pending proposals.
            self._activate_plan()
        self._remember(
            name,
            metadata_before,
            self._plan_edit_snapshot() if field == "anatomy" else {},
            [(field, indices, before, after)],
        )
        self.views.refresh_slices()
        self._assignment_counts()
        self._status("Edit applied. Refresh 3D surfaces when ready.")
        if field == "anatomy":
            self._remesh_timer.start(1200)

    def _roi(self):
        assert self.doc is not None
        if not self.roi_check.isChecked():
            return None
        start = [int(s.value()) for s in self.roi_start]
        end = [int(s.value()) for s in self.roi_end]
        if any(
            a >= b or b > size
            for a, b, size in zip(start, end, self.doc.geometry.shape[::-1], strict=True)
        ):
            raise ValueError("Editing region must be nonempty and inside the volume.")
        return tuple(slice(a, b) for a, b in zip(start[::-1], end[::-1], strict=True))

    def _center_roi(self):
        if self.doc is None or self.last_point is None:
            self._status("Pick a point in a view first.")
            return
        center = self.doc.geometry.world_to_index(self.last_point)
        extent = 40 / np.asarray(self.doc.geometry.spacing)
        for i in range(3):
            self.roi_start[i].setValue(max(0, int(center[i] - extent[i])))
            self.roi_end[i].setValue(
                min(self.doc.geometry.shape[2 - i], int(center[i] + extent[i] + 1))
            )
        self.roi_check.setChecked(True)

    def _separate(self):
        if self.doc is None:
            return
        try:
            self.jobs.submit("separate", self.doc, roi=self._roi())
            self._status("Separating seeded bone regions at native resolution…")
        except ValueError as error:
            self._error(str(error))

    def _accept_separation(self):
        assert self.doc is not None
        if self.pending_separation is None:
            return
        proposal, meshes, counts = (
            self.pending_separation,
            self._separation_meshes,
            self._separation_counts,
        )
        before = self.doc.anatomy
        metadata_before = self._plan_edit_snapshot()
        self.doc.anatomy = proposal
        self._loaded_meshes[False] = meshes
        self._invalidate(all_plans=True)
        self._anatomy_counts = counts
        self._remember(
            "Accept separation",
            {"anatomy": before, **metadata_before},
            {"anatomy": proposal, **self._plan_edit_snapshot()},
        )
        self._remesh_timer.stop()
        self._display(meshes, False)
        self._assignment_counts()
        self._sync_enabled()
        self._status(
            "Separation accepted. Resolve unassigned bone or mark it excluded before cutting."
        )

    def _discard_proposal(self):
        self.pending_separation = None
        self._separation_meshes = self._separation_counts = None
        self._stage_changed(self.tabs.currentIndex())
        self._sync_enabled()

    def _refresh_surfaces(self):
        if self.doc and not self.jobs.busy:
            self._remesh_timer.stop()
            mode = self.doc.fragments is not None and self.tabs.currentIndex() != 1
            if mode in self._loaded_meshes:
                self._stage_changed(self.tabs.currentIndex())
                return
            self.jobs.submit("surfaces", self.doc, anatomy=not mode)

    def _stage_changed(self, index):
        if self.doc is None:
            return
        self.pick_mode = None
        self.views.set_tool("navigate")
        self.views.set_plane([0, 0, 0], [0, 0, 1], False)
        if (
            self.doc.plan.accepted
            and self.doc.fragments is None
            and index != 1
            and not self.jobs.busy
        ):
            self.jobs.submit("prepare_plan", self.doc)
            self._status(f"Preparing {self.doc.plan.name} in the background…")
        mode = self.doc.fragments is not None and index != 1
        labels = self.doc.fragments if mode else self.doc.anatomy
        meshes = self._loaded_meshes.get(mode)
        if index == 1 and self.pending_separation is not None:
            meshes, labels = self._separation_meshes, self.pending_separation
        elif index == 2 and self.pending_fragments is not None:
            mode, meshes, labels = True, self._fragment_meshes, self.pending_fragments
        if meshes is not None:
            self._display(meshes, mode, labels)
        else:
            # Do not leave a stale cut scene selectable while its replacement is built.
            self._display({}, mode, labels)
            if not self.jobs.busy:
                self._refresh_surfaces()
        if index != 3:
            self.views.set_transforms({})
        if index == 1:
            self._tool_changed()
        elif index == 2:
            self._load_plane_controls()
        elif index == 3:
            self._load_motion()
            self._update_transforms()

    def _cut_config(self):
        assert self.doc is not None
        return self.doc.plan.cuts[self.operation_combo.currentData()]

    def _selected_plane(self):
        config = self._cut_config()
        key = self.plane_combo.currentData()
        if self.operation_combo.currentData() == "bsso":
            side, plane = key.split(".")
            return config[side][plane]
        if self.operation_combo.currentData() == "lefort":
            return config["planes"][key]
        return config["plane"]

    def _load_cut_controls(self):
        if self.doc is None:
            return
        self.loading = True
        config = self._cut_config()
        bsso = self.operation_combo.currentData() == "bsso"
        self.cut_enabled.setChecked(config["enabled"])
        self.plane_combo.clear()
        if bsso:
            for side in ("left", "right"):
                for key in ("front", "upper", "sagittal"):
                    self.plane_combo.addItem(f"{side.title()} {key}", f"{side}.{key}")
        elif self.operation_combo.currentData() == "lefort":
            for key in config["planes"]:
                self.plane_combo.addItem(key.replace("_", " ").title(), key)
        else:
            self.plane_combo.addItem("Cut plane", "plane")
        for i in range(3):
            for widgets, key in ((self.region_min, "min"), (self.region_max, "max")):
                widgets[i].setEnabled(not bsso)
                if not bsso:
                    widgets[i].setValue(config["region"][key][i])
        self.midline.setEnabled(bsso)
        if bsso:
            self.midline.setValue(config["midline"])
        self.loading = False
        self._load_plane_controls()

    def _load_plane_controls(self):
        if self.loading or self.doc is None or self.plane_combo.currentData() is None:
            return
        plane = self._selected_plane()
        self.loading = True
        for controls, key in ((self.plane_origin, "origin"), (self.plane_normal, "normal")):
            for spin, value in zip(controls, plane[key], strict=True):
                spin.setValue(value)
        self.loading = False
        self._show_cut_plane()

    def _show_cut_plane(self):
        assert self.doc is not None
        if self.tabs.currentIndex() != 2:
            return
        plane = self._selected_plane()
        origin = self.doc.plan_to_world(plane["origin"])
        normal = self.doc.frame[:3, :3].T @ np.asarray(plane["normal"])
        self.views.set_plane(origin, normal, self._cut_config()["enabled"])

    def _plan_edit_snapshot(self):
        assert self.doc is not None
        return {"plans": deepcopy(self.doc.plans), "scenarios": deepcopy(self.doc.scenarios)}

    def _cut_toggled(self, enabled):
        if self.loading or self.doc is None:
            return
        before = self._plan_edit_snapshot()
        self._cut_config()["enabled"] = enabled
        self._invalidate()
        self._remember("Enable operation", before, self._plan_edit_snapshot())
        self._load_plan_selectors()
        self._show_cut_plane()

    def _plane_controls_changed(self):
        if self.loading or self.doc is None:
            return
        normal = np.array([s.value() for s in self.plane_normal])
        if np.linalg.norm(normal) < 1e-8:
            self._error("A cut plane needs a nonzero normal.")
            self._load_plane_controls()
            return
        before = self._plan_edit_snapshot()
        plane = self._selected_plane()
        plane["origin"] = [s.value() for s in self.plane_origin]
        plane["normal"] = (normal / np.linalg.norm(normal)).tolist()
        if before == self._plan_edit_snapshot():
            return
        self._invalidate()
        self._remember("Adjust cut plane", before, self._plan_edit_snapshot())
        self._show_cut_plane()

    def _region_changed(self):
        if self.loading or self.doc is None:
            return
        before = self._plan_edit_snapshot()
        config = self._cut_config()
        if self.operation_combo.currentData() == "bsso":
            config["midline"] = self.midline.value()
        else:
            config["region"] = {
                "min": [s.value() for s in self.region_min],
                "max": [s.value() for s in self.region_max],
            }
        if before == self._plan_edit_snapshot():
            return
        self._invalidate()
        self._remember("Adjust cut region", before, self._plan_edit_snapshot())

    def _plane_dragged(self, origin, normal):
        if self.doc is None or self.loading or self.tabs.currentIndex() != 2:
            return
        plan = self.doc.world_to_plan(origin)
        direction = self.doc.frame[:3, :3] @ np.asarray(normal)
        for spins, values in ((self.plane_origin, plan), (self.plane_normal, direction)):
            for spin, value in zip(spins, values, strict=True):
                spin.setValue(float(value))
        self._plane_controls_changed()

    def _build_cuts(self, accepted):
        if self.doc is None or self.jobs.busy:
            return
        if accepted:
            if (
                self.pending_fragments is None
                or self._fragment_counts is None
                or self._anatomy_counts is None
                or self._pending_key != self.doc.configuration_key
            ):
                return
            from .engine import validate_fragment_preview

            try:
                validate_fragment_preview(
                    self.doc, self.pending_fragments, self._fragment_counts, self._anatomy_counts
                )
            except ValueError as error:
                self._error(str(error))
                return
            self.doc.fragments = self.pending_fragments
            self.doc.fragments_key = self.doc.configuration_key
            self.doc.plan.accepted = True
            self._loaded_meshes[True] = self._fragment_meshes
            self.pending_fragments = None
            self._pending_key = None
            self._fragment_meshes = self._fragment_counts = None
            self._remesh_timer.stop()
            self._mark_dirty()
            self._cache_active_plan()
            self._load_plan_selectors()
            if self.tabs.currentIndex() == 3:
                self._stage_changed(3)
            else:
                self.tabs.setCurrentIndex(3)
            self._sync_enabled()
            self._status("Cuts accepted. Move fragments with numeric controls and a visible pivot.")
            return
        self.jobs.submit("cuts_preview", self.doc)
        self._status("Partitioning bone fragments at native resolution…")

    def _arm_pick(self, mode):
        if self.doc is None:
            return
        self.pick_mode = mode
        self.views.set_tool("pick")
        self._status(f"Pick a point in a slice or the 3D view to set {mode.replace('_', ' ')}.")
        self._sync_enabled()

    def _fragment_selected(self, value):
        self.selected_fragment = value

    def _picked_voxel(self, point, labels):
        """Surface picks sit between voxels; resolve them inside the picked actor."""
        assert self.doc is not None
        xyz = np.rint(self.doc.geometry.world_to_index(point)).astype(int)
        shape = np.asarray(self.doc.geometry.shape[::-1])
        actor = self.views.last_pick_fragment
        if actor is None:
            if np.any(xyz < 0) or np.any(xyz >= shape):
                return None, 0
            return xyz, int(labels[tuple(xyz[::-1])])
        candidates = np.array(
            [xyz + [x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
        )
        candidates = candidates[np.all((candidates >= 0) & (candidates < shape), axis=1)]
        if not len(candidates):
            return None, 0
        candidate_labels = labels[tuple(candidates[:, ::-1].T)]
        candidates = candidates[candidate_labels == actor]
        if not len(candidates):
            return None, 0
        distance = np.linalg.norm(self.doc.geometry.index_to_world(candidates) - point, axis=1)
        return candidates[np.argmin(distance)], actor

    def _point_picked(self, point):
        if self.doc is None or self.jobs.busy:
            return
        self.last_point = point
        plan = self.doc.world_to_plan(point).tolist()
        mode = self.pick_mode
        self.views.set_crosshair(point)
        if mode == "component":
            xyz = np.rint(self.doc.geometry.world_to_index(point)).astype(int).tolist()
            self.jobs.submit("component", self.doc, xyz=xyz, label=self.label_combo.currentData())
            return
        if mode == "plane":
            for spin, value in zip(self.plane_origin, plan, strict=True):
                spin.setValue(value)
            self._plane_controls_changed()
        elif mode in ("region_min", "region_max"):
            if self.operation_combo.currentData() == "bsso":
                self._status("BSSO uses a midline and three planes per side.")
            else:
                for spin, value in zip(getattr(self, mode), plan, strict=True):
                    spin.setValue(value)
                self._region_changed()
        elif mode == "confirm":
            if self.pending_fragments is None:
                self._status("Preview fragments before confirming pieces.")
                return
            xyz, value = self._picked_voxel(point, self.pending_fragments)
            if xyz is None:
                self._status("Pick inside the fragment or on its displayed surface.")
                return
            expected = self.confirm_combo.currentData()
            if value != expected:
                self._status(
                    f"That point is {self.doc.fragment_names.get(value, 'background')}; select a point inside {self.doc.fragment_names[expected]}."
                )
                return
            self.doc.plan.confirmations[str(expected)] = self.doc.world_to_plan(
                self.doc.geometry.index_to_world(xyz)
            ).tolist()
            self._mark_dirty()
            self.confirm_label.setText(
                "Confirmed: "
                + ", ".join(self.doc.fragment_names[int(i)] for i in self.doc.plan.confirmations)
            )
        elif mode == "pivot":
            before = deepcopy(self.doc.scenarios)
            motion = self.doc.motions[self.motion_combo.currentData()]
            motion.pivot, motion.pivot_set = plan, True
            self._remember(
                "Place pivot", {"scenarios": before}, {"scenarios": deepcopy(self.doc.scenarios)}
            )
            self._load_motion()
            self._update_transforms()
        elif mode == "landmark":
            scenarios_before = deepcopy(self.doc.scenarios)
            if self.doc.fragments is None:
                self._status("Accept fragments before adding measurement landmarks.")
                return
            _, fragment = self._picked_voxel(point, self.doc.fragments)
            if not fragment:
                self._status("Pick a bone fragment for a measurement landmark.")
                return
            self.doc.scenario.landmarks.append(
                {
                    "name": f"Point {len(self.doc.scenario.landmarks) + 1}",
                    "point": plan,
                    "fragment": fragment,
                }
            )
            self._remember(
                "Add landmark",
                {"scenarios": scenarios_before},
                {"scenarios": deepcopy(self.doc.scenarios)},
            )
            self._update_transforms()
        else:
            self._status(
                f"Planning coordinates: R {plan[0]:.2f}, A {plan[1]:.2f}, S {plan[2]:.2f} mm"
            )
        if mode:
            self.pick_mode = None
            self.views.set_tool("navigate")

    def _frame_changed(self):
        if self.loading or self.doc is None:
            return
        from scipy.spatial.transform import Rotation

        from .engine import default_cuts
        from .model import CutPlan

        if any(plan.accepted or plan.confirmations for plan in self.doc.plans.values()) or any(
            m.pivot_set or any(m.translation) or any(m.angles)
            for scenario in self.doc.scenarios.values()
            for m in scenario.motions.values()
        ):
            answer = QMessageBox.question(
                self,
                "Change head reference",
                "Changing the reference resets cuts, pivots, landmarks and movement scenarios. Keep the segmentation and reset these settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._load_document_controls()
                return
        before = {
            key: deepcopy(getattr(self.doc, key))
            for key in ("frame", "plans", "scenarios", "active_scenario")
        }
        self.doc.frame = np.eye(4)
        self.doc.frame[:3, :3] = Rotation.from_euler(
            "xyz", [s.value() for s in self.frame_spins], degrees=True
        ).as_matrix() @ np.diag([-1.0, -1.0, 1.0])
        self.doc.plans = {"plan1": CutPlan("Le Fort I", cuts=default_cuts(self.doc))}
        self.doc.scenarios = {"Scenario 1": new_scenario()}
        self.doc.active_scenario = "Scenario 1"
        self._invalidate(all_plans=True)
        after = {key: deepcopy(getattr(self.doc, key)) for key in before}
        self.history.push(Edit("Change reference", [], before, after))
        self._mark_dirty()
        self._load_document_controls()

    def _load_motion(self):
        if self.doc is None:
            return
        motion = self.doc.motions[self.motion_combo.currentData()]
        self.loading = True
        values = [
            motion.translation[0],
            motion.translation[1],
            -motion.translation[2],
            *motion.angles,
        ]
        for spin, value in zip(self.motion_spins, values, strict=True):
            spin.setValue(value)
        self.loading = False
        self.pivot_label.setText(
            "Pivot R/A/S: " + ", ".join(f"{v:.2f}" for v in motion.pivot) + " mm"
            if motion.pivot_set
            else "Place a pivot to enable rotation."
        )
        self._sync_enabled()

    def _motion_changed(self):
        if self.loading or self.doc is None or self.doc.fragments is None:
            return
        if self._motion_before is None:
            self._motion_before = deepcopy(self.doc.scenarios)
        motion = self.doc.motions[self.motion_combo.currentData()]
        v = [s.value() for s in self.motion_spins]
        motion.translation = [v[0], v[1], -v[2]]
        motion.angles = v[3:]
        self._mark_dirty()
        self._update_transforms()

    def _finish_motion_edit(self):
        if self.doc is None:
            return
        if self._motion_before is not None:
            self.history.push(
                Edit(
                    "Move fragment",
                    [],
                    {"scenarios": self._motion_before},
                    {"scenarios": deepcopy(self.doc.scenarios)},
                )
            )
            self._motion_before = None

    def _update_transforms(self):
        if self.doc is None:
            return
        from .engine import fragment_transforms

        transforms = fragment_transforms(self.doc)
        self.views.set_transforms(
            transforms if self.doc.fragments is not None and self.tabs.currentIndex() == 3 else {}
        )
        motion = self.doc.motions[self.motion_combo.currentData()]
        self.views.set_pivot(self.doc.plan_to_world(motion.pivot) if motion.pivot_set else None)
        lines = []
        displayed_landmarks = []
        for landmark in self.doc.scenario.landmarks:
            point = self.doc.plan_to_world(landmark["point"])
            matrix = (
                transforms.get(landmark["fragment"], np.eye(4))
                if self.tabs.currentIndex() == 3 and self.doc.fragments is not None
                else np.eye(4)
            )
            moved = matrix[:3, :3] @ point + matrix[:3, 3]
            delta = self.doc.frame[:3, :3] @ (moved - point)
            displayed_landmarks.append({"name": landmark["name"], "point": point, "moved": moved})
            lines.append(
                f"{landmark['name']}: {np.linalg.norm(delta):.2f} mm (R {delta[0]:+.2f}, A {delta[1]:+.2f}, S {delta[2]:+.2f})"
            )
        self.landmark_label.setText(
            "\n".join(lines) or "Pick landmarks to measure displacement from baseline."
        )
        self.views.set_landmarks(displayed_landmarks)

    def _share_pivot(self):
        if self.doc is None:
            return
        pivot = self.doc.motions[self.motion_combo.currentData()]
        if not pivot.pivot_set:
            self._status("Place a pivot first.")
            return
        before = deepcopy(self.doc.scenarios)
        for motion in self.doc.motions.values():
            motion.pivot = list(pivot.pivot)
            motion.pivot_set = True
        self._remember(
            "Share pivot", {"scenarios": before}, {"scenarios": deepcopy(self.doc.scenarios)}
        )
        self._update_transforms()
        self._load_motion()

    def _reset_pivot(self):
        if self.doc is None or self.jobs.busy:
            return
        self._finish_motion_edit()
        if self.pick_mode == "pivot":
            self.pick_mode = None
            self.views.set_tool("navigate")
            self._status("Pivot pick cancelled.")
        motion = self.doc.motions[self.motion_combo.currentData()]
        if not (motion.pivot_set or any(motion.pivot) or any(motion.angles)):
            self._sync_enabled()
            return
        before = deepcopy(self.doc.scenarios)
        motion.pivot, motion.angles, motion.pivot_set = [0.0] * 3, [0.0] * 3, False
        self._remember(
            "Reset pivot", {"scenarios": before}, {"scenarios": deepcopy(self.doc.scenarios)}
        )
        self._load_motion()
        self._update_transforms()
        self._status("Pivot and rotation reset; translation retained.")

    def _reset_motion(self):
        if self.doc is None:
            return
        before = deepcopy(self.doc.scenarios)
        motion = self.doc.motions[self.motion_combo.currentData()]
        motion.translation, motion.angles = [0.0] * 3, [0.0] * 3
        self._remember(
            "Reset movement", {"scenarios": before}, {"scenarios": deepcopy(self.doc.scenarios)}
        )
        self._load_motion()
        self._update_transforms()

    def _scenario_changed(self, name):
        if (
            self.loading
            or self.doc is None
            or self.jobs.busy
            or name not in self.doc.scenarios
            or name == self.doc.active_scenario
        ):
            return
        self._finish_motion_edit()
        self.doc.active_scenario = name
        self._mark_dirty()
        self._activate_plan()

    def _duplicate_scenario(self):
        if self.doc is None:
            return
        name, ok = QInputDialog.getText(
            self,
            "Duplicate scenario",
            "Scenario name",
            text=f"Scenario {len(self.doc.scenarios) + 1}",
        )
        name = name.strip()
        if not ok or not name:
            return
        if name in self.doc.scenarios:
            self._error("That scenario name is already in use.")
            return
        before = deepcopy(self.doc.scenarios)
        self.doc.scenarios[name] = deepcopy(self.doc.scenario)
        self._remember(
            "Duplicate scenario", {"scenarios": before}, {"scenarios": deepcopy(self.doc.scenarios)}
        )
        self._scenario_changed(name)

    def _undo(self, forward):
        if self.doc is None or self.jobs.busy:
            return
        self._finish_motion_edit()
        stack = self.history.redo_stack if forward else self.history.undo_stack
        if not stack:
            return
        edit = stack[-1]
        anatomy_changed = "anatomy" in edit.before or any(
            patch[0] == "anatomy" for patch in edit.patches
        )
        if self.history.apply(self.doc, forward):
            if anatomy_changed:
                self._loaded_meshes.pop(False, None)
                if "anatomy" in edit.before:
                    self._anatomy_counts = None
                elif self._anatomy_counts is not None:
                    for field, _, before, after in edit.patches:
                        if field == "anatomy":
                            delta = np.bincount(after, minlength=5) - np.bincount(
                                before, minlength=5
                            )
                            self._anatomy_counts += delta if forward else -delta
            if self.doc.fragments is None:
                self._loaded_meshes.pop(True, None)
                self.views.invalidate_surfaces(True)
            self.pending_fragments = self.pending_separation = None
            self._separation_meshes = self._separation_counts = None
            self._fragment_meshes = self._fragment_counts = None
            self._mark_dirty()
            self._load_document_controls()
            self._activate_plan()

    def closeEvent(self, event):
        if self.jobs.busy:
            answer = QMessageBox.question(
                self,
                "Computation running",
                "Cancel the running computation and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            event.ignore()
            self._close_after_cancel = True
            self.jobs.close()
            return
        if self.dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved project",
                "Save changes before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                event.ignore()
                self._after_save = self.close
                self.save_project()
                return
        self.jobs.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="BIMAX binary-volume jaw simulator")
    parser.add_argument("path", nargs="?", help="Binary NIfTI volume or .bimax project")
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("BIMAX")
    app.setDesktopFileName("app.bimax.simulator")
    bundled_icon = Path(__file__).resolve().parent / "icons" / "bimax.svg"
    fallback = QIcon(str(bundled_icon)) if bundled_icon.is_file() else QIcon()
    icon = QIcon.fromTheme("app.bimax.simulator", fallback)
    if not icon.isNull():
        app.setWindowIcon(icon)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    window.views.initialize()
    if args.path:
        QTimer.singleShot(0, lambda: window.load_path(args.path))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
