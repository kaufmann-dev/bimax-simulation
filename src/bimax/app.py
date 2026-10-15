"""Native desktop workflow for binary-volume jaw simulation."""

import argparse
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCursor, QIcon, QKeySequence, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
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
from .model import ANATOMY, CUT_TYPES, PROXIMAL_IDS, Document, new_scenario

SCENE_CACHE_BYTES = 768 * 1024**2

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
        self.ai_control = None
        self.doc: Document | None = None
        self.project_path = None
        self.project_version = None
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
        self._scene_cache = OrderedDict()
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
        self.scenario_combo = QComboBox()
        self.scenario_combo.currentTextChanged.connect(self._scenario_changed)
        self.scenario_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.scenario_combo.setMinimumContentsLength(12)
        self.scenario_combo.setMinimumWidth(0)
        self.scenario_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tabs.currentChanged.connect(self._stage_changed)
        split = QSplitter()
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        scenario_form = form_layout()
        scenario_form.addRow("Scenario", self.scenario_combo)
        sidebar_layout.addLayout(scenario_form)
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
        from .control import Control
        self.ai_control = Control(self)
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
        self.ai_banner = QLabel()
        toolbar.addWidget(self.ai_banner)
        self.take_control_button = QPushButton("Take control")
        self.take_control_button.clicked.connect(lambda: self.ai_control and self.ai_control.release())
        toolbar.addWidget(self.take_control_button)
        self.take_control_button.hide()

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
        layout = self._tab("Orient")
        note("BIMAX", layout).setStyleSheet("font-size: 24px; font-weight: 600;")
        note("Explore bone movement using an unsegmented binary bone volume.", layout)
        button("Import binary NIfTI…", self.import_volume, layout)
        button("Open BIMAX project…", self.open_project, layout)
        self.info_label = note(
            "Input: one 3D .nii or .nii.gz file with values 0 and 1. Prepare the bone volume with medsurface before import.",
            layout,
        )
        group = QGroupBox("Whole-skull orientation")
        form = form_layout(group)
        self.orientation_spins = vector_inputs(
            form, "Pitch / roll / yaw °", self._orientation_changed, -180, 180
        )
        for spin, axis in zip(self.orientation_spins, ("Pitch", "Roll", "Yaw"), strict=True):
            spin.setAccessibleName(f"Whole-skull {axis.lower()} in degrees")
            spin.setToolTip(f"{axis} in degrees")
        layout.addWidget(group)
        note("Align the whole skull for viewing and image export. This orientation is shared by all scenarios.", layout)
        button("Reset orientation", self._reset_orientation, layout)
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
            "Adjust geometric cuts in the unmoved anatomy. BSSO uses a stepped split; each proximal segment can rotate independently about a picked condyle pivot.",
            layout,
        )
        self.cut_checks = {}
        for cut_id, (name, _, _) in CUT_TYPES.items():
            check = QCheckBox(name)
            check.toggled.connect(lambda enabled, key=cut_id: self._select_cut(key, enabled))
            self.cut_checks[cut_id] = check
            layout.addWidget(check)
        self.cut_note = note("Edit each shared cut once. Scenarios select combinations of these cuts.", layout)
        form = form_layout()
        self.operation_combo = QComboBox()
        for cut_id, (name, _, _) in CUT_TYPES.items():
            self.operation_combo.addItem(name, cut_id)
        self.operation_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.operation_combo.setMinimumContentsLength(12)
        self.operation_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.operation_combo.currentIndexChanged.connect(self._load_cut_controls)
        form.addRow("Edit shared cut", self.operation_combo)
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
        confirm_form.addRow("Confirm piece", self.confirm_combo)
        layout.addLayout(confirm_form)
        button("Identify piece at next pick", lambda: self._arm_pick("confirm"), layout)
        self.confirm_label = note(
            "Confirm the enabled fragments by picking a point inside each one.", layout
        )
        self.accept_cuts_button = button("Accept combination", lambda: self._build_cuts(True), layout)
        button("Continue to movement", lambda: self.tabs.setCurrentIndex(3), layout)
        layout.addStretch()

    def _build_move(self):
        layout = self._tab("Move")
        note("Move the jaws", layout).setStyleSheet("font-size: 18px;")
        form = form_layout()
        self.move_cut_note = QLabel()
        self.move_cut_note.setWordWrap(True)
        form.addRow(self.move_cut_note)
        self.motion_combo = QComboBox()
        self.motion_combo.addItem("Linked jaws", "linked")
        self.motion_combo.addItem("Distal mandible", "mandible")
        self.motion_combo.addItem("Chin", "chin")
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
        self.motion_form = form
        layout.addLayout(form)
        self._motion_before = None
        note(
            "Positive pitch is counterclockwise in the right-profile view (anterior to screen-right). Place a pivot before rotating.",
            layout,
        )
        self.proximal_note = note("For BSSO proximal segments, pick a condyle pivot: it is a geometric approximation of the joint center. Only the selected segment rotates; its baseline pivot stays fixed.", layout)
        self.proximal_note.hide()
        self.pivot_label = note("No pivot selected.", layout)
        button("Place pivot at next pick", lambda: self._arm_pick("pivot"), layout)
        self.reset_pivot_button = button("Reset pivot", self._reset_pivot, layout)
        self.reset_pivot_button.setToolTip(
            "Clear the selected control's pivot and rotation; keep its translation. "
            "Also cancel a pending pivot pick."
        )
        self.share_pivot_button = button("Use this pivot for all jaw controls", self._share_pivot, layout)
        button("Reset selected movement", self._reset_motion, layout)
        button("Duplicate scenario…", self._duplicate_scenario, layout)
        self.baseline_check = QCheckBox("Show baseline overlay")
        self.baseline_check.toggled.connect(self.views.show_baseline)
        layout.addWidget(self.baseline_check)
        button("Right-profile view", self.views.profile_view, layout)
        button("Add measurement landmark at next pick", self._arm_new_landmark, layout)
        self.landmark_label = note("Landmarks show displacement from the baseline.", layout)
        self._build_measurements(layout)
        button("Save project…", self.save_project, layout)
        self.image_export_button = button("Export scenario images…", self._export_images, layout)
        self.report_button = button("Export scenario report (PDF)…", self._export_report, layout)
        self.export_button = button("Export fragment labelmaps…", self._export, layout)
        layout.addStretch()

    def _build_measurements(self, layout):
        box = QGroupBox("Symmetry measurements")
        form = form_layout(box)
        self.symmetry_reference_label = QLabel("Reference midline is unset.")
        self.symmetry_reference_label.setWordWrap(True)
        form.addRow(self.symmetry_reference_label)
        self.symmetry_right_spin = QDoubleSpinBox()
        self.symmetry_right_spin.setRange(-10000, 10000)
        self.symmetry_right_spin.setDecimals(2)
        form.addRow("Reference Right (mm)", self.symmetry_right_spin)
        set_reference = QPushButton("Set reference offset")
        set_reference.clicked.connect(lambda: self._planning_action("set_symmetry_reference", right=self.symmetry_right_spin.value()))
        form.addRow(set_reference)
        pick_reference = QPushButton("Set reference at next pick")
        pick_reference.clicked.connect(lambda: self._arm_pick("symmetry_reference"))
        form.addRow(pick_reference)
        self.landmark_combo = QComboBox()
        self.landmark_combo.currentTextChanged.connect(self._landmark_selected)
        form.addRow("Landmark", self.landmark_combo)
        self.landmark_name = QLineEdit()
        form.addRow("Landmark name", self.landmark_name)
        self.landmark_fragment_combo = QComboBox()
        form.addRow("Attached fragment", self.landmark_fragment_combo)
        self.landmark_point_spins = vector_inputs(form, "Baseline R/A/S (mm)", lambda: None)
        for label, callback in (
            ("New landmark", self._new_landmark),
            ("Save landmark", self._save_landmark),
            ("Place named landmark at next pick", lambda: self._arm_pick("landmark")),
            ("Delete landmark", self._delete_landmark),
        ):
            widget = QPushButton(label)
            widget.clicked.connect(callback)
            form.addRow(widget)
        self.measurement_combo = QComboBox()
        self.measurement_combo.currentTextChanged.connect(self._measurement_selected)
        form.addRow("Measurement", self.measurement_combo)
        self.measurement_name = QLineEdit()
        form.addRow("Measurement name", self.measurement_name)
        self.measurement_kind_combo = QComboBox()
        for label, kind in (("Midline deviation", "midline"), ("Paired symmetry", "paired"), ("Ramus inclination", "ramus")):
            self.measurement_kind_combo.addItem(label, kind)
        self.measurement_kind_combo.currentIndexChanged.connect(self._measurement_kind_changed)
        form.addRow("Measure", self.measurement_kind_combo)
        self.measurement_role_labels: list[QLabel] = []
        self.measurement_role_combos: list[QComboBox] = []
        for _ in range(4):
            role_label, combo = QLabel(), QComboBox()
            form.addRow(role_label, combo)
            self.measurement_role_labels.append(role_label)
            self.measurement_role_combos.append(combo)
        for label, callback in (
            ("New measurement", lambda: self.measurement_name.clear()),
            ("Save measurement", self._save_measurement),
            ("Delete measurement", self._delete_measurement),
        ):
            widget = QPushButton(label)
            widget.clicked.connect(callback)
            form.addRow(widget)
        self.measurement_results_label = QLabel()
        self.measurement_results_label.setWordWrap(True)
        self.measurement_results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.measurement_results_label)
        explanation = QLabel("Planning R/A/S measurements are independent of the camera. Midline: positive Right. Paired differences: left minus right; height = Superior, front = Anterior, lateral = distance from reference (mm). Ramus: condyle minus angle, frontal positive outward and sagittal positive anterior (degrees); differences left minus right.")
        explanation.setWordWrap(True)
        form.addRow(explanation)
        layout.addWidget(box)
        self._measurement_kind_changed()

    def _planning_action(self, operation, **arguments):
        if self.doc is None or self.jobs.busy or (self.ai_control and self.ai_control.session):
            return
        try:
            self._command(operation, **arguments)
            if operation == "set_landmark" and not arguments.get("remove"):
                self.landmark_combo.setCurrentText(arguments["name"])
            if operation == "set_measurement" and not arguments.get("remove"):
                self.measurement_combo.setCurrentText(arguments["name"])
        except ValueError as error:
            self._error(str(error))

    def _arm_new_landmark(self):
        self._new_landmark()
        self._arm_pick("landmark")

    def _new_landmark(self):
        self.landmark_name.clear()
        for spin in self.landmark_point_spins:
            spin.setValue(0)

    def _save_landmark(self):
        if self.doc:
            self._planning_action("set_landmark", scenario=self.doc.active_scenario,
                                  name=self.landmark_name.text().strip(),
                                  fragment=self.landmark_fragment_combo.currentData(),
                                  point=[spin.value() for spin in self.landmark_point_spins])

    def _delete_landmark(self):
        if self.doc:
            self._planning_action("set_landmark", scenario=self.doc.active_scenario,
                                  name=self.landmark_name.text().strip(), remove=True)

    def _landmark_selected(self, name=None):
        if self.loading or self.doc is None:
            return
        name = self.landmark_combo.currentText() if name is None else name
        item = next((item for item in self.doc.scenario.landmarks if item["name"] == name), None)
        if item:
            self.landmark_name.setText(item["name"])
            self.landmark_fragment_combo.setCurrentIndex(self.landmark_fragment_combo.findData(item["fragment"]))
            for spin, value in zip(self.landmark_point_spins, item["point"], strict=True):
                spin.setValue(value)

    def _measurement_kind_changed(self):
        from .symmetry import ROLES
        roles = ROLES[self.measurement_kind_combo.currentData()]
        for index, (label, combo) in enumerate(zip(self.measurement_role_labels, self.measurement_role_combos, strict=True)):
            label.setVisible(index < len(roles))
            combo.setVisible(index < len(roles))
            if index < len(roles):
                label.setText(roles[index].replace("_", " ").title())

    def _measurement_selected(self, name=None):
        if self.loading or self.doc is None:
            return
        name = self.measurement_combo.currentText() if name is None else name
        definition = self.doc.scenario.measurements.get(name)
        if definition:
            from .symmetry import ROLES
            self.measurement_name.setText(name)
            self.measurement_kind_combo.setCurrentIndex(self.measurement_kind_combo.findData(definition.kind))
            for index, role in enumerate(ROLES[definition.kind]):
                self.measurement_role_combos[index].setCurrentText(definition.landmarks[role])

    def _save_measurement(self):
        if self.doc:
            from .symmetry import ROLES
            kind = self.measurement_kind_combo.currentData()
            self._planning_action("set_measurement", scenario=self.doc.active_scenario,
                                  name=self.measurement_name.text().strip(), kind=kind,
                                  landmarks={role: self.measurement_role_combos[index].currentText() for index, role in enumerate(ROLES[kind])})

    def _delete_measurement(self):
        if self.doc:
            self._planning_action("set_measurement", scenario=self.doc.active_scenario,
                                  name=self.measurement_name.text().strip(), remove=True)

    def _load_measurement_controls(self):
        if self.doc is None:
            return
        from .symmetry import active_fragment_ids
        previous, self.loading = self.loading, True
        for combo, names in ((self.landmark_combo, [item["name"] for item in self.doc.scenario.landmarks]),
                             (self.measurement_combo, list(self.doc.scenario.measurements))):
            selected = combo.currentText()
            combo.clear()
            combo.addItems(names)
            combo.setCurrentIndex(max(0, combo.findText(selected)))
        selected = self.landmark_fragment_combo.currentData()
        self.landmark_fragment_combo.clear()
        for fragment in self.doc.fragment_descriptors.values():
            if fragment.id in active_fragment_ids(self.doc) - {"excluded", "unassigned"}:
                self.landmark_fragment_combo.addItem(fragment.name, fragment.id)
        self.landmark_fragment_combo.setCurrentIndex(max(0, self.landmark_fragment_combo.findData(selected)))
        names = [item["name"] for item in self.doc.scenario.landmarks]
        for combo in self.measurement_role_combos:
            selected = combo.currentText()
            combo.clear()
            combo.addItems(names)
            combo.setCurrentIndex(max(0, combo.findText(selected)))
        if self.doc.symmetry_right is not None:
            self.symmetry_right_spin.setValue(self.doc.symmetry_right)
        self.loading = previous
        self._landmark_selected()
        self._measurement_selected()
        self._show_measurement_results()

    def _show_measurement_results(self):
        if self.doc is None:
            return
        from .symmetry import measurement_results
        self.symmetry_reference_label.setText(f"Reference plane: R = {self.doc.symmetry_right:.2f} mm" if self.doc.symmetry_right is not None else "Reference midline is unset; select a point or set an offset.")
        lines = []
        for name, result in measurement_results(self.doc).items():
            lines.append(name)
            if result["status"] == "invalid":
                lines.append(result["error"])
            else:
                for field, baseline in result["baseline"].items():
                    unit = "°" if field.endswith("_deg") else "mm"
                    title = field.removesuffix("_deg").removesuffix("_mm").replace("_", " ")
                    lines.append(f"{title}: baseline {baseline:+.2f}, moved {result['moved'][field]:+.2f} {unit}")
        self.measurement_results_label.setText("\n".join(lines) or "Create a named measurement to compare baseline and moved results.")

    def _status(self, message):
        self.statusBar().showMessage(message)

    def _error(self, message):
        self._after_save = None
        self._status(message)
        if self.ai_control and self.ai_control.session:
            return
        QMessageBox.warning(self, "BIMAX", message)

    def _busy_changed(self, busy):
        self.cancel_button.setVisible(busy)
        self.tabs.setEnabled(not busy)
        self.scenario_combo.setEnabled(self.doc is not None and not busy)
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
        locked = bool(self.ai_control and self.ai_control.session)
        self.tabs.setEnabled(not busy and not locked)
        self.scenario_combo.setEnabled(has_doc and not busy and not locked)
        for spin in self.orientation_spins:
            spin.setEnabled(has_doc and not busy)
        for name, action in (
            self.toolbar_actions.items() if hasattr(self, "toolbar_actions") else []
        ):
            action.setEnabled((not busy and not locked and (has_doc or name in ("Import volume", "Open project"))) or name == "Right profile")
        for index in range(1, 4):
            self.tabs.setTabEnabled(index, has_doc)
        if hasattr(self, "accept_separation_button"):
            self.accept_separation_button.setEnabled(self.pending_separation is not None)
            self.discard_separation_button.setEnabled(self.pending_separation is not None)
        if hasattr(self, "export_button"):
            can_export = self.doc is not None and self.doc.fragments is not None and not busy
            self.export_button.setEnabled(can_export)
            self.image_export_button.setEnabled(can_export)
            self.report_button.setEnabled(can_export and self.doc is not None and self.doc.cuts_accepted)
            self.accept_cuts_button.setEnabled(self.pending_fragments is not None)
            can_move = self.doc is not None and self.doc.fragments is not None and not busy
            self.motion_combo.setEnabled(can_move)
            if self.doc is not None:
                chin_enabled = "genio" in self.doc.scenario.cut_ids
                motion_model = self.motion_combo.model()
                if isinstance(motion_model, QStandardItemModel):
                    motion_model.item(self.motion_combo.findData("chin")).setEnabled(chin_enabled)
                if not chin_enabled and self.motion_combo.currentData() == "chin":
                    self.motion_combo.setCurrentIndex(0)
            motion = self._selected_motion() if self.doc else None
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
                        (i < 3 and self.motion_combo.currentData() not in PROXIMAL_IDS)
                        or (
                            i >= 3 and
                            self.doc is not None
                            and self._selected_motion().pivot_set
                        )
                    )
                )

    def _command(self, operation, scenario=None, **args):
        from .commands import Commands
        if self.doc is None:
            return {}
        self._finish_motion_edit()
        self._cache_active_scene()
        result = Commands(self.doc, self.history).execute(operation, scenario, **args)
        if result["changed"]:
            self.dirty = True
            self.setWindowTitle(f"BIMAX — {self.doc.source_name} *")
        self._refresh_command(operation)
        return result

    def _refresh_command(self, operation):
        if self.doc is None:
            return
        if self.doc.fragments is None:
            self._loaded_meshes.pop(True, None)
            self.pending_fragments = None
            self._pending_key = None
            self._fragment_meshes = self._fragment_counts = None
            self.views.invalidate_surfaces(True)
        self._load_document_controls()
        if operation == "set_orientation":
            self.views.profile_view()
        if self.ai_control and self.ai_control.session:
            self.ai_control.show_scene()
        else:
            self._activate_scene()
        self._sync_enabled()

    def _mark_dirty(self):
        if self.doc:
            self.doc.revision += 1
        self.dirty = True
        self.setWindowTitle(f"BIMAX — {self.doc.source_name if self.doc else ''} *")

    def _invalidate(self, all_anatomy=False, cut_id=None):
        if self.doc is None:
            return
        key = cut_id or self.operation_combo.currentData()
        affected = list(self.doc.cuts) if all_anatomy else [key]
        for cid in affected:
            cut = self.doc.cuts[cid]
            cut.confirmations = {}
            cut.accepted = False
            cut.revision += 1
        for scenario in self.doc.scenarios.values():
            if all_anatomy or key in scenario.cut_ids:
                scenario.landmarks = []
        if all_anatomy:
            self.doc.anatomy_revision += 1
            self.doc.reviews.clear()
            self._scene_cache.clear()
        if all_anatomy or key in self.doc.scenario.cut_ids:
            self.doc.fragments = None
            self.doc.fragments_key = None
            self._loaded_meshes.pop(True, None)
            self.views.invalidate_surfaces(True)
            self.views.set_transforms({})
        self.pending_fragments = None
        self.pending_separation = None
        self._separation_meshes = self._separation_counts = None
        self._fragment_meshes = self._fragment_counts = None
        self._pending_key = None
        if self.doc.fragments is None:
            self._display(self._loaded_meshes.get(False, {}), False, self.doc.anatomy)
        self.confirm_label.setText("Confirm new pieces by picking a point inside each one.")
        self._load_scenario_controls()
        self._sync_enabled()

    def _cache_active_scene(self):
        if self.doc is None or self.doc.fragments is None or True not in self._loaded_meshes:
            return
        key = self.doc.configuration_key if self.doc.cuts_accepted else None
        if self.doc.fragments_key != key or not self.doc.cuts_accepted:
            return
        self._scene_cache[key] = (self.doc.fragments, self._loaded_meshes[True])
        self._scene_cache.move_to_end(key)
        def size(entry):
            labels, meshes = entry
            return labels.nbytes + sum(mesh.vertices.nbytes + mesh.faces.nbytes + mesh.normals.nbytes
                                      for mesh in meshes.values())
        while self._scene_cache and sum(size(entry) for entry in self._scene_cache.values()) > SCENE_CACHE_BYTES:
            self._scene_cache.popitem(last=False)

    def _activate_scene(self):
        if self.doc is None:
            return
        self.pending_fragments = self.pending_separation = None
        self._fragment_meshes = self._fragment_counts = None
        self._separation_meshes = self._separation_counts = None
        self._pending_key = None
        self.pick_mode = None
        key = self.doc.configuration_key if self.doc.cuts_accepted else None
        if self.doc.fragments_key != key or not self.doc.cuts_accepted:
            self.doc.fragments = None
            self.doc.fragments_key = None
            self._loaded_meshes.pop(True, None)
            self.views.invalidate_surfaces(True)
        if self.doc.cuts_accepted and self.doc.fragments is None and key in self._scene_cache:
            self.doc.fragments, self._loaded_meshes[True] = self._scene_cache[key]
            self.doc.fragments_key = key
            self._scene_cache.move_to_end(key)
        self._load_scenario_controls()
        self._load_cut_controls()
        self._load_motion()
        self._stage_changed(self.tabs.currentIndex())

    def _load_scenario_controls(self):
        if self.doc is None:
            return
        from .engine import required_fragment_labels
        previous, self.loading = self.loading, True
        self.scenario_combo.clear()
        self.scenario_combo.addItems(list(self.doc.scenarios))
        self.scenario_combo.setCurrentText(self.doc.active_scenario)
        self.scenario_combo.setToolTip(self.doc.active_scenario)
        for key, check in self.cut_checks.items():
            check.setChecked(key in self.doc.scenario.cut_ids)
            check.setToolTip(self.doc.cuts[key].name)
        selected = " + ".join(self.doc.cuts[key].name for key in self.doc.scenario.cut_ids) or "No cuts selected"
        self.move_cut_note.setText(selected + (" · accepted" if self.doc.cuts_accepted else " · preview and accept first"))
        self.cut_note.setText("Shared edits affect every scenario using that cut. Modified templates need review before use.")
        old = self.motion_combo.currentData()
        self.motion_combo.clear()
        self.motion_combo.addItem("Linked jaws", "linked")
        for key in reversed(self.doc.upper_cuts):
            label = self.doc.fragment_labels[f"upper:{key}"]
            self.motion_combo.addItem(self.doc.fragment_names[label], key)
        self.motion_combo.addItem("Distal mandible", "mandible")
        self.motion_combo.addItem("Chin", "chin")
        if "bsso" in self.doc.scenario.cut_ids:
            self.motion_combo.addItem("Left proximal segment", "left-proximal")
            self.motion_combo.addItem("Right proximal segment", "right-proximal")
        self.motion_combo.setCurrentIndex(max(0, self.motion_combo.findData(old)))
        old = self.confirm_combo.currentData()
        self.confirm_combo.clear()
        required = required_fragment_labels(self.doc)
        if self._anatomy_counts is not None and self._anatomy_counts[3]:
            required.add(5)
        for label in sorted(required):
            self.confirm_combo.addItem(self.doc.fragment_names[label], label)
        self.confirm_combo.setCurrentIndex(max(0, self.confirm_combo.findData(old)))
        self.loading = previous

    def _select_cut(self, cut_id, enabled):
        if self.ai_control and self.ai_control.session:
            return
        if self.loading or self.doc is None or self.jobs.busy:
            return
        selected = list(self.doc.scenario.cut_ids)
        if enabled and cut_id not in selected:
            cut = self.doc.cuts[cut_id]
            if cut.kind == "lefort":
                selected = [key for key in selected if not (self.doc.cuts[key].kind == "lefort" and self.doc.cuts[key].level == cut.level)]
            selected.append(cut_id)
        elif not enabled and cut_id in selected:
            selected.remove(cut_id)
        else:
            return
        self._command("select_cuts", self.doc.active_scenario, cut_ids=selected)

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
        if self.ai_control and self.ai_control.session:
            return
        if self.jobs.busy or not self._confirm_replace():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import binary bone volume", "", "Binary NIfTI (*.nii *.nii.gz)"
        )
        if path:
            self.load_path(path)

    def open_project(self):
        if self.ai_control and self.ai_control.session:
            return
        if self.jobs.busy or not self._confirm_replace():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "BIMAX project (*.bimax)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        path = Path(path).absolute()
        operation = "open" if str(path).endswith(".bimax") else "import"
        self.jobs.submit(operation, path=str(path))
        self._status("Loading volume and extracting display surfaces…")

    def save_project(self, save_as=False):
        if self.ai_control and self.ai_control.session:
            return
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
        from .storage import file_version
        version = self.project_version if path == self.project_path else file_version(path)
        self.jobs.submit("save", self.doc, path=path, expected_version=version)
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

    def _case_details(self):
        from .report import CaseInfo

        dialog = QDialog(self)
        dialog.setWindowTitle("Scenario report details")
        form = QFormLayout(dialog)
        form.addRow(QLabel("Optional. Printed in the report only; not saved in the project."))
        fields = {key: QLineEdit() for key in ("patient", "birth_date", "case_id", "surgeon")}
        for key, label in zip(fields, ("Patient", "Date of birth", "Case ID", "Surgeon / planner"), strict=True):
            form.addRow(label, fields[key])
        notes = QPlainTextEdit()
        notes.setMaximumHeight(110)
        form.addRow("Notes", notes)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return CaseInfo(**{key: field.text().strip() for key, field in fields.items()},
                        notes=notes.toPlainText().strip())

    def _export_report(self):
        if self.doc is None or self.doc.fragments is None or self.jobs.busy:
            return
        if not self.doc.cuts_accepted:
            self._error("Review and accept the active cut combination before exporting a report.")
            return
        case = self._case_details()
        if case is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export scenario report", f"{self.doc.active_scenario}-report.pdf", "PDF (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        from .report import export_report

        self._status("Rendering pre-op and planned views for the report…")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            exported = export_report(
                self.views, self.doc, path, project_path=self.project_path, dirty=self.dirty, case=case
            )
        except Exception as error:
            failure = str(error) or "Could not export the scenario report."
        else:
            failure = ""
        finally:
            QApplication.restoreOverrideCursor()
        if failure:
            self._error(failure)
        else:
            self._status(f"Scenario report exported: {exported}")

    def _job_finished(self, operation, revision, result):
        if "_control_session" in result:
            control = self.ai_control
            if not control or result["_control_session"] != control.session or result["_control_job"] != control.active_job:
                self._status("Discarded a result from a revoked editing session.")
                return
            if not control.owns_result(operation, revision, result):
                self._status("Discarded an outdated AI computation result.")
                return
            if operation in ("cuts_preview", "load_prepared"):
                return  # Control publishes provisional geometry without accepting it.
        if operation not in ("import", "open") and self.doc is None:
            return
        if operation not in ("import", "open") and self.doc and operation != "load_prepared" and revision != self.doc.revision:
            self._status("Discarded an outdated computation result.")
            return
        if self.doc is not None and "document_id" in result and result["document_id"] != self.doc._identity:
            self._status("Discarded a result for a different document.")
            return
        if self.doc is not None and "anatomy_fingerprint" in result:
            digest = result["anatomy_fingerprint"]
            if digest["revision"] == self.doc.anatomy_revision:
                self.doc._anatomy_digest = digest["digest"]
                self.doc._digest_revision = digest["revision"]
        if (
            self.doc
            and "configuration_key" in result
            and result["configuration_key"] != self.doc.configuration_key
        ):
            self._status("Discarded a result for different cuts or anatomy.")
            return
        if self.doc is not None and "artifacts" in result:
            self.doc.artifacts.update(result["artifacts"])
        if operation in ("import", "open"):
            self.doc = result["document"]
            assert self.doc is not None
            self.doc.source.setflags(write=False)
            self.project_path = result.get("path") if operation == "open" else None
            self.project_version = result.get("file_version")
            self._loaded_meshes = result["surface_sets"]
            self._scene_cache.clear()
            self._cache_active_scene()
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
            self.tabs.setCurrentIndex(0)
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
            self._pending_key = result["configuration_key"]
            self._load_scenario_controls()
            self._display(result["surfaces"], True, self.pending_fragments)
            self.views.set_transforms({})
            self._status(
                "Fragment preview ready. Existing confirmations are reused; identify any new pieces, then accept."
            )
        elif operation in ("prepare_scene", "load_prepared"):
            assert self.doc is not None
            if result.get("needs_preparation"):
                self.jobs.submit("prepare_scene", self.doc)
                self._status("Preparing this combination once…")
                return
            from .engine import validate_fragment_preview
            try:
                validate_fragment_preview(self.doc, result["fragments"], result["fragment_counts"], result["anatomy_counts"])
            except ValueError as error:
                self._error(str(error))
                return
            self.doc.fragments = result["fragments"]
            self.doc.fragments_key = result["configuration_key"]
            self._loaded_meshes[True] = result["surfaces"]
            self._cache_active_scene()
            self._stage_changed(self.tabs.currentIndex())
            self._status("Saved combination ready.")
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
            self.project_version = result.get("file_version")
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
        self._load_scenario_controls()
        # Decode the orthonormal planning rotation relative to the initial LPS→RAS basis.
        from scipy.spatial.transform import Rotation

        rotation = self.doc.frame[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
        angles = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        for spin, value in zip(self.frame_spins, angles, strict=True):
            spin.setValue(float(value))
        for spin, value in zip(self.orientation_spins, self.doc.view_angles, strict=True):
            spin.setValue(value)
        self.loading = False
        self._load_cut_controls()
        self._load_motion()
        self._assignment_counts()
        self._load_measurement_controls()
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
        if self.ai_control and self.ai_control.session:
            return
        tool = self.tool_combo.currentData()
        self.pick_mode = "component" if tool == "component" else None
        visual_tool = {"component": "pick", "unbarrier": "barrier", "unseed": "seed"}.get(
            tool, tool
        )
        self.views.set_tool(visual_tool)
        self.views.set_label(self.label_combo.currentData())

    def _paint(self, payload):
        if self.ai_control and self.ai_control.session:
            return
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
        metadata_before = self._cut_edit_snapshot() if field == "anatomy" else {}
        array[indices] = after
        if field == "anatomy" and self._anatomy_counts is not None:
            self._anatomy_counts += np.bincount(after, minlength=5) - np.bincount(
                before, minlength=5
            )
        if field == "anatomy":
            self._loaded_meshes.pop(False, None)
            self._invalidate(all_anatomy=True)
        else:
            # Editing separation hints does not change the accepted anatomy or
            # any accepted cut geometry. Only discard the now-stale pending proposals.
            self._activate_scene()
        self._remember(
            name,
            metadata_before,
            self._cut_edit_snapshot() if field == "anatomy" else {},
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
        metadata_before = self._cut_edit_snapshot()
        self.doc.anatomy = proposal
        self._loaded_meshes[False] = meshes
        self._invalidate(all_anatomy=True)
        self._anatomy_counts = counts
        self._remember(
            "Accept separation",
            {"anatomy": before, **metadata_before},
            {"anatomy": proposal, **self._cut_edit_snapshot()},
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
            self.doc.cuts_accepted
            and self.doc.fragments is None
            and index != 1
            and not self.jobs.busy
        ):
            from .prepared import load_options
            if self.doc.configuration_key in self.doc.artifacts:
                self.jobs.submit("load_prepared", None, **load_options(self.doc))
                self._status("Loading saved combination…")
            else:
                self.jobs.submit("prepare_scene", self.doc)
                self._status("Preparing this combination once…")
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
        return self.doc.cuts[self.operation_combo.currentData()].geometry

    def _selected_plane(self):
        assert self.doc is not None
        config = self._cut_config()
        key = self.plane_combo.currentData()
        if self.operation_combo.currentData() == "bsso":
            side, plane = key.split(".")
            return config[side][plane]
        if self.doc.cuts[self.operation_combo.currentData()].kind == "lefort":
            return config["planes"][key]
        return config["plane"]

    def _load_cut_controls(self):
        if self.doc is None:
            return
        self.loading = True
        config = self._cut_config()
        bsso = self.operation_combo.currentData() == "bsso"
        selected_plane = self.plane_combo.currentData()
        self.plane_combo.clear()
        if bsso:
            for side in ("left", "right"):
                for key in ("front", "upper", "sagittal"):
                    self.plane_combo.addItem(f"{side.title()} {key}", f"{side}.{key}")
        elif self.doc.cuts[self.operation_combo.currentData()].kind == "lefort":
            for key in config["planes"]:
                self.plane_combo.addItem(key.replace("_", " ").title(), key)
        else:
            self.plane_combo.addItem("Cut plane", "plane")
        self.plane_combo.setCurrentIndex(max(0, self.plane_combo.findData(selected_plane)))
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
        self.views.set_plane(origin, normal, self.operation_combo.currentData() in self.doc.scenario.cut_ids)

    def _cut_edit_snapshot(self):
        assert self.doc is not None
        return {"cuts": deepcopy(self.doc.cuts), "scenarios": deepcopy(self.doc.scenarios),
                "reviews": deepcopy(self.doc.reviews)}

    def _plane_controls_changed(self):
        if self.ai_control and self.ai_control.session:
            return
        if self.loading or self.doc is None:
            return
        normal = np.array([s.value() for s in self.plane_normal])
        if np.linalg.norm(normal) < 1e-8:
            self._error("A cut plane needs a nonzero normal.")
            self._load_plane_controls()
            return
        key = self.operation_combo.currentData()
        geometry = deepcopy(self._cut_config())
        # The selected plane path differs between the seven shared definitions.
        if key == "bsso":
            side, plane_id = self.plane_combo.currentData().split(".")
            plane = geometry[side][plane_id]
        elif key == "genio":
            plane = geometry["plane"]
        else:
            plane = geometry["planes"][self.plane_combo.currentData()]
        plane["origin"] = [s.value() for s in self.plane_origin]
        plane["normal"] = (normal / np.linalg.norm(normal)).tolist()
        self._command("replace_cut", cut_id=key, geometry=geometry, action_label="Adjust cut plane")
        self._show_cut_plane()

    def _region_changed(self):
        if self.ai_control and self.ai_control.session:
            return
        if self.loading or self.doc is None:
            return
        config = deepcopy(self._cut_config())
        key = self.operation_combo.currentData()
        if key == "bsso":
            config["midline"] = self.midline.value()
        else:
            config["region"] = {
                "min": [spin.value() for spin in self.region_min],
                "max": [spin.value() for spin in self.region_max],
            }
        try:
            self._command("replace_cut", cut_id=key, geometry=config, action_label="Adjust cut region")
        except ValueError as error:
            self._error(str(error))
            self._load_cut_controls()

    def _plane_dragged(self, origin, normal):
        if self.ai_control and self.ai_control.session:
            return
        if self.doc is None or self.loading or self.tabs.currentIndex() != 2:
            return
        plan = self.doc.world_to_plan(origin)
        direction = self.doc.frame[:3, :3] @ np.asarray(normal)
        for spins, values in ((self.plane_origin, plan), (self.plane_normal, direction)):
            for spin, value in zip(spins, values, strict=True):
                spin.setValue(float(value))
        self._plane_controls_changed()

    def _build_cuts(self, accepted):
        if self.ai_control and self.ai_control.session:
            return
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
            from .commands import Commands

            try:
                Commands(self.doc, self.history).execute(
                    "accept_combination", self.doc.active_scenario,
                    fragments=self.pending_fragments, fragment_counts=self._fragment_counts,
                    anatomy_counts=self._anatomy_counts,
                )
            except ValueError as error:
                self._error(str(error))
                return
            self.doc.fragments = self.pending_fragments
            self.doc.fragments_key = self.doc.configuration_key
            self._loaded_meshes[True] = self._fragment_meshes
            self.pending_fragments = None
            self._pending_key = None
            self._fragment_meshes = self._fragment_counts = None
            self._remesh_timer.stop()
            self.dirty = True
            self.setWindowTitle(f"BIMAX — {self.doc.source_name} *")
            self._cache_active_scene()
            self._load_scenario_controls()
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
        if self.ai_control and self.ai_control.session:
            return
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
        if self.ai_control and self.ai_control.session:
            return
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
            from .commands import Commands
            Commands(self.doc, self.history).execute(
                "confirm_fragment", self.doc.active_scenario,
                fragment=self.doc.fragment_descriptors[expected].id,
                point=self.doc.world_to_plan(self.doc.geometry.index_to_world(xyz)).tolist(),
            )
            self.dirty = True
            self.setWindowTitle(f"BIMAX — {self.doc.source_name} *")
            self.confirm_label.setText(
                "Confirmed: "
                + ", ".join(self.doc.fragment_names[self.doc.fragment_labels[i]] for i in self.doc.review.confirmations if i in self.doc.fragment_labels)
            )
        elif mode == "symmetry_reference":
            self._planning_action("set_symmetry_reference", point=plan)
        elif mode == "pivot":
            target = self.motion_combo.currentData()
            if target in PROXIMAL_IDS:
                _, fragment = self._picked_voxel(point, self.doc.fragments)
                if fragment != self.doc.fragment_labels[target]:
                    self._status("Pick a condyle point on the selected proximal segment.")
                    return
            self._planning_action("set_proximal_rotations" if target in PROXIMAL_IDS else "set_movements", scenario=self.doc.active_scenario,
                          targets={self.motion_combo.currentData(): {"pivot": plan}},
                          action_label="Place pivot")
        elif mode == "landmark":
            if self.doc.fragments is None:
                self._status("Accept fragments before adding measurement landmarks.")
                return
            _, fragment = self._picked_voxel(point, self.doc.fragments)
            if not fragment:
                self._status("Pick a bone fragment for a measurement landmark.")
                return
            self._planning_action("set_landmark", scenario=self.doc.active_scenario,
                          name=self.landmark_name.text().strip() or f"Point {len(self.doc.scenario.landmarks) + 1}",
                          point=plan, fragment=self.doc.fragment_descriptors[fragment].id,
                          action_label="Add landmark")
        else:
            self._status(
                f"Planning coordinates: R {plan[0]:.2f}, A {plan[1]:.2f}, S {plan[2]:.2f} mm"
            )
        if mode:
            self.pick_mode = None
            self.views.set_tool("navigate")

    def _orientation_changed(self):
        if self.ai_control and self.ai_control.session:
            return
        if self.loading or self.doc is None or self.jobs.busy:
            return
        angles = [spin.value() for spin in self.orientation_spins]
        if angles != self.doc.view_angles:
            self._command("set_orientation", rotation=dict(zip(("pitch", "roll", "yaw"), angles, strict=True)))

    def _reset_orientation(self):
        if self.doc is None or self.jobs.busy:
            return
        for spin in self.orientation_spins:
            spin.setValue(0)
        self._orientation_changed()

    def _frame_changed(self):
        if self.ai_control and self.ai_control.session:
            return
        if self.loading or self.doc is None:
            return
        from scipy.spatial.transform import Rotation

        from .engine import default_cuts

        if self.doc.symmetry_right is not None or any(scenario.landmarks or scenario.measurements for scenario in self.doc.scenarios.values()) or any(cut.accepted or cut.confirmations for cut in self.doc.cuts.values()) or any(
            m.pivot_set or any(m.translation) or any(m.angles)
            for scenario in self.doc.scenarios.values()
            for m in scenario.motions.values()
        ):
            answer = QMessageBox.question(
                self,
                "Change head reference",
                "Changing the reference resets cuts, pivots, landmarks, symmetry measurements and movement scenarios. Keep the segmentation and reset these settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._load_document_controls()
                return
        before = {
            key: deepcopy(getattr(self.doc, key))
            for key in ("frame", "cuts", "scenarios", "active_scenario", "reviews", "symmetry_right")
        }
        self.doc.frame = np.eye(4)
        self.doc.frame[:3, :3] = Rotation.from_euler(
            "xyz", [s.value() for s in self.frame_spins], degrees=True
        ).as_matrix() @ np.diag([-1.0, -1.0, 1.0])
        self.doc.symmetry_right = None
        self.doc.cuts = default_cuts(self.doc)
        self.doc.scenarios = {"Scenario 1": new_scenario()}
        self.doc.active_scenario = "Scenario 1"
        self._invalidate(all_anatomy=True)
        after = {key: deepcopy(getattr(self.doc, key)) for key in before}
        self.history.push(Edit("Change reference", [], before, after))
        self._mark_dirty()
        self._load_document_controls()

    def _selected_motion(self):
        assert self.doc is not None
        key = self.motion_combo.currentData()
        return self.doc.scenario.proximal_rotations[key] if key in PROXIMAL_IDS else self.doc.motions[key]

    def _load_motion(self):
        if self.doc is None or self.loading:
            return
        motion = self._selected_motion()
        proximal = self.motion_combo.currentData() in PROXIMAL_IDS
        self.proximal_note.setVisible(proximal)
        self.share_pivot_button.setEnabled(not proximal)
        for spin in self.motion_spins[:3]:
            spin.setVisible(not proximal)
            self.motion_form.labelForField(spin).setVisible(not proximal)
        self.loading = True
        translation = [0, 0, 0] if proximal else [motion.translation[0], motion.translation[1], -motion.translation[2]]
        values = [*translation, *motion.angles]
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
        if self.ai_control and self.ai_control.session:
            return
        if self.loading or self.doc is None or self.doc.fragments is None:
            return
        if self._motion_before is None:
            self._motion_before = deepcopy(self.doc.scenarios)
        from .commands import Commands
        values = [spin.value() for spin in self.motion_spins]
        target = self.motion_combo.currentData()
        rotation = dict(zip(("pitch", "roll", "yaw"), values[3:], strict=True))
        if target in PROXIMAL_IDS:
            result = Commands(self.doc, self.history).execute(
                "set_proximal_rotations", self.doc.active_scenario, record_history=False,
                targets={target: {"rotation": rotation}},
            )
        else:
            result = Commands(self.doc, self.history).execute(
                "set_movements", self.doc.active_scenario, record_history=False,
                targets={target: {"translation": [values[0], values[1], -values[2]], "rotation": rotation}},
            )
        if result["changed"]:
            self.dirty = True
            self.setWindowTitle(f"BIMAX — {self.doc.source_name} *")
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
        motion = self._selected_motion()
        self.views.set_pivot(self.doc.plan_to_world(motion.pivot) if motion.pivot_set else None)
        lines = []
        displayed_landmarks = []
        from .symmetry import landmark_results
        for landmark in landmark_results(self.doc):
            if landmark["status"] != "valid":
                lines.append(f"{landmark['name']}: {landmark['error']}")
                continue
            point = self.doc.plan_to_world(landmark["baseline"])
            moved = self.doc.plan_to_world(landmark["moved"]) if self.tabs.currentIndex() == 3 and self.doc.fragments is not None else point
            delta = self.doc.frame[:3, :3] @ (moved - point)
            displayed_landmarks.append({"name": landmark["name"], "point": point, "moved": moved})
            lines.append(f"{landmark['name']}: {np.linalg.norm(delta):.2f} mm (R {delta[0]:+.2f}, A {delta[1]:+.2f}, S {delta[2]:+.2f})")
        self.landmark_label.setText(
            "\n".join(lines) or "Pick landmarks to measure displacement from baseline."
        )
        self.views.set_landmarks(displayed_landmarks)
        self._show_measurement_results()

    def _share_pivot(self):
        if self.ai_control and self.ai_control.session:
            return
        if self.doc is None:
            return
        if self.motion_combo.currentData() in PROXIMAL_IDS:
            return
        pivot = self.doc.motions[self.motion_combo.currentData()]
        if not pivot.pivot_set:
            self._status("Place a pivot first.")
            return
        self._command("set_movements", self.doc.active_scenario,
                      targets={key: {"pivot": list(pivot.pivot)} for key in self.doc.motions},
                      action_label="Share pivot")

    def _reset_pivot(self):
        if self.ai_control and self.ai_control.session:
            return
        if self.doc is None or self.jobs.busy:
            return
        self._finish_motion_edit()
        if self.pick_mode == "pivot":
            self.pick_mode = None
            self.views.set_tool("navigate")
            self._status("Pivot pick cancelled.")
        motion = self._selected_motion()
        if not (motion.pivot_set or any(motion.pivot) or any(motion.angles)):
            self._sync_enabled()
            return
        if self.motion_combo.currentData() in PROXIMAL_IDS:
            self._command("reset_proximal_rotations", self.doc.active_scenario,
                          targets=[self.motion_combo.currentData()], clear_pivots=True, action_label="Reset proximal pivot")
        else:
            self._command("reset_pivot", self.doc.active_scenario,
                          target=self.motion_combo.currentData(), action_label="Reset pivot")
        self._status("Pivot and rotation reset; translation retained.")

    def _reset_motion(self):
        if self.ai_control and self.ai_control.session:
            return
        if self.doc is None:
            return
        if self.motion_combo.currentData() in PROXIMAL_IDS:
            self._command("reset_proximal_rotations", self.doc.active_scenario,
                          targets=[self.motion_combo.currentData()], action_label="Reset proximal rotation")
        else:
            self._command("set_movements", self.doc.active_scenario,
                          targets={self.motion_combo.currentData(): {
                              "translation": [0.0] * 3, "rotation": {"pitch": 0, "roll": 0, "yaw": 0},
                          }}, action_label="Reset movement")

    def _scenario_changed(self, name):
        if self.ai_control and self.ai_control.session:
            return
        if (
            self.loading
            or self.doc is None
            or self.jobs.busy
            or name not in self.doc.scenarios
            or name == self.doc.active_scenario
        ):
            return
        self._command("select_scenario", name)

    def _duplicate_scenario(self):
        if self.ai_control and self.ai_control.session:
            return
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
        self._command("create_scenario", name=name, source=self.doc.active_scenario)

    def _undo(self, forward):
        if self.ai_control and self.ai_control.session:
            return
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
            if "view_angles" in edit.before:
                self._mark_dirty()
                self._load_document_controls()
                self.views.profile_view()
                self._sync_enabled()
                return
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
            self._activate_scene()

    def closeEvent(self, event):
        if self.ai_control and self.ai_control.session:
            self.ai_control.release()
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
        if self.ai_control:
            self.ai_control.close()
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
