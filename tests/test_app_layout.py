"""Native layout and view-mode controls, without the import/export workflow."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
)

from bimax.app import STYLE, MainWindow
from bimax.engine import default_cuts
from bimax.model import Document, Geometry, new_scenario
from bimax.surfaces import make_surfaces


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        pytest.skip("Native layout verification requires DISPLAY; use xvfb-run")
    app = QApplication.instance() or QApplication([])
    previous = app.styleSheet()
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    yield app
    app.setStyleSheet(previous)


@pytest.fixture
def window(qapp):
    result = MainWindow()
    result.resize(1280, 900)
    result.show()
    result.views.initialize()
    geometry = Geometry((12, 12, 12), (0.4,) * 3, (-134, -294.4, -680), tuple(np.eye(3).ravel()))
    source = np.zeros(geometry.shape, dtype=np.uint8)
    source[2:10, 2:10, 2:10] = 1
    anatomy = source * 2
    anatomy[:6] = source[:6] * 3
    doc = Document(geometry, source, anatomy, np.zeros_like(source), np.zeros_like(source), source_name="skull_merged.nii.gz")
    doc.cuts = default_cuts(doc)
    for level in (2, 3):
        doc.scenarios[f"Le Fort {level}"] = new_scenario([f"lefort{level}"])
    result._job_finished("open", -1, {
        "document": doc,
        "surface_sets": {False: make_surfaces(anatomy, geometry)},
        "anatomy_counts": np.bincount(anatomy.ravel(), minlength=5),
        "fragment_mode": False,
    })
    QTest.qWait(30)
    yield result
    result.jobs.close()
    result.dirty = False
    result.views.close()
    result.close()
    qapp.processEvents()


@pytest.mark.parametrize("width", [350, 375, 460])
def test_sidebar_forms_fit_without_horizontal_scrolling(window, qapp, width):
    sidebar = window.tabs.parentWidget()
    sidebar.setFixedWidth(width)
    for stage in range(4):
        window.tabs.setCurrentIndex(stage)
        operations = ("lefort1", "lefort2", "lefort3", "modified-lefort2", "modified-lefort3", "bsso", "genio") if stage == 2 else ("lefort1",)
        for operation in operations:
            window.operation_combo.setCurrentIndex(window.operation_combo.findData(operation))
            QTest.qWait(20)
            qapp.processEvents()
            scroll = window.tabs.widget(stage)
            page = scroll.widget()
            assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            assert scroll.horizontalScrollBar().maximum() == 0
            assert page.width() <= scroll.viewport().width()
            assert window.tabs.width() <= sidebar.width()
            for child in page.findChildren(QWidget):
                if isinstance(child, (QLabel, QPushButton, QComboBox, QDoubleSpinBox, QGroupBox)) and child.isVisibleTo(page):
                    left = child.mapTo(page, QPoint()).x()
                    assert left >= 0
                    assert left + child.width() <= page.width(), (stage, type(child), child.width())


@pytest.mark.parametrize("level", [1, 2, 3])
def test_cut_plan_plane_selectors_fit_narrow_sidebar(window, qapp, level):
    window.tabs.parentWidget().setFixedWidth(350)
    window.tabs.setCurrentIndex(2)
    window.operation_combo.setCurrentIndex(window.operation_combo.findData(f"lefort{level}"))
    QTest.qWait(30)
    qapp.processEvents()
    assert window.doc.cuts[f"lefort{level}"].level == level
    expected = window.doc.cuts[f"lefort{level}"].geometry["planes"]
    assert {window.plane_combo.itemData(index) for index in range(window.plane_combo.count())} == set(expected)
    scroll = window.tabs.widget(2)
    assert scroll.horizontalScrollBar().maximum() == 0
    assert scroll.widget().width() <= scroll.viewport().width()


def test_long_plan_and_scenario_names_do_not_widen_sidebar(window, qapp):
    name = "Le Fort III + BSSO + genioplasty (approximate independent alternative)"
    window.doc.cuts["lefort3"].name = name
    scenario = window.doc.scenarios.pop(window.doc.active_scenario)
    window.doc.scenarios[name] = scenario
    window.doc.active_scenario = name
    window._load_scenario_controls()
    window.tabs.parentWidget().setFixedWidth(350)
    for stage in (2, 3):
        window.tabs.setCurrentIndex(stage)
        QTest.qWait(30)
        scroll = window.tabs.widget(stage)
        page = scroll.widget()
        assert scroll.horizontalScrollBar().maximum() == 0
        assert page.width() <= scroll.viewport().width()
        for combo in page.findChildren(QComboBox):
            left = combo.mapTo(page, QPoint()).x()
            assert left >= 0 and left + combo.width() <= page.width()
    assert window.scenario_combo.toolTip() == name


def test_expand_3d_toolbar_and_keyboard_restore_views(window, qapp):
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window)
    action = window.expand_3d_action
    stage = window.tabs.currentIndex()
    assert not action.isChecked()
    assert not window.restore_views_action.isEnabled()
    action.trigger()
    qapp.processEvents()
    assert window.views.is_3d_expanded
    assert action.text() == "Show four views"
    assert window.restore_views_action.isEnabled()
    assert window.tabs.isVisible()
    assert all(not view.isVisible() for view in window.views.slices)
    window.views.vtk.setFocus()
    QTest.keyClick(window.views.vtk, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not action.isChecked()
    assert not window.views.is_3d_expanded
    assert all(view.isVisible() for view in window.views.slices)
    QTest.keyClick(window.views.vtk, Qt.Key.Key_F11)
    qapp.processEvents()
    assert window.views.is_3d_expanded
    QTest.keyClick(window.views.vtk, Qt.Key.Key_F11)
    qapp.processEvents()
    assert not window.views.is_3d_expanded
    assert window.tabs.currentIndex() == stage
    assert not window.dirty
