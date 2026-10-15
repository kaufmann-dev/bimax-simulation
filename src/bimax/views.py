"""Patient-aligned slice editing and an embedded native VTK fragment viewer."""

import os
import shutil
import tempfile
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

# Registers the OpenGL rendering backend used by the embedded window.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QGridLayout, QLabel, QSlider, QVBoxLayout, QWidget
from scipy.ndimage import binary_dilation
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkCommand, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPlane, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersHybrid import vtkRenderLargeImage
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTerrain
from vtkmodules.vtkInteractionWidgets import vtkImplicitPlaneRepresentation, vtkImplicitPlaneWidget2
from vtkmodules.vtkRenderingAnnotation import vtkCaptionActor2D
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkCellPicker,
    vtkPolyDataMapper,
    vtkRenderer,
)

from bimax.model import ANATOMY_COLORS, COLORS, Document, Geometry
from bimax.surfaces import Mesh


def world_bounds(geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(list(product(*[(-0.5, n - 0.5) for n in geometry.shape[::-1]])))
    world = geometry.index_to_world(corners)
    return world.min(axis=0), world.max(axis=0)


def sample_indices(geometry: Geometry, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor sampling without negative indexing outside the volume."""
    xyz = np.floor(geometry.world_to_index(world) + 0.5).astype(np.int64)
    valid = np.all((xyz >= 0) & (xyz < np.asarray(geometry.shape[::-1])), axis=-1)
    safe = np.clip(xyz, 0, np.asarray(geometry.shape[::-1]) - 1)
    flat = np.ravel_multi_index(safe[..., ::-1].reshape(-1, 3).T, geometry.shape)
    return np.asarray(flat).reshape(valid.shape), valid


EXPORT_VIEWS = (
    ("right", "Right", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ("left", "Left", np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ("front", "Front", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ("birds-eye", "Bird's-eye", np.array([0.0, np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40))]), np.array([0.0, 0.0, 1.0])),
)
EXPORT_IMAGE_SIZE = 2048
# Report pages add obliques and a submental view to the standard exports.
REPORT_VIEWS = EXPORT_VIEWS + (
    ("front-right", "Front-right oblique", np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0]), np.array([0.0, 0.0, 1.0])),
    ("front-left", "Front-left oblique", np.array([-np.sqrt(0.5), np.sqrt(0.5), 0.0]), np.array([0.0, 0.0, 1.0])),
    ("inferior", "Inferior", np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
)
REPORT_IMAGE_SIZE = 1200


class SliceView(QWidget):
    """A physical patient plane, independent of the stored image direction."""

    pointPicked = Signal(object)
    strokeFinished = Signal(object)
    sliceMoved = Signal(int, float)

    ORIENTATIONS = (
        ("Axial", 2, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), "R", "L", "A", "P"),
        ("Coronal", 1, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), "R", "L", "S", "I"),
        ("Sagittal · right profile", 0, np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0]), "P", "A", "S", "I"),
    )

    def __init__(self, orientation: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.orientation = orientation
        self.title, self.axis, self.u, self.v, *_ = self.ORIENTATIONS[orientation]
        self.document: Document | None = None
        self.labels: np.ndarray | None = None
        self.fragment_mode = False
        self.tool = "navigate"
        self.label = 3
        self.brush_mm = 3.0
        self.zoom = 1.0
        self._pan = np.zeros(2)
        self._panning = False
        self._pan_anchor = QPointF()
        self.crosshair = np.zeros(3)
        self._bounds = (np.zeros(3), np.ones(3))
        self._uv_bounds = (np.zeros(2), np.ones(2))
        self._image = QImage()
        self._cached_size = (-1, -1)
        self._stroke: set[int] = set()
        self._dragging = False
        self._last_point: np.ndarray | None = None
        self._cursor: QPointF | None = None
        self._polygon: list[QPointF] = []
        self._plane: tuple[np.ndarray, np.ndarray] | None = None
        self._pivot: np.ndarray | None = None
        self._landmarks: list[dict[str, Any]] = []
        self.setMinimumSize(180, 150)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName(f"{self.title} slice editor")
        self.setToolTip("Wheel or Page Up/Down: change slice. Ctrl+wheel or +/-: zoom. Middle drag: pan. Home: fit volume. Arrow keys: move crosshair. Enter: pick. Escape: cancel polygon. Seeds use anatomy colors with a white outline; barriers are magenta.")

    def set_document(self, document: Document) -> None:
        self.document = document
        self.zoom = 1.0
        self._pan[:] = 0
        self.labels = document.anatomy
        self._bounds = world_bounds(document.geometry)
        corners = np.asarray(list(product(*zip(*self._bounds, strict=True))))
        projected = np.stack((corners @ self.u, corners @ self.v), axis=1)
        self._uv_bounds = (projected.min(axis=0), projected.max(axis=0))
        self.crosshair = np.mean(self._bounds, axis=0)
        self.invalidate()

    def invalidate(self) -> None:
        self._cached_size = (-1, -1)
        self.update()

    def content_rect(self) -> QRectF:
        return QRectF(24, 27, max(1, self.width() - 48), max(1, self.height() - 49))

    def image_rect(self) -> QRectF:
        area = self.content_rect()
        extent = self._uv_bounds[1] - self._uv_bounds[0]
        scale = min(area.width() / extent[0], area.height() / extent[1]) * self.zoom
        width, height = extent * scale
        return QRectF(area.center().x() - width / 2 + self._pan[0], area.center().y() - height / 2 + self._pan[1], width, height)

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        """Zoom around a screen position while retaining its physical location."""
        anchor = self.content_rect().center() if anchor is None else anchor
        world = self.screen_to_world(anchor)
        polygon = [self.screen_to_world(point) for point in self._polygon]
        self.zoom = float(np.clip(zoom, 1, 20))
        moved = self.world_to_screen(world)
        self._pan += [anchor.x() - moved.x(), anchor.y() - moved.y()]
        self._polygon = [self.world_to_screen(point) for point in polygon]
        self.invalidate()

    def fit_volume(self) -> None:
        polygon = [self.screen_to_world(point) for point in self._polygon]
        self.zoom = 1.0
        self._pan[:] = 0
        self._polygon = [self.world_to_screen(point) for point in polygon]
        self.invalidate()

    def resizeEvent(self, event: Any) -> None:
        self.invalidate()
        super().resizeEvent(event)

    def screen_to_world(self, point: QPointF) -> np.ndarray:
        rect = self.image_rect()
        fraction = np.array([(point.x() - rect.left()) / rect.width(), (point.y() - rect.top()) / rect.height()])
        uv = self._uv_bounds[0] + fraction * (self._uv_bounds[1] - self._uv_bounds[0])
        world = self.u * uv[0] + self.v * uv[1]
        world[self.axis] = self.crosshair[self.axis]
        return world

    def world_to_screen(self, world: np.ndarray) -> QPointF:
        uv = np.array([world @ self.u, world @ self.v])
        fraction = (uv - self._uv_bounds[0]) / (self._uv_bounds[1] - self._uv_bounds[0])
        rect = self.image_rect()
        return QPointF(rect.left() + fraction[0] * rect.width(), rect.top() + fraction[1] * rect.height())

    def pixel_world(self, width: int, height: int) -> np.ndarray:
        low, high = self._uv_bounds
        us = low[0] + (np.arange(width) + 0.5) / width * (high[0] - low[0])
        vs = low[1] + (np.arange(height) + 0.5) / height * (high[1] - low[1])
        world = us[None, :, None] * self.u + vs[:, None, None] * self.v
        world[..., self.axis] = self.crosshair[self.axis]
        return world

    def _render_image(self, width: int, height: int, viewport: QRectF | None = None) -> None:
        if self.document is None:
            return
        if viewport is None:
            world = self.pixel_world(width, height)
        else:
            first = self.screen_to_world(viewport.topLeft())
            last = self.screen_to_world(viewport.bottomRight())
            us = (np.arange(width) + 0.5) / width * ((last - first) @ self.u)
            vs = (np.arange(height) + 0.5) / height * ((last - first) @ self.v)
            world = first + us[None, :, None] * self.u + vs[:, None, None] * self.v
        flat, valid = sample_indices(self.document.geometry, world)
        source = self.document.source.reshape(-1)[flat] * valid
        rgb = np.repeat((source * 175).astype(np.uint8)[..., None], 3, axis=2)
        if self.labels is not None:
            labels = self.labels.reshape(-1)[flat] * valid
            colors = COLORS if self.fragment_mode else ANATOMY_COLORS
            for label, color in colors.items():
                rgb[labels == label] = np.asarray(color) * 255
        seeds = self.document.seeds.reshape(-1)[flat] * valid
        barriers = self.document.barriers.reshape(-1)[flat] * valid
        # Anatomy colors distinguish competing seeds; a white outline identifies
        # them over already assigned anatomy, even for a single-pixel seed.
        rgb[binary_dilation(seeds > 0)] = [255, 255, 255]
        for label, color in ANATOMY_COLORS.items():
            rgb[seeds == label] = np.asarray(color) * 255
        rgb[barriers > 0] = [255, 60, 200]
        self._image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._cached_size = (width, height)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(10, 18, self.title)
        if self.document is None:
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Import a binary bone volume")
            return
        rect = self.image_rect()
        viewport = rect.intersected(self.content_rect())
        painter.save()
        painter.setClipRect(self.content_rect())
        if not viewport.isEmpty():
            size = (max(1, round(viewport.width())), max(1, round(viewport.height())))
            if self._cached_size != size:
                self._render_image(*size, viewport)
            painter.drawImage(viewport, self._image)
        cross = self.world_to_screen(self.crosshair)
        painter.setPen(QPen(QColor(240, 240, 240, 145), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(rect.left(), cross.y()), QPointF(rect.right(), cross.y()))
        painter.drawLine(QPointF(cross.x(), rect.top()), QPointF(cross.x(), rect.bottom()))
        self._draw_plane(painter, viewport)
        self._draw_markers(painter, self.content_rect())
        painter.restore()
        _, _, _, _, left, right, top, bottom = self.ORIENTATIONS[self.orientation]
        painter.setPen(QColor("#ffffff"))
        area = self.content_rect()
        painter.drawText(7, round(area.center().y()), left)
        painter.drawText(self.width() - 18, round(area.center().y()), right)
        painter.drawText(round(area.center().x()), 26, top)
        painter.drawText(round(area.center().x()), self.height() - 5, bottom)
        painter.drawText(10, self.height() - 5, f"{self.crosshair[self.axis]:.1f} mm · {self.zoom:.1f}×")
        if self._cursor is not None and self.tool in {"seed", "paint", "barrier", "erase"}:
            radius = self.brush_mm * rect.width() / (self._uv_bounds[1][0] - self._uv_bounds[0][0]) / 2
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawEllipse(self._cursor, radius, radius)
        if self._polygon:
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawPolyline(QPolygonF(self._polygon))
        if self.hasFocus():
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawRect(self.rect().adjusted(1, 1, -2, -2))

    def _draw_markers(self, painter: QPainter, rect: QRectF) -> None:
        markers = [(entry["name"], np.asarray(entry["moved"]), QColor("#ffffff")) for entry in self._landmarks]
        if self._pivot is not None:
            markers.append(("Pivot", self._pivot, QColor("#ffef62")))
        for name, world, color in markers:
            position = self.world_to_screen(world)
            if not rect.contains(position):
                continue
            distance = world[self.axis] - self.crosshair[self.axis]
            label = name if abs(distance) < 0.05 else f"{name} ({distance:+.1f} mm off slice)"
            painter.setPen(QPen(QColor("#000000"), 4))
            painter.drawEllipse(position, 5, 5)
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(position, 5, 5)
            text = QRectF(position.x() + 9, position.y() - 13, 230, 20)
            painter.fillRect(text, QColor(0, 0, 0, 190))
            painter.drawText(text, Qt.AlignmentFlag.AlignVCenter, label)

    def _draw_plane(self, painter: QPainter, rect: QRectF) -> None:
        if self._plane is None or rect.isEmpty():
            return
        origin, normal = self._plane
        vertices = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
        points = [self.screen_to_world(point) for point in vertices]
        distances = [float((point - origin) @ normal) for point in points]
        crossings = []
        for index in range(4):
            following = (index + 1) % 4
            a, b = distances[index], distances[following]
            if a * b <= 0 and abs(a - b) > 1e-10:
                fraction = a / (a - b)
                crossings.append(QPointF(vertices[index].x() + fraction * (vertices[following].x() - vertices[index].x()), vertices[index].y() + fraction * (vertices[following].y() - vertices[index].y())))
        if len(crossings) >= 2:
            painter.setPen(QPen(QColor("#ffef62"), 2))
            painter.drawLine(crossings[0], crossings[1])

    def brush_indices(self, point: np.ndarray) -> np.ndarray:
        """Physical circular brush with one native-voxel patient-plane thickness."""
        if self.document is None:
            return np.array([], dtype=np.int64)
        geometry = self.document.geometry
        radius = self.brush_mm / 2
        # Project each voxel edge onto the patient-plane normal. This includes all
        # voxels intersected by the slice even with anisotropy or oblique storage.
        half_thickness = float(np.sum(np.abs(geometry.affine[self.axis, :3]))) / 2
        extents = np.full(3, radius)
        extents[self.axis] = half_thickness
        corners = point + np.asarray(list(product(*[(-v, v) for v in extents])))
        indices = geometry.world_to_index(corners)
        low = np.maximum(0, np.floor(indices.min(axis=0)).astype(int))
        high = np.minimum(np.asarray(geometry.shape[::-1]) - 1, np.ceil(indices.max(axis=0)).astype(int))
        if np.any(high < low):
            return np.array([], dtype=np.int64)
        xyz = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(low, high, strict=True)], indexing="ij"), axis=-1).reshape(-1, 3)
        delta = geometry.index_to_world(xyz) - point
        in_plane = (delta @ self.u) ** 2 + (delta @ self.v) ** 2
        select = (in_plane <= radius**2 + 1e-7) & (np.abs(delta[:, self.axis]) <= half_thickness + 1e-7)
        return np.ravel_multi_index(xyz[select, ::-1].T, geometry.shape)

    def _append_stroke(self, point: np.ndarray) -> None:
        previous = self._last_point
        if previous is None:
            steps = np.asarray([point])
        else:
            count = max(1, int(np.linalg.norm(point - previous) / max(0.2, self.brush_mm / 4)) + 1)
            steps = np.linspace(previous, point, count + 1)[1:]
        for sample in steps:
            self._stroke.update(self.brush_indices(sample).tolist())
        self._last_point = point

    def _emit_stroke(self) -> None:
        if self._stroke:
            self.strokeFinished.emit({"tool": self.tool, "label": self.label, "indices": np.fromiter(self._stroke, dtype=np.int64), "finish": True})
        self._stroke.clear()
        self._last_point = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        if self.document is None:
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_anchor = event.position()
            return
        if not self.image_rect().contains(event.position()) or not self.content_rect().contains(event.position()):
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._polygon.clear()
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.screen_to_world(event.position())
        if self.tool == "polygon":
            self._polygon.append(event.position())
            self.update()
        elif self.tool in {"paint", "seed", "erase", "barrier"}:
            self._dragging = True
            self._append_stroke(point)
        else:
            self._dragging = True
            self.pointPicked.emit(point.tolist())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._cursor = event.position()
        if self._panning:
            delta = event.position() - self._pan_anchor
            self._pan += [delta.x(), delta.y()]
            self._polygon = [point + delta for point in self._polygon]
            self._pan_anchor = event.position()
            self.invalidate()
            return
        if self._dragging and self.document is not None and self.image_rect().contains(event.position()) and self.content_rect().contains(event.position()):
            point = self.screen_to_world(event.position())
            if self.tool in {"paint", "seed", "erase", "barrier"}:
                self._append_stroke(point)
            elif self.tool == "navigate":
                self.pointPicked.emit(point.tolist())
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            return
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._emit_stroke()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.tool == "polygon" and len(self._polygon) >= 3:
            self._finish_polygon()

    def _finish_polygon(self) -> None:
        if self.document is None or len(self._polygon) < 3:
            return
        # Rasterize at native physical resolution so zoom never drops thin bone.
        low, high = self._uv_bounds
        step = min(self.document.geometry.spacing) / 2
        width, height = np.maximum(1, np.ceil((high - low) / step).astype(int))
        rect = self.image_rect()
        vertices = np.array([[(p.x() - rect.left()) / rect.width() * width - 0.5, (p.y() - rect.top()) / rect.height() * height - 0.5] for p in self._polygon])
        from skimage.draw import polygon

        rows, columns = polygon(vertices[:, 1], vertices[:, 0], shape=(height, width))
        world = self.u * (low[0] + (columns[:, None] + 0.5) / width * (high[0] - low[0])) + self.v * (low[1] + (rows[:, None] + 0.5) / height * (high[1] - low[1]))
        world[:, self.axis] = self.crosshair[self.axis]
        indices, valid = sample_indices(self.document.geometry, world)
        self._stroke.update(indices[valid].tolist())
        self._emit_stroke()
        self._polygon.clear()
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.document is not None:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.set_zoom(self.zoom * 1.2 ** (event.angleDelta().y() / 120), event.position())
            else:
                self.sliceMoved.emit(self.axis, event.angleDelta().y() / 120 * min(self.document.geometry.spacing))
            event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.document is None:
            return super().keyPressEvent(event)
        step = min(self.document.geometry.spacing)
        if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus):
            self.set_zoom(self.zoom * (1 / 1.2 if event.key() == Qt.Key.Key_Minus else 1.2))
        elif event.key() == Qt.Key.Key_Home:
            self.fit_volume()
        elif event.key() in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            self.sliceMoved.emit(self.axis, step if event.key() == Qt.Key.Key_PageUp else -step)
        elif event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            directions: dict[int, np.ndarray] = {Qt.Key.Key_Left: -self.u, Qt.Key.Key_Right: self.u, Qt.Key.Key_Up: -self.v, Qt.Key.Key_Down: self.v}
            direction = directions[event.key()]
            self.pointPicked.emit((self.crosshair + direction * step).tolist())
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.tool == "polygon":
                self._finish_polygon()
            elif self.tool in {"paint", "seed", "erase", "barrier"}:
                self._append_stroke(self.crosshair)
                self._emit_stroke()
            else:
                self.pointPicked.emit(self.crosshair.tolist())
        elif event.key() == Qt.Key.Key_Escape:
            self._polygon.clear()
            self._stroke.clear()
            self._dragging = False
            self.update()
        else:
            super().keyPressEvent(event)


class FourViews(QWidget):
    pointPicked = Signal(object)
    paintRequested = Signal(object)
    selectedFragment = Signal(int)
    planeChanged = Signal(object, object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.document: Document | None = None
        self.last_pick_fragment: int | None = None
        self.tool = "navigate"
        self.fragment_mode = False
        self.slices = [SliceView(i, self) for i in range(3)]
        self.sliders: list[QSlider] = []
        self.actors: dict[int, vtkActor] = {}
        self.baseline_actors: dict[int, vtkActor] = {}
        # One immutable scene per interpretation; stage navigation keeps the
        # same actors, mappers and uploaded GPU buffers. A replacement evicts
        # that mode's previous scene, and a new document clears both modes.
        self._scene_cache: dict[
            bool, tuple[dict[int, Mesh], dict[int, vtkActor], dict[int, vtkActor]]
        ] = {}
        self.transforms: dict[int, np.ndarray] = {}
        self._visibility: dict[bool, dict[int, bool]] = {False: {}, True: {}}
        self.landmark_actors: list[vtkCaptionActor2D] = []
        self._baseline = False
        self._plane_updating = False
        self._camera_initialized = False
        self._3d_expanded = False
        self._slice_panels: list[QWidget] = []
        layout = self._grid = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for i, view in enumerate(self.slices):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(0)
            panel_layout.addWidget(view, 1)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setAccessibleName(f"{view.title} slice position")
            slider.valueChanged.connect(lambda value, target=view: self._slider_moved(target, value))
            panel_layout.addWidget(slider)
            self.sliders.append(slider)
            self._slice_panels.append(panel)
            layout.addWidget(panel, i // 2, i % 2)
            view.pointPicked.connect(self._point_picked)
            view.strokeFinished.connect(self.paintRequested.emit)
            view.sliceMoved.connect(self._slice_moved)
        vtk_panel = self._vtk_panel = QWidget()
        vtk_layout = QVBoxLayout(vtk_panel)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        vtk_layout.setSpacing(0)
        self.vtk = QVTKRenderWindowInteractor(vtk_panel)
        self.vtk.setMinimumSize(180, 150)
        self.vtk.setAccessibleName("3D bone fragments")
        self.vtk.setToolTip("Drag: orbit. Middle drag: pan. Wheel: zoom. Pick tool: select anatomy.")
        vtk_layout.addWidget(self.vtk, 1)
        self.view_label = QLabel("3D · drag to orbit · wheel to zoom")
        self.view_label.setStyleSheet("background: #000000; color: #ffffff; padding: 5px;")
        vtk_layout.addWidget(self.view_label)
        layout.addWidget(vtk_panel, 1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.vtk.GetRenderWindow().AddRenderer(self.renderer)
        self.pivot_actor = self._make_caption("Pivot", (1, 0.94, 0.38))
        self.pivot_actor.SetVisibility(False)
        self.renderer.AddViewProp(self.pivot_actor)
        self.interactor = self.vtk.GetRenderWindow().GetInteractor()
        orbit = vtkInteractorStyleTerrain()
        # Terrain orbit retains the anatomical up direction, avoiding the roll
        # accumulated by trackball rotation. Supply its missing wheel zoom.
        orbit.AddObserver(vtkCommand.MouseWheelForwardEvent, lambda *_: self._zoom_3d(1.21))
        orbit.AddObserver(vtkCommand.MouseWheelBackwardEvent, lambda *_: self._zoom_3d(1 / 1.21))
        self.interactor.SetInteractorStyle(orbit)
        self.interactor.AddObserver(vtkCommand.LeftButtonPressEvent, self._pick_3d, 1.0)
        self.plane_rep = vtkImplicitPlaneRepresentation()
        self.plane_rep.SetPlaceFactor(1.1)
        self.plane_rep.DrawPlaneOff()
        self.plane_rep.GetOutlineProperty().SetOpacity(0.15)
        self.plane_rep.GetPlaneProperty().SetColor(1, 0.9, 0.2)
        self.plane_widget = vtkImplicitPlaneWidget2()
        self.plane_widget.SetInteractor(self.interactor)
        self.plane_widget.SetRepresentation(self.plane_rep)
        self.plane_widget.AddObserver(vtkCommand.EndInteractionEvent, self._plane_changed)

    @property
    def is_3d_expanded(self) -> bool:
        return self._3d_expanded

    def set_3d_expanded(self, enabled: bool) -> None:
        """Fill this workspace with 3D, preserving the camera and slice editors."""
        if enabled == self._3d_expanded:
            return
        self._3d_expanded = enabled
        for panel in self._slice_panels:
            panel.setVisible(not enabled)
        # Keep the native VTK window parented and alive while changing its span.
        self._grid.removeWidget(self._vtk_panel)
        if enabled:
            self._grid.addWidget(self._vtk_panel, 0, 0, 2, 2)
        else:
            self._grid.addWidget(self._vtk_panel, 1, 1)
        self._grid.activate()
        self.request_render()

    def initialize(self) -> None:
        self.vtk.Initialize()
        self.request_render()

    def request_render(self) -> None:
        """Let Qt coalesce state changes and exposes into one native paint."""
        # QVTK's paintEvent renders the native scene. A separate render timer
        # would draw it again when the queued Qt paint is subsequently handled.
        self.vtk.update()

    def _zoom_3d(self, factor: float) -> None:
        camera = self.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            camera.SetParallelScale(camera.GetParallelScale() / factor)
        else:
            camera.Dolly(factor)
        self.renderer.ResetCameraClippingRange()
        self.request_render()

    @staticmethod
    def _make_caption(name: str, color: tuple[float, float, float]) -> vtkCaptionActor2D:
        caption = vtkCaptionActor2D()
        caption.SetCaption(name)
        caption.BorderOff()
        caption.ThreeDimensionalLeaderOff()
        caption.SetPosition(16, 14)
        caption.SetWidth(0.2)
        caption.SetHeight(0.04)
        caption.GetTextActor().SetTextScaleModeToNone()
        text = caption.GetCaptionTextProperty()
        text.SetFontSize(15)
        text.BoldOn()
        text.ItalicOff()
        text.ShadowOff()
        text.SetColor(*color)
        text.SetBackgroundColor(0, 0, 0)
        text.SetBackgroundOpacity(0.7)
        caption.GetProperty().SetColor(*color)
        glyph = vtkSphereSource()
        glyph.SetRadius(0.5)
        caption.SetLeaderGlyphConnection(glyph.GetOutputPort())
        caption.SetMaximumLeaderGlyphSize(9)
        caption.SetLeaderGlyphSize(0.014)
        caption.PickableOff()
        return caption

    def set_pivot(self, point_lps: Any | None) -> None:
        """Display a baseline-space rotation pivot, including through bone."""
        point = None if point_lps is None else np.asarray(point_lps, dtype=float)
        self.pivot_actor.SetVisibility(point is not None)
        if point is not None:
            self.pivot_actor.SetAttachmentPoint(*point)
        for view in self.slices:
            view._pivot = point
            view.update()
        self.request_render()

    def set_landmarks(self, landmarks: list[dict[str, Any]]) -> None:
        """Accept named baseline ``point`` and transformed ``moved`` LPS points."""
        for actor in self.landmark_actors:
            self.renderer.RemoveViewProp(actor)
        self.landmark_actors.clear()
        for landmark in landmarks:
            actor = self._make_caption(landmark["name"], (1, 1, 1))
            actor.SetAttachmentPoint(*landmark["moved"])
            self.renderer.AddViewProp(actor)
            self.landmark_actors.append(actor)
        for view in self.slices:
            view._landmarks = landmarks
            view.update()
        self.request_render()

    def set_visibility(self, label: int, visible: bool) -> None:
        """Toggle a 3D surface; slice ownership remains available for editing."""
        self._visibility[self.fragment_mode][label] = visible
        if label in self.actors:
            self.actors[label].SetVisibility(visible)
            self.baseline_actors[label].SetVisibility(visible and self._baseline)
        self.request_render()

    def is_visible(self, label: int) -> bool:
        return self._visibility[self.fragment_mode].get(label, True)

    def set_document(self, document: Document) -> None:
        for actor in list(self.actors.values()) + list(self.baseline_actors.values()):
            self.renderer.RemoveActor(actor)
        self.actors = {}
        self.baseline_actors = {}
        self._scene_cache.clear()
        self.document = document
        self.transforms = {}
        self._visibility = {False: {}, True: {}}
        self.set_pivot(None)
        self.set_landmarks([])
        self._camera_initialized = False
        self.fragment_mode = False
        for view in self.slices:
            view.set_document(document)
        low, high = world_bounds(document.geometry)
        self.plane_rep.PlaceWidget(np.column_stack((low, high)).reshape(-1).tolist())
        for view, slider in zip(self.slices, self.sliders, strict=True):
            slider.blockSignals(True)
            slider.setRange(0, round((high[view.axis] - low[view.axis]) / min(document.geometry.spacing)))
            slider.blockSignals(False)
        self.set_crosshair(np.mean([low, high], axis=0))

    def _slider_moved(self, view: SliceView, value: int) -> None:
        if self.document is None:
            return
        low, _ = world_bounds(self.document.geometry)
        point = view.crosshair.copy()
        point[view.axis] = low[view.axis] + value * min(self.document.geometry.spacing)
        self.set_crosshair(point)

    def _slice_moved(self, axis: int, amount: float) -> None:
        point = self.slices[0].crosshair.copy()
        point[axis] += amount
        self.set_crosshair(point)

    def _point_picked(self, point: Any) -> None:
        self.last_pick_fragment = None
        self.set_crosshair(point)
        self.pointPicked.emit(point)

    def set_crosshair(self, world: Any) -> None:
        if self.document is None:
            return
        low, high = world_bounds(self.document.geometry)
        point = np.clip(np.asarray(world, dtype=float), low, high)
        for view, slider in zip(self.slices, self.sliders, strict=True):
            changed_slice = view.crosshair[view.axis] != point[view.axis]
            view.crosshair = point.copy()
            if changed_slice:
                view.invalidate()
            else:
                view.update()
            slider.blockSignals(True)
            slider.setValue(round((point[view.axis] - low[view.axis]) / min(self.document.geometry.spacing)))
            slider.blockSignals(False)

    def refresh_slices(self, labels: np.ndarray | None = None) -> None:
        if self.document is None:
            return
        selected = labels if labels is not None else (self.document.fragments if self.fragment_mode and self.document.fragments is not None else self.document.anatomy)
        for view in self.slices:
            view.labels = selected
            view.fragment_mode = self.fragment_mode
            view.invalidate()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        for view in self.slices:
            view.tool = tool
            view._polygon.clear()
            view.update()

    def set_label(self, label: int) -> None:
        for view in self.slices:
            view.label = label

    def set_brush_mm(self, diameter: float) -> None:
        for view in self.slices:
            view.brush_mm = max(0.1, float(diameter))
            view.update()

    def invalidate_surfaces(self, fragment_mode: bool) -> None:
        """Release stale display geometry, including its selectable actors."""
        cached = self._scene_cache.pop(fragment_mode, None)
        if cached is not None and self.actors is cached[1]:
            for actor in list(self.actors.values()) + list(self.baseline_actors.values()):
                self.renderer.RemoveActor(actor)
            self.actors = {}
            self.baseline_actors = {}
            self.request_render()

    def set_surfaces(self, meshes: dict[int, Mesh], fragment_mode: bool = True) -> None:
        self.fragment_mode = fragment_mode
        cached = self._scene_cache.get(fragment_mode)
        if cached is None or cached[0] is not meshes:
            cached = (meshes, *self._make_actors(meshes, fragment_mode))
            self._scene_cache[fragment_mode] = cached
        scene_changed = self.actors is not cached[1]
        if scene_changed:
            for actor in list(self.actors.values()) + list(self.baseline_actors.values()):
                self.renderer.RemoveActor(actor)
            self.actors, self.baseline_actors = cached[1:]
            for actor in list(self.actors.values()) + list(self.baseline_actors.values()):
                self.renderer.AddActor(actor)
        for label, actor in self.actors.items():
            actor.SetVisibility(self.is_visible(label))
            self.baseline_actors[label].SetVisibility(self._baseline and self.is_visible(label))
        self.set_transforms(self.transforms)
        if not self._camera_initialized and meshes:
            self.profile_view()
            self._camera_initialized = True
        if self.document is not None:
            selected = self.document.fragments if fragment_mode and self.document.fragments is not None else self.document.anatomy
            if any(view.labels is not selected or view.fragment_mode != fragment_mode for view in self.slices):
                self.refresh_slices()
        if scene_changed:
            self.request_render()

    def _make_actors(
        self, meshes: dict[int, Mesh], fragment_mode: bool
    ) -> tuple[dict[int, vtkActor], dict[int, vtkActor]]:
        actors: dict[int, vtkActor] = {}
        baselines: dict[int, vtkActor] = {}
        colors = COLORS if fragment_mode else ANATOMY_COLORS
        for label, mesh in meshes.items():
            points = vtkPoints()
            points.SetData(numpy_to_vtk(mesh.vertices, deep=False))
            cells = vtkCellArray()
            cells.SetData(3, numpy_to_vtk(mesh.faces.reshape(-1), deep=False))
            poly = vtkPolyData()
            poly.SetPoints(points)
            poly.SetPolys(cells)
            poly.GetPointData().SetNormals(numpy_to_vtk(mesh.normals, deep=False))
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*colors.get(label, (0.8, 0.8, 0.8)))
            actor.GetProperty().SetInterpolationToPhong()
            actor.SetVisibility(self._visibility[fragment_mode].get(label, True))
            actors[label] = actor
            baseline = vtkActor()
            baseline.SetMapper(mapper)
            baseline.GetProperty().SetColor(0.85, 0.85, 0.85)
            baseline.GetProperty().SetOpacity(0.16)
            baseline.GetProperty().SetRepresentationToWireframe()
            baseline.PickableOff()
            baseline.SetVisibility(self._baseline and self._visibility[fragment_mode].get(label, True))
            baselines[label] = baseline
        return actors, baselines

    def set_transforms(self, transforms: dict[int, np.ndarray]) -> None:
        self.transforms = transforms if self.fragment_mode else {}
        changed = False
        for label, actor in self.actors.items():
            matrix = np.asarray(self.transforms.get(label, np.eye(4)))
            existing = actor.GetUserMatrix()
            if existing is not None and all(
                existing.GetElement(row, column) == matrix[row, column]
                for row in range(4)
                for column in range(4)
            ):
                continue
            vtk_matrix = vtkMatrix4x4()
            for row in range(4):
                for column in range(4):
                    vtk_matrix.SetElement(row, column, float(matrix[row, column]))
            actor.SetUserMatrix(vtk_matrix)
            changed = True
        if changed:
            self.renderer.ResetCameraClippingRange()
            self.request_render()

    def show_baseline(self, visible: bool) -> None:
        self._baseline = visible
        for label, actor in self.baseline_actors.items():
            actor.SetVisibility(visible and self._visibility[self.fragment_mode].get(label, True))
        self.request_render()

    def _export_direction(self, planning_vector: np.ndarray) -> np.ndarray:
        """Convert a planning-frame direction into a normalized LPS vector."""
        assert self.document is not None
        origin = self.document.plan_to_world([0, 0, 0])
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_euler("xyz", self.document.view_angles, degrees=True)
        direction = self.document.plan_to_world(rotation.inv().apply(planning_vector)) - origin
        return direction / np.linalg.norm(direction)

    @staticmethod
    def _visible_bounds(actors: list[vtkActor]) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.asarray([actor.GetBounds() for actor in actors], dtype=float).reshape(-1, 3, 2)
        return bounds[:, :, 0].min(axis=0), bounds[:, :, 1].max(axis=0)

    def _set_export_camera(
        self, direction: np.ndarray, up: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> None:
        camera = self.renderer.GetActiveCamera()
        center = (low + high) / 2
        distance = max(float(np.max(high - low)), 1.0) * 2
        camera.ParallelProjectionOn()
        self.renderer.ResetCamera(
            float(low[0]),
            float(high[0]),
            float(low[1]),
            float(high[1]),
            float(low[2]),
            float(high[2]),
        )
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + direction * distance))
        camera.SetViewUp(*up)
        self.renderer.ResetCameraClippingRange()

    @staticmethod
    def _image_from_vtk(image: Any) -> QImage:
        dimensions = image.GetDimensions()
        width, height = int(dimensions[0]), int(dimensions[1])
        scalars = image.GetPointData().GetScalars()
        if width <= 0 or height <= 0 or scalars is None:
            raise RuntimeError("VTK did not return an image for export.")
        pixels = vtk_to_numpy(scalars)
        components = int(image.GetNumberOfScalarComponents())
        if components not in (3, 4) or pixels.size != width * height * components:
            raise RuntimeError("VTK returned an unsupported image format for export.")
        pixels = np.ascontiguousarray(np.flipud(pixels.reshape(height, width, components)))
        format_ = QImage.Format.Format_RGB888 if components == 3 else QImage.Format.Format_RGBA8888
        return QImage(pixels.data, width, height, pixels.strides[0], format_).copy()

    @staticmethod
    def _save_png(image: QImage, path: Path) -> None:
        if image.isNull() or not image.save(str(path)):
            raise RuntimeError(f"Could not write {path.name}.")

    @staticmethod
    def _contact_sheet(images: list[tuple[str, QImage]]) -> QImage:
        size = EXPORT_IMAGE_SIZE
        sheet = QImage(size * 2, size * 2, QImage.Format.Format_RGB32)
        sheet.fill(QColor("#000000"))
        painter = QPainter(sheet)
        font = QFont()
        font.setPixelSize(38)
        font.setBold(True)
        painter.setFont(font)
        for index, (label, image) in enumerate(images):
            x, y = (index % 2) * size, (index // 2) * size
            painter.drawImage(x, y, image)
            painter.fillRect(x + 18, y + 18, 290, 58, QColor(0, 0, 0, 190))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(x + 32, y + 59, label)
        painter.end()
        return sheet

    def _actor_matrices(self) -> dict[int, Any]:
        return {label: actor.GetUserMatrix() for label, actor in self.actors.items()}

    def _restore_matrices(self, matrices: dict[int, Any]) -> None:
        for label, matrix in matrices.items():
            self.actors[label].SetUserMatrix(matrix)

    def _clear_matrices(self) -> None:
        for actor in self.actors.values():
            actor.SetUserMatrix(vtkMatrix4x4())

    def comparison_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Visible bounds covering both baseline and moved positions."""
        visible = [actor for actor in self.actors.values() if actor.GetVisibility()]
        if not visible:
            raise ValueError("Select at least one visible structure before exporting scenario images.")
        moved = self._visible_bounds(visible)
        matrices = self._actor_matrices()
        try:
            self._clear_matrices()
            baseline = self._visible_bounds(visible)
        finally:
            self._restore_matrices(matrices)
        return np.minimum(moved[0], baseline[0]), np.maximum(moved[1], baseline[1])

    def render_views(
        self,
        views: tuple[tuple[str, str, np.ndarray, np.ndarray], ...],
        size: int,
        *,
        baseline: bool = False,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[tuple[str, QImage]]:
        """Render standardized square views, restoring the interactive scene."""
        if self.document is None or self.document.fragments is None or not self.fragment_mode:
            raise ValueError("Build and review the selected cuts before exporting scenario images.")
        visible = [actor for actor in self.actors.values() if actor.GetVisibility()]
        if not visible:
            raise ValueError("Select at least one visible structure before exporting scenario images.")
        camera = self.renderer.GetActiveCamera()
        camera_before = vtkCamera()
        camera_before.DeepCopy(camera)
        viewport_before = self.renderer.GetViewport()
        hidden = list(self.baseline_actors.values()) + self.landmark_actors + [self.pivot_actor]
        visibility_before = [(actor, actor.GetVisibility()) for actor in hidden]
        plane_enabled = bool(self.plane_widget.GetEnabled())
        matrices = self._actor_matrices()
        try:
            for actor in hidden:
                actor.SetVisibility(False)
            if plane_enabled:
                self.plane_widget.SetEnabled(False)
            if baseline:
                self._clear_matrices()

            window_size = self.vtk.GetRenderWindow().GetSize()
            side = min(window_size)
            if side <= 0:
                raise RuntimeError("The 3D view is not ready for image export.")
            # Large-image capture tiles the whole window. An inset viewport
            # repeats its black margins inside every tile, cutting bands out
            # of the anatomy. Capture the full viewport and crop only afterward.
            self.renderer.SetViewport(0, 0, 1, 1)
            low, high = bounds if bounds is not None else self._visible_bounds(visible)
            magnification = max(1, -(-size // side))
            rendered: list[tuple[str, QImage]] = []
            for _, label, direction, up in views:
                self._set_export_camera(
                    self._export_direction(direction), self._export_direction(up), low, high
                )
                # Fit a bounding sphere inside the final square crop, including
                # portrait windows, independently of the interactive camera.
                camera.SetParallelScale(
                    max(float(np.linalg.norm(high - low)), 1.0) * 0.55
                    * window_size[1] / side
                )
                image_filter = vtkRenderLargeImage()
                image_filter.SetInput(self.renderer)
                image_filter.SetMagnification(magnification)
                image_filter.Update()
                image = self._image_from_vtk(image_filter.GetOutput())
                crop_size = min(image.width(), image.height())
                image = image.copy(
                    (image.width() - crop_size) // 2,
                    (image.height() - crop_size) // 2,
                    crop_size,
                    crop_size,
                ).scaled(
                    size,
                    size,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                rendered.append((label, image))
            return rendered
        finally:
            self._restore_matrices(matrices)
            self.renderer.SetViewport(*viewport_before)
            camera.DeepCopy(camera_before)
            for actor, visible_before in visibility_before:
                actor.SetVisibility(visible_before)
            if plane_enabled:
                self.plane_widget.SetEnabled(True)
            self.request_render()

    def export_scenario_images(self, output_dir: str | Path) -> list[Path]:
        """Export four standardized moved-fragment views and a contact sheet."""
        if self.document is None or self.document.fragments is None or not self.fragment_mode:
            raise ValueError("Build and review the selected cuts before exporting scenario images.")
        if not any(actor.GetVisibility() for actor in self.actors.values()):
            raise ValueError("Select at least one visible structure before exporting scenario images.")

        destination = Path(output_dir).absolute()
        if destination.is_symlink() or destination.exists():
            raise ValueError("Image export destination must be a new folder.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".bimax-image-export-", dir=destination.parent))
        try:
            rendered = self.render_views(EXPORT_VIEWS, EXPORT_IMAGE_SIZE)
            for (filename, *_), (_, image) in zip(EXPORT_VIEWS, rendered, strict=True):
                self._save_png(image, temporary / f"{filename}.png")
            self._save_png(self._contact_sheet(rendered), temporary / "contact-sheet.png")
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return [destination / f"{filename}.png" for filename, *_ in EXPORT_VIEWS] + [
            destination / "contact-sheet.png"
        ]

    def profile_view(self) -> None:
        if self.document is None:
            return
        low, high = world_bounds(self.document.geometry)
        center = (low + high) / 2
        # Look from planning-frame right; anterior is screen-right and superior
        # is up, including after the user corrects the head orientation.
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        right = self._export_direction(np.asarray([1, 0, 0]))
        superior = self._export_direction(np.asarray([0, 0, 1]))
        camera.SetPosition(*(center + right * max(high - low) * 2))
        camera.SetViewUp(*superior)
        camera.ParallelProjectionOn()
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.view_label.setText("Right profile · anterior → · positive pitch: counterclockwise")
        self.request_render()

    def set_plane(self, origin_lps: Any, normal_lps: Any, enabled: bool = True) -> None:
        self._plane_updating = True
        origin = np.asarray(origin_lps, dtype=float)
        normal = np.asarray(normal_lps, dtype=float)
        normal /= max(1e-12, np.linalg.norm(normal))
        self.plane_rep.SetOrigin(*origin)
        self.plane_rep.SetNormal(*normal)
        self.plane_widget.SetEnabled(enabled)
        for view in self.slices:
            view._plane = (origin, normal) if enabled else None
            view.update()
        self._plane_updating = False
        self.request_render()

    def _plane_changed(self, _caller: Any, _event: Any) -> None:
        if self._plane_updating:
            return
        plane = vtkPlane()
        self.plane_rep.GetPlane(plane)
        origin, normal = list(plane.GetOrigin()), list(plane.GetNormal())
        for view in self.slices:
            view._plane = (np.asarray(origin), np.asarray(normal))
            view.update()
        self.planeChanged.emit(origin, normal)

    def _pick_3d(self, _caller: Any, _event: Any) -> None:
        if self.tool not in {"pick", "seed", "paint"}:
            return
        x, y = self.interactor.GetEventPosition()
        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        if not picker.Pick(x, y, 0, self.renderer):
            return
        picked_actor = picker.GetActor()
        for label, actor in self.actors.items():
            if actor == picked_actor:
                world = np.asarray([*picker.GetPickPosition(), 1.0])
                baseline = np.linalg.inv(self.transforms.get(label, np.eye(4))) @ world
                self.last_pick_fragment = label
                self.selectedFragment.emit(label)
                self.set_crosshair(baseline[:3])
                self.pointPicked.emit(baseline[:3].tolist())
                break

    def closeEvent(self, event: Any) -> None:
        self.plane_widget.SetEnabled(False)
        self.vtk.Finalize()
        super().closeEvent(event)
