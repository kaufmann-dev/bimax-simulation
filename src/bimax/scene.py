"""Offscreen VTK scene of a prepared scenario.

Scenario images, report images, previews and picking render here without Qt widgets or a
display, so the desktop app and ``bimax-cli`` produce identical pictures. Meshes and actor
matrices are in physical LPS millimetres; picks return baseline planning Right/Anterior/Superior
millimetres.
"""

import os
import shutil
import tempfile
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

# Registers the OpenGL rendering backend (EGL or OSMesa when there is no display).
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from scipy.spatial.transform import Rotation
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkWindowToImageFilter,
)

from .engine import fragment_transforms, motion_matrix
from .model import COLORS, Document, Geometry, Motion
from .surfaces import Mesh

EXPORT_VIEWS = (
    ("right", "Right", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ("left", "Left", np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ("front", "Front", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ("birds-eye", "Bird's-eye", np.array([0.0, np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40))]), np.array([0.0, 0.0, 1.0])),
)
EXPORT_IMAGE_SIZE = 2048
# Report views: lateral and frontal first, then obliques and the submental view.
REPORT_VIEWS = (
    EXPORT_VIEWS[0],
    EXPORT_VIEWS[2],
    EXPORT_VIEWS[1],
    ("front-right", "Front-right oblique", np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0]), np.array([0.0, 0.0, 1.0])),
    ("front-left", "Front-left oblique", np.array([-np.sqrt(0.5), np.sqrt(0.5), 0.0]), np.array([0.0, 0.0, 1.0])),
    ("inferior", "Inferior", np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
)
REPORT_IMAGE_SIZE = 1600
SLICE_VIEWS = {"axial": (2, 0, 1), "coronal": (1, 0, 2), "sagittal": (0, 1, 2)}  # normal, u, v
VIEW_NAMES = tuple(name for name, *_ in EXPORT_VIEWS) + tuple(SLICE_VIEWS)
PREVIEW_VIEWS = ("contact-sheet", *VIEW_NAMES)


def sample_indices(geometry: Geometry, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor sampling without negative indexing outside the volume."""
    xyz = np.floor(geometry.world_to_index(world) + 0.5).astype(np.int64)
    valid = np.all((xyz >= 0) & (xyz < np.asarray(geometry.shape[::-1])), axis=-1)
    safe = np.clip(xyz, 0, np.asarray(geometry.shape[::-1]) - 1)
    flat = np.ravel_multi_index(safe[..., ::-1].reshape(-1, 3).T, geometry.shape)
    return np.asarray(flat).reshape(valid.shape), valid


def surface_actor(mesh: Mesh, color: tuple[float, float, float]) -> vtkActor:
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
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetInterpolationToPhong()
    return actor


def vtk_matrix(matrix: np.ndarray) -> vtkMatrix4x4:
    result = vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            result.SetElement(row, column, float(matrix[row, column]))
    return result


def export_direction(doc: Document, planning_vector: np.ndarray) -> np.ndarray:
    """Convert a planning-frame direction into a normalized LPS vector, honoring the saved
    whole-skull display orientation."""
    origin = doc.plan_to_world([0, 0, 0])
    rotation = Rotation.from_euler("xyz", doc.view_angles, degrees=True)
    direction = doc.plan_to_world(rotation.inv().apply(planning_vector)) - origin
    return direction / np.linalg.norm(direction)


def visible_bounds(actors: list[vtkActor]) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray([actor.GetBounds() for actor in actors], dtype=float).reshape(-1, 3, 2)
    return bounds[:, :, 0].min(axis=0), bounds[:, :, 1].max(axis=0)


def image_from_vtk(image: Any) -> QImage:
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


def save_png(image: QImage, path: Path) -> None:
    if image.isNull() or not image.save(str(path)):
        raise RuntimeError(f"Could not write {path.name}.")


def contact_sheet(images: list[tuple[str, QImage]]) -> QImage:
    """Two-by-two sheet of equally sized square views, each labelled in its top-left corner."""
    size = images[0][1].width()
    sheet = QImage(size * 2, size * 2, QImage.Format.Format_RGB32)
    sheet.fill(QColor("#000000"))
    scale = size / EXPORT_IMAGE_SIZE
    font_px = max(12, round(38 * scale))
    pad, box_w, box_h = max(6, round(18 * scale)), max(80, round(290 * scale)), max(font_px + 8, round(58 * scale))
    painter = QPainter(sheet)
    font = QFont()
    font.setPixelSize(font_px)
    font.setBold(True)
    painter.setFont(font)
    for index, (label, image) in enumerate(images):
        x, y = (index % 2) * size, (index // 2) * size
        painter.drawImage(x, y, image)
        painter.fillRect(x + pad, y + pad, box_w, box_h, QColor(0, 0, 0, 190))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(x + pad + round(box_h * 0.24), y + pad + round(box_h * 0.71), label)
    painter.end()
    return sheet


def _view(name: str) -> tuple[str, str, np.ndarray, np.ndarray]:
    return next(view for view in EXPORT_VIEWS if view[0] == name)


class Scene:
    """Prepared fragment surfaces of the active scenario, moved by its planned transforms."""

    fragment_mode = True

    def __init__(self, doc: Document, meshes: dict[int, Mesh], hidden: frozenset[int] | set[int] = frozenset()):
        if doc.fragments is None:
            raise ValueError("Prepare the scenario's cut combination first.")
        self.document = doc
        self.transforms = fragment_transforms(doc)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.window = vtkRenderWindow()
        self.window.SetOffScreenRendering(True)
        self.window.SetMultiSamples(0)
        self.window.AddRenderer(self.renderer)
        self.actors: dict[int, vtkActor] = {}
        for label, mesh in sorted(meshes.items()):
            actor = surface_actor(mesh, COLORS.get(label, (0.8, 0.8, 0.8)))
            actor.SetVisibility(label not in hidden)
            actor.SetUserMatrix(vtk_matrix(self.transforms.get(label, np.eye(4))))
            self.renderer.AddActor(actor)
            self.actors[label] = actor

    def __enter__(self) -> Scene:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.window.Finalize()

    def is_visible(self, label: int) -> bool:
        return label in self.actors and bool(self.actors[label].GetVisibility())

    def _visible(self) -> list[vtkActor]:
        visible = [actor for actor in self.actors.values() if actor.GetVisibility()]
        if not visible:
            raise ValueError("Select at least one visible structure before rendering.")
        return visible

    def _set_matrices(self, baseline: bool) -> None:
        for label, actor in self.actors.items():
            actor.SetUserMatrix(vtk_matrix(np.eye(4) if baseline else self.transforms.get(label, np.eye(4))))

    def comparison_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Visible bounds covering both baseline and moved positions."""
        visible = self._visible()
        moved = visible_bounds(visible)
        try:
            self._set_matrices(baseline=True)
            baseline = visible_bounds(visible)
        finally:
            self._set_matrices(baseline=False)
        return np.minimum(moved[0], baseline[0]), np.maximum(moved[1], baseline[1])

    def _camera(self, direction: np.ndarray, up: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
        """Parallel projection along a planning direction, framing a bounding sphere."""
        camera = self.renderer.GetActiveCamera()
        center = (low + high) / 2
        camera.ParallelProjectionOn()
        self.renderer.ResetCamera(*(float(value) for pair in zip(low, high, strict=True) for value in pair))
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + export_direction(self.document, direction) * max(float(np.max(high - low)), 1.0) * 2))
        camera.SetViewUp(*export_direction(self.document, up))
        camera.SetParallelScale(max(float(np.linalg.norm(high - low)), 1.0) * 0.55)
        self.renderer.ResetCameraClippingRange()

    def _capture(self, size: int) -> QImage:
        self.window.SetSize(size, size)
        self.window.Render()
        grab = vtkWindowToImageFilter()
        grab.SetInput(self.window)
        grab.ReadFrontBufferOff()
        grab.Update()
        return image_from_vtk(grab.GetOutput())

    def render_views(
        self,
        views: tuple[tuple[str, str, np.ndarray, np.ndarray], ...],
        size: int,
        *,
        baseline: bool = False,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[tuple[str, QImage]]:
        """Square views; `baseline` shows every fragment at its pre-operative position."""
        low, high = bounds if bounds is not None else visible_bounds(self._visible())
        rendered = []
        try:
            self._set_matrices(baseline)
            for _, label, direction, up in views:
                self._camera(direction, up, low, high)
                rendered.append((label, self._capture(size)))
        finally:
            self._set_matrices(baseline=False)
        return rendered

    def export_scenario_images(self, output_dir: str | Path) -> list[Path]:
        """Four standardized moved-fragment views and a contact sheet in a new folder."""
        self._visible()
        destination = Path(output_dir).absolute()
        if destination.is_symlink() or destination.exists():
            raise ValueError("Image export destination must be a new folder.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".bimax-image-export-", dir=destination.parent))
        try:
            rendered = self.render_views(EXPORT_VIEWS, EXPORT_IMAGE_SIZE)
            for (filename, *_), (_, image) in zip(EXPORT_VIEWS, rendered, strict=True):
                save_png(image, temporary / f"{filename}.png")
            save_png(contact_sheet(rendered), temporary / "contact-sheet.png")
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return [destination / f"{filename}.png" for filename, *_ in EXPORT_VIEWS] + [
            destination / "contact-sheet.png"
        ]

    def preview(self, view: str, size: int, position: float | None = None) -> QImage:
        """The contact sheet (right, left, front, bird's-eye), one 3D view, or one slice."""
        if view == "contact-sheet":
            return contact_sheet(self.render_views(EXPORT_VIEWS, size // 2))
        if view in SLICE_VIEWS:
            return self._slice(view, size, position)[0]
        if view not in VIEW_NAMES:
            raise ValueError(f"Unknown view {view!r}; use one of {', '.join(PREVIEW_VIEWS)}.")
        return self.render_views((_view(view),), size)[0][1]

    def pick(self, view: str, x: float, y: float, size: int, position: float | None = None) -> dict:
        """Fragment and baseline planning point under normalized top-left image coordinates."""
        if view not in VIEW_NAMES or not np.isfinite([x, y]).all() or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Pick in a single view with normalized coordinates between 0 and 1.")
        if view in SLICE_VIEWS:
            _, labels, points = self._slice(view, size, position)
            row, col = min(int(y * size), size - 1), min(int(x * size), size - 1)
            label, point = int(labels[row, col]), points[row, col]
        else:
            picker = vtkCellPicker()
            picker.SetTolerance(0.001)
            self._set_matrices(baseline=False)
            self._camera(*_view(view)[2:], *visible_bounds(self._visible()))
            self.window.SetSize(size, size)
            self.window.Render()
            if not picker.Pick(min(x * size, size - 1), min((1 - y) * size, size - 1), 0, self.renderer):
                raise ValueError("No fragment at this image point. Pick a visible bone surface.")
            actor = picker.GetActor()
            label = next(key for key, value in self.actors.items() if value is actor)
            inverse = np.linalg.inv(self.transforms.get(label, np.eye(4)))
            point = inverse[:3, :3] @ np.array(picker.GetPickPosition()) + inverse[:3, 3]
        if not label:
            raise ValueError("No fragment at this image point.")
        return {
            "fragment": self.document.fragment_descriptors[label].id,
            "point": self.document.world_to_plan(point).tolist(),
            "coordinates": "baseline planning Right/Anterior/Superior millimetres",
        }

    def _slice(self, name: str, size: int, position: float | None) -> tuple[QImage, np.ndarray, np.ndarray]:
        """Moved fragments cut by a plane; returns the image, labels and baseline LPS points."""
        if position is None or not np.isfinite(position):
            raise ValueError("Slice views require a finite planning position in millimetres.")
        doc = self.document
        axis, u, v = SLICE_VIEWS[name]
        # Slice position is measured along the oriented planning normal.
        rotation = motion_matrix(Motion(angles=doc.view_angles, pivot_set=True))[:3, :3]
        corners = doc.world_to_plan(doc.geometry.index_to_world(
            np.array(list(product(*[(-0.5, n - 0.5) for n in doc.geometry.shape[::-1]])))
        )) @ rotation
        low, high = corners.min(axis=0), corners.max(axis=0)
        extent = max(high[u] - low[u], high[v] - low[v]) * 1.2
        coordinates = (np.arange(size) + 0.5) / size * extent - extent / 2
        plane = np.zeros((size, size, 3))
        plane[..., axis] = position
        plane[..., u] = (low[u] + high[u]) / 2 + coordinates[None, :]
        plane[..., v] = (low[v] + high[v]) / 2 - coordinates[:, None]
        moved_world = doc.plan_to_world(plane @ rotation.T)
        baseline = np.zeros_like(moved_world)
        labels = np.zeros((size, size), dtype=np.uint8)
        fragments = doc.fragments
        assert fragments is not None
        for label in self.actors:
            if not self.is_visible(label):
                continue
            inverse = np.linalg.inv(self.transforms.get(label, np.eye(4)))
            points = moved_world @ inverse[:3, :3].T + inverse[:3, 3]
            flat, valid = sample_indices(doc.geometry, points)
            selected = valid & (fragments.reshape(-1)[flat] == label)
            labels[selected] = label
            baseline[selected] = points[selected]
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        for label, color in COLORS.items():
            rgb[labels == label] = np.array(color) * 255
        image = QImage(rgb.data, size, size, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        return image, labels, baseline
