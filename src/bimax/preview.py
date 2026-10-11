"""Main-thread PNG captures with stable cameras and baseline-space picking."""

import base64
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QColor, QFont, QImage, QImageWriter, QPainter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkRenderer,
    vtkRenderWindow,
    vtkWindowToImageFilter,
)

from .engine import fragment_transforms, motion_matrix
from .model import COLORS, Document, Motion
from .views import EXPORT_VIEWS, FourViews, sample_indices


def png(image: QImage) -> str:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    writer = QImageWriter(buffer, b"png")
    if not writer.write(image):
        raise RuntimeError("PNG capture failed.")
    return base64.b64encode(buffer.data().data()).decode("ascii")


@dataclass
class Capture:
    doc: Document
    views: Any
    names: list[str]
    position: float | None
    size: int
    images: dict[str, QImage] = field(default_factory=dict)
    scenes: dict[str, tuple[Any, Any, dict[Any, int]]] = field(default_factory=dict)
    slices: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def render(self, name: str) -> None:
        if name in ("axial", "coronal", "sagittal"):
            self._slice(name)
            return
        renderer = vtkRenderer()
        renderer.SetBackground(0, 0, 0)
        window = vtkRenderWindow()
        window.SetOffScreenRendering(True)
        window.SetSize(self.size, self.size)
        window.SetMultiSamples(0)
        window.AddRenderer(renderer)
        actor_labels = {}
        actors = []
        for label, original in self.views.actors.items():
            actor = vtkActor()
            actor.SetMapper(original.GetMapper())
            actor.GetProperty().DeepCopy(original.GetProperty())
            actor.SetUserMatrix(original.GetUserMatrix())
            actor.SetVisibility(original.GetVisibility())
            renderer.AddActor(actor)
            actor_labels[actor] = label
            if actor.GetVisibility():
                actors.append(actor)
        if not actors:
            window.Finalize()
            raise ValueError("Enable at least one fragment in Visible structures.")
        low, high = FourViews._visible_bounds(actors)
        _, _, direction, up = next(item for item in EXPORT_VIEWS if item[0] == name)
        camera = renderer.GetActiveCamera()
        center = (low + high) / 2
        direction = self.views._export_direction(direction)
        up = self.views._export_direction(up)
        camera.SetPosition(*(center + direction * max(float(np.linalg.norm(high - low)), 1) * 3))
        camera.SetFocalPoint(*center)
        camera.SetViewUp(*up)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(max(float(np.linalg.norm(high - low)), 1) * 0.55)
        renderer.ResetCameraClippingRange()
        window.Render()
        image_filter = vtkWindowToImageFilter()
        image_filter.SetInput(window)
        image_filter.ReadFrontBufferOff()
        image_filter.Update()
        self.images[name] = FourViews._image_from_vtk(image_filter.GetOutput())
        self.scenes[name] = (window, renderer, actor_labels)

    def _slice(self, name: str) -> None:
        if self.position is None or not np.isfinite(self.position):
            raise ValueError("Slice previews require a finite planning position in millimetres.")
        # Slice position is measured along the oriented planning normal.
        axis, u, v = {"axial": (2, 0, 1), "coronal": (1, 0, 2), "sagittal": (0, 1, 2)}[name]
        rotation = motion_matrix(Motion(angles=self.doc.view_angles, pivot_set=True))[:3, :3]
        corners = (
            self.doc.world_to_plan(
                self.doc.geometry.index_to_world(
                    np.array(
                        list(product(*[(-0.5, n - 0.5) for n in self.doc.geometry.shape[::-1]]))
                    )
                )
            )
            @ rotation
        )
        low, high = corners.min(axis=0), corners.max(axis=0)
        extent = max(high[u] - low[u], high[v] - low[v]) * 1.2
        coordinates = (np.arange(self.size) + 0.5) / self.size * extent - extent / 2
        plane = np.zeros((self.size, self.size, 3))
        plane[..., axis] = self.position
        plane[..., u] = (low[u] + high[u]) / 2 + coordinates[None, :]
        plane[..., v] = (low[v] + high[v]) / 2 - coordinates[:, None]
        moved_world = self.doc.plan_to_world(plane @ rotation.T)
        baseline = np.zeros_like(moved_world)
        labels = np.zeros((self.size, self.size), dtype=np.uint8)
        fragments = self.doc.fragments
        assert fragments is not None
        for label, matrix in fragment_transforms(self.doc).items():
            if not self.views.is_visible(label):
                continue
            inverse = np.linalg.inv(matrix)
            points = moved_world @ inverse[:3, :3].T + inverse[:3, 3]
            flat, valid = sample_indices(self.doc.geometry, points)
            selected = valid & (fragments.reshape(-1)[flat] == label)
            labels[selected] = label
            baseline[selected] = points[selected]
        rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for label, color in COLORS.items():
            rgb[labels == label] = np.array(color) * 255
        self.images[name] = QImage(
            rgb.data, self.size, self.size, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.slices[name] = (labels, baseline)

    def result(self) -> str:
        if len(self.names) == 1:
            return png(self.images[self.names[0]])
        sheet = QImage(1024, 1024, QImage.Format.Format_RGB32)
        sheet.fill(QColor("black"))
        painter = QPainter(sheet)
        font = QFont()
        font.setPixelSize(18)
        painter.setFont(font)
        for index, name in enumerate(self.names):
            x, y = index % 2 * 512, index // 2 * 512
            painter.drawImage(
                x, y, self.images[name].scaled(512, 512, Qt.AspectRatioMode.IgnoreAspectRatio)
            )
            painter.setPen(QColor("white"))
            painter.drawText(x + 12, y + 24, name)
        painter.end()
        return png(sheet)

    def pick(self, name: str, x: float, y: float) -> dict:
        if (
            name not in self.images
            or not np.isfinite([x, y]).all()
            or not (0 <= x <= 1 and 0 <= y <= 1)
        ):
            raise ValueError("Use a captured view name and normalized coordinates between 0 and 1.")
        if name in self.slices:
            labels, points = self.slices[name]
            row, col = (
                min(int(y * self.size), self.size - 1),
                min(int(x * self.size), self.size - 1),
            )
            label = int(labels[row, col])
            point = points[row, col]
        else:
            _, renderer, actor_labels = self.scenes[name]
            picker = vtkCellPicker()
            picker.SetTolerance(0.001)
            if not picker.Pick(
                min(x * self.size, self.size - 1),
                min((1 - y) * self.size, self.size - 1),
                0,
                renderer,
            ):
                raise ValueError("No fragment at this image point. Pick a visible bone surface.")
            actor = picker.GetActor()
            label = actor_labels[actor]
            inverse = np.linalg.inv(fragment_transforms(self.doc)[label])
            point = inverse[:3, :3] @ np.array(picker.GetPickPosition()) + inverse[:3, 3]
        if not label:
            raise ValueError("No fragment at this image point.")
        return {
            "fragment": self.doc.fragment_descriptors[label].id,
            "point": self.doc.world_to_plan(point).tolist(),
            "conventions": "baseline planning Right/Anterior/Superior millimetres",
        }

    def close(self) -> None:
        for window, _, _ in self.scenes.values():
            window.Finalize()
        self.scenes.clear()
