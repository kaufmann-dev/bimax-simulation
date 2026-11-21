"""Offscreen scene: exports, previews and picking render without a display."""

import os

import numpy as np
import pytest
from factories import confirm, document
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from scipy.spatial.transform import Rotation

import bimax.scene as scene_module
from bimax.prepared import prepare_geometry
from bimax.scene import EXPORT_VIEWS, REPORT_VIEWS, Scene


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def planned(view_angles=(0.0, 0.0, 0.0)):
    """Factory volume with a Le Fort I maxilla (z 9..11) moved 1 mm up."""
    doc = document()
    doc.scenario.cut_ids = ["lefort1"]
    confirm(doc)
    doc.view_angles = list(view_angles)
    doc.motions["lefort1"].translation = [0, 0, 1]
    result = prepare_geometry(doc)
    doc.fragments, doc.fragments_key = result["fragments"], doc.configuration_key
    return doc, result["surfaces"]


def foreground(image: QImage) -> int:
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
    return int((pixels[:, : image.width() * 3].reshape(image.height(), image.width(), 3).max(axis=2) > 20).sum())


@pytest.mark.parametrize("orientation", [[0.0, 0.0, 0.0], [17.0, -9.0, 23.0]])
def test_cameras_follow_the_planning_frame_and_orientation(qapp, monkeypatch, orientation):
    doc, meshes = planned(orientation)
    seen = []

    def capture(self, size):
        camera = self.renderer.GetActiveCamera()
        position, focal = np.asarray(camera.GetPosition()), np.asarray(camera.GetFocalPoint())
        seen.append(((position - focal) / np.linalg.norm(position - focal), np.asarray(camera.GetViewUp())))
        return QImage(size, size, QImage.Format.Format_RGB32)

    monkeypatch.setattr(Scene, "_capture", capture)
    with Scene(doc, meshes) as scene:
        rendered = scene.render_views(EXPORT_VIEWS, 32)
    assert [label for label, _ in rendered] == [label for _, label, *_ in EXPORT_VIEWS]
    origin = doc.plan_to_world([0, 0, 0])
    inverse = Rotation.from_euler("xyz", orientation, degrees=True).inv()

    def expected(vector):
        moved = doc.plan_to_world(inverse.apply(vector)) - origin
        return moved / np.linalg.norm(moved)

    for (direction, up), (_, _, view_direction, view_up) in zip(seen, EXPORT_VIEWS, strict=True):
        np.testing.assert_allclose(direction, expected(view_direction), atol=1e-9)
        np.testing.assert_allclose(up / np.linalg.norm(up), expected(view_up), atol=1e-9)


def test_baseline_views_share_bounds_and_restore_the_planned_positions(qapp, monkeypatch):
    doc, meshes = planned()
    matrices = []

    def capture(self, size):
        matrices.append(self.actors[2].GetUserMatrix().GetElement(2, 3))
        return QImage(size, size, QImage.Format.Format_RGB32)

    monkeypatch.setattr(Scene, "_capture", capture)
    with Scene(doc, meshes) as scene:
        low, high = scene.comparison_bounds()
        scene.render_views(REPORT_VIEWS, 16, baseline=True, bounds=(low, high))
        assert matrices == [0.0] * len(REPORT_VIEWS)
        assert scene.actors[2].GetUserMatrix().GetElement(2, 3) == 1.0
        moved = scene_module.visible_bounds([scene.actors[2]])
        assert high[2] >= moved[1][2] and low[2] <= moved[0][2] - 1.0 + 1e-9


def test_export_scenario_images_writes_four_views_and_a_contact_sheet(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(scene_module, "EXPORT_IMAGE_SIZE", 96)
    doc, meshes = planned()
    with Scene(doc, meshes) as scene:
        exported = scene.export_scenario_images(tmp_path / "images")
        with pytest.raises(ValueError, match="new folder"):
            scene.export_scenario_images(tmp_path / "images")
    assert [path.name for path in exported] == [
        "right.png", "left.png", "front.png", "birds-eye.png", "contact-sheet.png"
    ]
    for path in exported[:4]:
        image = QImage(str(path))
        assert image.width() == image.height() == 96
        assert foreground(image) > 500, "the bone renders"
    assert QImage(str(exported[-1])).width() == 192
    with Scene(doc, meshes, hidden=set(meshes)) as scene:
        with pytest.raises(ValueError, match="visible structure"):
            scene.export_scenario_images(tmp_path / "hidden")


def test_previews_render_sheets_views_and_slices(qapp):
    doc, meshes = planned()
    with Scene(doc, meshes) as scene:
        assert scene.preview("contact-sheet", 128).width() == 128
        assert foreground(scene.preview("front", 64)) > 300
        axial = scene.preview("axial", 64, position=10.5)
        assert axial.width() == 64 and foreground(axial) > 300
        with pytest.raises(ValueError, match="position"):
            scene.preview("coronal", 64)
        with pytest.raises(ValueError, match="Unknown view"):
            scene.preview("top", 64)


def test_picks_return_the_fragment_and_its_baseline_point(qapp):
    doc, meshes = planned()
    with Scene(doc, meshes) as scene:
        # the maxilla moved up 1 mm: at planned height 11.5 the slice shows its baseline 10.5
        sliced = scene.pick("axial", 0.5, 0.5, 256, position=11.5)
        assert sliced["fragment"] == "upper:lefort1"
        assert sliced["point"][2] == pytest.approx(10.5)
        surface = scene.pick("front", 0.5, 0.5, 256)
        assert surface["fragment"] == "upper:lefort1"
        assert 9 - 0.5 <= surface["point"][2] <= 11 + 0.5
        with pytest.raises(ValueError, match="No fragment"):
            scene.pick("front", 0.01, 0.01, 256)
