import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from bimax.model import Document, Geometry
from bimax.surfaces import make_surfaces
from bimax.views import FourViews, SliceView, sample_indices, world_bounds


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        pytest.skip("Native Qt/VTK verification needs DISPLAY; use xvfb-run")
    app = QApplication.instance() or QApplication([])
    yield app


def document(direction=None):
    geometry = Geometry((12, 14, 16), (0.8, 1.2, 1.7), (4.0, -7.0, 2.0), tuple(np.eye(3).ravel()) if direction is None else tuple(direction.ravel()))
    source = np.zeros(geometry.shape, dtype=np.uint8)
    source[2:10, 2:12, 2:14] = 1
    return Document(geometry, source, source * 2, np.zeros_like(source), np.zeros_like(source))


@pytest.mark.parametrize("direction", [np.eye(3), np.diag([-1.0, 1.0, 1.0]), np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]])])
def test_affine_nearest_sampling(direction):
    doc = document(direction)
    xyz = np.array([[0, 0, 0], [3, 5, 7], [15, 13, 11], [-1, 0, 0]])
    world = doc.geometry.index_to_world(xyz)
    flat, valid = sample_indices(doc.geometry, world)
    assert valid.tolist() == [True, True, True, False]
    np.testing.assert_array_equal(flat[:3], np.ravel_multi_index(xyz[:3, ::-1].T, doc.geometry.shape))


def test_surfaces_close_volume_boundary_and_preserve_affine():
    geometry = Geometry((3, 4, 5), (0.5, 1.0, 2.0), (10, 20, 30), tuple(np.diag([-1.0, 1.0, 1.0]).ravel()))
    labels = np.ones(geometry.shape, dtype=np.uint8)
    original = labels.copy()
    vertices, faces = make_surfaces(labels, geometry)[1]
    low, high = world_bounds(geometry)
    np.testing.assert_allclose(vertices.min(axis=0), low)
    np.testing.assert_allclose(vertices.max(axis=0), high)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), "A boundary object must produce a closed manifold"
    np.testing.assert_array_equal(labels, original)


@pytest.mark.parametrize("orientation", [0, 1, 2])
def test_oblique_slice_screen_world_roundtrip(qapp, orientation):
    doc = document(np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, -1]]))
    view = SliceView(orientation)
    view.resize(520, 320)
    view.set_document(doc)
    point = view.crosshair + view.u * 2.1 + view.v * 1.3
    np.testing.assert_allclose(view.screen_to_world(view.world_to_screen(point)), point)
    world = view.pixel_world(20, 30)
    assert world.shape == (30, 20, 3)
    np.testing.assert_allclose(world[..., view.axis], view.crosshair[view.axis])
    view.show()
    qapp.processEvents()
    assert not view.grab().isNull()
    view.close()


def test_slice_brush_is_physical_and_emits_stroke(qapp):
    doc = document()
    view = SliceView(0)
    view.resize(500, 400)
    view.set_document(doc)
    center = doc.geometry.index_to_world([7, 6, 5])
    view.crosshair = center.copy()
    view.brush_mm = 2.1
    indices = view.brush_indices(center)
    z, y, x = np.unravel_index(indices, doc.geometry.shape)
    assert set(z) == {5}
    assert set(y) == {6}
    assert set(x) == {6, 7, 8}
    requests = []
    view.strokeFinished.connect(requests.append)
    view.tool = "seed"
    view.label = 3
    view.show()
    qapp.processEvents()
    pixel = view.world_to_screen(center).toPoint()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=pixel)
    QTest.mouseMove(view, pixel + QPoint(20, 0))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=pixel + QPoint(20, 0))
    assert len(requests) == 1
    assert requests[0]["tool"] == "seed"
    assert requests[0]["finish"] is True
    assert len(requests[0]["indices"]) >= len(indices)
    view.close()


def test_zoom_pan_preserve_physical_picking_and_bound_raster_memory(qapp):
    view = SliceView(0)
    view.resize(520, 360)
    view.set_document(document())
    view.show()
    qapp.processEvents()
    anchor = view.content_rect().center() + QPoint(37, 22)
    world = view.screen_to_world(anchor)
    before_brush = np.sort(view.brush_indices(world))
    view.set_zoom(20, anchor)
    qapp.processEvents()
    np.testing.assert_allclose(view.screen_to_world(anchor), world)
    np.testing.assert_array_equal(np.sort(view.brush_indices(world)), before_brush)
    assert view._image.width() <= view.width()
    assert view._image.height() <= view.height()
    start = anchor.toPoint()
    finish = start + QPoint(40, 20)
    QTest.mousePress(view, Qt.MouseButton.MiddleButton, pos=start)
    QTest.mouseMove(view, finish)
    QTest.mouseRelease(view, Qt.MouseButton.MiddleButton, pos=finish)
    np.testing.assert_allclose(view.screen_to_world(anchor + QPoint(40, 20)), world)
    QTest.keyClick(view, Qt.Key.Key_Home)
    assert view.zoom == 1
    np.testing.assert_allclose(view._pan, 0)
    QTest.keyClick(view, Qt.Key.Key_Plus)
    assert view.zoom > 1
    view.close()


def test_native_four_views_mesh_transform_plane_and_pick(qapp):
    doc = document()
    widget = FourViews()
    widget.resize(900, 700)
    widget.show()
    widget.initialize()
    widget.set_document(doc)
    meshes = make_surfaces(doc.anatomy, doc.geometry)
    widget.set_surfaces(meshes, fragment_mode=True)
    transform = np.eye(4)
    transform[:3, 3] = [0, -3.0, 2.0]
    widget.set_transforms({2: transform})
    actor_matrix = widget.actors[2].GetUserMatrix()
    assert actor_matrix.GetElement(1, 3) == -3.0
    widget.show_baseline(True)
    assert widget.baseline_actors[2].GetVisibility()
    center = doc.geometry.index_to_world([7, 6, 5])
    widget.set_plane(center, [0, 0, 1], True)
    assert widget.plane_widget.GetEnabled()
    assert all(view._plane is not None for view in widget.slices)
    widget.set_plane(center, [0, 0, 1], False)
    widget.profile_view()
    qapp.processEvents()
    # Project a visible face on the right side into the native VTK render window.
    baseline_surface = doc.geometry.index_to_world([1.5, 6, 5])
    moved_surface = baseline_surface + transform[:3, 3]
    widget.renderer.SetWorldPoint(*moved_surface, 1)
    widget.renderer.WorldToDisplay()
    x, y, _ = widget.renderer.GetDisplayPoint()
    picked = []
    widget.pointPicked.connect(picked.append)
    widget.set_tool("pick")
    widget.interactor.SetEventPosition(round(x), round(y))
    widget._pick_3d(None, None)
    assert picked
    np.testing.assert_allclose(picked[-1], baseline_surface, atol=0.15)
    assert widget.last_pick_fragment == 2
    widget._point_picked(center.tolist())
    assert widget.last_pick_fragment is None
    widget.set_pivot(center)
    assert widget.pivot_actor.GetVisibility()
    np.testing.assert_allclose(widget.pivot_actor.GetAttachmentPoint(), center)
    assert all(view._pivot is not None for view in widget.slices)
    widget.set_landmarks([{"name": "Point 1", "point": center.tolist(), "moved": moved_surface.tolist()}])
    np.testing.assert_allclose(widget.landmark_actors[0].GetAttachmentPoint(), moved_surface)
    widget.set_visibility(2, False)
    assert not widget.actors[2].GetVisibility()
    assert not widget.baseline_actors[2].GetVisibility()
    widget.show_baseline(False)
    widget.show_baseline(True)
    assert not widget.baseline_actors[2].GetVisibility()
    widget.set_surfaces(meshes, fragment_mode=False)
    widget.set_transforms({2: transform})
    assert widget.actors[2].GetVisibility(), "Anatomy and fragment visibility are independent"
    assert widget.actors[2].GetUserMatrix().GetElement(1, 3) == 0
    widget.set_surfaces(meshes, fragment_mode=True)
    assert not widget.actors[2].GetVisibility(), "Visibility persists across surface rebuilds"
    widget.set_pivot(None)
    assert not widget.pivot_actor.GetVisibility()
    widget.set_landmarks([])
    assert not widget.landmark_actors
    assert not widget.grab().isNull()
    widget.close()


def test_expand_3d_preserves_rendering_and_slice_state(qapp):
    doc = document()
    widget = FourViews()
    widget.resize(900, 700)
    widget.show()
    widget.initialize()
    widget.set_document(doc)
    widget.set_surfaces(make_surfaces(doc.anatomy, doc.geometry))
    widget.set_crosshair(doc.geometry.index_to_world([6, 5, 4]))
    for view in widget.slices:
        view.set_zoom(2.5, view.content_rect().center() + QPoint(20, 15))
    camera = widget.renderer.GetActiveCamera()
    camera.Azimuth(23)
    camera.Elevation(11)
    camera.Zoom(1.3)
    qapp.processEvents()
    camera_state = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp(), camera.GetParallelScale())
    slice_state = [(view.crosshair.copy(), view.zoom, view._pan.copy()) for view in widget.slices]
    slider_values = [slider.value() for slider in widget.sliders]
    original_size = widget.vtk.size()
    native_window = widget.vtk.GetRenderWindow()
    actor = widget.actors[2]
    assert not widget.is_3d_expanded

    for _ in range(2):
        widget.set_3d_expanded(True)
        widget.set_3d_expanded(True)
        qapp.processEvents()
        assert widget.is_3d_expanded
        assert all(not view.isVisible() for view in widget.slices)
        assert all(not slider.isVisible() for slider in widget.sliders)
        assert widget.vtk.isVisible()
        assert widget.vtk.width() > original_size.width() * 1.8
        assert widget.vtk.height() > original_size.height() * 1.8
        assert widget.vtk.parentWidget().geometry() == widget.rect()
        assert widget.vtk.GetRenderWindow() is native_window
        assert widget.actors[2] is actor
        assert (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp(), camera.GetParallelScale()) == camera_state

        widget.set_3d_expanded(False)
        widget.set_3d_expanded(False)
        qapp.processEvents()
        assert not widget.is_3d_expanded
        assert all(view.isVisible() for view in widget.slices)
        assert all(slider.isVisible() for slider in widget.sliders)
        assert widget.vtk.size() == original_size
        assert (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp(), camera.GetParallelScale()) == camera_state
        for view, (crosshair, zoom, pan) in zip(widget.slices, slice_state, strict=True):
            np.testing.assert_array_equal(view.crosshair, crosshair)
            assert view.zoom == zoom
            np.testing.assert_array_equal(view._pan, pan)
        assert [slider.value() for slider in widget.sliders] == slider_values
    widget.close()


def test_expand_3d_before_loading_document(qapp):
    widget = FourViews()
    widget.set_3d_expanded(True)
    widget.resize(900, 700)
    widget.show()
    widget.initialize()
    widget.set_document(document())
    qapp.processEvents()
    assert widget.is_3d_expanded
    assert all(not view.isVisible() for view in widget.slices)
    assert widget.vtk.isVisible()
    widget.set_3d_expanded(False)
    qapp.processEvents()
    assert all(view.isVisible() for view in widget.slices)
    widget.close()
