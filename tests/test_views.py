import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

import bimax.surfaces as surfaces
import bimax.views as views
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
    mesh = make_surfaces(labels, geometry)[1]
    vertices, faces = mesh.vertices, mesh.faces
    low, high = world_bounds(geometry)
    np.testing.assert_allclose(vertices.min(axis=0), low)
    np.testing.assert_allclose(vertices.max(axis=0), high)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), "A boundary object must produce a closed manifold"
    np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1, atol=1e-6)
    assert np.all(np.sum((vertices - (low + high) / 2) * mesh.normals, axis=1) > 0)
    np.testing.assert_array_equal(labels, original)


def test_mesh_install_reuses_geometry_and_coalesces_native_renders(qapp):
    doc = document()
    widget = FourViews()
    widget.resize(900, 700)
    widget.show()
    widget.initialize()
    widget.set_document(doc)
    meshes = make_surfaces(doc.anatomy, doc.geometry)
    widget.set_surfaces(meshes)
    QTest.qWait(100)
    actor = widget.actors[2]
    poly = actor.GetMapper().GetInput()
    assert poly.GetPointData().GetNormals() is not None
    assert np.shares_memory(vtk_to_numpy(poly.GetPoints().GetData()), meshes[2].vertices)
    assert np.shares_memory(vtk_to_numpy(poly.GetPolys().GetConnectivityArray()), meshes[2].faces)
    assert np.shares_memory(vtk_to_numpy(poly.GetPointData().GetNormals()), meshes[2].normals)
    renders = []
    widget.vtk.GetRenderWindow().AddObserver("StartEvent", lambda *_: renders.append(True))
    widget.set_transforms({})
    widget.set_pivot(None)
    widget.set_landmarks([])
    widget.set_plane([0, 0, 0], [0, 0, 1], False)
    widget.set_surfaces(meshes)
    assert not renders, "State setters must not synchronously render dense geometry"
    QTest.qWait(100)
    QTest.qWait(50)
    assert len(renders) == 1
    assert widget.actors[2] is actor
    renders.clear()
    previous_matrix = actor.GetUserMatrix()
    widget.set_surfaces(meshes)
    widget.set_transforms({})
    QTest.qWait(40)
    assert not renders, "Revisiting the same unchanged scene needs no native redraw"
    assert actor.GetUserMatrix() is previous_matrix

    widget.set_surfaces(meshes, fragment_mode=False)
    anatomy_actor = widget.actors[2]
    assert anatomy_actor is not actor
    widget.set_surfaces(meshes, fragment_mode=True)
    assert widget.actors[2] is actor
    assert len(widget._scene_cache) == 2
    replacement = make_surfaces(doc.anatomy, doc.geometry)
    widget.set_surfaces(replacement, fragment_mode=True)
    assert widget.actors[2] is not actor
    assert len(widget._scene_cache) == 2, "Only one current scene is retained per mode"
    current = widget.actors[2]
    widget.invalidate_surfaces(False)
    assert set(widget._scene_cache) == {True}
    assert widget.actors[2] is current
    widget.invalidate_surfaces(True)
    assert not widget._scene_cache
    assert not widget.actors
    assert not widget.baseline_actors
    assert not widget.renderer.HasViewProp(current)
    widget.invalidate_surfaces(True)
    widget.set_surfaces(replacement, fragment_mode=True)
    widget.set_document(document())
    assert not widget._scene_cache
    assert not widget.actors
    assert not widget.baseline_actors
    renders.clear()
    widget.request_render()
    widget.close()
    QTest.qWait(50)
    assert not renders, "A queued update must not render a closed native window"


def test_circular_orbit_keeps_head_upright_and_supports_pan_zoom(qapp):
    widget = FourViews()
    widget.resize(900, 700)
    widget.show()
    widget.initialize()
    qapp.processEvents()
    camera = widget.renderer.GetActiveCamera()
    up = np.array([0, 0.6, 0.8])
    position = np.array([0, -8, 6])
    camera.SetPosition(*position)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(*up)
    camera.ParallelProjectionOn()
    interactor = widget.interactor
    interactor.SetEventPosition(200, 160)
    interactor.InvokeEvent("LeftButtonPressEvent")
    for x, y in [(230, 160), (230, 190), (200, 190), (200, 160)] * 8:
        interactor.SetEventPosition(x, y)
        interactor.InvokeEvent("MouseMoveEvent")
    interactor.InvokeEvent("LeftButtonReleaseEvent")
    np.testing.assert_allclose(camera.GetViewUp(), up, atol=1e-10)
    np.testing.assert_allclose(camera.GetPosition(), position, atol=1e-10)
    scale = camera.GetParallelScale()
    interactor.InvokeEvent("MouseWheelForwardEvent")
    assert camera.GetParallelScale() < scale
    interactor.InvokeEvent("MouseWheelBackwardEvent")
    assert camera.GetParallelScale() == pytest.approx(scale)
    interactor.InvokeEvent("MiddleButtonPressEvent")
    interactor.SetEventPosition(220, 180)
    interactor.InvokeEvent("MouseMoveEvent")
    interactor.InvokeEvent("MiddleButtonReleaseEvent")
    assert np.linalg.norm(camera.GetFocalPoint()) > 0
    np.testing.assert_allclose(camera.GetViewUp(), up)
    widget.close()


def test_birds_eye_is_elevated_front():
    _, _, direction, up = views.EXPORT_VIEWS[3]
    assert direction[0] == 0
    assert direction[1] > 0
    assert np.degrees(np.arctan2(direction[2], direction[1])) == pytest.approx(40)
    np.testing.assert_array_equal(up, [0, 0, 1])


def test_export_scenario_images_uses_standard_cameras_and_restores_scene(qapp, tmp_path, monkeypatch):
    class FakeRenderLargeImage:
        cameras = []

        def SetInput(self, renderer):
            self.renderer = renderer

        def SetMagnification(self, value):
            self.magnification = value

        def Update(self):
            camera = self.renderer.GetActiveCamera()
            position = np.asarray(camera.GetPosition())
            focal = np.asarray(camera.GetFocalPoint())
            up = np.asarray(camera.GetViewUp())
            self.cameras.append(((position - focal) / np.linalg.norm(position - focal), up))
            self.image = vtkImageData()
            self.image.SetDimensions(48, 36, 1)
            pixels = np.full((36 * 48, 3), 180, dtype=np.uint8)
            self.image.GetPointData().SetScalars(numpy_to_vtk(pixels, deep=True))

        def GetOutput(self):
            return self.image

    monkeypatch.setattr(views, "EXPORT_IMAGE_SIZE", 96)
    monkeypatch.setattr(views, "vtkRenderLargeImage", FakeRenderLargeImage)
    doc = document()
    doc.fragments = doc.anatomy.copy()
    doc.accept_review()
    doc.fragments_key = doc.configuration_key
    widget = FourViews()
    widget.resize(900, 700)
    widget.show()
    widget.initialize()
    widget.set_document(doc)
    widget.set_surfaces(make_surfaces(doc.fragments, doc.geometry), fragment_mode=True)
    transform = np.eye(4)
    transform[:3, 3] = [0, 4, 0]
    widget.set_transforms({2: transform})
    widget.show_baseline(True)
    widget.set_pivot(doc.geometry.index_to_world([6, 7, 8]))
    camera = widget.renderer.GetActiveCamera()
    camera.Azimuth(17)
    camera_state = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())
    viewport = widget.renderer.GetViewport()

    exported = widget.export_scenario_images(tmp_path / "images")

    assert [path.name for path in exported] == [
        "right.png",
        "left.png",
        "front.png",
        "birds-eye.png",
        "contact-sheet.png",
    ]
    for path in exported[:4]:
        image = QImage(str(path))
        assert image.size().width() == image.size().height() == 96
    sheet = QImage(str(exported[-1]))
    assert sheet.size().width() == sheet.size().height() == 192
    directions = [entry[0] for entry in FakeRenderLargeImage.cameras]
    ups = [entry[1] / np.linalg.norm(entry[1]) for entry in FakeRenderLargeImage.cameras]
    origin = doc.plan_to_world([0, 0, 0])
    expected_directions = [
        (doc.plan_to_world(direction) - origin) / np.linalg.norm(doc.plan_to_world(direction) - origin)
        for _, _, direction, _ in views.EXPORT_VIEWS
    ]
    expected_ups = [
        (doc.plan_to_world(up) - origin) / np.linalg.norm(doc.plan_to_world(up) - origin)
        for _, _, _, up in views.EXPORT_VIEWS
    ]
    np.testing.assert_allclose(directions, expected_directions)
    np.testing.assert_allclose(ups, expected_ups)
    assert (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()) == camera_state
    assert widget.renderer.GetViewport() == viewport
    assert widget.actors[2].GetUserMatrix().GetElement(1, 3) == 4
    assert widget.baseline_actors[2].GetVisibility()
    assert widget.pivot_actor.GetVisibility()
    widget.set_visibility(2, False)
    with pytest.raises(ValueError, match="visible structure"):
        widget.export_scenario_images(tmp_path / "hidden")
    widget.close()


@pytest.mark.parametrize("size", [(900, 700), (700, 1100)])
def test_export_scenario_images_renders_native_vtk_scene(qapp, tmp_path, monkeypatch, size):
    # Exercise tiled capture: the output must exceed the native viewport size.
    monkeypatch.setattr(views, "EXPORT_IMAGE_SIZE", 2048)
    doc = document()
    doc.fragments = doc.anatomy.copy()
    doc.accept_review()
    doc.fragments_key = doc.configuration_key
    widget = FourViews()
    widget.resize(*size)
    widget.show()
    widget.initialize()
    widget.set_document(doc)
    widget.set_surfaces(make_surfaces(doc.fragments, doc.geometry), fragment_mode=True)
    QTest.qWait(50)

    exported = widget.export_scenario_images(tmp_path / "native-images")

    for path in exported:
        assert path.stat().st_size > 0
    assert QImage(str(exported[0])).size().width() == 2048
    assert QImage(str(exported[-1])).size().width() == 4096
    for path in exported[:4]:
        image = QImage(str(path)).convertToFormat(QImage.Format.Format_RGB888)
        pixels = np.asarray(image.constBits()).reshape(image.height(), image.bytesPerLine())
        row = pixels[image.height() // 2, :image.width() * 3].reshape(-1, 3)
        foreground = np.flatnonzero(row.max(axis=1) > 20)
        assert foreground.size > 100
        assert np.all(np.diff(foreground) == 1), "Solid bone has black gaps at tile boundaries"
        column = pixels[:, :image.width() * 3].reshape(image.height(), image.width(), 3)[:, image.width() // 2]
        foreground = np.flatnonzero(column.max(axis=1) > 20)
        assert foreground.size > 100
        assert np.all(np.diff(foreground) == 1), "Solid bone has horizontal tile gaps"
    widget.close()


def test_large_display_surface_preserves_topology_and_voxel_scale_geometry(monkeypatch):
    geometry = Geometry((32, 32, 32), (0.4, 0.6, 0.8), (10, -7, 3), tuple(np.eye(3).ravel()))
    coordinates = np.indices(geometry.shape) - 15.5
    labels = (np.sum(coordinates**2, axis=0) <= 13**2).astype(np.uint8)
    original_labels = labels.copy()
    exact = make_surfaces(labels, geometry)[1]
    monkeypatch.setattr(surfaces, "DISPLAY_TRIANGLE_TARGET", 300)
    reduced = make_surfaces(labels, geometry)[1]
    assert len(reduced.faces) < len(exact.faces)
    # Topology/error constraints take precedence over the requested triangle target.
    assert len(reduced.faces) > surfaces.DISPLAY_TRIANGLE_TARGET
    edges = np.sort(np.concatenate([reduced.faces[:, pair] for pair in ([0, 1], [1, 2], [2, 0])]), axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(edge_counts == 2)
    assert len(reduced.vertices) - len(unique_edges) + len(reduced.faces) == 2
    for reference, candidate in ((exact, reduced), (reduced, exact)):
        points = vtkPoints()
        points.SetData(numpy_to_vtk(reference.vertices, deep=False))
        faces = vtkCellArray()
        faces.SetData(3, numpy_to_vtk(reference.faces.ravel(), deep=False))
        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(faces)
        distance = vtkImplicitPolyDataDistance()
        distance.SetInput(poly)
        samples = np.concatenate((candidate.vertices, candidate.vertices[candidate.faces].mean(axis=1)))
        maximum = max(abs(distance.EvaluateFunction(point)) for point in samples)
        assert maximum <= min(geometry.spacing) * 0.25 + 1e-5
    np.testing.assert_array_equal(labels, original_labels)


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
