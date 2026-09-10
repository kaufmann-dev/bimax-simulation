"""Worker-safe surface extraction; mesh vertices are in physical LPS millimetres."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkDecimatePro, vtkPolyDataNormals
from vtkmodules.vtkFiltersGeneral import vtkDiscreteFlyingEdges3D

from bimax.model import Geometry


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray


DISPLAY_TRIANGLE_TARGET = 150_000
DISPLAY_ERROR_VOXEL_FRACTION = 0.25


def make_surfaces(
    labels: np.ndarray,
    geometry: Geometry,
    progress: Callable[[str], None] | None = None,
) -> dict[int, Mesh]:
    """Extract display surfaces without changing the authoritative voxel labels.

    A one-voxel background border closes objects touching the image boundary.
    VTK works in index space; the complete affine is applied after extraction,
    including oblique or reflected direction matrices.

    Large surfaces are simplified with an accumulated geometric error budget
    of one quarter of the smallest voxel spacing. Topology and boundaries are
    preserved even when that prevents reaching the display triangle target.
    """
    if labels.shape != geometry.shape:
        raise ValueError("Surface label shape does not match the volume geometry")
    padded = np.pad(np.asarray(labels, dtype=np.uint8), 1)
    image = vtkImageData()
    image.SetDimensions(*padded.shape[::-1])
    image.GetPointData().SetScalars(numpy_to_vtk(padded.ravel(), deep=False))
    extractor = vtkDiscreteFlyingEdges3D()
    extractor.SetInputData(image)
    extractor.ComputeNormalsOff()
    extractor.ComputeGradientsOff()
    extractor.ComputeScalarsOff()
    result: dict[int, Mesh] = {}
    for value in range(1, int(labels.max(initial=0)) + 1):
        if progress:
            progress(f"Building surface {value}")
        extractor.SetValue(0, value)
        extractor.Update()
        mesh = extractor.GetOutput()
        if not mesh.GetNumberOfPoints():
            continue
        vertices = geometry.index_to_world(vtk_to_numpy(mesh.GetPoints().GetData()) - 1)
        cells = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3).copy()
        if np.linalg.det(geometry.affine[:3, :3]) < 0:
            cells = cells[:, ::-1].copy()
        # Compute lighting normals here, in the computation process. A lazy
        # normals filter attached to the GUI mapper executes during its first
        # bounds query/render and can freeze Qt for a large project.
        vertices = vertices.astype(np.float32)
        cells = cells.astype(np.int32)
        points = vtkPoints()
        points.SetData(numpy_to_vtk(vertices, deep=False))
        polygons = vtkCellArray()
        polygons.SetData(3, numpy_to_vtk(cells.ravel(), deep=False))
        physical = vtkPolyData()
        physical.SetPoints(points)
        physical.SetPolys(polygons)
        if len(cells) > DISPLAY_TRIANGLE_TARGET:
            if progress:
                progress(f"Preparing display surface {value}")
            decimator = vtkDecimatePro()
            decimator.SetInputData(physical)
            decimator.SetTargetReduction(1 - DISPLAY_TRIANGLE_TARGET / len(cells))
            decimator.PreserveTopologyOn()
            decimator.SplittingOff()
            decimator.BoundaryVertexDeletionOff()
            decimator.AccumulateErrorOn()
            decimator.SetErrorIsAbsolute(True)
            decimator.SetAbsoluteError(min(geometry.spacing) * DISPLAY_ERROR_VOXEL_FRACTION)
            decimator.Update()
            physical = decimator.GetOutput()
        normals = vtkPolyDataNormals()
        normals.SetInputData(physical)
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.Update()
        shaded = normals.GetOutput()
        result[value] = Mesh(
            vtk_to_numpy(shaded.GetPoints().GetData()).copy(),
            vtk_to_numpy(shaded.GetPolys().GetConnectivityArray())
            .reshape(-1, 3)
            .astype(np.int32),
            vtk_to_numpy(shaded.GetPointData().GetNormals()).copy(),
        )
    return result
