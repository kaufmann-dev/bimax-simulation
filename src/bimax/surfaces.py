"""Worker-safe surface extraction; mesh vertices are in physical LPS millimetres."""

from collections.abc import Callable

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersGeneral import vtkDiscreteFlyingEdges3D

from bimax.model import Geometry

Mesh = tuple[np.ndarray, np.ndarray]


def make_surfaces(
    labels: np.ndarray,
    geometry: Geometry,
    progress: Callable[[str], None] | None = None,
) -> dict[int, Mesh]:
    """Extract closed surfaces without changing the authoritative voxel labels.

    A one-voxel background border closes objects touching the image boundary.
    VTK works in index space; the complete affine is applied after extraction,
    including oblique or reflected direction matrices.
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
        result[value] = (vertices.astype(np.float32), cells.astype(np.int32))
    return result
