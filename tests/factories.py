"""Small shared-cut fixtures with explicit review and physical geometry."""

import numpy as np

from bimax.engine import build_fragments, default_cuts, required_fragment_labels
from bimax.model import Document, Geometry


def document():
    source = np.ones((21, 21, 21), dtype=np.uint8)
    anatomy = np.where(np.indices(source.shape)[0] < 9, 3, 2).astype(np.uint8)
    doc = Document(Geometry(source.shape, (1.0,) * 3, (0.0,) * 3, tuple(np.eye(3).ravel())),
                   source, anatomy, np.zeros_like(source), np.zeros_like(source))
    doc.cuts = default_cuts(doc)
    doc.cuts["lefort1"].geometry["planes"]["upper"]["origin"][2] = 11
    return doc


def confirm(doc, fragments=None):
    fragments = build_fragments(doc, validate=False) if fragments is None else fragments
    for label in required_fragment_labels(doc):
        xyz = np.argwhere(fragments == label)[0][::-1]
        doc.review.confirmations[doc.fragment_descriptors[label].id] = doc.world_to_plan(
            doc.geometry.index_to_world(xyz)).tolist()
    doc.accept_review()
    return fragments


def upper_geometry(height, *, normal=(0, 0, -1)):
    return {"planes": {"upper": {"origin": [0, 0, height], "normal": list(normal)}},
            "groups": [["upper"]], "region": {"min": [-100] * 3, "max": [100] * 3}}
