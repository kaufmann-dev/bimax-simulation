import json
import tomllib
from copy import deepcopy

import numpy as np
import pytest
from factories import confirm, document

from bimax.aligners import aligner_movements, export_aligner_movements, read_landmarks
from bimax.engine import fragment_transforms

MAXILLA = {  # LPS points on the maxilla fragment of the factory volume (z 9..11, y 0..13)
    "incisor_midline": [10.0, 1.0, 10.0],
    "canine_right": [6.0, 3.0, 10.0],
    "canine_left": [14.0, 3.0, 10.0],
    "molar_right": [4.0, 12.0, 10.0],
    "molar_left": [16.0, 12.0, 10.0],
}
CT_BITE = np.array([0.0, -0.5, 0.0])  # the CT mandible sits 0.5 mm anterior of the occlusion


def landmarks():
    mandible = {name: [x, y, 4.0] for name, (x, y, _) in MAXILLA.items()}
    return {
        "format": "aligners-surgery-landmarks",
        "version": 1,
        "coordinates": "DICOM patient LPS mm of the CT",
        "head_frame": {
            "origin": [10.0, 1.0, 10.0],
            "x_patient_right": [-1.0, 0.0, 0.0],
            "y_anterior": [0.0, -1.0, 0.0],
            "z_superior": [0.0, 0.0, 1.0],
        },
        "head_pitch_deg": 1.5,
        "jaws": {
            "maxilla": {n: {"reference_lps": p, "ct_lps": p} for n, p in MAXILLA.items()},
            "mandible": {
                n: {"reference_lps": p, "ct_lps": (np.asarray(p) + CT_BITE).tolist()}
                for n, p in mandible.items()
            },
        },
    }


def planned(cut_ids=("lefort1",)):
    doc = document()
    doc.scenario.cut_ids = list(cut_ids)
    doc.fragments = confirm(doc)
    return doc


def test_translations_become_landmark_tables_measured_from_the_occlusion():
    doc = planned()
    doc.motions["lefort1"].translation = [0, 3, 2]  # planning R/A/S: 3 forward, 2 up
    doc.motions["mandible"].translation = [1, -4, 0]  # 1 right, 4 back
    movements = aligner_movements(doc, landmarks())
    assert movements.fragments == {"maxilla": "Maxilla", "mandible": "Distal mandible"}
    for move in movements.tables["maxilla"].values():
        assert move == {"forward_mm": 3.0, "up_mm": 2.0, "left_mm": 0.0}
    for move in movements.tables["mandible"].values():
        # the CT mandible starts 0.5 mm anterior of the occlusion the table is measured from
        assert move == {"forward_mm": -3.5, "up_mm": 0.0, "left_mm": -1.0}
    assert movements.head_pitch_deg == 1.5


def test_rotations_are_evaluated_at_each_landmark():
    doc = planned()
    motion = doc.motions["lefort1"]
    motion.angles = [2.0, -3.0, 4.0]
    motion.pivot = doc.world_to_plan(np.array(MAXILLA["incisor_midline"])).tolist()
    motion.pivot_set = True
    motion.translation = [0.5, 2.0, -1.0]
    T = fragment_transforms(doc)[2]
    table = aligner_movements(doc, landmarks()).tables["maxilla"]
    for name, p in MAXILLA.items():
        d = T[:3, :3] @ p + T[:3, 3] - np.asarray(p)
        assert table[name] == {
            "forward_mm": round(-d[1], 1) + 0.0,
            "up_mm": round(d[2], 1) + 0.0,
            "left_mm": round(d[0], 1) + 0.0,
        }
    assert len({tuple(m.values()) for m in table.values()}) > 1


def test_without_a_le_fort_cut_the_maxilla_stays_with_the_skull():
    doc = planned(cut_ids=())
    doc.motions["mandible"].translation = [0, 2, 0]
    movements = aligner_movements(doc, landmarks())
    assert movements.fragments["maxilla"] == "Fixed skull"
    assert all(m == {"forward_mm": 0.0, "up_mm": 0.0, "left_mm": 0.0}
               for m in movements.tables["maxilla"].values())


def test_export_writes_plan_toml_tables(tmp_path):
    doc = planned()
    doc.motions["lefort1"].translation = [0, 3, 2]
    source = tmp_path / "landmarks.json"
    source.write_text(json.dumps(landmarks()))
    path = export_aligner_movements(doc, source, tmp_path / "movements.toml")
    text = path.read_text()
    assert '"Scenario 1"' in text and "head_pitch_deg = 1.5" in text
    tables = tomllib.loads(text)["surgery"]
    assert tables == aligner_movements(doc, landmarks()).tables
    assert list(tables["maxilla"]) == list(MAXILLA)


def test_landmarks_from_another_ct_or_split_dentitions_are_rejected():
    doc = planned()
    far = landmarks()
    far["jaws"]["maxilla"]["molar_left"]["ct_lps"] = [200.0, 12.0, 10.0]
    with pytest.raises(ValueError, match="same CT"):
        aligner_movements(doc, far)
    split = landmarks()
    split["jaws"]["maxilla"]["molar_left"]["ct_lps"] = [16.0, 12.0, 14.0]  # in the skull
    with pytest.raises(ValueError, match="different fragments"):
        aligner_movements(doc, split)
    unaccepted = deepcopy(doc)
    unaccepted.reviews.clear()
    with pytest.raises(ValueError, match="accept"):
        aligner_movements(unaccepted, landmarks())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(format="other"), "not an aligners landmarks"),
        (lambda d: d.update(version=2), "version 2"),
        (lambda d: d["head_frame"].update(y_anterior=[1.0, 0.0, 0.0]), "orthonormal"),
        (lambda d: d["jaws"]["maxilla"].update(pogonion=d["jaws"]["maxilla"]["canine_left"]), "unknown"),
        (lambda d: d["jaws"]["mandible"]["molar_left"].pop("ct_lps"), "ct_lps"),
    ],
)
def test_invalid_landmark_files_fail_clearly(tmp_path, change, message):
    data = landmarks()
    change(data)
    path = tmp_path / "landmarks.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        read_landmarks(path)
