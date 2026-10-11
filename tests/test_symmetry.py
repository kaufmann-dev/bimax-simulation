"""Fixed condyle pivots, independent rotations, and explicit geometric symmetry."""

from copy import deepcopy

import numpy as np
import pytest
from factories import document

from bimax.commands import Commands, inspect_document
from bimax.engine import fragment_transforms, motion_matrix
from bimax.history import History
from bimax.storage import load_project, save_project
from bimax.symmetry import measurement_results


def setup_measurements():
    doc, history = document(), History()
    commands = Commands(doc, history)
    commands.execute("select_cuts", "Scenario 1", cut_ids=["bsso", "genio"])
    commands.execute("set_symmetry_reference", point=[10, -4, 7])
    landmarks = {
        "Left condyle": ("left-proximal", [4, 2, 20]),
        "Left angle": ("left-proximal", [5, 0, 10]),
        "Right condyle": ("right-proximal", [17, 1, 22]),
        "Right angle": ("right-proximal", [15, 0, 10]),
        "Chin": ("chin", [12, 4, 2]),
    }
    for name, (fragment, point) in landmarks.items():
        commands.execute("set_landmark", "Scenario 1", name=name, fragment=fragment, point=point)
    for name, kind, landmarks in (
        ("Chin midline", "midline", {"point": "Chin"}),
        ("Condyle pair", "paired", {"left": "Left condyle", "right": "Right condyle"}),
        (
            "Ramus",
            "ramus",
            {
                "left_condyle": "Left condyle",
                "left_angle": "Left angle",
                "right_condyle": "Right condyle",
                "right_angle": "Right angle",
            },
        ),
    ):
        commands.execute("set_measurement", "Scenario 1", name=name, kind=kind, landmarks=landmarks)
    return doc, history, commands


def test_proximal_rotation_fixed_pivot_independence_and_no_inheritance():
    doc, history, commands = setup_measurements()
    commands.execute(
        "set_movements",
        "Scenario 1",
        targets={
            "linked": {"translation": [4, 7, 2], "rotation": {"yaw": 9}, "pivot": [3, 4, 5]},
            "mandible": {"translation": [0, 6, 0]},
            "chin": {"translation": [0, 3, 0]},
        },
    )
    before = fragment_transforms(doc)
    np.testing.assert_allclose(before[3], np.eye(4))
    np.testing.assert_allclose(before[4], np.eye(4))
    key = doc.configuration_key
    fragments = np.zeros_like(doc.source)
    doc.fragments, doc.fragments_key = fragments, key
    commands.execute(
        "set_proximal_rotations",
        "Scenario 1",
        targets={
            "left-proximal": {"pivot": [4, 2, 20], "rotation": {"pitch": 12, "roll": 7, "yaw": -11}}
        },
    )
    after = fragment_transforms(doc)
    pivot = doc.plan_to_world([4, 2, 20])
    np.testing.assert_allclose(after[3][:3, :3] @ pivot + after[3][:3, 3], pivot, atol=1e-12)
    for label in (1, 4, 5, 6):
        np.testing.assert_allclose(after[label], before[label])
    assert doc.fragments is fragments and doc.configuration_key == key
    expected_chin = (
        np.linalg.inv(doc.frame)
        @ motion_matrix(doc.motions["linked"])
        @ motion_matrix(doc.motions["mandible"])
        @ motion_matrix(doc.motions["chin"])
        @ doc.frame
    )
    np.testing.assert_allclose(after[6], expected_chin)
    landmarks = inspect_document(doc)["scenarios"]["Scenario 1"]["landmarks"]
    left_condyle = next(item for item in landmarks if item["name"] == "Left condyle")
    np.testing.assert_allclose(left_condyle["moved"], left_condyle["baseline"], atol=1e-12)
    left_angle = next(item for item in landmarks if item["name"] == "Left angle")
    assert left_angle["distance_mm"] > 0
    right_condyle = next(item for item in landmarks if item["name"] == "Right condyle")
    np.testing.assert_allclose(right_condyle["moved"], right_condyle["baseline"])
    assert history.apply(doc, False)
    np.testing.assert_allclose(fragment_transforms(doc)[3], before[3])
    assert history.apply(doc, True)
    np.testing.assert_allclose(fragment_transforms(doc)[3], after[3])
    commands.execute("reset_proximal_rotations", "Scenario 1", targets=["left-proximal"])
    assert doc.scenario.proximal_rotations["left-proximal"].angles == [0, 0, 0]
    assert doc.scenario.proximal_rotations["left-proximal"].pivot_set
    np.testing.assert_allclose(fragment_transforms(doc)[3], np.eye(4))


def test_proximal_validation_is_atomic_and_translation_is_unavailable():
    doc, history = document(), History()
    commands = Commands(doc, history)
    with pytest.raises(ValueError, match="Select BSSO"):
        commands.execute(
            "set_proximal_rotations", "Scenario 1", targets={"left-proximal": {"pivot": [0, 0, 0]}}
        )
    commands.execute("select_cuts", "Scenario 1", cut_ids=["bsso"])
    before = deepcopy(doc.scenarios)
    with pytest.raises(ValueError, match="pivot"):
        commands.execute(
            "set_proximal_rotations",
            "Scenario 1",
            targets={
                "left-proximal": {"pivot": [1, 2, 3], "rotation": {"pitch": 9}},
                "right-proximal": {"rotation": {"yaw": 3}},
            },
        )
    assert doc.scenarios == before
    with pytest.raises(ValueError, match="translation"):
        commands.execute(
            "set_proximal_rotations",
            "Scenario 1",
            targets={"left-proximal": {"translation": [1, 0, 0]}},
        )
    with pytest.raises(ValueError, match="Unknown movement"):
        commands.execute(
            "set_movements", "Scenario 1", targets={"left-proximal": {"translation": [1, 0, 0]}}
        )
    commands.execute(
        "set_proximal_rotations",
        "Scenario 1",
        targets={
            "left-proximal": {"pivot": [4, 2, 20], "rotation": {"pitch": 10}},
            "right-proximal": {"pivot": [17, 1, 22], "rotation": {"yaw": -9}},
        },
    )
    assert doc.scenario.proximal_rotations["right-proximal"].angles == [0, 0, -9]


def test_symmetry_formulas_signs_baseline_moved_and_orientation_independence():
    doc, history, commands = setup_measurements()
    results = measurement_results(doc)
    midline = results["Chin midline"]
    assert midline["baseline"] == {"signed_lateral_mm": 2, "distance_mm": 2}
    pair = results["Condyle pair"]["baseline"]
    assert pair == {
        "height_difference_mm": -2,
        "anterior_difference_mm": 1,
        "lateral_distance_difference_mm": -1,
    }
    ramus = results["Ramus"]["baseline"]
    assert ramus["left_frontal_deg"] == pytest.approx(np.rad2deg(np.arctan2(1, 10)))
    assert ramus["right_frontal_deg"] == pytest.approx(np.rad2deg(np.arctan2(2, 12)))
    assert ramus["left_sagittal_deg"] == pytest.approx(np.rad2deg(np.arctan2(2, 10)))
    assert ramus["right_sagittal_deg"] == pytest.approx(np.rad2deg(np.arctan2(1, 12)))
    assert ramus["frontal_difference_deg"] == pytest.approx(
        ramus["left_frontal_deg"] - ramus["right_frontal_deg"]
    )
    commands.execute(
        "set_movements",
        "Scenario 1",
        targets={"mandible": {"translation": [-1, 0, 0]}, "chin": {"translation": [-1, 0, 0]}},
    )
    moved = measurement_results(doc)
    assert moved["Chin midline"]["baseline"] == midline["baseline"]
    assert moved["Chin midline"]["moved"] == {"signed_lateral_mm": 0, "distance_mm": 0}
    commands.execute("set_orientation", rotation={"pitch": 60, "roll": -20, "yaw": 37})
    assert measurement_results(doc) == moved
    commands.execute(
        "set_proximal_rotations",
        "Scenario 1",
        targets={"left-proximal": {"pivot": [4, 2, 20], "rotation": {"pitch": 10}}},
    )
    after = measurement_results(doc)["Ramus"]
    assert after["baseline"] == ramus
    assert after["moved"]["left_sagittal_deg"] != ramus["left_sagittal_deg"]
    assert after["moved"]["right_sagittal_deg"] == pytest.approx(ramus["right_sagittal_deg"])


def test_measurements_reject_missing_unsuitable_coincident_and_silent_retargeting():
    doc, history, commands = setup_measurements()
    with pytest.raises(ValueError, match="Missing measurement"):
        commands.execute(
            "set_measurement",
            "Scenario 1",
            name="Broken",
            kind="midline",
            landmarks={"point": "Missing"},
        )
    with pytest.raises(ValueError, match="referenced"):
        commands.execute("set_landmark", "Scenario 1", name="Left condyle", remove=True)
    with pytest.raises(ValueError, match="left proximal"):
        commands.execute(
            "set_landmark",
            "Scenario 1",
            name="Left condyle",
            fragment="right-proximal",
            point=[4, 2, 20],
        )
    with pytest.raises(ValueError, match="coincide"):
        commands.execute(
            "set_landmark",
            "Scenario 1",
            name="Left angle",
            fragment="left-proximal",
            point=[4, 2, 20],
        )
    with pytest.raises(ValueError, match="definitions explicitly"):
        commands.execute("select_cuts", "Scenario 1", cut_ids=[])
    commands.execute("set_measurement", "Scenario 1", name="Ramus", remove=True)
    commands.execute("set_measurement", "Scenario 1", name="Condyle pair", remove=True)
    commands.execute("set_landmark", "Scenario 1", name="Left condyle", remove=True)
    doc.scenario.landmarks.clear()
    invalid = measurement_results(doc)["Chin midline"]
    assert invalid["status"] == "invalid" and "Missing" in invalid["error"]
    with pytest.raises(ValueError, match="symmetry reference"):
        Commands(document(), History()).execute(
            "set_measurement",
            "Scenario 1",
            name="Unset",
            kind="midline",
            landmarks={"point": "Missing"},
        )


def test_symmetry_and_proximal_state_roundtrip_and_invalid_state_refuses_save(tmp_path):
    doc, history, commands = setup_measurements()
    commands.execute(
        "set_proximal_rotations",
        "Scenario 1",
        targets={
            "left-proximal": {"pivot": [4, 2, 20], "rotation": {"pitch": 12, "roll": 4}},
            "right-proximal": {"pivot": [17, 1, 22], "rotation": {"yaw": 5}},
        },
    )
    path = tmp_path / "symmetry.bimax"
    save_project(doc, path)
    restored = load_project(path)
    assert restored.symmetry_right == 10
    assert restored.scenarios == doc.scenarios
    assert measurement_results(restored) == measurement_results(doc)
    original = path.read_bytes()
    doc.scenario.proximal_rotations["left-proximal"].pivot_set = False
    with pytest.raises(ValueError, match="pivot"):
        save_project(doc, path)
    assert path.read_bytes() == original
    doc.scenario.proximal_rotations["left-proximal"].pivot_set = True
    doc.scenario.landmarks.pop()
    with pytest.raises(ValueError, match="Missing"):
        save_project(doc, path)
    assert path.read_bytes() == original
