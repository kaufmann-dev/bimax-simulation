"""Atomic planning validation, named targeting, inheritance, and cache preservation."""

from copy import deepcopy

import numpy as np
import pytest
from factories import document, upper_geometry

from bimax.commands import Commands, inspect_document
from bimax.engine import fragment_transforms
from bimax.history import History
from bimax.model import new_scenario


def test_atomic_movements_named_scenario_and_pivot_validation():
    doc, history = document(), History()
    doc.scenarios["Alternative"] = new_scenario()
    commands = Commands(doc, history)
    before = deepcopy(doc.scenarios)
    with pytest.raises(ValueError, match="pivot"):
        commands.execute(
            "set_movements",
            "Alternative",
            targets={"linked": {"translation": [0, 7, 0]}, "mandible": {"rotation": {"pitch": 9}}},
        )
    assert doc.scenarios == before
    assert doc.active_scenario == "Scenario 1"
    assert not history.undo_stack
    commands.execute(
        "set_movements",
        "Alternative",
        targets={
            "linked": {"translation": [0, 7, 0]},
            "mandible": {"rotation": {"pitch": 9}, "pivot": [1, 2, 3]},
        },
    )
    assert doc.active_scenario == "Alternative"
    assert doc.scenarios["Scenario 1"] == before["Scenario 1"]
    assert len(history.undo_stack) == 1
    commands.execute("set_movements", "Alternative", targets={"mandible": {"rotation": {"yaw": 2}}})
    assert doc.motions["mandible"].angles == [9, 0, 2]
    assert doc.motions["mandible"].pivot == [1, 2, 3]
    with pytest.raises(ValueError, match="scenario name"):
        commands.execute("set_movements", targets={"linked": {"translation": [0, 1, 0]}})


def test_history_keeps_provisional_geometry_for_movement_only_edits():
    doc, history = document(), History()
    fragments = np.zeros_like(doc.source)
    doc.fragments, doc.fragments_key = fragments, doc.configuration_key
    commands = Commands(doc, history)
    commands.execute("set_movements", "Scenario 1", targets={"linked": {"translation": [1, 2, 3]}})
    assert doc.fragments is fragments
    assert history.apply(doc, False)
    assert doc.fragments is fragments
    assert doc.motions["linked"].translation == [0, 0, 0]
    assert history.apply(doc, True)
    assert doc.fragments is fragments
    assert doc.motions["linked"].translation == [1, 2, 3]


def test_shared_cut_impacts_and_invalid_definitions_are_atomic():
    doc, history = document(), History()
    doc.scenarios["Uses cut"] = new_scenario(["lefort1"])
    commands = Commands(doc, history)
    key = doc.configuration_key
    result = commands.execute("replace_cut", cut_id="lefort1", geometry=upper_geometry(12))
    assert result["affected_scenarios"] == ["Uses cut"]
    assert doc.configuration_key == key
    before = deepcopy(doc.cuts)
    broken = upper_geometry(13)
    broken["planes"]["upper"]["normal"] = [0, 0, 0]
    with pytest.raises(ValueError, match="normal"):
        commands.execute("replace_cut", cut_id="lefort1", geometry=broken)
    assert doc.cuts == before
    commands.execute("select_cuts", "Uses cut", cut_ids=["lefort1"])
    assert doc.active_scenario == "Uses cut"
    with pytest.raises(ValueError, match="one variant"):
        commands.execute("select_cuts", "Uses cut", cut_ids=["lefort2", "modified-lefort2"])
    assert doc.scenario.cut_ids == ["lefort1"]


def test_landmark_results_use_nested_movements_and_inspection_is_read_only():
    doc, history = document(), History()
    commands = Commands(doc, history)
    commands.execute("select_cuts", "Scenario 1", cut_ids=["genio", "lefort1", "lefort3"])
    commands.execute(
        "set_movements",
        "Scenario 1",
        targets={
            "linked": {"translation": [1, 2, 3]},
            "mandible": {"translation": [0, 4, 0]},
            "chin": {"translation": [0, 5, 0]},
            "lefort3": {"translation": [0, 6, 0]},
            "lefort1": {"translation": [0, 7, 0]},
        },
    )
    commands.execute("set_landmark", "Scenario 1", name="Chin", fragment="chin", point=[0, 0, 0])
    before = deepcopy(doc.reviews)
    state = inspect_document(doc)
    assert doc.reviews == before
    point = state["scenarios"]["Scenario 1"]["landmarks"][0]
    assert point["displacement"] == [1, 11, 3]
    matrices = fragment_transforms(doc)
    np.testing.assert_allclose(doc.frame[:3, :3] @ matrices[2][:3, 3], [1, 15, 3])
    key = doc.configuration_key
    commands.execute("set_orientation", rotation={"pitch": 10, "roll": 20, "yaw": 30})
    assert doc.configuration_key == key
    assert inspect_document(doc)["scenarios"]["Scenario 1"]["landmarks"][0] == point
    commands.execute("set_landmark", "Scenario 1", name="Chin", remove=True)
    assert doc.scenario.landmarks == []
