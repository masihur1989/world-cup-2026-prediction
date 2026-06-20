import pytest
from src.fixtures_bracket import assign_third_place


def test_assignment_respects_eligibility():
    slot_eligibles = [
        frozenset({"A", "B", "C"}),
        frozenset({"B", "C", "D"}),
        frozenset({"A", "D"}),
        frozenset({"C", "D", "E"}),
    ]
    best_thirds = [("A", "T_A"), ("B", "T_B"), ("D", "T_D"), ("E", "T_E")]
    result = assign_third_place(slot_eligibles, best_thirds)
    assert len(result) == 4
    group_of = {team: g for g, team in best_thirds}
    for slot_idx, team in enumerate(result):
        assert group_of[team] in slot_eligibles[slot_idx]
    assert sorted(result) == sorted(t for _, t in best_thirds)


def test_assignment_raises_when_infeasible():
    slot_eligibles = [frozenset({"A"}), frozenset({"A"})]
    best_thirds = [("A", "T_A"), ("B", "T_B")]
    with pytest.raises(ValueError):
        assign_third_place(slot_eligibles, best_thirds)
