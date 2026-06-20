import pandas as pd
import pytest
from src.fixtures_bracket import (
    normalize_team, parse_slot, Slot, load_tournament, Tournament,
)


def test_normalize_team_known_aliases():
    assert normalize_team("USA") == "United States"
    assert normalize_team("Turkiye") == "Turkey"
    assert normalize_team("Curaçao") == "Curacao"


def test_normalize_team_passthrough():
    assert normalize_team("Brazil") == "Brazil"


def test_parse_slot_winner():
    s = parse_slot("Winner C")
    assert s.kind == "winner" and s.group == "C" and s.eligible is None


def test_parse_slot_runner_up():
    s = parse_slot("Runner-up F")
    assert s.kind == "runner_up" and s.group == "F"


def test_parse_slot_third():
    s = parse_slot("3rd Place (A/B/C/D/F)")
    assert s.kind == "third"
    assert s.eligible == frozenset({"A", "B", "C", "D", "F"})
    assert s.group is None


def test_load_tournament_groups(tmp_fixtures):
    t = load_tournament(tmp_fixtures)
    assert isinstance(t, Tournament)
    assert len(t.groups) == 12
    assert all(len(v) == 4 for v in t.groups.values())
    assert "United States" in t.groups["D"]
    assert "Turkey" in t.groups["D"]


def test_load_tournament_group_matches(tmp_fixtures):
    t = load_tournament(tmp_fixtures)
    assert len(t.group_matches) == 72
    a, b, g = t.group_matches[0]
    assert g == "A"


def test_load_tournament_r32_slots(tmp_fixtures):
    t = load_tournament(tmp_fixtures)
    assert len(t.r32_slots) == 16
    first = t.r32_slots[0]
    assert isinstance(first[0], Slot) and isinstance(first[1], Slot)


@pytest.fixture
def tmp_fixtures(tmp_path):
    return "data/raw/wc2026_fixtures.csv"
