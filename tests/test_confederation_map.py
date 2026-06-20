from src.feature_builder import CONFEDERATION, CONFEDERATIONS
from src.fixtures_bracket import load_tournament, normalize_team


def test_all_fixture_teams_have_confederation():
    t = load_tournament("data/raw/wc2026_fixtures.csv")
    teams = {tm for grp in t.groups.values() for tm in grp}
    missing = sorted(tm for tm in teams if tm not in CONFEDERATION)
    assert missing == [], f"teams missing confederation: {missing}"


def test_confederation_values_valid():
    assert all(v in CONFEDERATIONS for v in CONFEDERATION.values())


def test_known_assignments():
    assert CONFEDERATION["United States"] == "CONCACAF"
    assert CONFEDERATION["Norway"] == "UEFA"
    assert CONFEDERATION["Ghana"] == "CAF"
    assert CONFEDERATION["Qatar"] == "AFC"
    assert CONFEDERATION["New Zealand"] == "OFC"
