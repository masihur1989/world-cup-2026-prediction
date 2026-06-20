import pandas as pd
from app.dashboard import list_snapshots, bracket_boxes


def test_list_snapshots_reads_index(tmp_path):
    idx = tmp_path / "predictions"
    idx.mkdir()
    pd.DataFrame([{"as_of_date": "2026-06-10", "label": "pre_tournament",
                   "n_simulations": 100, "generated_at": "x",
                   "path": "2026-06-10__pre_tournament.csv",
                   "bracket_path": "2026-06-10__pre_tournament__bracket.csv"}]).to_csv(
        idx / "index.csv", index=False)
    snaps = list_snapshots(str(idx / "index.csv"))
    assert snaps[0]["label"] == "pre_tournament"


def test_bracket_boxes_top_n():
    bracket = pd.DataFrame([
        {"round": "R32", "match_index": 0, "team": "Brazil", "p_reach": 0.9},
        {"round": "R32", "match_index": 0, "team": "Serbia", "p_reach": 0.7},
        {"round": "R32", "match_index": 0, "team": "Ghana", "p_reach": 0.3},
        {"round": "R32", "match_index": 0, "team": "Haiti", "p_reach": 0.1},
    ])
    boxes = bracket_boxes(bracket, top_n=2)
    cell = boxes[("R32", 0)]
    assert cell == [("Brazil", 0.9), ("Serbia", 0.7)]


def test_team_flag_emoji():
    from app.dashboard import team_flag
    assert team_flag("Brazil") == "🇧🇷"
    assert team_flag("Germany") == "🇩🇪"
    assert team_flag("Unknownland") == ""


def test_all_fixture_teams_have_flags():
    from app.dashboard import team_flag
    from src.fixtures_bracket import load_tournament
    teams = {t for g in load_tournament("data/raw/wc2026_fixtures.csv").groups.values() for t in g}
    missing = sorted(t for t in teams if not team_flag(t))
    assert missing == [], f"teams without flags: {missing}"
