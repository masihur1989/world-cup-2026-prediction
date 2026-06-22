"""
Streamlit dashboard — reads only from data/processed/, no model code at runtime.
Run: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

# Ensure the project root is importable when Streamlit runs this script directly
# (Streamlit sets the working dir to app/, so `src` isn't on sys.path otherwise).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="WC 2026 Predictor", layout="wide")

CONF_COLORS = {
    "UEFA":      "#1f77b4",
    "CONMEBOL":  "#ffd700",
    "CAF":       "#2ca02c",
    "AFC":       "#d62728",
    "CONCACAF":  "#ff7f0e",
    "OFC":       "#9467bd",
}

# Canonical team name -> ISO-3166 alpha-2 code, for emoji flags. England and
# Scotland are not ISO countries; their flag emoji are handled in _FLAG_OVERRIDE.
TEAM_ISO2 = {
    "Czechia": "CZ", "Mexico": "MX", "South Africa": "ZA", "South Korea": "KR",
    "Bosnia and Herzegovina": "BA", "Canada": "CA", "Qatar": "QA", "Switzerland": "CH",
    "Brazil": "BR", "Haiti": "HT", "Morocco": "MA",
    "Australia": "AU", "Paraguay": "PY", "Turkey": "TR", "United States": "US",
    "Curacao": "CW", "Ecuador": "EC", "Germany": "DE", "Ivory Coast": "CI",
    "Japan": "JP", "Netherlands": "NL", "Sweden": "SE", "Tunisia": "TN",
    "Belgium": "BE", "Egypt": "EG", "Iran": "IR", "New Zealand": "NZ",
    "Cape Verde": "CV", "Saudi Arabia": "SA", "Spain": "ES", "Uruguay": "UY",
    "France": "FR", "Iraq": "IQ", "Norway": "NO", "Senegal": "SN",
    "Algeria": "DZ", "Argentina": "AR", "Austria": "AT", "Jordan": "JO",
    "Colombia": "CO", "DR Congo": "CD", "Portugal": "PT", "Uzbekistan": "UZ",
    "Croatia": "HR", "Ghana": "GH", "Panama": "PA",
}
# Subdivision flags (tag sequences) for non-ISO members.
_FLAG_OVERRIDE = {
    "England": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "Scotland": "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
}


def team_flag(team: str) -> str:
    """Return the emoji flag for a team, or '' if unknown."""
    if team in _FLAG_OVERRIDE:
        return _FLAG_OVERRIDE[team]
    iso2 = TEAM_ISO2.get(team)
    if not iso2:
        return ""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso2.upper())


def with_flag(team: str) -> str:
    """Return 'FLAG Team' (or just 'Team' if no flag known)."""
    f = team_flag(team)
    return f"{f} {team}".strip()

def list_snapshots(index_path="data/processed/predictions/index.csv"):
    """Return snapshots (newest first) from the predictions index, or []."""
    from pathlib import Path
    if not Path(index_path).exists():
        return []
    idx = pd.read_csv(index_path)
    idx = idx.sort_values("generated_at", ascending=False)
    return idx.to_dict("records")


def bracket_boxes(bracket_df, top_n=3):
    """Map (round, match_index) -> list of (team, p_reach) top-N descending."""
    boxes = {}
    for (rnd, mi), grp in bracket_df.groupby(["round", "match_index"]):
        top = grp.sort_values("p_reach", ascending=False).head(top_n)
        boxes[(rnd, mi)] = list(zip(top["team"], top["p_reach"]))
    return boxes


@st.cache_data
def load_sim_results():
    return pd.read_csv("data/processed/simulation_results.csv")

@st.cache_data
def load_features():
    return pd.read_csv("data/processed/features.csv", parse_dates=["date"])


@st.cache_data
def load_fixtures():
    return pd.read_csv("data/raw/wc2026_fixtures.csv", parse_dates=["date"])

@st.cache_data
def load_bracket():
    try:
        return pd.read_csv("data/processed/bracket.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_matchups():
    try:
        return pd.read_csv("data/processed/matchups.csv")
    except FileNotFoundError:
        return None


def lookup_matchup(matchups_df, team_a, team_b):
    """Return (p_win_a, p_draw, p_win_b) for team_a vs team_b, or None."""
    row = matchups_df[(matchups_df["team_a"] == team_a) & (matchups_df["team_b"] == team_b)]
    if not row.empty:
        r = row.iloc[0]
        return float(r["p_win_a"]), float(r["p_draw"]), float(r["p_win_b"])
    rev = matchups_df[(matchups_df["team_a"] == team_b) & (matchups_df["team_b"] == team_a)]
    if not rev.empty:
        r = rev.iloc[0]
        return float(r["p_win_b"]), float(r["p_draw"]), float(r["p_win_a"])
    return None


def tab_match_predictor(matchups_df):
    st.header("Match Outcome Predictor")
    if matchups_df is None or matchups_df.empty:
        st.warning("No matchup predictions yet. Run the pipeline simulate stage.")
        return

    from src.fixtures_bracket import load_tournament
    tournament = load_tournament("data/raw/wc2026_fixtures.csv")
    teams = sorted({t for g in tournament.groups.values() for t in g})

    mode = st.radio("Mode", ["Scheduled fixture", "Any matchup"], horizontal=True)

    if mode == "Scheduled fixture":
        fixtures = tournament.group_matches  # (a, b, group)
        options = [f"{with_flag(a)}  vs  {with_flag(b)}   ·  Group {g}"
                   for (a, b, g) in fixtures]
        pick = st.selectbox("Fixture", options, index=0)
        team_a, team_b, _g = fixtures[options.index(pick)]
    else:
        c1, c2 = st.columns(2)
        with c1:
            team_a = st.selectbox("Team A", teams, index=0,
                                  format_func=with_flag)
        with c2:
            team_b = st.selectbox("Team B", [t for t in teams if t != team_a],
                                  index=0, format_func=with_flag)

    probs = lookup_matchup(matchups_df, team_a, team_b)
    if probs is None:
        st.warning(f"No prediction available for {team_a} vs {team_b}.")
        return
    p_win_a, p_draw, p_win_b = probs

    st.subheader(f"{with_flag(team_a)}  vs  {with_flag(team_b)}")
    fig = go.Figure(go.Bar(
        x=[f"{team_a} win", "Draw", f"{team_b} win"],
        y=[p_win_a, p_draw, p_win_b],
        marker_color=["#2ca02c", "#aec7e8", "#d62728"],
        text=[f"{p:.0%}" for p in (p_win_a, p_draw, p_win_b)],
        textposition="outside",
    ))
    fig.update_layout(yaxis=dict(range=[0, 1], title="Probability", tickformat=".0%"),
                      height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Calibrated W/D/L probabilities from the 3-class XGBoost model "
               "(group-stage perspective). Knockout ties are broken separately in the simulator.")


def tab_champion_probabilities(sim_results: pd.DataFrame):
    from src.fixtures_bracket import load_tournament

    st.header("Champion Probabilities")
    st.caption("Each cell is a team, shaded by its simulated probability of winning the tournament.")

    wc_groups = load_tournament("data/raw/wc2026_fixtures.csv").groups
    p_by_team = dict(zip(sim_results["team"], sim_results["p_champion"]))
    groups = sorted(wc_groups.keys())
    max_slots = max(len(t) for t in wc_groups.values())

    # Build a group (rows) × slot (cols) grid of championship probabilities,
    # with each team sorted within its group by probability (strongest first).
    z, labels, customdata = [], [], []
    for g in groups:
        teams = sorted(wc_groups[g], key=lambda t: p_by_team.get(t, 0.0), reverse=True)
        z_row, lab_row, cd_row = [], [], []
        for slot in range(max_slots):
            if slot < len(teams):
                team = teams[slot]
                p = p_by_team.get(team, 0.0)
                z_row.append(p)
                lab_row.append(f"{with_flag(team)}<br>{p:.1%}")
                cd_row.append(team)
            else:
                z_row.append(None); lab_row.append(""); cd_row.append("")
        z.append(z_row); labels.append(lab_row); customdata.append(cd_row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"#{i+1}" for i in range(max_slots)],
        y=[f"Group {g}" for g in groups],
        text=labels,
        texttemplate="%{text}",
        textfont=dict(size=11),
        customdata=customdata,
        hovertemplate="%{customdata}<br>P(champion): %{z:.2%}<extra></extra>",
        colorscale="YlOrRd",
        colorbar=dict(title="P(champion)", tickformat=".0%"),
        hoverongaps=False,
    ))
    fig.update_layout(
        height=max(400, 34 * len(groups)),
        xaxis=dict(title="Rank within group", side="top"),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=80, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def tab_bracket(bracket_df):
    st.header("Tournament Bracket (probabilistic)")
    if bracket_df is None or bracket_df.empty:
        st.warning("No bracket data yet. Run the pipeline simulate stage.")
        return
    top_n = st.slider("Teams shown per match", 1, 4, 3)
    boxes = bracket_boxes(bracket_df, top_n=top_n)
    round_order = ["R32", "R16", "QF", "SF", "Final"]
    cols = st.columns(len(round_order))
    for col, rnd in zip(cols, round_order):
        with col:
            st.subheader(rnd)
            match_indices = sorted({mi for (r, mi) in boxes if r == rnd})
            for mi in match_indices:
                lines = "<br>".join(f"{with_flag(t)} · {p:.0%}" for t, p in boxes[(rnd, mi)])
                st.markdown(
                    f"<div style='border:1px solid #888;border-radius:6px;"
                    f"padding:6px;margin-bottom:6px;font-size:12px'>{lines}</div>",
                    unsafe_allow_html=True,
                )


def tab_group_standings(sim_results: pd.DataFrame):
    st.header("Simulated Group Stage Standings")
    from src.fixtures_bracket import load_tournament
    groups = load_tournament("data/raw/wc2026_fixtures.csv").groups

    cols = st.columns(3)
    for i, group in enumerate(sorted(groups.keys())):
        teams = groups[group]
        gdf = sim_results[sim_results["team"].isin(teams)].copy()
        gdf = gdf.sort_values("p_advance", ascending=False)
        gdf["P(advance)"] = (gdf["p_advance"] * 100).round(0).astype(int).astype(str) + "%"
        gdf["P(win grp)"] = (gdf["p_group_winner"] * 100).round(0).astype(int).astype(str) + "%"
        gdf["Team"] = gdf["team"].map(with_flag)
        with cols[i % 3]:
            st.markdown(f"**Group {group}**")
            st.dataframe(gdf[["Team", "P(advance)", "P(win grp)"]].reset_index(drop=True),
                         hide_index=True, use_container_width=True)


def main():
    st.title("FIFA World Cup 2026 — AI Prediction System")
    st.caption("All predictions based on historical data through 2024. No live data sources.")

    def _read(name):
        try:
            return pd.read_csv(f"data/processed/predictions/{name}")
        except (FileNotFoundError, TypeError, ValueError):
            return None

    snaps = list_snapshots()
    matchups_df = None
    if snaps:
        labels = [f"{s['as_of_date']} · {s['label']}" for s in snaps]
        choice = st.sidebar.selectbox("Prediction snapshot", labels, index=0)
        chosen = snaps[labels.index(choice)]
        sim_results = pd.read_csv(f"data/processed/predictions/{chosen['path']}")
        bracket_df = pd.read_csv(f"data/processed/predictions/{chosen['bracket_path']}")
        mp = chosen.get("matchups_path")
        matchups_df = _read(mp) if isinstance(mp, str) and mp else None
    else:
        sim_results = load_sim_results()
        bracket_df = load_bracket()

    if matchups_df is None:
        matchups_df = load_matchups()  # fall back to 'latest'

    tab1, tab2, tab3, tab4 = st.tabs([
        "Match Predictor", "Champion Probabilities", "Group Stage Standings",
        "Tournament Bracket",
    ])
    with tab1:
        tab_match_predictor(matchups_df)
    with tab2:
        tab_champion_probabilities(sim_results)
    with tab3:
        tab_group_standings(sim_results)
    with tab4:
        tab_bracket(bracket_df)


if __name__ == "__main__":
    main()
