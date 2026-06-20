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
def load_shap():
    try:
        return pd.read_csv("data/processed/shap_values.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_fixtures():
    return pd.read_csv("data/raw/wc2026_fixtures.csv", parse_dates=["date"])

@st.cache_data
def load_bracket():
    try:
        return pd.read_csv("data/processed/bracket.csv")
    except FileNotFoundError:
        return None


def tab_match_predictor(features: pd.DataFrame, sim_results: pd.DataFrame):
    st.header("Match Outcome Predictor")
    all_teams = sorted(sim_results["team"].tolist())

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", all_teams, index=0)
    with col2:
        team_b = st.selectbox("Team B", [t for t in all_teams if t != team_a], index=0)

    # Look up most recent feature row for this matchup
    mask = (features["team_a"] == team_a) & (features["team_b"] == team_b)
    mask_rev = (features["team_a"] == team_b) & (features["team_b"] == team_a)

    if mask.any():
        row = features[mask].sort_values("date").iloc[-1]
        elo_diff = row.get("elo_diff", 0)
        lambda_a = row.get("lambda_a", 1.5)
        p_poisson = row.get("p_win_poisson", 0.45)
    elif mask_rev.any():
        row = features[mask_rev].sort_values("date").iloc[-1]
        elo_diff = -row.get("elo_diff", 0)
        lambda_a = row.get("lambda_a", 1.2)
        p_poisson = 1 - row.get("p_win_poisson", 0.45)
    else:
        elo_diff = 0; lambda_a = 1.5; p_poisson = 0.45

    # Derive simple probability display from Elo diff
    p_win_elo  = 1 / (1 + 10 ** (-elo_diff / 400))
    p_loss_elo = 1 - p_win_elo
    p_draw_elo = 0.28  # empirical average draw rate
    total = p_win_elo + p_loss_elo + p_draw_elo
    p_win_elo /= total; p_loss_elo /= total; p_draw_elo /= total

    st.subheader(f"{team_a} vs {team_b}")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Poisson Model (Stage 1)**")
        p_draw_poisson = max(0.0, 1 - p_poisson - (1 - p_poisson))
        # Poisson win/loss with a nominal draw band
        p_win_p = p_poisson * 0.85
        p_loss_p = (1 - p_poisson) * 0.85
        p_draw_p = 1 - p_win_p - p_loss_p
        fig_p = go.Figure(go.Bar(
            x=["Win A", "Draw", "Win B"],
            y=[p_win_p, p_draw_p, p_loss_p],
            marker_color=["#2ca02c", "#aec7e8", "#d62728"],
        ))
        fig_p.update_layout(yaxis=dict(range=[0, 1], title="Probability"), height=300)
        st.plotly_chart(fig_p, use_container_width=True)

    with c2:
        st.markdown("**Elo-derived Estimate**")
        fig_e = go.Figure(go.Bar(
            x=["Win A", "Draw", "Win B"],
            y=[p_win_elo, p_draw_elo, p_loss_elo],
            marker_color=["#2ca02c", "#aec7e8", "#d62728"],
        ))
        fig_e.update_layout(yaxis=dict(range=[0, 1], title="Probability"), height=300)
        st.plotly_chart(fig_e, use_container_width=True)

    st.caption(f"Elo differential: {elo_diff:+.0f} | Expected goals A: {lambda_a:.2f}")


def tab_champion_probabilities(sim_results: pd.DataFrame):
    from src.simulator import WC2026_GROUPS  # type: ignore

    st.header("Champion Probabilities")
    st.caption("Each cell is a team, shaded by its simulated probability of winning the tournament.")

    p_by_team = dict(zip(sim_results["team"], sim_results["p_champion"]))
    groups = sorted(WC2026_GROUPS.keys())
    max_slots = max(len(t) for t in WC2026_GROUPS.values())

    # Build a group (rows) × slot (cols) grid of championship probabilities,
    # with each team sorted within its group by probability (strongest first).
    z, labels, customdata = [], [], []
    for g in groups:
        teams = sorted(WC2026_GROUPS[g], key=lambda t: p_by_team.get(t, 0.0), reverse=True)
        z_row, lab_row, cd_row = [], [], []
        for slot in range(max_slots):
            if slot < len(teams):
                team = teams[slot]
                p = p_by_team.get(team, 0.0)
                z_row.append(p)
                lab_row.append(f"{team}<br>{p:.1%}")
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
                lines = "<br>".join(f"{t} · {p:.0%}" for t, p in boxes[(rnd, mi)])
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
        with cols[i % 3]:
            st.markdown(f"**Group {group}**")
            st.dataframe(gdf[["team", "P(advance)", "P(win grp)"]].reset_index(drop=True),
                         hide_index=True, use_container_width=True)


def tab_shap(shap_df):
    st.header("SHAP Feature Importance")
    if shap_df is None:
        st.warning("SHAP values not yet computed. Run the full pipeline first.")
        return
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=True)
    fig = go.Figure(go.Bar(
        x=shap_df["mean_abs_shap"],
        y=shap_df["feature"],
        orientation="h",
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        height=500,
        xaxis=dict(title="Mean |SHAP value|"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Computed on WC 2022 holdout set using the 3-class XGBoost model. "
        "Higher = more influential for match outcome predictions."
    )


def main():
    st.title("FIFA World Cup 2026 — AI Prediction System")
    st.caption("All predictions based on historical data through 2024. No live data sources.")

    snaps = list_snapshots()
    if snaps:
        labels = [f"{s['as_of_date']} · {s['label']}" for s in snaps]
        choice = st.sidebar.selectbox("Prediction snapshot", labels, index=0)
        chosen = snaps[labels.index(choice)]
        sim_results = pd.read_csv(f"data/processed/predictions/{chosen['path']}")
        bracket_df = pd.read_csv(f"data/processed/predictions/{chosen['bracket_path']}")
    else:
        sim_results = load_sim_results()
        bracket_df = load_bracket()

    features = load_features()
    shap_df = load_shap()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Match Predictor", "Champion Probabilities", "Group Stage Standings",
        "Tournament Bracket", "SHAP Feature Importance",
    ])
    with tab1:
        tab_match_predictor(features, sim_results)
    with tab2:
        tab_champion_probabilities(sim_results)
    with tab3:
        tab_group_standings(sim_results)
    with tab4:
        tab_bracket(bracket_df)
    with tab5:
        tab_shap(shap_df)


if __name__ == "__main__":
    main()
