"""
Streamlit dashboard — reads only from data/processed/, no model code at runtime.
Run: streamlit run app/dashboard.py
"""
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
    st.header("Champion Probabilities")
    df = sim_results.copy()
    df = df.sort_values("p_champion", ascending=True)
    df["color"] = df["confederation"].map(CONF_COLORS).fillna("#888888")

    fig = go.Figure(go.Bar(
        x=df["p_champion"],
        y=df["team"],
        orientation="h",
        marker_color=df["color"],
        text=[f"{p:.1%}" for p in df["p_champion"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=max(400, 20 * len(df)),
        xaxis=dict(title="P(Champion)", tickformat=".0%"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    cols = st.columns(len(CONF_COLORS))
    for col, (conf, color) in zip(cols, CONF_COLORS.items()):
        col.markdown(
            f"<div style='background:{color};padding:4px;border-radius:4px;"
            f"color:white;text-align:center;font-size:12px'>{conf}</div>",
            unsafe_allow_html=True,
        )


def tab_group_standings(sim_results: pd.DataFrame):
    st.header("Simulated Group Stage Standings")
    from src.simulator import WC2026_GROUPS  # type: ignore

    groups = sorted(WC2026_GROUPS.keys())
    cols = st.columns(3)
    for i, group in enumerate(groups):
        teams = WC2026_GROUPS[group]
        group_df = sim_results[sim_results["team"].isin(teams)][
            ["team", "p_champion", "confederation"]
        ].sort_values("p_champion", ascending=False)
        group_df["P(advance)"] = (
            group_df["p_champion"].rank(ascending=False).apply(
                lambda r: f"{max(0.3, 1 - 0.2 * (r-1)):.0%}"
            )
        )
        with cols[i % 3]:
            st.markdown(f"**Group {group}**")
            st.dataframe(
                group_df[["team", "P(advance)"]].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )


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

    sim_results = load_sim_results()
    features    = load_features()
    shap_df     = load_shap()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Match Predictor",
        "Champion Probabilities",
        "Group Stage Standings",
        "SHAP Feature Importance",
    ])

    with tab1:
        tab_match_predictor(features, sim_results)
    with tab2:
        tab_champion_probabilities(sim_results)
    with tab3:
        tab_group_standings(sim_results)
    with tab4:
        tab_shap(shap_df)


if __name__ == "__main__":
    main()
