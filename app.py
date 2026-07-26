"""
RosterIQ — All-in-one Fantasy Football Draft & Season Manager
================================================================
Single-file Streamlit application.

Modules:
    1. Draft Room (Value-Based Drafting)
    2. Weekly Start/Sit — Monte Carlo Matchup Engine
    3. Trade Evaluator
    4. Waiver Wire & Injuries
    5. Deep Player Stats
    6. League Standings & Playoff Odds

The app tries to pull real player data via `nfl_data_py`. If that
package / network call is unavailable (e.g. offline dev, blocked
egress, or a season with no data yet), RosterIQ falls back to a
realistic synthetic dataset so every tab keeps working and the user
always sees a clean message instead of a crash.
"""

import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RosterIQ | Fantasy Football Command Center",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / CUSTOM CSS — dark stadium theme (charcoal/navy + emerald/orange)
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --rq-bg-0: #0b0f14;
    --rq-bg-1: #101720;
    --rq-bg-2: #151d29;
    --rq-card: #131b24;
    --rq-border: #223041;
    --rq-emerald: #00e6a0;
    --rq-emerald-dim: #04725a;
    --rq-orange: #ff7a1a;
    --rq-orange-dim: #7a3d10;
    --rq-text: #e9eef3;
    --rq-text-dim: #90a4b7;
    --rq-red: #ff4d5e;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--rq-text);
}

h1, h2, h3, h4, .rq-heading {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.3px;
}

/* App background */
.stApp {
    background: radial-gradient(circle at 15% 0%, #14202c 0%, #0b0f14 45%, #090c10 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d131b 0%, #0a0e13 100%);
    border-right: 1px solid var(--rq-border);
}
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label {
    color: var(--rq-text-dim);
}

/* Hide default Streamlit chrome we don't want */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

/* Top banner */
.rq-banner {
    background: linear-gradient(120deg, #0f1922 0%, #101a24 60%, #17222c 100%);
    border: 1px solid var(--rq-border);
    border-radius: 18px;
    padding: 22px 28px;
    margin-bottom: 18px;
    box-shadow: 0 0 0 1px rgba(0,230,160,0.05), 0 8px 30px rgba(0,0,0,0.35);
}
.rq-banner-title {
    font-size: 30px;
    font-weight: 700;
    color: var(--rq-text);
    margin: 0;
}
.rq-banner-title span { color: var(--rq-emerald); }
.rq-banner-sub {
    color: var(--rq-text-dim);
    margin-top: 4px;
    font-size: 14px;
}

/* Generic card */
.rq-card {
    background: var(--rq-card);
    border: 1px solid var(--rq-border);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
.rq-card h4 {
    margin-top: 0;
    color: var(--rq-text);
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--rq-text-dim);
}

/* Metric badge */
.rq-metric-row { display: flex; gap: 12px; flex-wrap: wrap; }
.rq-badge {
    flex: 1;
    min-width: 130px;
    background: linear-gradient(160deg, #121b24 0%, #0f161e 100%);
    border: 1px solid var(--rq-border);
    border-radius: 14px;
    padding: 14px 16px;
    text-align: left;
}
.rq-badge .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--rq-text-dim);
    margin-bottom: 6px;
}
.rq-badge .value {
    font-size: 22px;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--rq-emerald);
}
.rq-badge.orange .value { color: var(--rq-orange); }
.rq-badge.red .value { color: var(--rq-red); }

/* Tier pill */
.rq-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.rq-pill.t1 { background: rgba(0,230,160,0.15); color: var(--rq-emerald); border: 1px solid var(--rq-emerald-dim); }
.rq-pill.t2 { background: rgba(255,122,26,0.15); color: var(--rq-orange); border: 1px solid var(--rq-orange-dim); }
.rq-pill.t3 { background: rgba(144,164,183,0.15); color: var(--rq-text-dim); border: 1px solid var(--rq-border); }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, var(--rq-emerald) 0%, #00b884 100%);
    color: #06110c;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.5rem 1rem;
    transition: all 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    filter: brightness(1.08);
    box-shadow: 0 0 18px rgba(0,230,160,0.35);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--rq-border);
}
.stTabs [data-baseweb="tab"] {
    background: var(--rq-bg-1);
    border: 1px solid var(--rq-border);
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    color: var(--rq-text-dim);
    padding: 10px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: var(--rq-emerald) !important;
    background: var(--rq-card) !important;
    border-color: var(--rq-emerald-dim) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--rq-border);
    border-radius: 12px;
    overflow: hidden;
}

/* Sliders / inputs */
.stSlider [data-baseweb="slider"] div div { background: var(--rq-emerald) !important; }

/* Divider */
.rq-divider { border-top: 1px solid var(--rq-border); margin: 18px 0; }

/* Footer visitor strip */
.rq-footer {
    text-align: center;
    color: var(--rq-text-dim);
    font-size: 12px;
    padding: 18px 0 6px 0;
}

/* Share box */
.rq-share code {
    display: block;
    background: #0a0f14;
    border: 1px solid var(--rq-border);
    border-radius: 8px;
    padding: 6px 8px;
    font-size: 12px;
    color: var(--rq-emerald);
    word-break: break-all;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
POSITIONS = ["QB", "RB", "WR", "TE"]
LEAGUE_SIZE_OPTIONS = [8, 10, 12, 14, 16]
SCORING_OPTIONS = ["PPR", "Half-PPR", "Standard"]
CURRENT_SEASON = 2025
TOTAL_WEEKS = 17

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
]

FIRST_NAMES = [
    "James", "Michael", "Chris", "Josh", "Trevor", "Jordan", "Marcus", "DeAndre",
    "Tyler", "Justin", "Aaron", "Derrick", "Jalen", "Cooper", "Devon", "Tee",
    "Amari", "CeeDee", "Nico", "Puka", "Garrett", "Sam", "Brock", "Bijan",
    "Kenneth", "Rhamondre", "Javonte", "Isiah", "George", "Travis", "Mark",
    "Dallas", "Kyle", "Zay", "DK", "Terry", "Chris", "Drake", "Rachaad", "Joe",
]
LAST_NAMES = [
    "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller", "Wilson", "Moore",
    "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Clark",
    "Lewis", "Young", "Allen", "King", "Wright", "Scott", "Hill", "Adams",
    "Baker", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips",
]

RNG_SEED = 42

# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _generate_synthetic_dataset(seed: int = RNG_SEED) -> pd.DataFrame:
    """Builds a realistic-shaped synthetic player pool so the app always has
    data to work with, regardless of live data availability."""
    rng = np.random.default_rng(seed)
    rows = []
    name_pool = set()

    pos_counts = {"QB": 34, "RB": 65, "WR": 90, "TE": 34}
    # base scoring profiles per position (weekly mean/std in PPR terms)
    pos_profile = {
        "QB": dict(mean=(12, 24), std=(4, 7)),
        "RB": dict(mean=(3, 20), std=(3, 8)),
        "WR": dict(mean=(3, 19), std=(3, 9)),
        "TE": dict(mean=(2, 15), std=(2, 6)),
    }

    player_id = 1000
    for pos, count in pos_counts.items():
        # generate a descending "talent curve" so tiers emerge naturally
        talent = np.sort(rng.beta(2.0, 3.2, size=count))[::-1]
        lo_m, hi_m = pos_profile[pos]["mean"]
        lo_s, hi_s = pos_profile[pos]["std"]
        for i in range(count):
            while True:
                nm = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                if nm not in name_pool:
                    name_pool.add(nm)
                    break
            t = talent[i]
            mean_pts = lo_m + t * (hi_m - lo_m) + rng.normal(0, 0.6)
            mean_pts = max(mean_pts, 0.5)
            std_pts = lo_s + (1 - t) * 0.4 * (hi_s - lo_s) + rng.normal(0, 0.4)
            std_pts = max(std_pts, 1.0)

            target_share = np.clip(rng.normal(0.06 + t * 0.24, 0.03), 0, 0.4) if pos in ("WR", "TE", "RB") else 0.0
            snap_pct = np.clip(rng.normal(0.35 + t * 0.55, 0.08), 0.05, 0.98)
            redzone_touches = max(0, rng.poisson(1 + t * 4))
            air_yards = max(0, rng.normal(300 + t * 900, 150)) if pos in ("WR", "TE") else max(0, rng.normal(20 + t * 60, 20))
            adp = round(1 + (1 - t) * 220 + rng.normal(0, 4), 1)
            bye = int(rng.integers(5, 14))
            injury_status = rng.choice(
                ["Healthy", "Healthy", "Healthy", "Healthy", "Questionable", "Doubtful", "Out", "IR"],
                p=[0.55, 0.15, 0.1, 0.05, 0.08, 0.03, 0.03, 0.01],
            )

            rows.append(dict(
                player_id=player_id,
                player_name=nm,
                position=pos,
                team=rng.choice(NFL_TEAMS),
                proj_mean=round(float(mean_pts), 2),
                proj_std=round(float(std_pts), 2),
                target_share=round(float(target_share), 3),
                snap_pct=round(float(snap_pct), 3),
                redzone_touches=int(redzone_touches),
                air_yards=round(float(air_yards), 1),
                adp=adp,
                bye_week=bye,
                injury_status=injury_status,
                talent=round(float(t), 4),
            ))
            player_id += 1

    df = pd.DataFrame(rows).sort_values("adp").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_player_data():
    """Attempts to load live player data via nfl_data_py. Falls back to a
    synthetic dataset (with a clear on-screen notice) if unavailable.

    Returns:
        (DataFrame, str source_label, str|None warning_message)
    """
    try:
        import nfl_data_py as nfl  # noqa: F401

        seasonal = nfl.import_seasonal_data([CURRENT_SEASON - 1])
        rosters = nfl.import_seasonal_rosters([CURRENT_SEASON - 1])

        merged = seasonal.merge(
            rosters[["player_id", "player_name", "position", "team"]],
            on="player_id",
            how="left",
        )
        merged = merged[merged["position"].isin(POSITIONS)].copy()
        if merged.empty:
            raise ValueError("nfl_data_py returned no usable rows for current positions.")

        games = merged["games"].replace(0, np.nan)
        merged["proj_mean"] = (merged.get("fantasy_points_ppr", merged.get("fantasy_points", 0)) / games).fillna(0)
        merged["proj_std"] = (merged["proj_mean"] * 0.35).clip(lower=1.0)
        merged["target_share"] = merged.get("tgt_sh", 0).fillna(0)
        merged["snap_pct"] = merged.get("offense_pct", 0).fillna(0)
        merged["redzone_touches"] = merged.get("rz_tgt", merged.get("rz_carries", 0)).fillna(0)
        merged["air_yards"] = merged.get("air_yards", 0).fillna(0)
        merged["adp"] = merged["proj_mean"].rank(ascending=False, method="first")
        merged["bye_week"] = np.random.default_rng(RNG_SEED).integers(5, 14, size=len(merged))
        merged["injury_status"] = "Healthy"
        merged["player_id"] = merged["player_id"]

        keep_cols = [
            "player_id", "player_name", "position", "team", "proj_mean", "proj_std",
            "target_share", "snap_pct", "redzone_touches", "air_yards", "adp",
            "bye_week", "injury_status",
        ]
        out = merged[keep_cols].dropna(subset=["player_name"]).reset_index(drop=True)
        out["talent"] = (out["proj_mean"] - out["proj_mean"].min()) / (
            out["proj_mean"].max() - out["proj_mean"].min() + 1e-9
        )
        if len(out) < 40:
            raise ValueError("Live dataset too small after filtering — using synthetic fallback.")
        return out, "live (nfl_data_py)", None

    except Exception as exc:  # noqa: BLE001 — intentionally broad: any failure -> graceful fallback
        synthetic = _generate_synthetic_dataset()
        warning = (
            "Live NFL data could not be loaded right now "
            f"({type(exc).__name__}: {exc}). RosterIQ is running on a realistic "
            "synthetic dataset so every tool below still works — swap in your own "
            "feed any time via `load_player_data()`."
        )
        return synthetic, "synthetic (fallback)", warning


# ----------------------------------------------------------------------------
# SCORING / LEAGUE-SETTINGS HELPERS
# ----------------------------------------------------------------------------
def scoring_multiplier(scoring: str) -> float:
    """Rough PPR-equivalent multiplier applied to receptions-driven scoring."""
    return {"PPR": 1.0, "Half-PPR": 0.85, "Standard": 0.7}.get(scoring, 1.0)


def apply_scoring(df: pd.DataFrame, scoring: str) -> pd.DataFrame:
    df = df.copy()
    mult = scoring_multiplier(scoring)
    # WR/TE/RB benefit more from PPR since receptions matter; QB unaffected
    receiving_bump = df["position"].isin(["WR", "TE", "RB"]).astype(float)
    factor = 1 + (mult - 0.7) * 0.35 * receiving_bump
    df["proj_mean_scored"] = (df["proj_mean"] * factor).round(2)
    df["proj_std_scored"] = (df["proj_std"] * factor).round(2)
    return df


def replacement_levels(df: pd.DataFrame, league_size: int, roster: dict) -> dict:
    """Compute replacement-level (baseline) weekly points for each position
    given how many starters of that position get drafted league-wide,
    including a share of FLEX slots split across RB/WR/TE by usage."""
    starters_needed = {
        "QB": roster["QB"] * league_size,
        "RB": roster["RB"] * league_size,
        "WR": roster["WR"] * league_size,
        "TE": roster["TE"] * league_size,
    }
    flex_total = roster["FLEX"] * league_size
    # split flex slots proportionally to typical flex usage: RB 55%, WR 40%, TE 5%
    flex_split = {"RB": 0.55, "WR": 0.40, "TE": 0.05}
    for pos, share in flex_split.items():
        starters_needed[pos] += flex_total * share

    baselines = {}
    for pos in POSITIONS:
        pos_df = df[df["position"] == pos].sort_values("proj_mean_scored", ascending=False)
        idx = int(round(starters_needed.get(pos, 0)))
        idx = max(idx - 1, 0)
        if len(pos_df) == 0:
            baselines[pos] = 0.0
        else:
            idx = min(idx, len(pos_df) - 1)
            baselines[pos] = float(pos_df.iloc[idx]["proj_mean_scored"])
    return baselines


def tier_label(vbd_value: float, pos_max: float) -> str:
    if pos_max <= 0:
        return "t3"
    ratio = vbd_value / pos_max
    if ratio >= 0.66:
        return "t1"
    if ratio >= 0.33:
        return "t2"
    return "t3"


# ----------------------------------------------------------------------------
# MONTE CARLO ENGINE
# ----------------------------------------------------------------------------
def run_matchup_sim(
    mean_a, std_a, mean_b, std_b,
    opp_def_a=1.0, opp_def_b=1.0,
    weather_a=1.0, weather_b=1.0,
    iterations=1000, seed=RNG_SEED,
):
    rng = np.random.default_rng(seed)
    adj_mean_a = mean_a * opp_def_a * weather_a
    adj_mean_b = mean_b * opp_def_b * weather_b
    sims_a = rng.normal(adj_mean_a, max(std_a, 0.5), iterations).clip(min=0)
    sims_b = rng.normal(adj_mean_b, max(std_b, 0.5), iterations).clip(min=0)

    win_prob_a = float((sims_a > sims_b).mean())
    boom_thresh_a = adj_mean_a + std_a
    bust_thresh_a = max(adj_mean_a - std_a, 0)
    boom_prob_a = float((sims_a >= boom_thresh_a).mean())
    bust_prob_a = float((sims_a <= bust_thresh_a).mean())

    boom_thresh_b = adj_mean_b + std_b
    bust_thresh_b = max(adj_mean_b - std_b, 0)
    boom_prob_b = float((sims_b >= boom_thresh_b).mean())
    bust_prob_b = float((sims_b <= bust_thresh_b).mean())

    return dict(
        sims_a=sims_a, sims_b=sims_b,
        win_prob_a=win_prob_a, win_prob_b=1 - win_prob_a,
        boom_prob_a=boom_prob_a, bust_prob_a=bust_prob_a,
        boom_prob_b=boom_prob_b, bust_prob_b=bust_prob_b,
        adj_mean_a=adj_mean_a, adj_mean_b=adj_mean_b,
    )


# ----------------------------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------------------------
def init_session_state():
    if "visit_counted" not in st.session_state:
        st.session_state.visit_counted = True
        st.session_state.visitor_count = st.session_state.get("visitor_count", 0) + 1
    if "visitor_count" not in st.session_state:
        st.session_state.visitor_count = 1
    if "drafted_players" not in st.session_state:
        st.session_state.drafted_players = set()
    if "my_team" not in st.session_state:
        st.session_state.my_team = set()


init_session_state()

# ----------------------------------------------------------------------------
# SIDEBAR — LEAGUE SETTINGS + SHARE BOX
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏈 League Settings")
    league_size = st.select_slider(
        "League Size", options=LEAGUE_SIZE_OPTIONS, value=12,
        help="Number of teams in your fantasy league. Drives replacement-level baselines league-wide.",
    )
    scoring_system = st.selectbox("Scoring System", SCORING_OPTIONS, index=0)

    st.markdown("**Starting Roster Slots**")
    col_a, col_b = st.columns(2)
    with col_a:
        n_qb = st.number_input("QB", min_value=0, max_value=3, value=1, step=1)
        n_wr = st.number_input("WR", min_value=0, max_value=5, value=2, step=1)
        n_flex = st.number_input("FLEX", min_value=0, max_value=4, value=1, step=1)
    with col_b:
        n_rb = st.number_input("RB", min_value=0, max_value=5, value=2, step=1)
        n_te = st.number_input("TE", min_value=0, max_value=3, value=1, step=1)

    roster_settings = {"QB": n_qb, "RB": n_rb, "WR": n_wr, "TE": n_te, "FLEX": n_flex}

    st.markdown("<div class='rq-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🔗 Share This Tool")
    app_url = st.text_input(
        "Your deployed app URL", value="https://your-app-name.streamlit.app",
        help="Once deployed on Streamlit Community Cloud, paste your live URL here so teammates can grab it.",
        label_visibility="collapsed",
    )
    st.markdown(f"<div class='rq-share'><code>{app_url}</code></div>", unsafe_allow_html=True)
    st.caption("Copy the link above and drop it in your league group chat.")

# ----------------------------------------------------------------------------
# LOAD + PREP DATA
# ----------------------------------------------------------------------------
raw_df, data_source, data_warning = load_player_data()

if raw_df is None or raw_df.empty:
    st.error("RosterIQ could not load any player data (live or synthetic). Please refresh the page.")
    st.stop()

if data_warning:
    st.info(f"ℹ️ Data source: **{data_source}**. {data_warning}")

df = apply_scoring(raw_df, scoring_system)
baselines = replacement_levels(df, league_size, roster_settings)
df["replacement_pts"] = df["position"].map(baselines).fillna(0)
df["vbd"] = (df["proj_mean_scored"] - df["replacement_pts"]).round(2)

pos_max_vbd = df.groupby("position")["vbd"].max().to_dict()
df["tier"] = df.apply(lambda r: tier_label(r["vbd"], pos_max_vbd.get(r["position"], 1)), axis=1)

# ----------------------------------------------------------------------------
# BANNER
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="rq-banner">
        <p class="rq-banner-title">Roster<span>IQ</span> 🏈</p>
        <p class="rq-banner-sub">
            {league_size}-team &middot; {scoring_system} &middot; Draft, start/sit, trades, waivers,
            deep stats &amp; standings — all powered by probabilistic modeling.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_draft, tab_sim, tab_trade, tab_waiver, tab_stats, tab_standings = st.tabs([
    "🎯 Draft Room",
    "🎲 Start/Sit — Monte Carlo",
    "🔁 Trade Evaluator",
    "📈 Waiver Wire & Injuries",
    "📊 Deep Player Stats",
    "🏆 Standings & Playoff Odds",
])

# =============================================================================
# TAB 1 — DRAFT ROOM (VBD)
# =============================================================================
with tab_draft:
    st.markdown("#### Value-Based Drafting")
    st.caption(
        "Replacement baselines are computed live from your league size and roster slots. "
        "VBD = projected points above the last realistic starter at that position."
    )

    badge_cols = st.columns(4)
    for i, pos in enumerate(POSITIONS):
        with badge_cols[i]:
            st.markdown(
                f"""
                <div class="rq-badge {'orange' if i % 2 else ''}">
                    <div class="label">{pos} Replacement Level</div>
                    <div class="value">{baselines.get(pos, 0):.1f} pts</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='rq-divider'></div>", unsafe_allow_html=True)

    col_filter, col_search = st.columns([1, 2])
    with col_filter:
        pos_filter = st.multiselect("Filter position", POSITIONS, default=POSITIONS)
    with col_search:
        search_term = st.text_input("Search player name", "")

    board = df[df["position"].isin(pos_filter)].copy()
    if search_term.strip():
        board = board[board["player_name"].str.contains(search_term.strip(), case=False, na=False)]
    board = board.sort_values("vbd", ascending=False)
    board["Drafted"] = board["player_id"].isin(st.session_state.drafted_players)
    board_display = board[~board["Drafted"]].reset_index(drop=True)

    st.markdown("##### Draft Board")
    if board_display.empty:
        st.warning("No players match your filters (or everyone's been drafted). Adjust filters above.")
    else:
        show_cols = ["player_name", "position", "team", "adp", "proj_mean_scored", "vbd", "tier", "bye_week"]
        pretty = board_display[show_cols].rename(columns={
            "player_name": "Player", "position": "Pos", "team": "Team", "adp": "ADP",
            "proj_mean_scored": "Proj Pts/G", "vbd": "VBD", "tier": "Tier", "bye_week": "Bye",
        }).head(60)
        st.dataframe(pretty, use_container_width=True, height=420)

        st.markdown("##### Mark Players Drafted")
        pick_options = board_display["player_name"] + " (" + board_display["position"] + " - " + board_display["team"] + ")"
        pick_map = dict(zip(pick_options, board_display["player_id"]))
        picks = st.multiselect("Select player(s) to mark as drafted", list(pick_options))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Mark Drafted"):
                for p in picks:
                    st.session_state.drafted_players.add(pick_map[p])
                st.rerun()
        with c2:
            if st.button("↩️ Reset Draft Board"):
                st.session_state.drafted_players = set()
                st.rerun()

    st.markdown("<div class='rq-divider'></div>", unsafe_allow_html=True)
    st.markdown("##### Positional Tier Lists")
    tier_tabs = st.tabs(POSITIONS)
    for pos, t in zip(POSITIONS, tier_tabs):
        with t:
            pos_board = df[(df["position"] == pos) & (~df["player_id"].isin(st.session_state.drafted_players))]
            pos_board = pos_board.sort_values("vbd", ascending=False).head(24)
            for tier_id, tier_name in [("t1", "Tier 1 — Elite"), ("t2", "Tier 2 — Solid Starters"), ("t3", "Tier 3 — Depth/Flex")]:
                tier_players = pos_board[pos_board["tier"] == tier_id]
                if tier_players.empty:
                    continue
                names = ", ".join(tier_players["player_name"].tolist())
                st.markdown(
                    f"<span class='rq-pill {tier_id}'>{tier_name}</span> &nbsp; {names}",
                    unsafe_allow_html=True,
                )

# =============================================================================
# TAB 2 — WEEKLY START/SIT MONTE CARLO
# =============================================================================
with tab_sim:
    st.markdown("#### Monte Carlo Matchup Engine")
    st.caption("Simulate two players head-to-head using mean scoring, volatility, opponent defense, and weather.")

    all_names = df.sort_values("player_name")["player_name"].tolist()
    if len(all_names) < 2:
        st.error("Not enough players loaded to run a comparison.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            player_a_name = st.selectbox("Player A", all_names, index=0, key="sim_a")
        with c2:
            default_b_idx = 1 if len(all_names) > 1 else 0
            player_b_name = st.selectbox("Player B", all_names, index=default_b_idx, key="sim_b")

        iterations = st.slider("Monte Carlo Iterations", min_value=500, max_value=1000, value=1000, step=100)

        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            opp_def_a = st.slider("Player A — Opponent Defense Factor", 0.7, 1.3, 1.0, 0.01,
                                   help="< 1.0 = tougher matchup, > 1.0 = softer matchup")
        with adv2:
            opp_def_b = st.slider("Player B — Opponent Defense Factor", 0.7, 1.3, 1.0, 0.01)
        with adv3:
            weather_a = st.slider("Weather Impact (A)", 0.8, 1.0, 1.0, 0.01)
        weather_b = st.slider("Weather Impact (B)", 0.8, 1.0, 1.0, 0.01)

        row_a = df[df["player_name"] == player_a_name].iloc[0]
        row_b = df[df["player_name"] == player_b_name].iloc[0]

        if st.button("▶️ Run Simulation", type="primary"):
            result = run_matchup_sim(
                row_a["proj_mean_scored"], row_a["proj_std_scored"],
                row_b["proj_mean_scored"], row_b["proj_std_scored"],
                opp_def_a, opp_def_b, weather_a, weather_b,
                iterations=iterations,
            )

            recommended = player_a_name if result["win_prob_a"] >= result["win_prob_b"] else player_b_name
            st.success(f"⭐ **Start Recommendation: {recommended}**")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="rq-badge"><div class="label">{player_a_name} Win Prob</div>
                <div class="value">{result['win_prob_a']*100:.1f}%</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="rq-badge orange"><div class="label">{player_b_name} Win Prob</div>
                <div class="value">{result['win_prob_b']*100:.1f}%</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="rq-badge"><div class="label">{player_a_name} Boom / Bust</div>
                <div class="value">{result['boom_prob_a']*100:.0f}% / {result['bust_prob_a']*100:.0f}%</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="rq-badge orange"><div class="label">{player_b_name} Boom / Bust</div>
                <div class="value">{result['boom_prob_b']*100:.0f}% / {result['bust_prob_b']*100:.0f}%</div></div>""", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=result["sims_a"], name=player_a_name, opacity=0.65,
                                        marker_color="#00e6a0", nbinsx=40))
            fig.add_trace(go.Histogram(x=result["sims_b"], name=player_b_name, opacity=0.65,
                                        marker_color="#ff7a1a", nbinsx=40))
            fig.update_layout(
                barmode="overlay",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Simulated Fantasy Point Distributions",
                xaxis_title="Fantasy Points",
                yaxis_title="Frequency",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3 — TRADE EVALUATOR
# =============================================================================
with tab_trade:
    st.markdown("#### Multi-Player Trade Evaluator")
    st.caption("Compares net projected points across remaining weeks, scarcity-adjusted for your league size.")

    current_week = st.slider("Current Week", 1, TOTAL_WEEKS, 6)
    remaining_weeks = max(TOTAL_WEEKS - current_week, 1)
    st.caption(f"Remaining weeks this season: **{remaining_weeks}**")

    all_names_sorted = df.sort_values("player_name")["player_name"].tolist()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**You Send**")
        send_names = st.multiselect("Players you give up", all_names_sorted, key="send")
    with c2:
        st.markdown("**You Receive**")
        recv_names = st.multiselect("Players you receive", all_names_sorted, key="recv")

    # Scarcity weighting: smaller leagues reward elite/top-tier talent more;
    # larger leagues reward roster depth (more replacement-level players matter).
    def scarcity_weight(vbd_value: float, league_size: int) -> float:
        elite_bias = np.clip((16 - league_size) / 8, 0, 1)  # 1.0 at 8-team, 0 at 16-team
        depth_bias = 1 - elite_bias
        # elite players (high VBD) get amplified in small leagues;
        # all players get a flat depth credit in large leagues
        return (1 + elite_bias * max(vbd_value, 0) / 40) * (1 + depth_bias * 0.15)

    def trade_side_value(names):
        if not names:
            return 0.0, pd.DataFrame()
        sub = df[df["player_name"].isin(names)].copy()
        sub["weight"] = sub["vbd"].apply(lambda v: scarcity_weight(v, league_size))
        sub["season_value"] = sub["proj_mean_scored"] * remaining_weeks * sub["weight"]
        return float(sub["season_value"].sum()), sub

    send_value, send_df = trade_side_value(send_names)
    recv_value, recv_df = trade_side_value(recv_names)
    net_value = recv_value - send_value

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="rq-badge"><div class="label">Value Sent</div>
        <div class="value">{send_value:,.0f} pts</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="rq-badge orange"><div class="label">Value Received</div>
        <div class="value">{recv_value:,.0f} pts</div></div>""", unsafe_allow_html=True)
    with m3:
        cls = "" if net_value >= 0 else "red"
        st.markdown(f"""<div class="rq-badge {cls}"><div class="label">Net Trade Value</div>
        <div class="value">{net_value:+,.0f} pts</div></div>""", unsafe_allow_html=True)

    if send_names or recv_names:
        if net_value > 15:
            st.success("📈 This trade **clearly favors you** based on projected rest-of-season value.")
        elif net_value < -15:
            st.error("📉 This trade **clearly favors the other side** — proceed with caution.")
        else:
            st.info("⚖️ This trade is roughly **fair value** — decide based on team needs/roster construction.")

        with st.expander("See player-level breakdown"):
            cols_show = ["player_name", "position", "proj_mean_scored", "vbd", "weight", "season_value"]
            if not send_df.empty:
                st.markdown("**Sending:**")
                st.dataframe(send_df[cols_show].round(2), use_container_width=True)
            if not recv_df.empty:
                st.markdown("**Receiving:**")
                st.dataframe(recv_df[cols_show].round(2), use_container_width=True)
    else:
        st.warning("Select at least one player on each side to evaluate a trade.")

# =============================================================================
# TAB 4 — WAIVER WIRE & INJURIES
# =============================================================================
with tab_waiver:
    st.markdown("#### Waiver Wire Rankings & Injury Report")
    st.caption("Ranked by a composite of target share, snap count %, and red zone involvement.")

    drafted_ids = st.session_state.drafted_players
    available = df[~df["player_id"].isin(drafted_ids)].copy() if drafted_ids else df.copy()

    def norm(series):
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    available["composite_score"] = (
        0.40 * norm(available["target_share"])
        + 0.30 * norm(available["snap_pct"])
        + 0.30 * norm(available["redzone_touches"])
    ) * 100

    # FAAB suggestion scaled by position scarcity for the chosen league size:
    # smaller leagues -> less scarcity -> lower bids; larger leagues -> higher bids
    scarcity_factor = league_size / 12.0
    available["faab_pct"] = (available["composite_score"] * 0.35 * scarcity_factor).clip(0, 100).round(1)

    pos_filter_w = st.multiselect("Position", POSITIONS, default=POSITIONS, key="waiver_pos")
    show_injured_only = st.checkbox("Show only players with a non-Healthy injury status")

    w_board = available[available["position"].isin(pos_filter_w)]
    if show_injured_only:
        w_board = w_board[w_board["injury_status"] != "Healthy"]
    w_board = w_board.sort_values("composite_score", ascending=False).head(50)

    if w_board.empty:
        st.warning("No players match these filters.")
    else:
        pretty_w = w_board[[
            "player_name", "position", "team", "target_share", "snap_pct",
            "redzone_touches", "composite_score", "faab_pct", "injury_status",
        ]].rename(columns={
            "player_name": "Player", "position": "Pos", "team": "Team",
            "target_share": "Tgt Share", "snap_pct": "Snap %",
            "redzone_touches": "RZ Touches", "composite_score": "Score",
            "faab_pct": "Suggested FAAB %", "injury_status": "Status",
        }).round(2)
        st.dataframe(pretty_w, use_container_width=True, height=460)

# =============================================================================
# TAB 5 — DEEP PLAYER STATS
# =============================================================================
with tab_stats:
    st.markdown("#### Deep Player Stats & Simulation Outcomes")

    stat_player = st.selectbox("Choose a player", df.sort_values("player_name")["player_name"].tolist(), key="deep_player")
    prow = df[df["player_name"] == stat_player].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    consistency_index = round(100 * (1 - min(prow["proj_std_scored"] / max(prow["proj_mean_scored"], 0.1), 1)), 1)
    with m1:
        st.markdown(f"""<div class="rq-badge"><div class="label">Air Yards</div>
        <div class="value">{prow['air_yards']:.0f}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="rq-badge orange"><div class="label">Target Share</div>
        <div class="value">{prow['target_share']*100:.1f}%</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="rq-badge"><div class="label">Red Zone Touches</div>
        <div class="value">{prow['redzone_touches']:.0f}</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="rq-badge orange"><div class="label">Consistency Index</div>
        <div class="value">{consistency_index}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='rq-divider'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### Target Share vs Air Yards (Position Peers)")
        peers = df[df["position"] == prow["position"]]
        fig1 = px.scatter(
            peers, x="air_yards", y="target_share", size="redzone_touches",
            color="vbd", hover_name="player_name",
            color_continuous_scale=["#223041", "#00e6a0"],
            template="plotly_dark",
        )
        fig1.add_scatter(
            x=[prow["air_yards"]], y=[prow["target_share"]], mode="markers",
            marker=dict(size=18, color="#ff7a1a", symbol="star"), name=stat_player,
        )
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        st.markdown("##### Simulated Weekly Outcome — Bell Curve")
        rng = np.random.default_rng(RNG_SEED)
        sims = rng.normal(prow["proj_mean_scored"], max(prow["proj_std_scored"], 0.5), 2000).clip(min=0)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=sims, nbinsx=40, marker_color="#00e6a0", opacity=0.85))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Simulated Fantasy Points", yaxis_title="Frequency",
        )
        st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# TAB 6 — LEAGUE STANDINGS & PLAYOFF ODDS
# =============================================================================
with tab_standings:
    st.markdown("#### League Standings & Playoff Odds")
    st.caption(
        "Synthetic league schedule generator: builds fictional team power ratings, computes "
        "all-play records, and simulates the rest of the season to estimate playoff odds."
    )

    if "team_power" not in st.session_state or st.session_state.get("standings_league_size") != league_size:
        rng = np.random.default_rng(RNG_SEED)
        st.session_state.team_power = {
            f"Team {i+1}": float(rng.normal(100, 12)) for i in range(league_size)
        }
        st.session_state.standings_league_size = league_size

    team_power = st.session_state.team_power
    current_week_s = st.slider("Weeks Completed", 1, TOTAL_WEEKS - 1, 6, key="standings_week")
    playoff_spots = st.slider("Number of Playoff Spots", 2, max(league_size // 2, 2), min(4, league_size // 2))

    rng = np.random.default_rng(RNG_SEED + current_week_s)
    team_names = list(team_power.keys())
    records = {t: {"wins": 0, "losses": 0, "all_play_wins": 0, "all_play_games": 0, "pf": 0.0} for t in team_names}

    # simulate completed weeks
    for wk in range(current_week_s):
        week_scores = {t: max(rng.normal(team_power[t], 14), 0) for t in team_names}
        shuffled = team_names[:]
        rng.shuffle(shuffled)
        pairs = list(zip(shuffled[::2], shuffled[1::2]))
        for a, b in pairs:
            records[a]["pf"] += week_scores[a]
            records[b]["pf"] += week_scores[b]
            if week_scores[a] > week_scores[b]:
                records[a]["wins"] += 1
                records[b]["losses"] += 1
            else:
                records[b]["wins"] += 1
                records[a]["losses"] += 1
        # all-play: compare each team's score to every other team's score that week
        for t in team_names:
            others = [week_scores[o] for o in team_names if o != t]
            records[t]["all_play_wins"] += sum(1 for o in others if week_scores[t] > o)
            records[t]["all_play_games"] += len(others)

    standings_df = pd.DataFrame([
        dict(
            Team=t,
            Wins=records[t]["wins"],
            Losses=records[t]["losses"],
            PF=round(records[t]["pf"], 1),
            AllPlayWinPct=round(records[t]["all_play_wins"] / max(records[t]["all_play_games"], 1), 3),
        )
        for t in team_names
    ]).sort_values(["Wins", "PF"], ascending=False).reset_index(drop=True)
    standings_df.index = standings_df.index + 1

    st.markdown("##### Current Standings")
    st.dataframe(standings_df, use_container_width=True, height=380)

    st.markdown("<div class='rq-divider'></div>", unsafe_allow_html=True)
    st.markdown("##### Playoff Odds (Monte Carlo — remaining schedule)")

    n_sims = st.slider("Season Simulations", 200, 2000, 800, step=100)
    remaining = TOTAL_WEEKS - current_week_s
    playoff_hits = {t: 0 for t in team_names}

    for _ in range(n_sims):
        sim_wins = {t: records[t]["wins"] for t in team_names}
        sim_pf = {t: records[t]["pf"] for t in team_names}
        for wk in range(remaining):
            week_scores = {t: max(rng.normal(team_power[t], 14), 0) for t in team_names}
            shuffled = team_names[:]
            rng.shuffle(shuffled)
            pairs = list(zip(shuffled[::2], shuffled[1::2]))
            for a, b in pairs:
                sim_pf[a] += week_scores[a]
                sim_pf[b] += week_scores[b]
                if week_scores[a] > week_scores[b]:
                    sim_wins[a] += 1
                else:
                    sim_wins[b] += 1
        ranked = sorted(team_names, key=lambda t: (sim_wins[t], sim_pf[t]), reverse=True)
        for t in ranked[:playoff_spots]:
            playoff_hits[t] += 1

    odds_df = pd.DataFrame([
        dict(Team=t, PlayoffOdds=round(100 * playoff_hits[t] / n_sims, 1))
        for t in team_names
    ]).sort_values("PlayoffOdds", ascending=False).reset_index(drop=True)

    fig3 = px.bar(
        odds_df, x="Team", y="PlayoffOdds", color="PlayoffOdds",
        color_continuous_scale=["#223041", "#00e6a0"],
        template="plotly_dark",
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Playoff Odds (%)", xaxis_title="",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# FOOTER — VISITOR COUNTER
# ----------------------------------------------------------------------------
st.markdown("<div class='rq-divider'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="rq-footer">
        RosterIQ 🏈 &middot; Session visits this browser session: <b>{st.session_state.visitor_count}</b>
        &middot; Data source: {data_source} &middot; Built with Streamlit + Plotly
        <br/>Not affiliated with the NFL. For entertainment / decision-support purposes only.
    </div>
    """,
    unsafe_allow_html=True,
)
