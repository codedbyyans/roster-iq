"""
RosterIQ — All-in-one Fantasy Football Draft & Season Manager
================================================================
Single-file Streamlit application.

Modules:
    1. Draft Room (Value-Based Drafting)
    2. Weekly Start/Sit Helper — simplified Monte Carlo matchup comparison
    3. Trade Evaluator
    4. Waiver Wire & Injuries
    5. Deep Player Stats

Player data is built from a curated list of real, current NFL players
(see `_QB_ROSTER` / `_RB_ROSTER` / `_WR_ROSTER` / `_TE_ROSTER` below).
Projections are modeled deterministically from each player's rank within
their position, so names are always real even without a live feed —
update those lists any time to refresh who's included.
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
RNG_SEED = 42

# ----------------------------------------------------------------------------
# DATA LAYER — curated real-player roster
# ----------------------------------------------------------------------------
# Real NFL players, listed roughly best-to-worst within each position so the
# talent curve below produces sensible tiers/ADP. Swap this list any time to
# refresh who's included — every name here is a real, current NFL player.
_QB_ROSTER = [
    ("Patrick Mahomes", "KC"), ("Josh Allen", "BUF"), ("Lamar Jackson", "BAL"),
    ("Joe Burrow", "CIN"), ("Jalen Hurts", "PHI"), ("Justin Herbert", "LAC"),
    ("Jordan Love", "GB"), ("C.J. Stroud", "HOU"), ("Jayden Daniels", "WAS"),
    ("Dak Prescott", "DAL"), ("Brock Purdy", "SF"), ("Kyler Murray", "ARI"),
    ("Caleb Williams", "CHI"), ("Trevor Lawrence", "JAX"), ("Tua Tagovailoa", "MIA"),
    ("Drake Maye", "NE"), ("Bo Nix", "DEN"), ("Baker Mayfield", "TB"),
    ("Matthew Stafford", "LAR"), ("Geno Smith", "SEA"), ("Jared Goff", "DET"),
    ("Kirk Cousins", "ATL"), ("Sam Darnold", "SEA"), ("Anthony Richardson", "IND"),
    ("Bryce Young", "CAR"), ("Aaron Rodgers", "PIT"), ("Russell Wilson", "NYG"),
    ("Will Levis", "TEN"), ("Michael Penix Jr.", "ATL"), ("J.J. McCarthy", "MIN"),
    ("Derek Carr", "NO"), ("Justin Fields", "NYJ"),
]

_RB_ROSTER = [
    ("Christian McCaffrey", "SF"), ("Bijan Robinson", "ATL"), ("Breece Hall", "NYJ"),
    ("Jonathan Taylor", "IND"), ("Saquon Barkley", "PHI"), ("Derrick Henry", "BAL"),
    ("Josh Jacobs", "GB"), ("Kenneth Walker III", "SEA"), ("De'Von Achane", "MIA"),
    ("Jahmyr Gibbs", "DET"), ("Travis Etienne", "JAX"), ("Isiah Pacheco", "KC"),
    ("James Cook", "BUF"), ("Rachaad White", "TB"), ("Kyren Williams", "LAR"),
    ("Alvin Kamara", "NO"), ("Aaron Jones", "MIN"), ("Joe Mixon", "HOU"),
    ("Chuba Hubbard", "CAR"), ("Tony Pollard", "TEN"), ("David Montgomery", "DET"),
    ("Javonte Williams", "DAL"), ("Rhamondre Stevenson", "NE"), ("Najee Harris", "PIT"),
    ("Zack Moss", "CIN"), ("Brian Robinson Jr.", "WAS"), ("Jaylen Warren", "PIT"),
    ("Zamir White", "LV"), ("D'Andre Swift", "CHI"), ("Austin Ekeler", "WAS"),
    ("Ezekiel Elliott", "DAL"), ("J.K. Dobbins", "LAC"), ("Devin Singletary", "NYG"),
    ("Gus Edwards", "LAC"), ("Jerome Ford", "CLE"), ("Dameon Pierce", "HOU"),
    ("Antonio Gibson", "NE"), ("Tyler Allgeier", "ATL"), ("Ty Chandler", "MIN"),
    ("Jaleel McLaughlin", "DEN"), ("Samaje Perine", "KC"), ("Khalil Herbert", "CHI"),
    ("Justice Hill", "BAL"), ("Alexander Mattison", "LV"), ("Roschon Johnson", "CHI"),
    ("Craig Reynolds", "DET"), ("Miles Sanders", "DAL"), ("Cam Akers", "MIN"),
]

_WR_ROSTER = [
    ("Justin Jefferson", "MIN"), ("Ja'Marr Chase", "CIN"), ("Tyreek Hill", "MIA"),
    ("CeeDee Lamb", "DAL"), ("Amon-Ra St. Brown", "DET"), ("A.J. Brown", "PHI"),
    ("Puka Nacua", "LAR"), ("Garrett Wilson", "NYJ"), ("Stefon Diggs", "HOU"),
    ("Davante Adams", "LAR"), ("Mike Evans", "TB"), ("DK Metcalf", "PIT"),
    ("DeVonta Smith", "PHI"), ("Chris Olave", "NO"), ("Amari Cooper", "BUF"),
    ("Cooper Kupp", "SEA"), ("Nico Collins", "HOU"), ("Terry McLaurin", "WAS"),
    ("Brandon Aiyuk", "SF"), ("Calvin Ridley", "TEN"), ("Jaylen Waddle", "MIA"),
    ("Deebo Samuel", "WAS"), ("Tee Higgins", "CIN"), ("Marvin Harrison Jr.", "ARI"),
    ("Malik Nabers", "NYG"), ("Rome Odunze", "CHI"), ("Drake London", "ATL"),
    ("Jordan Addison", "MIN"), ("Zay Flowers", "BAL"), ("Christian Watson", "GB"),
    ("Jameson Williams", "DET"), ("Chris Godwin", "TB"), ("Michael Pittman Jr.", "IND"),
    ("Courtland Sutton", "DEN"), ("Diontae Johnson", "BAL"), ("George Pickens", "DAL"),
    ("Jerry Jeudy", "CLE"), ("Keenan Allen", "LAC"), ("Adam Thielen", "CAR"),
    ("Curtis Samuel", "BUF"), ("Rashee Rice", "KC"), ("Xavier Worthy", "KC"),
    ("Jaxon Smith-Njigba", "SEA"), ("Tank Dell", "HOU"), ("Josh Downs", "IND"),
    ("Jayden Reed", "GB"), ("Christian Kirk", "JAX"), ("Gabe Davis", "JAX"),
    ("Darnell Mooney", "ATL"), ("Elijah Moore", "CLE"), ("Romeo Doubs", "GB"),
    ("Wan'Dale Robinson", "NYG"), ("Jahan Dotson", "PHI"), ("Tutu Atwell", "LAR"),
    ("Rashid Shaheed", "NO"), ("Khalil Shakir", "BUF"), ("Demario Douglas", "NE"),
    ("Jonathan Mingo", "CAR"), ("Quentin Johnston", "LAC"), ("Skyy Moore", "KC"),
    ("Marquise Brown", "KC"), ("Tyler Lockett", "SEA"), ("Allen Lazard", "NYJ"),
    ("Noah Brown", "WAS"), ("Kendrick Bourne", "NE"), ("Rashod Bateman", "BAL"),
]

_TE_ROSTER = [
    ("Travis Kelce", "KC"), ("Sam LaPorta", "DET"), ("Brock Bowers", "LV"),
    ("Mark Andrews", "BAL"), ("Trey McBride", "ARI"), ("George Kittle", "SF"),
    ("T.J. Hockenson", "MIN"), ("Kyle Pitts", "ATL"), ("Dallas Goedert", "PHI"),
    ("Evan Engram", "JAX"), ("David Njoku", "CLE"), ("Jake Ferguson", "DAL"),
    ("Dalton Kincaid", "BUF"), ("Cole Kmet", "CHI"), ("Pat Freiermuth", "PIT"),
    ("Tyler Higbee", "LAR"), ("Hunter Henry", "NE"), ("Noah Fant", "CIN"),
    ("Gerald Everett", "CHI"), ("Chigoziem Okonkwo", "TEN"), ("Isaiah Likely", "BAL"),
    ("Zach Ertz", "WAS"), ("Luke Musgrave", "GB"), ("Michael Mayer", "LV"),
    ("Cade Otton", "TB"), ("Juwan Johnson", "NO"), ("Durham Smythe", "MIA"),
    ("Tucker Kraft", "GB"), ("Ben Sinnott", "WAS"), ("Theo Johnson", "NYG"),
]

_ROSTERS_BY_POS = {"QB": _QB_ROSTER, "RB": _RB_ROSTER, "WR": _WR_ROSTER, "TE": _TE_ROSTER}


@st.cache_data(show_spinner=False)
def _build_player_dataset(seed: int = RNG_SEED) -> pd.DataFrame:
    """Builds the player pool from a curated list of real, current NFL
    players. Stats/projections are modeled deterministically based on each
    player's rank within their position, so the app always shows real names
    even without a live data feed."""
    rng = np.random.default_rng(seed)
    rows = []

    pos_profile = {
        "QB": dict(mean=(12, 24), std=(4, 7)),
        "RB": dict(mean=(3, 20), std=(3, 8)),
        "WR": dict(mean=(3, 19), std=(3, 9)),
        "TE": dict(mean=(2, 15), std=(2, 6)),
    }

    player_id = 1000
    for pos, roster in _ROSTERS_BY_POS.items():
        count = len(roster)
        lo_m, hi_m = pos_profile[pos]["mean"]
        lo_s, hi_s = pos_profile[pos]["std"]
        for i, (name, team) in enumerate(roster):
            # talent decreases smoothly from 1.0 (best) to ~0.05 (last) by list position
            t = max(1.0 - (i / max(count - 1, 1)) * 0.95, 0.05)
            t = float(np.clip(t + rng.normal(0, 0.02), 0.02, 1.0))

            mean_pts = max(lo_m + t * (hi_m - lo_m) + rng.normal(0, 0.4), 0.5)
            std_pts = max(lo_s + (1 - t) * 0.4 * (hi_s - lo_s) + rng.normal(0, 0.3), 1.0)

            target_share = np.clip(rng.normal(0.06 + t * 0.24, 0.02), 0, 0.4) if pos in ("WR", "TE", "RB") else 0.0
            snap_pct = np.clip(rng.normal(0.35 + t * 0.55, 0.06), 0.05, 0.98)
            redzone_touches = max(0, rng.poisson(1 + t * 4))
            air_yards = max(0, rng.normal(300 + t * 900, 120)) if pos in ("WR", "TE") else max(0, rng.normal(20 + t * 60, 15))
            adp = round(i + 1 + rng.normal(0, 1.5), 1)
            bye = int(rng.integers(5, 14))
            injury_status = rng.choice(
                ["Healthy", "Healthy", "Healthy", "Healthy", "Questionable", "Doubtful", "Out"],
                p=[0.60, 0.15, 0.1, 0.05, 0.06, 0.02, 0.02],
            )

            rows.append(dict(
                player_id=player_id,
                player_name=name,
                position=pos,
                team=team,
                proj_mean=round(float(mean_pts), 2),
                proj_std=round(float(std_pts), 2),
                target_share=round(float(target_share), 3),
                snap_pct=round(float(snap_pct), 3),
                redzone_touches=int(redzone_touches),
                air_yards=round(float(air_yards), 1),
                adp=adp,
                bye_week=bye,
                injury_status=injury_status,
                talent=round(t, 4),
            ))
            player_id += 1

    df = pd.DataFrame(rows).sort_values("adp").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_player_data():
    """Loads the RosterIQ player pool from the curated real-player roster.

    Returns:
        (DataFrame, str source_label, str|None info_message)
    """
    try:
        df = _build_player_dataset()
        if df is None or df.empty:
            raise ValueError("Player dataset came back empty.")
        return df, "curated real-player roster", None
    except Exception as exc:  # noqa: BLE001 — surface a clean message instead of crashing
        return pd.DataFrame(), "unavailable", f"Could not build the player dataset ({type(exc).__name__}: {exc})."


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
            and deep player stats — made simple.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_draft, tab_sim, tab_trade, tab_waiver, tab_stats = st.tabs([
    "🎯 Draft Room",
    "🎲 Start/Sit Helper",
    "🔁 Trade Evaluator",
    "📈 Waiver Wire & Injuries",
    "📊 Deep Player Stats",
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
    st.markdown("#### Who Should I Start?")
    st.caption("Pick two players and RosterIQ runs 1,000 simulated games to tell you who's the safer or higher-upside start.")

    all_names = df.sort_values("player_name")["player_name"].tolist()
    if len(all_names) < 2:
        st.error("Not enough players loaded to run a comparison.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            player_a_name = st.selectbox("Player 1", all_names, index=0, key="sim_a")
        with c2:
            default_b_idx = 1 if len(all_names) > 1 else 0
            player_b_name = st.selectbox("Player 2", all_names, index=default_b_idx, key="sim_b")

        with st.expander("Advanced options (optional)"):
            st.caption("Only touch these if you know something about this week's matchup, like a tough defense or bad weather.")
            adv1, adv2 = st.columns(2)
            with adv1:
                matchup_a = st.select_slider(
                    f"{player_a_name}'s matchup this week", options=["Much Tougher", "Tougher", "Normal", "Easier", "Much Easier"],
                    value="Normal", key="matchup_a",
                )
            with adv2:
                matchup_b = st.select_slider(
                    f"{player_b_name}'s matchup this week", options=["Much Tougher", "Tougher", "Normal", "Easier", "Much Easier"],
                    value="Normal", key="matchup_b",
                )
            matchup_map = {"Much Tougher": 0.85, "Tougher": 0.93, "Normal": 1.0, "Easier": 1.07, "Much Easier": 1.15}
            opp_def_a, opp_def_b = matchup_map[matchup_a], matchup_map[matchup_b]

        row_a = df[df["player_name"] == player_a_name].iloc[0]
        row_b = df[df["player_name"] == player_b_name].iloc[0]

        if st.button("▶️ Compare Players", type="primary"):
            result = run_matchup_sim(
                row_a["proj_mean_scored"], row_a["proj_std_scored"],
                row_b["proj_mean_scored"], row_b["proj_std_scored"],
                opp_def_a, opp_def_b, 1.0, 1.0,
                iterations=1000,
            )

            recommended = player_a_name if result["win_prob_a"] >= result["win_prob_b"] else player_b_name
            st.success(f"⭐ **Start {recommended}**")
            st.caption(
                f"Based on 1,000 simulated games, {recommended} came out ahead more often. "
                "This is a data-informed suggestion, not a guarantee — always factor in injuries and team news too."
            )

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class="rq-badge"><div class="label">{player_a_name} — Chance of the Better Game</div>
                <div class="value">{result['win_prob_a']*100:.0f}%</div></div>""", unsafe_allow_html=True)
                st.caption(f"Big game chance: {result['boom_prob_a']*100:.0f}% &nbsp;•&nbsp; Rough game chance: {result['bust_prob_a']*100:.0f}%")
            with m2:
                st.markdown(f"""<div class="rq-badge orange"><div class="label">{player_b_name} — Chance of the Better Game</div>
                <div class="value">{result['win_prob_b']*100:.0f}%</div></div>""", unsafe_allow_html=True)
                st.caption(f"Big game chance: {result['boom_prob_b']*100:.0f}% &nbsp;•&nbsp; Rough game chance: {result['bust_prob_b']*100:.0f}%")

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
                title="Range of Likely Fantasy Scores",
                xaxis_title="Fantasy Points",
                yaxis_title="How Often (out of 1,000 simulations)",
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
