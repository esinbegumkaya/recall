from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from firebase_auth import (
    consume_auth_redirect,
    get_firebase_config,
    render_firebase_login,
)


# =========================================================
# Project path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recall.database import get_index_stats, initialize_database  # noqa: E402
from recall.engine import RecallEngine  # noqa: E402


# =========================================================
# App configuration
# =========================================================

APP_NAME = "Recall"
APP_TAGLINE = "Your knowledge, always at hand."

st.set_page_config(
    page_title="Recall · Local Knowledge Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Global styling — auto dark/light (follows the OS/browser
# color-scheme preference, independent of Streamlit's own
# theme setting), plus a more refined visual language.
# =========================================================

st.markdown(
    """
    <style>
        :root {
            /* Bridge onto Streamlit's own theme variables. These come from
               .streamlit/config.toml ([theme.light] / [theme.dark]) and
               switch automatically when the person picks "Use system
               setting" in the app's ⋮ → Settings → Theme menu — which then
               follows the OS/browser light/dark preference from then on. */
            --recall-bg-1: var(--st-background-color);
            --recall-bg-2: var(--st-background-color);
            --recall-bg-3: var(--st-secondary-background-color);

            --recall-text: var(--st-text-color);
            --recall-heading: var(--st-text-color);
            --recall-muted: var(--st-text-color);
            --recall-faint: var(--st-text-color);

            --recall-border: var(--st-border-color);
            --recall-card: var(--st-secondary-background-color);
            --recall-card-strong: var(--st-secondary-background-color);
            --recall-card-soft: var(--st-secondary-background-color);
            --recall-input: var(--st-secondary-background-color);
            --recall-input-border: var(--st-border-color);

            --recall-shadow: rgba(65, 75, 124, 0.10);
            --recall-watermark: rgba(88, 104, 214, 0.06);
            --recall-header: var(--st-background-color);

            /* Sidebar keeps its own fixed dark palette in both modes,
               matching [theme.light.sidebar] / [theme.dark.sidebar]. */
            --recall-sidebar-1: #171D31;
            --recall-sidebar-2: #111628;

            --recall-navy: #11172A;
            --recall-indigo: #5962D7;
            --recall-indigo-soft: #7278E8;
            --recall-blue: #6E9FFB;
            --recall-success: #20B486;
            --recall-warning: #F2A23A;
            --recall-danger: #D64545;
        }

        /* Note: muted/faint text share the same variable as the main text
           color (Streamlit doesn't expose separate "muted" theme tokens),
           so all body text uses one consistent color per theme. */

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        html, body, #root,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: transparent !important;
            color: var(--recall-text) !important;
        }

        .stApp {
            min-height: 100vh;
            color: var(--recall-text) !important;
            background:
                radial-gradient(circle at 72% 8%, rgba(112,126,236,0.16), transparent 27%),
                radial-gradient(circle at 94% 36%, rgba(90,145,230,0.12), transparent 24%),
                linear-gradient(135deg, var(--recall-bg-1) 0%, var(--recall-bg-2) 52%, var(--recall-bg-3) 100%) !important;
            background-attachment: fixed !important;
            transition: background 0.25s ease, color 0.25s ease;
        }

        .stApp::before {
            content: "∞";
            position: fixed;
            right: 4vw;
            top: 10vh;
            font-size: min(32vw, 490px);
            line-height: 0.8;
            font-family: Georgia, serif;
            color: var(--recall-watermark);
            transform: rotate(-4deg);
            z-index: 0;
            pointer-events: none;
            user-select: none;
        }

        [data-testid="stHeader"], .stApp > header {
            background: var(--recall-header) !important;
            color: var(--recall-text) !important;
            border-bottom: 1px solid var(--recall-border) !important;
            backdrop-filter: blur(18px);
            min-height: 3.5rem !important;
        }

        [data-testid="stAppViewContainer"] { padding-top: 0 !important; }

        [data-testid="stToolbar"], [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] svg, [data-testid="stHeaderActionElements"],
        [data-testid="stHeaderActionElements"] button,
        [data-testid="stHeaderActionElements"] svg,
        [data-testid="stMainMenu"] button, [data-testid="stStatusWidget"] {
            color: var(--recall-text) !important;
            fill: currentColor !important;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 4.65rem;
            padding-bottom: 3rem;
            position: relative;
            z-index: 1;
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 18% 10%, rgba(103,111,220,0.20), transparent 22%),
                linear-gradient(180deg, var(--recall-sidebar-1) 0%, var(--recall-sidebar-2) 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        [data-testid="stSidebar"] * { color: #EFF2FF; }

        [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display: none; }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 11px;
            margin-bottom: 0.18rem;
            padding-top: 0.30rem;
            padding-bottom: 0.30rem;
            transition: all 0.18s ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.08);
        }

        .recall-brand { padding: 0.4rem 0 0.2rem 0; }

        .recall-logo {
            font-size: 3.4rem;
            line-height: 0.8;
            font-family: Georgia, serif;
            background: linear-gradient(135deg,#D9DDFE,#6F79EA,#7EB6F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .recall-brand-name { font-size: 1.72rem; font-weight: 700; letter-spacing: -0.03em; }

        .recall-brand-tagline {
            color: rgba(236,240,255,0.70) !important;
            font-size: 0.78rem;
            margin-top: 0.10rem;
        }

        .local-card {
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.035);
            border-radius: 14px;
            padding: 0.9rem;
            margin-top: 0.55rem;
        }

        .local-card-title { font-size: 0.82rem; font-weight: 700; color: #F6F7FF !important; }

        .local-card-text {
            font-size: 0.72rem;
            line-height: 1.45;
            color: rgba(236,240,255,0.62) !important;
            margin-top: 0.22rem;
        }

        .page-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.17em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #7A83DF;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-size: clamp(2.2rem,4.2vw,4.1rem);
            line-height: 1.02;
            letter-spacing: -0.055em;
            font-weight: 760;
            color: var(--recall-heading);
            max-width: 820px;
            margin: 0;
        }

        .hero-title .accent {
            background: linear-gradient(90deg,#5A62D7,#878CF2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-copy {
            color: var(--recall-muted);
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.62;
            margin-top: 0.8rem;
            margin-bottom: 0.4rem;
        }

        .hero-panel, .metric-card, .surface, .answer-shell, .login-card {
            border: 1px solid var(--recall-border);
            background: var(--recall-card);
            color: var(--recall-text);
            box-shadow: 0 14px 38px var(--recall-shadow);
            backdrop-filter: blur(14px);
        }

        .hero-panel { border-radius: 24px; padding: 1.45rem 1.6rem; overflow: hidden; position: relative; }

        .hero-panel::after {
            content: "∞";
            position: absolute;
            right: 0.5rem;
            top: -2.2rem;
            font-family: Georgia, serif;
            font-size: 11rem;
            color: rgba(110,125,235,0.10);
            pointer-events: none;
        }

        .hero-mini { position: relative; z-index: 1; }
        .hero-mini-title { font-size: 1.18rem; font-weight: 760; color: var(--recall-heading); margin-bottom: 0.35rem; }
        .hero-mini-copy { font-size: 0.88rem; line-height: 1.5; color: var(--recall-muted); max-width: 360px; }

        .metric-card { border-radius: 17px; padding: 1rem 1.1rem; min-height: 118px; }

        .metric-icon {
            width: 42px; height: 42px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.15rem; color: #8189F1;
            background: rgba(93,102,220,0.13);
            margin-bottom: 0.60rem;
        }

        .metric-label {
            font-size: 0.74rem; font-weight: 650; color: var(--recall-muted);
            text-transform: uppercase; letter-spacing: 0.065em;
        }

        .metric-value { font-size: 1.55rem; font-weight: 760; color: var(--recall-heading); margin-top: 0.12rem; }
        .metric-caption { font-size: 0.76rem; color: var(--recall-faint); margin-top: 0.12rem; }

        .section-heading { font-size: 1.16rem; font-weight: 760; color: var(--recall-heading); margin-bottom: 0.7rem; }

        .surface { border-radius: 18px; padding: 1.1rem 1.2rem; }

        .feature {
            border: 1px solid var(--recall-border);
            border-radius: 15px;
            background: var(--recall-card-soft);
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.65rem;
        }

        .feature-title { font-weight: 720; font-size: 0.91rem; color: var(--recall-heading); }
        .feature-copy { color: var(--recall-muted); font-size: 0.78rem; line-height: 1.42; margin-top: 0.15rem; }

        .status-row {
            display: flex; justify-content: space-between; gap: 1rem; align-items: center;
            padding: 0.56rem 0; border-bottom: 1px solid var(--recall-border);
            font-size: 0.83rem; color: var(--recall-text);
        }

        .status-row:last-child { border-bottom: 0; }

        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #20B486; display: inline-block; margin-right: 0.42rem;
        }

        .answer-shell { border-radius: 18px; background: var(--recall-card-strong); padding: 1.2rem 1.3rem; }

        .source-badge {
            display: inline-flex; align-items: center;
            border: 1px solid rgba(115,126,235,0.23);
            background: rgba(91,101,212,0.13);
            color: #858DF2;
            border-radius: 999px;
            padding: 0.20rem 0.58rem;
            font-size: 0.72rem;
            font-weight: 720;
            margin-bottom: 0.4rem;
        }

        .pipeline-wrap {
            display: grid;
            grid-template-columns: repeat(7,minmax(110px,1fr));
            gap: 0.65rem;
            margin-top: 0.8rem;
        }

        .pipeline-step {
            border: 1px solid var(--recall-border);
            border-radius: 15px;
            background: var(--recall-card);
            padding: 0.9rem 0.7rem;
            text-align: center;
            font-size: 0.79rem;
            color: var(--recall-muted);
            box-shadow: 0 8px 20px var(--recall-shadow);
        }

        .pipeline-step b { display: block; font-size: 0.83rem; color: var(--recall-heading); margin-bottom: 0.2rem; }

        .profile-avatar {
            width: 72px; height: 72px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg,#5058C8,#7C82ED);
            color: #FFFFFF; font-size: 1.4rem; font-weight: 760;
            box-shadow: 0 10px 25px rgba(70,80,190,0.22);
        }

        .login-wrap { max-width: 1040px; margin: 5vh auto 0 auto; }
        .login-card { border-radius: 26px; background: var(--recall-card-strong); padding: 1.5rem; }

        .auth-provider-note {
            font-size: 0.72rem;
            color: var(--recall-faint);
            text-align: center;
            margin-top: 0.6rem;
        }

        [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stMainBlockContainer"], section.main, main {
            background: transparent !important;
            color: var(--recall-text) !important;
        }

        [data-testid="stSidebarContent"] { background: transparent !important; }

        .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, label,
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] { color: var(--recall-text) !important; }

        [data-testid="stCaptionContainer"] { color: var(--recall-muted) !important; }

        [data-testid="stTextInput"] > div > div,
        [data-testid="stTextArea"] > div > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {
            background: var(--recall-input) !important;
            color: var(--recall-text) !important;
            border-color: var(--recall-input-border) !important;
            box-shadow: none !important;
        }

        [data-testid="stTextInput"] > div > div {
            min-height: 3.15rem;
            border-radius: 16px !important;
            box-shadow: 0 10px 30px var(--recall-shadow) !important;
        }

        input, textarea, div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea, div[data-baseweb="select"] input {
            color: var(--recall-text) !important;
            -webkit-text-fill-color: var(--recall-text) !important;
            caret-color: var(--recall-text) !important;
        }

        input::placeholder, textarea::placeholder {
            color: var(--recall-faint) !important;
            -webkit-text-fill-color: var(--recall-faint) !important;
            opacity: 1 !important;
        }

        [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"] {
            background: var(--recall-card-strong) !important;
            color: var(--recall-text) !important;
        }

        [role="option"]:hover { background: var(--recall-card-soft) !important; }

        [data-testid="stForm"], [data-testid="stExpander"],
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--recall-card) !important;
            color: var(--recall-text) !important;
            border-color: var(--recall-border) !important;
        }

        [data-testid="stExpander"] { border-radius: 14px; }

        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 12px;
            min-height: 2.8rem;
            font-weight: 650;
            transition: all 0.15s ease;
        }

        div[data-testid="stButton"] > button:not([kind="primary"]),
        div[data-testid="stFormSubmitButton"] > button:not([kind="primary"]),
        div[data-testid="stDownloadButton"] > button {
            background: var(--recall-card-strong) !important;
            color: var(--recall-text) !important;
            border-color: var(--recall-border) !important;
        }

        div[data-testid="stButton"] > button:not([kind="primary"]):hover,
        div[data-testid="stDownloadButton"] > button:hover {
            background: var(--recall-card-soft) !important;
            border-color: rgba(120,130,230,0.38) !important;
            transform: translateY(-1px);
        }

        [data-testid="stAlert"] { color: var(--recall-text) !important; border-color: var(--recall-border) !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"] { color: var(--recall-text) !important; }
        [data-testid="stProgress"] { color: var(--recall-text) !important; }
        hr { border-color: var(--recall-border) !important; }

        @media (max-width: 1100px) {
            .pipeline-wrap { grid-template-columns: repeat(2,minmax(110px,1fr)); }
            .stApp::before { font-size: 54vw; right: -8vw; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helpers
# =========================================================

def safe_stats() -> Dict[str, Any]:
    try:
        return get_index_stats() or {}
    except Exception:
        return {}


def stat_value(stats: Dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = stats.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def metric_card(icon: str, label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def surface_title(title: str) -> None:
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)


def open_file(file_path: str) -> tuple[bool, str]:
    path = Path(file_path)
    if not path.exists():
        return False, "The file no longer exists at its indexed path."
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True, f"Opened {path.name}"
    except Exception as exc:
        return False, f"Could not open file: {exc}"


def open_folder(file_path: str) -> tuple[bool, str]:
    path = Path(file_path)
    folder = path.parent if path.suffix else path
    if not folder.exists():
        return False, "The folder no longer exists at its indexed path."
    try:
        if os.name == "nt":
            subprocess.Popen(["explorer", str(folder)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        return True, f"Opened {folder}"
    except Exception as exc:
        return False, f"Could not open folder: {exc}"


def notify_action(success: bool, message: str) -> None:
    if success:
        st.toast(message, icon="✅")
    else:
        st.error(message)


def source_location(source: Dict[str, Any]) -> str:
    pieces: List[str] = []
    page_number = source.get("page_number")
    section_name = source.get("section_name")
    chunk_index = source.get("chunk_index")
    if page_number is not None:
        pieces.append(f"Page {page_number}")
    if section_name:
        pieces.append(str(section_name))
    if chunk_index is not None:
        pieces.append(f"Chunk {chunk_index}")
    return " · ".join(pieces)


def render_source(source: Dict[str, Any], key_prefix: str) -> None:
    source_number = source.get("source_number", "?")
    file_path = str(source.get("file_path") or "")
    file_name = source.get("file_name") or Path(file_path).name or "Unknown file"

    with st.container(border=True):
        st.markdown(f'<span class="source-badge">Source {source_number}</span>', unsafe_allow_html=True)
        st.markdown(f"**{file_name}**")

        location = source_location(source)
        if location:
            st.caption(location)

        parent_text = source.get("parent_text")
        if parent_text:
            st.caption(f"Context: {parent_text}")

        evidence_text = source.get("text")
        if evidence_text:
            st.write(evidence_text)

        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            if st.button("Open file", key=f"{key_prefix}_open_file_{source_number}_{file_path}", use_container_width=True):
                success, message = open_file(file_path)
                notify_action(success, message)
        with c2:
            if st.button("Open folder", key=f"{key_prefix}_open_folder_{source_number}_{file_path}", use_container_width=True):
                success, message = open_folder(file_path)
                notify_action(success, message)


def render_result(result: Dict[str, Any], diagnostics: bool, key_prefix: str) -> None:
    st.markdown('<div class="answer-shell">', unsafe_allow_html=True)
    st.markdown("### Answer")
    st.markdown(result.get("answer") or "No answer was produced.")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Candidates", int(result.get("candidate_count") or 0))
    with c2:
        st.metric("Validated evidence", int(result.get("evidence_count") or 0))
    with c3:
        st.metric("Latency", f"{float(result.get('elapsed_seconds') or 0.0):.2f}s")
    with c4:
        st.metric("Grounding", "Abstained" if result.get("abstained") else "Supported")

    sources = result.get("sources") or []
    if sources:
        st.markdown("### Evidence")
        for source in sources:
            render_source(source, key_prefix)

    if diagnostics:
        with st.expander("Developer diagnostics", expanded=False):
            st.json(
                {
                    "abstained": result.get("abstained"),
                    "abstention_reason": result.get("abstention_reason"),
                    "candidate_count": result.get("candidate_count"),
                    "evidence_count": result.get("evidence_count"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "index_stats": result.get("index_stats"),
                }
            )


def session_average_latency() -> float:
    history = st.session_state.get("recall_history") or []
    values = [
        float(item.get("result", {}).get("elapsed_seconds") or 0.0)
        for item in history
        if item.get("result")
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def add_history(query: str, result: Dict[str, Any]) -> None:
    st.session_state.recall_history.insert(
        0,
        {
            "query": query,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "result": result,
        },
    )
    st.session_state.recall_history = st.session_state.recall_history[:20]


def initials(name: str) -> str:
    tokens = [token for token in name.strip().split() if token]
    if not tokens:
        return "R"
    if len(tokens) == 1:
        return tokens[0][0].upper()
    return (tokens[0][0] + tokens[-1][0]).upper()


# =========================================================
# Initialize
# =========================================================

initialize_database()


@st.cache_resource(show_spinner=False)
def get_recall_engine() -> RecallEngine:
    return RecallEngine()


engine = get_recall_engine()


DEFAULTS = {
    "authenticated": False,
    "profile_name": "",
    "profile_email": "",
    "profile_role": "Local User",
    "profile_workspace": "Personal Workspace",
    "auth_provider": "",
    "recall_query": "",
    "recall_result": None,
    "recall_error": None,
    "recall_history": [],
    "evaluation_result": None,
    "evaluation_query": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# Firebase auth redirect handling
#
# The Firebase widget signs the user in inside an iframe, then redirects
# the TOP-LEVEL browser window back to this app with the ID token attached
# as a query parameter (?fb_token=...). We check for that on every rerun,
# before anything else is rendered, so it works no matter which page/state
# the user lands back on.
# =========================================================

firebase_config = get_firebase_config()

if not st.session_state.authenticated:
    profile = consume_auth_redirect()
    if profile and profile.get("email"):
        st.session_state.authenticated = True
        st.session_state.profile_email = profile["email"]
        st.session_state.auth_provider = profile.get("provider", "password")

        display_name = profile.get("name") or ""
        if not display_name:
            display_name = (
                profile["email"].split("@", 1)[0].replace(".", " ").replace("_", " ").title()
            )
        st.session_state.profile_name = display_name

        st.rerun()


# =========================================================
# Sign in
# =========================================================

if not st.session_state.authenticated:
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
            """
            <div style="padding:1.4rem 0.5rem 0.5rem 0.5rem;">
                <div class="page-eyebrow">Private local AI</div>
                <div class="hero-title">
                    Your knowledge,<br>
                    <span class="accent">amplified.</span>
                </div>
                <div class="hero-copy">
                    Search, understand, and reason over your own files using
                    local retrieval, local models, and verifiable evidence.
                </div>
                <div style="font-family:Georgia,serif;font-size:9rem;line-height:0.75;
                            color:rgba(83,94,199,0.18);margin-top:1rem;">∞</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("## Welcome to Recall")
        st.caption("Sign in to open your private local knowledge workspace.")

        if firebase_config:
            render_firebase_login(firebase_config)
            st.markdown(
                '<div class="auth-provider-note">Secured by Firebase Authentication</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "Firebase yapılandırması bulunamadı. `.streamlit/secrets.toml` "
                "dosyasına `[firebase]` bölümünü ekleyip web app config "
                "değerlerini gir (bkz. `.streamlit/secrets.toml.example`)."
            )

        st.divider()
        st.markdown(
            """
            **Local-first privacy**

            Authentication is handled by Firebase; document retrieval and
            answer generation still run entirely inside your local Recall
            workflow — your files and their contents never leave this machine.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="recall-brand">
            <div class="recall-logo">∞</div>
            <div class="recall-brand-name">Recall</div>
            <div class="recall-brand-tagline">Your knowledge, always at hand.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        ["Home", "Ask Recall", "Knowledge Base", "Evaluation", "Architecture", "Profile", "Settings"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div class="local-card">
            <div class="local-card-title">🔒 Your data stays local.</div>
            <div class="local-card-text">
                Firebase handles sign-in only. Retrieval and generation stay local.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("Signed in as")
    st.markdown(f"**{st.session_state.profile_name or 'Recall User'}**")
    st.caption(st.session_state.profile_email or "Local session")
    if st.session_state.auth_provider:
        provider_label = "Google" if st.session_state.auth_provider == "google" else "E-posta/Şifre"
        st.caption(f"via {provider_label}")

    if st.button("Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.recall_result = None
        st.session_state.recall_error = None
        st.rerun()


# =========================================================
# Shared stats
# =========================================================

stats = safe_stats()
file_count = stat_value(stats, "files", "file_count")
chunk_count = stat_value(stats, "chunks", "chunk_count")
embedding_count = stat_value(stats, "embeddings", "embedding_count")
avg_latency = session_average_latency()


# =========================================================
# Home
# =========================================================

if page == "Home":
    st.markdown('<div style="height:0.25rem;"></div>', unsafe_allow_html=True)

    search_col, profile_col = st.columns([4.2, 1.1])

    with search_col:
        quick_query = st.text_input(
            "Search",
            placeholder="Search your documents or ask a question…",
            label_visibility="collapsed",
            key="home_quick_query",
        )

    with profile_col:
        st.markdown(
            f"""
            <div style="padding-top:0.35rem;display:flex;justify-content:flex-end;
                        align-items:center;gap:0.7rem;">
                <div style="width:42px;height:42px;border-radius:50%;
                            background:linear-gradient(135deg,#4F56C9,#777CEB);
                            color:white;display:flex;align-items:center;justify-content:center;
                            font-weight:700;">{initials(st.session_state.profile_name)}</div>
                <div>
                    <div style="font-weight:700;font-size:0.86rem;">
                        {st.session_state.profile_name or "Recall User"}
                    </div>
                    <div style="font-size:0.72rem;color:var(--recall-muted);">
                        {st.session_state.profile_workspace}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    hero_left, hero_right = st.columns([1.4, 1])

    with hero_left:
        first_name = (st.session_state.profile_name or "there").split()[0]
        st.markdown(
            f"""
            <div class="page-eyebrow">Welcome back, {first_name}</div>
            <div class="hero-title">
                Your knowledge,<br>
                <span class="accent">amplified.</span>
            </div>
            <div class="hero-copy">
                Recall helps you search, understand, and reason over your
                documents with AI — locally, securely, and with verifiable sources.
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Ask Recall →", type="primary", use_container_width=True):
                st.session_state.recall_query = quick_query.strip() if quick_query else ""
                st.session_state["Navigation"] = "Ask Recall"
                st.rerun()
        with c2:
            if st.button("View knowledge base", use_container_width=True):
                st.info("Open Knowledge Base from the left navigation.")

    with hero_right:
        st.markdown(
            """
            <div class="hero-panel">
                <div class="hero-mini">
                    <div style="font-family:Georgia,serif;font-size:6.3rem;line-height:0.8;
                                color:rgba(84,96,205,0.28);">∞</div>
                    <div class="hero-mini-title">Search · Understand · Trust</div>
                    <div class="hero-mini-copy">
                        Retrieval, evidence validation, local generation,
                        and citation verification work together before an answer is shown.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("▣", "Indexed files", f"{file_count:,}", "Private local corpus")
    with m2:
        metric_card("◉", "Text chunks", f"{chunk_count:,}", "Searchable retrieval units")
    with m3:
        metric_card("◈", "Embeddings", f"{embedding_count:,}", "Cached semantic vectors")
    with m4:
        metric_card("⚡", "Avg. response time", f"{avg_latency:.1f}s" if avg_latency else "—", "Current session")

    st.markdown("<br>", unsafe_allow_html=True)

    recent_col, explore_col, status_col = st.columns([1.25, 1.15, 1])

    with recent_col:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Recent activity")
        history = st.session_state.recall_history
        if not history:
            st.caption("No questions have been asked in this session yet.")
        else:
            for item in history[:5]:
                result = item.get("result") or {}
                st.markdown(f"**{item.get('query') or 'Untitled query'}**")
                st.caption(
                    f"{item.get('created_at') or ''} · "
                    f"{float(result.get('elapsed_seconds') or 0.0):.1f}s"
                )
                st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

    with explore_col:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Explore Recall")
        st.markdown(
            """
            <div class="feature">
                <div class="feature-title">Ask grounded questions</div>
                <div class="feature-copy">Get answers supported by validated local evidence.</div>
            </div>
            <div class="feature">
                <div class="feature-title">Inspect the knowledge base</div>
                <div class="feature-copy">Review index size, chunk coverage, and local storage.</div>
            </div>
            <div class="feature">
                <div class="feature-title">Run evaluations</div>
                <div class="feature-copy">Test answerability, abstention, and provenance behavior.</div>
            </div>
            <div class="feature">
                <div class="feature-title">Understand the architecture</div>
                <div class="feature-copy">See how retrieval, evidence gates, NLI, and generation connect.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with status_col:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("System status")
        st.markdown(
            """
            <div class="status-row"><span>Retrieval engine</span><span><span class="status-dot"></span>Online</span></div>
            <div class="status-row"><span>Embedding model</span><span><span class="status-dot"></span>Ready</span></div>
            <div class="status-row"><span>NLI verifier</span><span><span class="status-dot"></span>Ready</span></div>
            <div class="status-row"><span>Local database</span><span><span class="status-dot"></span>Connected</span></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("100% local workflow: your indexed document content stays on-device.")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Ask Recall
# =========================================================

elif page == "Ask Recall":
    st.markdown(
        """
        <div class="page-eyebrow">Grounded local Q&A</div>
        <div class="hero-title" style="font-size:2.7rem;">Ask Recall</div>
        <div class="hero-copy">
            Ask natural-language questions about your indexed files.
            Recall retrieves evidence, validates it, generates locally,
            and verifies citations before displaying the answer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.2, 1])

    with left:
        with st.form("recall_form"):
            query = st.text_area(
                "Question",
                value=st.session_state.recall_query,
                placeholder="What would you like to know?",
                height=120,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                ask_clicked = st.form_submit_button("Run grounded query", type="primary", use_container_width=True)
            with c2:
                clear_clicked = st.form_submit_button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.recall_query = ""
            st.session_state.recall_result = None
            st.session_state.recall_error = None
            st.rerun()

        if ask_clicked:
            query = (query or "").strip()
            st.session_state.recall_query = query
            st.session_state.recall_error = None

            if not query:
                st.warning("Enter a question first.")
            else:
                try:
                    with st.status("Running local reasoning pipeline…", expanded=True) as status:
                        st.write("Retrieving lexical and semantic candidates")
                        st.write("Filtering evidence by structure and topic")
                        st.write("Checking evidence with local NLI")
                        st.write("Generating a local grounded answer")
                        st.write("Validating citations and factual claims")

                        result_obj = engine.search(query)
                        result = result_obj.to_dict()

                        status.update(label="Answer ready", state="complete", expanded=False)

                    st.session_state.recall_result = result
                    add_history(query, result)

                except Exception as exc:
                    st.session_state.recall_result = None
                    st.session_state.recall_error = f"{type(exc).__name__}: {exc}"

        if st.session_state.recall_error:
            st.error("Recall could not complete the query.")
            with st.expander("Error details"):
                st.code(st.session_state.recall_error, language=None)

        if st.session_state.recall_result:
            st.markdown("---")
            render_result(st.session_state.recall_result, diagnostics=True, key_prefix="ask")

    with right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Try a question")
        examples = [
            "What AI experience does this person have?",
            "What programming languages are mentioned?",
            "Where was IBM Planning Analytics used?",
            "What cloud technologies are mentioned?",
        ]
        for index, example in enumerate(examples):
            if st.button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.recall_query = example
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Grounding controls")
        st.markdown(
            """
            - Structural evidence filtering
            - Topic-aware evidence admission
            - Provenance preservation
            - Local NLI verification
            - Citation-level claim validation
            - Safe deterministic fallback
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Knowledge Base
# =========================================================

elif page == "Knowledge Base":
    st.markdown(
        """
        <div class="page-eyebrow">Local corpus</div>
        <div class="hero-title" style="font-size:2.7rem;">Knowledge Base</div>
        <div class="hero-copy">Monitor the private index that powers Recall's retrieval layer.</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("▣", "Indexed files", f"{file_count:,}", "Stored in the local corpus")
    with c2:
        metric_card("◉", "Searchable chunks", f"{chunk_count:,}", "Text retrieval units")
    with c3:
        coverage = 100.0 * embedding_count / chunk_count if chunk_count else 0.0
        metric_card("◈", "Embedding coverage", f"{coverage:.1f}%", f"{embedding_count:,} embedded chunks")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Dense retrieval coverage")
        st.progress(
            min(max(coverage / 100.0, 0.0), 1.0),
            text=f"{coverage:.1f}% of indexed chunks have cached embeddings",
        )
        st.caption(
            "Lexical retrieval can search indexed text immediately. "
            "Dense semantic retrieval is available for chunks with embeddings."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Local storage")
        db_path = PROJECT_ROOT / "data" / "index" / "recall.db"
        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024 * 1024)
            st.metric("SQLite index", f"{db_size_mb:.1f} MB")
            st.caption(str(db_path))
        else:
            st.warning("Local SQLite index was not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="surface">', unsafe_allow_html=True)
    surface_title("Index lifecycle")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("**1 · Parse**")
        st.caption("Supported documents are converted into normalized text.")
    with p2:
        st.markdown("**2 · Chunk**")
        st.caption("Long documents are split into retrieval-sized evidence units.")
    with p3:
        st.markdown("**3 · Embed**")
        st.caption("Local vectors encode semantic meaning for dense search.")
    with p4:
        st.markdown("**4 · Index**")
        st.caption("SQLite stores metadata, text chunks, FTS rows, and embeddings.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Evaluation
# =========================================================

elif page == "Evaluation":
    st.markdown(
        """
        <div class="page-eyebrow">Reliability</div>
        <div class="hero-title" style="font-size:2.7rem;">Evaluation</div>
        <div class="hero-copy">
            Test whether Recall answers supported questions, abstains when
            evidence is missing, and preserves the correct attribution context.
        </div>
        """,
        unsafe_allow_html=True,
    )

    test_type = st.selectbox(
        "Evaluation case",
        ["Answerable question", "Unanswerable question", "Attribution / provenance", "Topic precision"],
    )

    defaults = {
        "Answerable question": "What AI experience does this person have?",
        "Unanswerable question": "What is this person's favorite restaurant?",
        "Attribution / provenance": "Where was IBM Planning Analytics used?",
        "Topic precision": "What experience does this person have with AI agents?",
    }

    eval_query = st.text_area("Query", value=defaults[test_type], height=105)

    expected = {
        "Answerable question": "Expected behavior: supported answer with validated citations.",
        "Unanswerable question": "Expected behavior: abstention or safe no-evidence behavior.",
        "Attribution / provenance": "Expected behavior: evidence remains associated with the correct role.",
        "Topic precision": "Expected behavior: only topic-specific evidence survives validation.",
    }

    st.info(expected[test_type])

    if st.button("Run evaluation", type="primary", use_container_width=True):
        try:
            with st.status("Evaluating production pipeline…", expanded=True) as status:
                result_obj = engine.search(eval_query.strip())
                result = result_obj.to_dict()
                status.update(label="Evaluation complete", state="complete", expanded=False)

            st.session_state.evaluation_result = result
            st.session_state.evaluation_query = eval_query.strip()

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    if st.session_state.evaluation_result:
        st.markdown("---")
        render_result(st.session_state.evaluation_result, diagnostics=True, key_prefix="evaluation")


# =========================================================
# Architecture
# =========================================================

elif page == "Architecture":
    st.markdown(
        """
        <div class="page-eyebrow">System design</div>
        <div class="hero-title" style="font-size:2.7rem;">Architecture</div>
        <div class="hero-copy">
            Recall separates retrieval, evidence admission, local generation,
            and post-generation verification into explicit stages.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pipeline-wrap">
            <div class="pipeline-step"><b>Query</b>Natural language input</div>
            <div class="pipeline-step"><b>Hybrid Retrieval</b>Lexical + dense</div>
            <div class="pipeline-step"><b>Evidence Gates</b>Structure + topic</div>
            <div class="pipeline-step"><b>NLI Admission</b>Semantic support</div>
            <div class="pipeline-step"><b>Generation</b>Foundry Local</div>
            <div class="pipeline-step"><b>Citation Repair</b>Safe source mapping</div>
            <div class="pipeline-step"><b>Validation</b>Claim + citation check</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Retrieval & evidence")
        st.markdown(
            """
            **Hybrid retrieval** combines exact lexical matching with semantic vector similarity.

            **Structural gates** reject evidence from the wrong content shape or category.

            **Topical gates** remove semantically nearby but off-topic evidence.

            **Provenance handling** preserves the role, organization, section,
            and parent context attached to each evidence unit.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Generation & verification")
        st.markdown(
            """
            **Local NLI** checks whether candidate evidence semantically supports the requested topic.

            **Foundry Local** generates an answer from validated evidence only.

            **Citation repair** safely restores omitted citations when support is unambiguous and near-extractive.

            **Final validation** checks cited claims before the answer is accepted.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="surface">', unsafe_allow_html=True)
    surface_title("Technology stack")
    st.table(
        [
            {"Layer": "Interface", "Technology": "Streamlit"},
            {"Layer": "Authentication", "Technology": "Firebase Authentication"},
            {"Layer": "Storage", "Technology": "SQLite + FTS"},
            {"Layer": "Embeddings", "Technology": "Qwen3 Embedding 0.6B"},
            {"Layer": "Generation", "Technology": "Qwen 3.5 2B · Foundry Local"},
            {"Layer": "Evidence verification", "Technology": "Local NLI model"},
            {"Layer": "Runtime", "Technology": "Python"},
        ]
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Profile
# =========================================================

elif page == "Profile":
    st.markdown(
        """
        <div class="page-eyebrow">Account</div>
        <div class="hero-title" style="font-size:2.7rem;">Profile</div>
        <div class="hero-copy">Manage your local workspace identity and review session activity.</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)

        avatar_col, identity_col = st.columns([0.35, 1])
        with avatar_col:
            st.markdown(f'<div class="profile-avatar">{initials(st.session_state.profile_name)}</div>', unsafe_allow_html=True)
        with identity_col:
            st.markdown(f"### {st.session_state.profile_name or 'Recall User'}")
            st.caption(st.session_state.profile_email or "Local session")
            if st.session_state.auth_provider:
                provider_label = "Google" if st.session_state.auth_provider == "google" else "E-posta/Şifre"
                st.caption(f"Signed in via {provider_label} (Firebase)")

        st.divider()

        name_value = st.text_input("Display name", value=st.session_state.profile_name, key="profile_name_editor")

        st.text_input("Email", value=st.session_state.profile_email, disabled=True)
        st.caption("Email is managed by Firebase and can't be edited here.")

        roles = ["Local User", "Student", "Developer", "Researcher", "Knowledge Worker"]
        current_role = st.session_state.profile_role if st.session_state.profile_role in roles else "Local User"
        role_value = st.selectbox("Role", roles, index=roles.index(current_role))

        workspace_value = st.text_input("Workspace", value=st.session_state.profile_workspace)

        if st.button("Save profile", type="primary", use_container_width=True):
            st.session_state.profile_name = name_value.strip() or "Recall User"
            st.session_state.profile_role = role_value
            st.session_state.profile_workspace = workspace_value.strip() or "Personal Workspace"
            st.success("Profile updated for this local session.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Workspace activity")

        a1, a2 = st.columns(2)
        with a1:
            st.metric("Queries this session", len(st.session_state.recall_history))
        with a2:
            st.metric("Average latency", f"{avg_latency:.1f}s" if avg_latency else "—")

        st.markdown("#### Privacy behavior")
        st.checkbox("Keep document processing local", value=True, disabled=True)
        st.checkbox("Show source citations", value=True, disabled=True)
        st.checkbox("Use safe fallback on validation failure", value=True, disabled=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Settings
# =========================================================

elif page == "Settings":
    st.markdown(
        """
        <div class="page-eyebrow">Runtime</div>
        <div class="hero-title" style="font-size:2.7rem;">Settings</div>
        <div class="hero-copy">Inspect local runtime information and manage the current session.</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Py", "Python", platform.python_version(), platform.system())
    with c2:
        metric_card("AI", "Chat model", "Qwen 3.5 2B", "Foundry Local")
    with c3:
        metric_card("◈", "Embedding model", "Qwen3 0.6B", "Local semantic retrieval")
    with c4:
        metric_card("◉", "Session queries", str(len(st.session_state.recall_history)), "Current workspace")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Appearance")
        st.caption(
            "Recall's colors come from .streamlit/config.toml. Open the ⋮ "
            "menu (top-right) → Settings → Theme and choose \"Use system "
            "setting\" once — after that, Recall automatically switches "
            "between light and dark to match your OS/browser."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Session history")
        history = st.session_state.recall_history
        if not history:
            st.caption("No query history in this session.")
        else:
            for item in history:
                result = item.get("result") or {}
                with st.expander(item.get("query") or "Untitled query", expanded=False):
                    st.caption(item.get("created_at") or "")
                    st.write(result.get("answer") or "")
                    st.caption(f"Latency: {float(result.get('elapsed_seconds') or 0.0):.2f}s")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        surface_title("Maintenance")

        export_payload = {
            "app": APP_NAME,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "index_stats": safe_stats(),
            "session_history": st.session_state.recall_history,
        }

        st.download_button(
            "Export session report",
            data=json.dumps(export_payload, indent=2, ensure_ascii=False),
            file_name="recall_session_report.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.button("Clear session history", use_container_width=True):
            st.session_state.recall_history = []
            st.session_state.recall_result = None
            st.session_state.evaluation_result = None
            st.success("Session history cleared.")

        if st.button("Reset cached engine", use_container_width=True):
            try:
                if hasattr(engine, "unload_chat_model"):
                    engine.unload_chat_model()
            except Exception:
                pass
            st.cache_resource.clear()
            st.success("Engine cache cleared. Local models will reload on the next query.")

        st.markdown("</div>", unsafe_allow_html=True)