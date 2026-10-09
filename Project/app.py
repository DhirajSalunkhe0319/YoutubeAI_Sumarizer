"""
YouTube Video Summarizer Web Application
Theme: Modern Dark Engineering Theme (#0e1117) with YouTube Red (#FF4B4B)
Powered by Streamlit and google/flan-t5-base running 100% locally.
"""

from __future__ import annotations

import html
import os
import sys
import time
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.youtube import YouTubeURLError, extract_video_id, fetch_video_metadata
from src.transcript import (
    TranscriptBlockedError,
    TranscriptError,
    TranscriptNotAvailableError,
    VideoNotAvailableError,
    fetch_transcript,
)
from src.utils import (
    chunk_snippets_with_timestamps,
    chunk_transcript,
    count_words,
    estimate_reading_time,
    format_final_notes,
    format_timestamp,
    format_transcript_paragraphs,
    parse_executive_notes,
)
from src.summarizer import (
    DEFAULT_MODEL_NAME,
    get_device,
    get_device_name,
    load_model,
    summarize_chunk,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube AI Summarizer • Local Research Notes",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Global Dark Theme CSS (#0e1117)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #060b13;
        --bg-panel: #0c1420;
        --bg-card: rgba(15, 22, 31, 0.94);
        --bg-card-soft: rgba(19, 28, 39, 0.96);
        --bg-nav: rgba(9, 12, 18, 0.98);
        --border-color: rgba(148, 163, 184, 0.15);
        --border-soft: rgba(255, 255, 255, 0.06);
        --accent-red: #ff4b4b;
        --accent-red-strong: #ff2d2d;
        --accent-glow: rgba(255, 75, 75, 0.24);
        --text-white: #f8fafc;
        --text-muted: #a4b2c8;
        --text-dim: #7d8fa9;
        --success: #21d18b;
        --shadow: rgba(0, 0, 0, 0.42);
    }

    html, body, [class*="css"], .stApp {
        background: radial-gradient(circle at 20% 0%, rgba(45, 70, 110, 0.35), transparent 28%),
                    radial-gradient(circle at 92% 12%, rgba(255, 75, 75, 0.22), transparent 18%),
                    #060b13 !important;
        color: var(--text-white) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 2.5rem !important;
        max-width: 1380px !important;
    }

    section[data-testid="stSidebar"] {
        background: rgba(7, 11, 18, 0.98) !important;
        border-right: 1px solid var(--border-color) !important;
        box-shadow: inset -1px 0 0 rgba(255,255,255,0.02);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.9rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.2rem 0.2rem 1rem;
        margin-bottom: 0.8rem;
    }

    .sidebar-logo {
        width: 28px;
        height: 28px;
        border-radius: 9px;
        background: linear-gradient(135deg, #ff4b4b, #ff2d2d);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.8rem;
        box-shadow: 0 8px 16px rgba(255, 75, 75, 0.38);
    }

    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #fff;
        letter-spacing: -0.02em;
    }

    .sidebar-version {
        display: inline-block;
        margin-left: 0.3rem;
        font-size: 0.7rem;
        color: #c8d0dc;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 0.15rem 0.4rem;
        vertical-align: middle;
    }

    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin: 0.9rem 0 1.4rem;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.7rem 0.8rem;
        border-radius: 10px;
        color: #dfe7f3;
        font-weight: 600;
        opacity: 0.8;
        border: 1px solid transparent;
    }

    .nav-item.active {
        background: rgba(255, 75, 75, 0.08);
        border-color: rgba(255, 75, 75, 0.12);
        color: #fff;
        opacity: 1;
    }

    .nav-item .icon {
        width: 20px;
        text-align: center;
        color: #dfe7f3;
        opacity: 0.9;
    }

    .sidebar-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.5rem;
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: rgba(25, 159, 102, 0.08);
        border: 1px solid rgba(33, 209, 139, 0.28);
        color: #72f0b4;
        font-size: 0.74rem;
        font-weight: 700;
    }

    .sidebar-pill .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #21d18b;
        box-shadow: 0 0 12px rgba(33, 209, 139, 0.6);
    }

    .sidebar-block {
        margin-top: 1.1rem;
    }

    .sidebar-label {
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.68rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .sidebar-box {
        background: rgba(17, 22, 31, 0.94);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.68rem 0.8rem;
        color: #dfe7f3;
        font-size: 0.79rem;
        line-height: 1.55;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }

    .metric-row:last-child {
        border-bottom: none;
    }

    .metric-key {
        color: var(--text-dim);
    }

    .metric-val {
        color: white;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
    }

    .sidebar-footer {
        position: sticky;
        bottom: 0;
        margin-top: 1.8rem;
        padding-top: 0.7rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        color: var(--text-dim);
        font-size: 0.72rem;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.8rem;
        padding: 1rem 0.2rem 0.6rem;
        margin-bottom: 0.5rem;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        color: #dfe7f3;
        background: rgba(15, 23, 33, 0.88);
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 0.75rem;
        font-weight: 700;
    }

    .status-chip .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #21d18b;
        box-shadow: 0 0 12px rgba(33, 209, 139, 0.7);
    }

    .deploy-btn {
        background: rgba(15, 23, 33, 0.88) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #fff !important;
        border-radius: 10px !important;
        padding: 0.5rem 0.9rem !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
    }

    .hero-shell {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 1rem 0 1.5rem;
        text-align: center;
    }

    .hero-logo {
        width: 56px;
        height: 56px;
        border-radius: 18px;
        background: linear-gradient(135deg, #ff4b4b, #ff2d2d);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: white;
        box-shadow: 0 12px 26px rgba(255, 75, 75, 0.34);
        flex-shrink: 0;
    }

    .hero-title {
        margin: 0;
        font-size: clamp(2.3rem, 4vw, 4rem);
        font-weight: 900;
        line-height: 0.98;
        letter-spacing: -0.06em;
        color: #ffffff;
    }

    .hero-title .gradient {
        color: #fff;
    }

    .hero-subtitle {
        margin: 0.7rem 0 0.1rem;
        font-size: clamp(1.5rem, 2.2vw, 2.3rem);
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #ecf1f9;
    }

    .hero-copy {
        position: relative;
        z-index: 1;
    }

    .hero-text {
        margin: 0 auto;
        max-width: 760px;
        color: rgba(214, 224, 236, 0.86);
        font-size: 1.06rem;
        line-height: 1.6;
    }

    .badge-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.2rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        background: rgba(16, 23, 35, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.15);
        color: #dfe7f3;
        font-size: 0.74rem;
        font-weight: 700;
    }

    .hero-badge.accent {
        background: rgba(255, 75, 75, 0.08);
        border-color: rgba(255, 75, 75, 0.2);
        color: #ff8a8a;
    }

    .hero-bg-shape {
        position: absolute;
        right: 7%;
        top: 18%;
        width: 150px;
        height: 150px;
        border-radius: 32px;
        transform: rotate(14deg);
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.08), rgba(135,206,250,0.02));
        border: 1px solid rgba(255,255,255,0.05);
        filter: blur(8px);
        z-index: 0;
    }

    .url-card-box {
        background: rgba(14, 22, 32, 0.92);
        border: 1px solid var(--border-color);
        border-radius: 18px;
        padding: 1.2rem 1.3rem 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }

    .url-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #edf4ff;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }

    .url-card-header .icon {
        color: var(--accent-red);
    }

    div[data-testid="stTextInput"] {
        margin-bottom: 0 !important;
    }

    div[data-testid="stTextInput"] > div {
        background: rgba(12, 18, 27, 0.92) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        box-shadow: none !important;
    }

    div[data-testid="stTextInput"] > div:focus-within {
        border-color: rgba(255, 75, 75, 0.6) !important;
        box-shadow: 0 0 0 1px rgba(255, 75, 75, 0.24), 0 0 0 4px rgba(255, 75, 75, 0.08) !important;
    }

    div[data-testid="stTextInput"] input {
        background: transparent !important;
        border: none !important;
        color: #f8fafc !important;
        caret-color: #ff8a8a !important;
        font-size: 1rem !important;
        padding: 0.95rem 1rem !important;
        min-height: 54px !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(148, 163, 184, 0.82) !important;
    }

    div[data-testid="stTextInput"] input::selection {
        background: rgba(255, 75, 75, 0.4) !important;
        color: #ffffff !important;
    }

    div[data-testid="stTextInput"] input:focus {
        outline: none !important;
    }

    div.stButton > button,
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff2d2d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.95rem 1.2rem !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        box-shadow: 0 8px 18px rgba(255, 75, 75, 0.28) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(255, 75, 75, 0.33) !important;
    }

    .feature-card {
        background: rgba(15, 22, 31, 0.92);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        min-height: 120px;
        padding: 1rem 1rem 0.9rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.12);
    }

    .feature-card-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: white;
        margin-bottom: 0.35rem;
        font-size: 1rem;
        font-weight: 800;
    }

    .feature-card-desc {
        color: var(--text-muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .how-it-works-box {
        background: rgba(16, 23, 34, 0.9);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 0.95rem 0.9rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin: 1.3rem 0 1.7rem;
    }

    .workflow-step {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        color: #dfe7f3;
        font-size: 0.74rem;
        font-weight: 700;
    }

    .workflow-num {
        min-width: 2.1rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: rgba(255, 75, 75, 0.12);
        border: 1px solid rgba(255, 75, 75, 0.22);
        color: #ff8a8a;
        padding: 0.25rem 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
    }

    .workflow-arrow {
        color: rgba(148, 163, 184, 0.9);
        font-weight: 700;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff4b4b, #ff8b61) !important;
    }

    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }

    .stTabs [role="tab"] {
        color: var(--text-muted) !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 0.55rem 1rem !important;
        font-weight: 700 !important;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: rgba(255, 75, 75, 0.08) !important;
        color: white !important;
        border-color: rgba(255, 75, 75, 0.25) !important;
    }

    .dashboard-card,
    .meta-hero-card,
    .exec-note-card,
    .transcript-dark-card {
        box-shadow: 0 8px 20px rgba(0,0,0,0.14);
    }

    .meta-hero-card {
        background: rgba(15, 22, 31, 0.94);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1rem 1rem 1.1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .meta-thumb-container {
        position: relative;
        width: 215px;
        min-width: 215px;
        height: 120px;
        border-radius: 12px;
        overflow: hidden;
        background: #000;
    }

    .meta-thumb-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .meta-duration-pill {
        position: absolute;
        right: 8px;
        bottom: 8px;
        background: rgba(0,0,0,0.8);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 6px;
        padding: 0.18rem 0.45rem;
        font-size: 0.68rem;
        font-weight: 800;
        color: white;
        font-family: 'JetBrains Mono', monospace;
    }

    .meta-video-title {
        font-size: 1.2rem;
        line-height: 1.35;
        font-weight: 800;
        color: white;
        margin-bottom: 0.7rem;
    }

    .meta-top-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255, 75, 75, 0.08);
        border: 1px solid rgba(255, 75, 75, 0.18);
        color: #ff8a8a;
        padding: 0.24rem 0.52rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .meta-stat-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem 1.4rem;
        color: var(--text-muted);
        font-size: 0.79rem;
    }

    .meta-stat-item {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }

    .dashboard-card {
        background: rgba(15, 22, 31, 0.94);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1rem 1.05rem 1.1rem;
        margin-top: 0.75rem;
    }

    .dashboard-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 0.7rem;
        margin-bottom: 0.8rem;
    }

    .dashboard-header-title {
        color: white;
        font-size: 1.02rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    .dashboard-header-meta {
        color: var(--text-dim);
        font-size: 0.72rem;
    }

    .exec-note-card {
        background: rgba(17, 23, 31, 0.85);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.75rem;
        display: flex;
        gap: 0.85rem;
        align-items: flex-start;
    }

    .exec-num-badge {
        width: 29px;
        height: 29px;
        min-width: 29px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 75, 75, 0.12);
        border: 1px solid rgba(255, 75, 75, 0.22);
        color: #ff8a8a;
        font-weight: 800;
        font-size: 0.72rem;
    }

    .exec-note-body {
        color: #e8eef6;
        line-height: 1.7;
        font-size: 0.94rem;
    }

    .exec-note-body strong {
        color: white;
        font-weight: 800;
    }

    .transcript-dark-card {
        background: rgba(7, 10, 15, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        overflow: hidden;
    }

    .transcript-top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        background: rgba(17, 22, 31, 0.9);
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        color: var(--text-muted);
        font-size: 0.72rem;
    }

    .transcript-scroll-area {
        padding: 1.1rem 1.1rem 1.3rem;
        max-height: 480px;
        overflow-y: auto;
        color: #edf5ff;
        line-height: 1.8;
        font-size: 0.92rem;
        white-space: normal;
        word-break: break-word;
    }

    .transcript-paragraph {
        margin: 0 0 1rem;
        color: #edf5ff;
    }

    .transcript-scroll-area::-webkit-scrollbar {
        width: 8px;
    }

    .transcript-scroll-area::-webkit-scrollbar-track {
        background: rgba(7,10,15,0.8);
    }

    .transcript-scroll-area::-webkit-scrollbar-thumb {
        background: rgba(148,163,184,0.35);
        border-radius: 10px;
    }

    .stDownloadButton > button,
    div[data-testid="stDownloadButton"] > button {
        background: rgba(21, 27, 36, 0.92) !important;
        color: white !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 10px !important;
        padding: 0.7rem 0.9rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        box-shadow: none !important;
    }

    .stDownloadButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: rgba(26, 35, 46, 0.96) !important;
        border-color: rgba(255,75,75,0.35) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ai_pipeline(model_name: str = DEFAULT_MODEL_NAME):
    """
    Loads and caches the local FLAN-T5 model and tokenizer in memory.
    Cached across user runs for instant inference.
    """
    device = get_device()
    tokenizer, model, dev = load_model(model_name=model_name, device=device)
    return tokenizer, model, dev


# ---------------------------------------------------------
# Sidebar: Structured Overview, Stack & Metrics
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">▶</div>
            <div class="sidebar-brand-title">YouTube AI Summarizer <span class="sidebar-version">v2.4.0</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-nav">
            <div class="nav-item active"><span class="icon">⌂</span> Home</div>
            <div class="nav-item"><span class="icon">i</span> About</div>
            <div class="nav-item"><span class="icon">⚙</span> Settings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-pill"><span class="dot"></span> Local AI • No API Key</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-block"><div class="sidebar-label">About</div></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-box">
            YouTube AI Summarizer turns long-form YouTube content into concise research notes using a locally running open-source AI model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-block"><div class="sidebar-label">Technology Stack</div></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-box">
            <div class="metric-row"><span class="metric-key">Python</span><span class="metric-val">3.10+</span></div>
            <div class="metric-row"><span class="metric-key">Streamlit</span><span class="metric-val">UI</span></div>
            <div class="metric-row"><span class="metric-key">FLAN-T5</span><span class="metric-val">base</span></div>
            <div class="metric-row"><span class="metric-key">PyTorch</span><span class="metric-val">GPU/CPU</span></div>
            <div class="metric-row"><span class="metric-key">Hugging Face</span><span class="metric-val">local</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    device_raw = get_device_name()
    is_cuda_active = "CUDA" in device_raw
    active_device_tag = "CUDA" if is_cuda_active else "CPU"

    st.markdown('<div class="sidebar-block"><div class="sidebar-label">Configuration</div></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-box">
            <div class="metric-row"><span class="metric-key">Chunk Size</span><span class="metric-val">300 words</span></div>
            <div class="metric-row"><span class="metric-key">Model</span><span class="metric-val">google/flan-t5-base</span></div>
            <div class="metric-row"><span class="metric-key">Device</span><span class="metric-val">{active_device_tag}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-footer">Built with <span style="color: #f2f6ff; font-weight: 700;">Streamlit</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Top Hero Section & Landing Page
# ---------------------------------------------------------
device_hero_tag = "GPU / CPU" if is_cuda_active else "GPU / CPU"

st.markdown(
    """
    <div class="topbar">
        <div class="status-chip"><span class="dot"></span> Device: CUDA</div>
        <div class="status-chip"><span class="dot"></span> 100% Local / Zero API Cost</div>
        <button class="deploy-btn">Deploy App</button>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-bg-shape"></div>
        <div class="hero-logo">▶</div>
        <div class="hero-copy">
            <h1 class="hero-title">YouTube <span class="gradient">AI</span> Summarizer</h1>
            <div class="hero-subtitle">Turn YouTube Videos Into Research Notes</div>
            <p class="hero-text">Understand hours of content in minutes with locally running open-source AI.</p>
            <div class="badge-row">
                <span class="hero-badge accent">✦ Powered by FLAN-T5</span>
                <span class="hero-badge">🔒 100% Local</span>
                <span class="hero-badge">⚡ {device_hero_tag}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. Main URL Input Card
# ---------------------------------------------------------
if "input_url" not in st.session_state:
    st.session_state["input_url"] = ""

st.markdown(
    """
    <div class="url-card-box">
        <div class="url-card-header">
            <span class="url-card-header-icon">🔗</span> Paste your YouTube video URL
        </div>
    """,
    unsafe_allow_html=True,
)

col_input, col_action = st.columns([4.8, 1.6])

with col_input:
    url_input_value = st.text_input(
        "YouTube Video URL",
        value=st.session_state["input_url"],
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        label_visibility="collapsed",
        key="main_url_input",
    )

with col_action:
    summarize_button_clicked = st.button("✨ Summarize Video", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. Feature Cards & How It Works (Before Run / Idle State)
# ---------------------------------------------------------
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-card-title"><span>⚡</span> Local AI</div>
            <div class="feature-card-desc">Runs using an open-source Hugging Face model directly on your device.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with f_col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-card-title"><span>🔒</span> No API Key</div>
            <div class="feature-card-desc">Zero OpenAI, Gemini, or Claude API keys required. 100% private.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with f_col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-card-title"><span>📝</span> Research Notes</div>
            <div class="feature-card-desc">Converts long videos into concise, structured notes and key takeaways.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with f_col4:
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-card-title"><span>🚀</span> GPU Accelerated</div>
            <div class="feature-card-desc">Uses CUDA when available ({active_device_tag}), automatically falls back to CPU.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# How It Works Workflow Bar
st.markdown(
    """
    <div class="how-it-works-box">
        <div class="workflow-step">
            <span class="workflow-num">01</span>
            <span>Transcript Extraction</span>
        </div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">
            <span class="workflow-num">02</span>
            <span>300-Word Chunks</span>
        </div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">
            <span class="workflow-num">03</span>
            <span>FLAN-T5 Beam Search</span>
        </div>
        <div class="workflow-arrow">➔</div>
        <div class="workflow-step">
            <span class="workflow-num">04</span>
            <span>Research Notes Dashboard</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 4 & 5. Execution Pipeline & Results Dashboard
# ---------------------------------------------------------
active_url = url_input_value.strip()

if summarize_button_clicked or ("last_result" in st.session_state and st.session_state.get("last_url") == active_url and active_url):
    if not active_url:
        st.warning("⚠️ Please paste a valid YouTube video URL first.")
        st.stop()

    # Step 1: Validate YouTube URL
    try:
        video_id = extract_video_id(active_url)
    except YouTubeURLError as err:
        st.error(f"❌ {err}")
        st.stop()

    # If new run triggered by button
    if summarize_button_clicked or "last_result" not in st.session_state or st.session_state.get("last_url") != active_url:
        progress_slot = st.empty()
        status_slot = st.empty()

        progress_bar = progress_slot.progress(0.10)
        status_slot.markdown(
            f"<div style='color: #94a3b8; font-size: 0.92rem; padding: 0.4rem 0;'>🔎 <b>Fetching transcript from YouTube ({video_id})...</b></div>",
            unsafe_allow_html=True,
        )

        start_time = time.time()

        try:
            # Step 2: Fetch video metadata (title, author, thumbnail)
            meta_info = fetch_video_metadata(video_id)

            # Step 3: Fetch transcript with timestamps
            progress_bar.progress(0.25)
            status_slot.markdown(
                "<div style='color: #94a3b8; font-size: 0.92rem; padding: 0.4rem 0;'>🧹 <b>Cleaning transcript and preparing 300-word chunks...</b></div>",
                unsafe_allow_html=True,
            )

            raw_text, transcript_meta = fetch_transcript(video_id)
            word_count = count_words(raw_text)

            if word_count == 0:
                st.error("⚠️ The extracted transcript appears to be empty.")
                st.stop()

            # Step 4: Chunking with timestamps (300 words default)
            snippets = transcript_meta.get("snippets", [])
            chunks_with_time = chunk_snippets_with_timestamps(snippets, chunk_size=300)
            if not chunks_with_time:
                plain_chunks = chunk_transcript(raw_text, chunk_size=300)
                chunks_with_time = [{"text": c, "start": 0.0, "start_formatted": "00:00", "word_count": count_words(c)} for c in plain_chunks]

            total_chunks = len(chunks_with_time)

            # Step 5: Load local model
            progress_bar.progress(0.40)
            status_slot.markdown(
                f"<div style='color: #94a3b8; font-size: 0.92rem; padding: 0.4rem 0;'>✂️ <b>Splitting transcript into {total_chunks} chunk(s) (approx. 300 words each)...</b></div>",
                unsafe_allow_html=True,
            )

            tokenizer, model, device = get_ai_pipeline()

            # Step 6: Generate summaries sequentially
            chunk_summaries = []
            for i, chunk_data in enumerate(chunks_with_time):
                pct = 0.45 + ((i + 1) / total_chunks) * 0.45
                progress_bar.progress(min(pct, 0.95))
                status_slot.markdown(
                    f"<div style='color: #94a3b8; font-size: 0.92rem; padding: 0.4rem 0;'>🤖 <b>Summarizing chunk {i + 1} of {total_chunks}...</b></div>",
                    unsafe_allow_html=True,
                )

                summary = summarize_chunk(
                    chunk=chunk_data["text"],
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    num_beams=4,
                    max_new_tokens=256,
                    min_new_tokens=45,
                )
                chunk_summaries.append(summary)

            status_slot.markdown(
                "<div style='color: #94a3b8; font-size: 0.92rem; padding: 0.4rem 0;'>📝 <b>Preparing research notes and dashboard...</b></div>",
                unsafe_allow_html=True,
            )
            progress_bar.progress(0.98)

            elapsed_seconds = round(time.time() - start_time, 1)

            # Step 7: Parse structured notes
            exec_notes = parse_executive_notes(chunk_summaries)
            final_markdown = format_final_notes(chunk_summaries)

            # Store in session state for persistence
            st.session_state["last_url"] = active_url
            st.session_state["last_result"] = {
                "video_id": video_id,
                "meta_info": meta_info,
                "transcript_meta": transcript_meta,
                "raw_text": raw_text,
                "word_count": word_count,
                "chunks_with_time": chunks_with_time,
                "chunk_summaries": chunk_summaries,
                "exec_notes": exec_notes,
                "final_markdown": final_markdown,
                "elapsed": elapsed_seconds,
            }

            progress_bar.progress(1.0)
            status_slot.empty()
            progress_slot.empty()

        except TranscriptNotAvailableError as err:
            progress_slot.empty()
            status_slot.empty()
            st.error(f"⚠️ **Transcript Unavailable:** {err}")
            st.info("💡 **Tip:** Some creators disable subtitles. Please test with a video that has closed captions (CC) enabled.")
            st.stop()
        except VideoNotAvailableError as err:
            progress_slot.empty()
            status_slot.empty()
            st.error(f"⚠️ **Video Unavailable:** {err}")
            st.stop()
        except TranscriptBlockedError as err:
            progress_slot.empty()
            status_slot.empty()
            st.error(f"⚠️ **Request Rate-Limited:** {err}")
            st.stop()
        except Exception as err:
            progress_slot.empty()
            status_slot.empty()
            st.error(f"⚠️ **Error:** {err}")
            st.stop()

    # Retrieve cached result
    res = st.session_state["last_result"]
    video_id = res["video_id"]
    meta_info = res["meta_info"]
    raw_text = res["raw_text"]
    word_count = res["word_count"]
    chunks_with_time = res["chunks_with_time"]
    chunk_summaries = res["chunk_summaries"]
    exec_notes = res["exec_notes"]
    final_markdown = res["final_markdown"]
    elapsed = res["elapsed"]

    duration_sec = res["transcript_meta"].get("duration_seconds", 0.0)
    duration_fmt = format_timestamp(duration_sec) if duration_sec > 0 else "12:30"
    reading_time_est = estimate_reading_time(word_count)

    thumbnail_url = meta_info.get("thumbnail", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
    video_title = meta_info.get("title", f"YouTube Video ({video_id})")
    author_name = meta_info.get("author", "YouTube Creator")
    lang_name = res["transcript_meta"].get("language", "English")

    # ---------------------------------------------------------
    # Top Video Metadata Card
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <div class="meta-hero-card">
            <div class="meta-thumb-container">
                <img src="{thumbnail_url}" class="meta-thumb-img" alt="Thumbnail" />
                <div class="meta-duration-pill">{duration_fmt}</div>
            </div>
            <div class="meta-details">
                <div class="meta-top-badge">
                    <span>⚡ Summarized locally with FLAN-T5 ({elapsed}s)</span>
                </div>
                <div class="meta-video-title">{video_title}</div>
                <div class="meta-stat-row">
                    <div class="meta-stat-item">
                        <span>📺</span> <span>{author_name}</span>
                    </div>
                    <div class="meta-stat-item">
                        <span>⏱️</span> <span>{duration_fmt} ({reading_time_est})</span>
                    </div>
                    <div class="meta-stat-item">
                        <span>📄</span> <span>{word_count:,} words</span>
                    </div>
                    <div class="meta-stat-item">
                        <span>✂️</span> <span>{len(chunks_with_time)} chunks</span>
                    </div>
                    <div class="meta-stat-item">
                        <span>⚡</span> <span>Device: {active_device_tag}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Tabs & High-Contrast Export Action Buttons
    # ---------------------------------------------------------
    tab_hdr_col, export_actions_col = st.columns([3.2, 2.8])

    with export_actions_col:
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            st.download_button(
                "📋 Copy .TXT",
                data=final_markdown,
                file_name=f"summary_{video_id}.txt",
                mime="text/plain",
                use_container_width=True,
                key="btn_copy_txt",
            )
        with btn_c2:
            st.download_button(
                "📥 Export .MD",
                data=f"# {video_title}\n\nURL: https://www.youtube.com/watch?v={video_id}\n\n{final_markdown}\n\n## Transcript\n\n{raw_text}",
                file_name=f"summary_{video_id}.md",
                mime="text/markdown",
                use_container_width=True,
                key="btn_export_md",
            )
        with btn_c3:
            st.download_button(
                "📝 Export Notion",
                data=f"# {video_title}\n\n> Summary extracted via local FLAN-T5\n\n{final_markdown}",
                file_name=f"notion_{video_id}.md",
                mime="text/markdown",
                use_container_width=True,
                key="btn_export_notion",
            )

    tab_exec, tab_chapters, tab_transcript = st.tabs([
        "📌 Executive Summary",
        "⏱️ Chapters & Timestamps",
        "📝 Full Cleaned Transcript",
    ])

    # ---------------------------------------------------------
    # Tab 1: 📌 Executive Summary (Rich card layout)
    # ---------------------------------------------------------
    with tab_exec:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-header">
                    <div class="dashboard-header-title">
                        <span style="color: #ff4b4b;">📌</span> Executive AI Notes Summary
                    </div>
                    <div class="dashboard-header-meta">
                        Synthesized using FLAN-T5 (300-word chunks)
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        for note in exec_notes:
            st.markdown(
                f"""
                <div class="exec-note-card">
                    <div class="exec-num-badge">{note['number']}</div>
                    <div class="exec-note-body">
                        <strong>{note['title']}:</strong> {note['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Tab 2: ⏱️ Chapters & Timestamps
    # ---------------------------------------------------------
    with tab_chapters:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-header">
                    <div class="dashboard-header-title">
                        <span style="color: #ff4b4b;">⏱️</span> Timeline Chapters & Breakdown
                    </div>
                    <div class="dashboard-header-meta">
                        {len(chunks_with_time)} timestamped chapters
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        for idx, (chunk_data, sum_txt) in enumerate(zip(chunks_with_time, chunk_summaries)):
            t_stamp = chunk_data.get("start_formatted", "00:00")
            st.markdown(
                f"""
                <div class="chap-row">
                    <div class="chap-time-pill">⏱️ {t_stamp}</div>
                    <div class="chap-desc">
                        {sum_txt}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Tab 3: 📝 Full Cleaned Transcript (Redesigned Dark Box)
    # ---------------------------------------------------------
    with tab_transcript:
        transcript_blocks = format_transcript_paragraphs(raw_text)
        transcript_html = "\n".join(
            f"<p class='transcript-paragraph'>{html.escape(block)}</p>"
            for block in transcript_blocks.split("\n\n")
            if block.strip()
        )

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-header">
                    <div class="dashboard-header-title">
                        <span style="color: #ff4b4b;">📝</span> Full Cleaned Transcript
                    </div>
                    <div class="dashboard-header-meta">
                        {word_count:,} words total ({reading_time_est})
                    </div>
                </div>
                <div class="transcript-dark-card">
                    <div class="transcript-top-bar">
                        <span>Original Cleaned Captions</span>
                        <span>Scrollable View</span>
                    </div>
                    <div class="transcript-scroll-area">{transcript_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.download_button(
            "📥 Download Transcript (.TXT)",
            data=raw_text,
            file_name=f"transcript_{video_id}.txt",
            mime="text/plain",
            use_container_width=True,
            key="btn_dl_raw_transcript",
        )
