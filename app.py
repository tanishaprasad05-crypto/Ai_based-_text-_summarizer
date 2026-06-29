"""
AI Based Text Summarizer  ·  v2.0
Real-world AI/NLP Portfolio Project
New: Sentence slider · Copy buttons · Keyword extractor · Readability score
     Session history · Download as .txt · Sentiment indicator · Teal/cyan theme
"""

import streamlit as st
import nltk
import time
import io
import re
import math
import collections
import datetime

import PyPDF2
from transformers import pipeline
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import nltk

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS  ·  Obsidian / Teal-Cyan / White
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;700;800;900&family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

/* ══ TOKENS ══════════════════════════════════════════════════ */
:root {
    --obs          : #080B12;
    --obs2         : #0C1018;
    --obs3         : #111827;
    --surface      : #131B2A;
    --surface2     : #1A2437;
    --surface3     : #1F2D42;
    --teal         : #14B8A6;
    --teal-bright  : #2DD4BF;
    --teal-dim     : #0F9688;
    --teal-glow    : rgba(20,184,166,0.18);
    --teal-glow2   : rgba(20,184,166,0.08);
    --cyan         : #06B6D4;
    --cyan-bright  : #22D3EE;
    --amber        : #F59E0B;
    --amber-dim    : rgba(245,158,11,0.15);
    --rose         : #F43F5E;
    --rose-dim     : rgba(244,63,94,0.12);
    --emerald      : #10B981;
    --emerald-dim  : rgba(16,185,129,0.12);
    --white        : #F8FAFC;
    --white-80     : rgba(248,250,252,0.80);
    --white-55     : rgba(248,250,252,0.55);
    --white-30     : rgba(248,250,252,0.30);
    --white-12     : rgba(248,250,252,0.12);
    --white-06     : rgba(248,250,252,0.06);
    --border       : rgba(20,184,166,0.18);
    --border-dim   : rgba(248,250,252,0.08);
    --shadow-deep  : 0 8px 40px rgba(0,0,0,0.55);
    --shadow-teal  : 0 4px 28px rgba(20,184,166,0.18);
    --shadow-teal2 : 0 8px 40px rgba(20,184,166,0.28);
}

/* ══ BASE ════════════════════════════════════════════════════ */
*, *::before, *::after          { box-sizing: border-box; }
html, body, .stApp              { background: var(--obs2) !important; color: var(--white); font-family: 'Manrope', sans-serif; }
.block-container                { padding: 2rem 2.8rem 5rem !important; max-width: 1220px !important; }
#MainMenu, footer, header        { visibility: hidden !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem !important; }

/* ══ SIDEBAR ═════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background  : linear-gradient(175deg, var(--surface) 0%, var(--obs) 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow  : 4px 0 48px rgba(0,0,0,0.6) !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    font-family   : 'Manrope', sans-serif !important;
    font-size     : 9px !important;
    font-weight   : 800 !important;
    letter-spacing: 0.26em !important;
    text-transform: uppercase !important;
    color         : var(--white-30) !important;
    padding       : 0 0 8px 2px !important;
    display       : block !important;
}
.stRadio > div { gap: 3px !important; }
.stRadio > div > label {
    background    : var(--white-06) !important;
    border        : 1px solid var(--border-dim) !important;
    border-radius : 10px !important;
    padding       : 11px 15px !important;
    color         : var(--white-55) !important;
    font-family   : 'Manrope', sans-serif !important;
    font-size     : 13px !important;
    font-weight   : 600 !important;
    transition    : all 0.18s ease !important;
    cursor        : pointer !important;
}
.stRadio > div > label:hover {
    background    : var(--teal-glow) !important;
    border-color  : var(--teal) !important;
    color         : var(--white) !important;
    transform     : translateX(5px) !important;
    box-shadow    : var(--shadow-teal) !important;
}

/* ══ SLIDER ══════════════════════════════════════════════════ */
.stSlider > div > div > div > div {
    background: var(--teal) !important;
}
.stSlider > div > div > div > div > div {
    background   : var(--teal-bright) !important;
    border       : 2px solid var(--teal) !important;
    box-shadow   : 0 0 14px rgba(20,184,166,0.5) !important;
}
.stSlider label {
    font-family   : 'Manrope', sans-serif !important;
    font-size     : 9px !important;
    font-weight   : 800 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color         : var(--white-30) !important;
}

/* ══ SELECTBOX ═══════════════════════════════════════════════ */
.stSelectbox > div > div {
    background    : var(--surface) !important;
    border        : 1px solid var(--border) !important;
    border-radius : 10px !important;
    color         : var(--white) !important;
    font-family   : 'Manrope', sans-serif !important;
}
.stSelectbox label {
    font-family   : 'Manrope', sans-serif !important;
    font-size     : 9px !important;
    font-weight   : 800 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color         : var(--white-30) !important;
}

/* ══ TYPOGRAPHY ══════════════════════════════════════════════ */
h1, h2, h3 { font-family: 'Cabinet Grotesk', 'Manrope', sans-serif !important; }
h1 { font-size: clamp(2rem,3.5vw,3rem) !important; font-weight: 900 !important; color: var(--white) !important; letter-spacing: -0.03em !important; }
h2 { font-size: 1.5rem !important; font-weight: 800 !important; color: var(--white-80) !important; }
h3 { font-size: 1.1rem !important; font-weight: 700 !important; color: var(--white-55) !important; }
p, li { color: var(--white-55); line-height: 1.75; font-size: 14.5px; }

/* ══ BRAND ═══════════════════════════════════════════════════ */
.sb-brand {
    font-family   : 'Cabinet Grotesk', sans-serif;
    font-weight   : 900;
    font-size     : 1.25rem;
    color         : var(--white);
    line-height   : 1.2;
    letter-spacing: -0.02em;
    display       : block;
    margin-bottom : 3px;
}
.sb-brand strong { color: var(--teal-bright); }
.sb-version {
    display       : inline-flex;
    align-items   : center;
    gap           : 5px;
    font-size     : 9px;
    font-weight   : 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color         : var(--white-30);
    margin-bottom : 22px;
    padding-left  : 1px;
    font-family   : 'Manrope', sans-serif;
}
.sb-version span {
    background    : var(--teal-glow);
    color         : var(--teal-bright);
    border        : 1px solid rgba(20,184,166,0.3);
    border-radius : 4px;
    padding       : 1px 6px;
    font-size     : 8px;
    font-weight   : 800;
    letter-spacing: 0.12em;
}

/* ══ PAGE HEADER ═════════════════════════════════════════════ */
.ph-wrap {
    padding-bottom: 28px;
    border-bottom : 1px solid var(--border-dim);
    margin-bottom : 32px;
}
.ph-eyebrow {
    display       : inline-flex;
    align-items   : center;
    gap           : 8px;
    font-size     : 9px;
    font-weight   : 800;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color         : var(--teal);
    margin-bottom : 12px;
    font-family   : 'Manrope', sans-serif;
}
.ph-eyebrow::before {
    content       : '';
    display       : inline-block;
    width         : 18px;
    height        : 2px;
    background    : var(--teal);
    border-radius : 999px;
}
.ph-title {
    font-family   : 'Cabinet Grotesk', sans-serif;
    font-weight   : 900;
    font-size     : clamp(2rem, 3.5vw, 3rem);
    color         : var(--white);
    line-height   : 1.02;
    letter-spacing: -0.03em;
    margin-bottom : 14px;
}
.ph-title .hl { color: var(--teal-bright); }
.ph-bar {
    width         : 44px;
    height        : 2px;
    background    : linear-gradient(90deg, var(--teal), var(--cyan), transparent);
    border-radius : 999px;
    margin-bottom : 16px;
}
.ph-desc {
    font-size     : 15px;
    color         : var(--white-30);
    max-width     : 620px;
    line-height   : 1.68;
    font-family   : 'Manrope', sans-serif;
    font-weight   : 400;
}

/* ══ CARDS ═══════════════════════════════════════════════════ */
.card {
    background    : var(--surface);
    border        : 1px solid var(--border-dim);
    border-radius : 18px;
    padding       : 24px 26px;
    margin        : 10px 0;
    box-shadow    : var(--shadow-deep);
    transition    : border-color 0.22s, box-shadow 0.22s;
}
.card:hover { border-color: rgba(20,184,166,0.35); }

.result-card {
    background    : var(--surface);
    border        : 1px solid var(--border-dim);
    border-radius : 18px;
    padding       : 26px 28px;
    position      : relative;
    overflow      : hidden;
    box-shadow    : var(--shadow-deep);
    transition    : all 0.22s;
}
.result-card:hover { border-color: rgba(20,184,166,0.45); box-shadow: var(--shadow-teal); }
.result-card::before {
    content       : '';
    position      : absolute;
    top: 0; left: 0; right: 0;
    height        : 2px;
    border-radius : 18px 18px 0 0;
}
.result-card.lsa::before  { background: linear-gradient(90deg, var(--white-30), transparent); }
.result-card.bart::before { background: linear-gradient(90deg, var(--teal), var(--cyan), transparent); box-shadow: 0 0 20px rgba(20,184,166,0.4); }

/* glow card */
.glow-card {
    background    : linear-gradient(135deg, var(--surface2), var(--surface));
    border        : 1px solid var(--border);
    border-radius : 18px;
    padding       : 26px 28px;
    box-shadow    : var(--shadow-teal);
    position      : relative;
    overflow      : hidden;
}
.glow-card::after {
    content       : '';
    position      : absolute;
    top: -60px; right: -60px;
    width         : 180px;
    height        : 180px;
    background    : radial-gradient(circle, rgba(20,184,166,0.12), transparent 70%);
    pointer-events: none;
}

/* ══ METRIC CARD ═════════════════════════════════════════════ */
.m-card {
    background    : var(--surface);
    border        : 1px solid var(--border-dim);
    border-radius : 14px;
    padding       : 20px 16px;
    text-align    : center;
    box-shadow    : var(--shadow-deep);
    transition    : all 0.22s;
    position      : relative;
    overflow      : hidden;
}
.m-card::after {
    content       : '';
    position      : absolute;
    bottom: 0; left: 0; right: 0;
    height        : 2px;
    opacity       : 0;
    background    : linear-gradient(90deg, var(--teal), var(--cyan));
    transition    : opacity 0.22s;
}
.m-card:hover { border-color: var(--teal); transform: translateY(-3px); box-shadow: var(--shadow-teal); }
.m-card:hover::after { opacity: 1; }
.m-val {
    font-family   : 'Cabinet Grotesk', sans-serif;
    font-weight   : 900;
    font-size     : 2.4rem;
    color         : var(--white);
    line-height   : 1;
    display       : block;
    margin-bottom : 5px;
    letter-spacing: -0.03em;
}
.m-val.teal    { color: var(--teal-bright); }
.m-val.cyan    { color: var(--cyan-bright); }
.m-val.amber   { color: var(--amber); }
.m-val.emerald { color: var(--emerald); }
.m-lbl {
    font-size     : 9px;
    font-weight   : 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color         : var(--white-30);
    font-family   : 'Manrope', sans-serif;
}

/* ══ PILLS / BADGES ══════════════════════════════════════════ */
.pill {
    display       : inline-flex;
    align-items   : center;
    padding       : 5px 14px;
    border-radius : 999px;
    font-size     : 9px;
    font-weight   : 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom : 16px;
    font-family   : 'Manrope', sans-serif;
}
.pill-white  { background: var(--white-12); color: var(--white); border: 1px solid var(--white-30); }
.pill-teal   { background: var(--teal-glow); color: var(--teal-bright); border: 1px solid rgba(20,184,166,0.35); }
.pill-cyan   { background: rgba(6,182,212,0.10); color: var(--cyan-bright); border: 1px solid rgba(6,182,212,0.30); }
.pill-amber  { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(245,158,11,0.30); }
.pill-rose   { background: var(--rose-dim); color: var(--rose); border: 1px solid rgba(244,63,94,0.30); }
.pill-emerald{ background: var(--emerald-dim); color: var(--emerald); border: 1px solid rgba(16,185,129,0.30); }

.win-badge {
    display       : inline-block;
    background    : var(--teal-glow);
    color         : var(--teal-bright);
    border        : 1px solid rgba(20,184,166,0.35);
    border-radius : 5px;
    padding       : 2px 8px;
    font-size      : 9px;
    font-weight   : 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-left   : 8px;
    vertical-align: middle;
    font-family   : 'Manrope', sans-serif;
}

.tag-chip {
    display       : inline-block;
    background    : var(--white-06);
    color         : var(--white-55);
    border        : 1px solid var(--border-dim);
    border-radius : 999px;
    padding       : 5px 15px;
    font-size     : 12.5px;
    font-weight   : 600;
    font-family   : 'Manrope', sans-serif;
    transition    : all 0.18s;
}
.tag-chip:hover { background: var(--teal-glow); border-color: var(--teal); color: var(--white); }

.stat-pill {
    display       : inline-flex;
    align-items   : center;
    gap           : 5px;
    background    : var(--white-06);
    border        : 1px solid var(--border-dim);
    border-radius : 999px;
    padding       : 4px 13px;
    font-size     : 12px;
    font-weight   : 600;
    color         : var(--white-30);
    font-family   : 'Manrope', sans-serif;
}
.stat-pill em { font-style: normal; color: var(--teal-bright); font-weight: 800; }

/* ══ FEATURE GRID ════════════════════════════════════════════ */
.feat-grid {
    display               : grid;
    grid-template-columns : repeat(3, 1fr);
    gap                   : 14px;
    margin                : 28px 0;
}
.feat-item {
    background    : var(--surface);
    border        : 1px solid var(--border-dim);
    border-radius : 16px;
    padding       : 22px 20px;
    transition    : all 0.22s;
    position      : relative;
    overflow      : hidden;
}
.feat-item::before {
    content       : '';
    position      : absolute;
    top: 0; left: 0; right: 0;
    height        : 2px;
    background    : linear-gradient(90deg, var(--teal), var(--cyan), transparent);
    opacity       : 0;
    transition    : opacity 0.22s;
}
.feat-item:hover {
    border-color  : rgba(20,184,166,0.40);
    box-shadow    : var(--shadow-teal);
    transform     : translateY(-3px);
}
.feat-item:hover::before { opacity: 1; }
.feat-icon  { font-size: 1.7rem; margin-bottom: 12px; }
.feat-title {
    font-family   : 'Cabinet Grotesk', sans-serif;
    font-size     : 1rem;
    font-weight   : 800;
    color         : var(--white);
    margin-bottom : 7px;
    letter-spacing: -0.01em;
}
.feat-new {
    display       : inline-block;
    background    : var(--teal-glow);
    color         : var(--teal-bright);
    border-radius : 4px;
    padding       : 1px 6px;
    font-size      : 8px;
    font-weight   : 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-left   : 6px;
    vertical-align: middle;
}
.feat-desc { font-size: 12.5px; color: var(--white-30); line-height: 1.65; font-family: 'Manrope', sans-serif; }

/* ══ KEYWORD CHIPS ═══════════════════════════════════════════ */
.kw-grid {
    display    : flex;
    flex-wrap  : wrap;
    gap        : 8px;
    margin-top : 12px;
}
.kw-chip {
    display       : inline-flex;
    align-items   : center;
    gap           : 6px;
    background    : var(--surface2);
    border        : 1px solid var(--border);
    border-radius : 8px;
    padding       : 5px 12px;
    font-size     : 12px;
    font-weight   : 600;
    color         : var(--white-80);
    font-family   : 'IBM Plex Mono', monospace;
    transition    : all 0.18s;
}
.kw-chip:hover { background: var(--teal-glow); border-color: var(--teal); color: var(--white); }
.kw-freq {
    background    : var(--teal-glow);
    color         : var(--teal-bright);
    border-radius : 4px;
    padding       : 1px 6px;
    font-size      : 10px;
    font-weight   : 800;
    min-width     : 20px;
    text-align    : center;
}

/* ══ COMPARISON TABLE ════════════════════════════════════════ */
.cmp-table {
    width          : 100%;
    border-collapse: collapse;
    font-size      : 13.5px;
    border-radius  : 14px;
    overflow       : hidden;
    border         : 1px solid var(--border-dim);
    box-shadow     : var(--shadow-deep);
    margin-top     : 14px;
}
.cmp-table th {
    background    : var(--surface2);
    color         : var(--white-55);
    font-family   : 'Manrope', sans-serif;
    font-size     : 9px;
    font-weight   : 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding       : 14px 20px;
    text-align    : left;
    border-bottom : 1px solid var(--border);
}
.cmp-table td {
    padding       : 12px 20px;
    border-bottom : 1px solid rgba(248,250,252,0.04);
    background    : var(--surface);
    color         : var(--white-55);
    font-family   : 'Manrope', sans-serif;
    transition    : background 0.15s;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table tr:hover td { background: var(--surface2); color: var(--white-80); }

/* ══ PROGRESS BARS ═══════════════════════════════════════════ */
.pbar-wrap { background: var(--white-06); border-radius: 999px; height: 6px; overflow: hidden; margin-top: 8px; }
.pbar-fill { height: 100%; border-radius: 999px; transition: width 0.8s cubic-bezier(.4,0,.2,1); }
.pbar-teal  { background: linear-gradient(90deg, var(--teal), var(--cyan-bright)); box-shadow: 0 0 12px rgba(20,184,166,0.35); }
.pbar-white { background: linear-gradient(90deg, var(--white-55), var(--white-12)); }

/* ══ STAT ROW INSIDE RESULT CARDS ═══════════════════════════ */
.stat-row {
    display    : flex;
    gap        : 22px;
    flex-wrap  : wrap;
    padding-top: 16px;
    border-top : 1px solid var(--border-dim);
    margin-top : 18px;
}
.su-val {
    font-family   : 'Cabinet Grotesk', sans-serif;
    font-weight   : 900;
    font-size     : 1.55rem;
    line-height   : 1;
    letter-spacing: -0.02em;
}
.su-lbl {
    font-size     : 9px;
    font-weight   : 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color         : var(--white-30);
    margin-top    : 3px;
    font-family   : 'Manrope', sans-serif;
}

/* ══ READABILITY SCORE BAR ═══════════════════════════════════ */
.read-bar-wrap {
    background    : var(--white-06);
    border-radius : 999px;
    height        : 10px;
    overflow      : hidden;
    margin        : 8px 0;
    position      : relative;
}
.read-bar-fill {
    height        : 100%;
    border-radius : 999px;
    transition    : width 1s cubic-bezier(.4,0,.2,1);
}

/* ══ HISTORY ITEM ════════════════════════════════════════════ */
.hist-item {
    background    : var(--surface2);
    border        : 1px solid var(--border-dim);
    border-radius : 12px;
    padding       : 14px 18px;
    cursor        : pointer;
    transition    : all 0.18s;
    margin-bottom : 8px;
}
.hist-item:hover { border-color: var(--teal); background: var(--surface3); }
.hist-ts {
    font-size     : 9.5px;
    font-weight   : 700;
    letter-spacing: 0.12em;
    color         : var(--white-30);
    text-transform: uppercase;
    font-family   : 'IBM Plex Mono', monospace;
    margin-bottom : 4px;
}
.hist-preview {
    font-size  : 12.5px;
    color      : var(--white-55);
    line-height: 1.5;
    font-family: 'Manrope', sans-serif;
    font-weight: 400;
}

/* ══ SENTIMENT BAR ═══════════════════════════════════════════ */
.sent-row {
    display      : flex;
    align-items  : center;
    gap          : 12px;
    margin-top   : 10px;
    font-family  : 'Manrope', sans-serif;
}
.sent-label { font-size: 12px; font-weight: 700; color: var(--white-55); min-width: 75px; }
.sent-bar-outer { flex: 1; background: var(--white-06); border-radius: 999px; height: 7px; overflow: hidden; }
.sent-bar-inner { height: 100%; border-radius: 999px; }

/* ══ DIVIDER WITH LABEL ══════════════════════════════════════ */
.div-label {
    display    : flex;
    align-items: center;
    gap        : 12px;
    margin     : 24px 0;
}
.div-label span {
    font-size     : 9px;
    font-weight   : 800;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color         : var(--white-30);
    white-space   : nowrap;
    font-family   : 'Manrope', sans-serif;
}
.div-label::before, .div-label::after {
    content   : '';
    flex      : 1;
    height    : 1px;
    background: var(--border-dim);
}

/* ══ INPUTS ══════════════════════════════════════════════════ */
.stTextArea textarea {
    background    : var(--surface) !important;
    border        : 1px solid var(--border) !important;
    border-radius : 14px !important;
    color         : var(--white) !important;
    font-family   : 'Manrope', sans-serif !important;
    font-size     : 14.5px !important;
    line-height   : 1.75 !important;
    box-shadow    : var(--shadow-deep) !important;
    caret-color   : var(--teal) !important;
}
.stTextArea textarea:focus {
    border-color  : var(--teal) !important;
    box-shadow    : 0 0 0 3px var(--teal-glow2), var(--shadow-deep) !important;
}
.stTextArea textarea::placeholder { color: var(--white-12) !important; }
.stTextArea label {
    font-family   : 'Manrope', sans-serif !important;
    font-size     : 9px !important;
    font-weight   : 800 !important;
    letter-spacing: 0.24em !important;
    text-transform: uppercase !important;
    color         : var(--white-30) !important;
}

/* ══ BUTTON ══════════════════════════════════════════════════ */
.stButton > button {
    background    : linear-gradient(135deg, var(--teal-dim), var(--teal)) !important;
    color         : #000 !important;
    font-family   : 'Manrope', sans-serif !important;
    font-weight   : 800 !important;
    font-size     : 13.5px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border        : none !important;
    border-radius : 10px !important;
    padding       : 13px 36px !important;
    box-shadow    : 0 6px 28px rgba(20,184,166,0.40) !important;
    transition    : all 0.2s ease !important;
}
.stButton > button:hover  { transform: translateY(-2px) !important; box-shadow: 0 12px 36px rgba(20,184,166,0.60) !important; background: linear-gradient(135deg, var(--teal), var(--teal-bright)) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ══ FILE UPLOADER ═══════════════════════════════════════════ */
.stFileUploader {
    background    : var(--surface) !important;
    border        : 2px dashed rgba(20,184,166,0.30) !important;
    border-radius : 16px !important;
    transition    : border-color 0.2s !important;
}
.stFileUploader:hover { border-color: var(--teal) !important; }
.stFileUploader label {
    font-family   : 'Manrope', sans-serif !important;
    font-size     : 9px !important;
    font-weight   : 800 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color         : var(--white-30) !important;
}

/* ══ ALERTS ══════════════════════════════════════════════════ */
.stAlert {
    background    : var(--surface2) !important;
    border-radius : 10px !important;
    font-family   : 'Manrope', sans-serif !important;
    font-size     : 14px !important;
    border        : 1px solid var(--border-dim) !important;
    color         : var(--white-55) !important;
}
.stSuccess { border-left: 3px solid var(--teal) !important; }
.stWarning { border-left: 3px solid var(--amber) !important; }

/* ══ SPINNER ══════════════════════════════════════════════════ */
.stSpinner > div > div { border-top-color: var(--teal) !important; }

/* ══ EXPANDER ════════════════════════════════════════════════ */
.streamlit-expanderHeader {
    background    : var(--surface2) !important;
    border-radius : 10px !important;
    color         : var(--white-55) !important;
    font-family   : 'Manrope', sans-serif !important;
    font-weight   : 700 !important;
    font-size     : 13px !important;
}

/* ══ DOWNLOAD BUTTON ═════════════════════════════════════════ */
.stDownloadButton > button {
    background    : var(--white-06) !important;
    color         : var(--white-55) !important;
    font-family   : 'Manrope', sans-serif !important;
    font-weight   : 700 !important;
    font-size     : 12.5px !important;
    letter-spacing: 0.06em !important;
    border        : 1px solid var(--border-dim) !important;
    border-radius : 9px !important;
    padding       : 9px 22px !important;
    transition    : all 0.18s !important;
    box-shadow    : none !important;
}
.stDownloadButton > button:hover {
    background    : var(--teal-glow) !important;
    border-color  : var(--teal) !important;
    color         : var(--white) !important;
    transform     : translateY(-1px) !important;
}

/* ══ SCROLLBAR ═══════════════════════════════════════════════ */
::-webkit-scrollbar        { width: 5px; }
::-webkit-scrollbar-track  { background: var(--obs); }
::-webkit-scrollbar-thumb  { background: var(--surface3); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal-dim); }

/* ══ MISC ════════════════════════════════════════════════════ */
hr { border-color: var(--border-dim) !important; margin: 18px 0 !important; }
code {
    font-family   : 'IBM Plex Mono', monospace !important;
    font-size     : 12px !important;
    color         : var(--teal-bright) !important;
    background    : var(--teal-glow2) !important;
    padding       : 2px 7px !important;
    border-radius : 5px !important;
}

/* ══ PIPE STEP ═══════════════════════════════════════════════ */
.pipe-step {
    background    : var(--surface2);
    border        : 1px solid var(--border-dim);
    border-radius : 12px;
    padding       : 16px 18px;
    text-align    : center;
    min-width     : 115px;
    transition    : all 0.2s;
}
.pipe-step:hover { border-color: var(--teal); box-shadow: var(--shadow-teal); transform: translateY(-2px); }
.pipe-step-icon  { font-size: 1.4rem; margin-bottom: 8px; }
.pipe-step-name  { font-family: 'Cabinet Grotesk', sans-serif; font-weight: 800; font-size: 12px; color: var(--white); letter-spacing: -0.01em; }
.pipe-step-sub   { font-size: 10.5px; color: var(--white-30); margin-top: 2px; font-family: 'Manrope', sans-serif; }
.pipe-arrow      { color: var(--teal); font-size: 18px; font-weight: 700; padding: 0 8px; }

/* ══ LIVE DOT ════════════════════════════════════════════════ */
.live-dot {
    display       : inline-block;
    width         : 7px;
    height        : 7px;
    border-radius : 50%;
    animation     : dot-pulse 2s ease infinite;
}
.dot-teal  { background: var(--teal); box-shadow: 0 0 10px rgba(20,184,166,0.7); }
.dot-white { background: var(--white-80); box-shadow: 0 0 8px rgba(255,255,255,0.5); }
@keyframes dot-pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.7); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HELPERS  &  NLP UTILITIES
# ─────────────────────────────────────────────────────────────

def page_header(eyebrow: str, title: str, hl: str, desc: str) -> None:
    st.markdown(f"""
    <div class="ph-wrap">
        <div class="ph-eyebrow">{eyebrow}</div>
        <div class="ph-title">{title} <span class="hl">{hl}</span></div>
        <div class="ph-bar"></div>
        <div class="ph-desc">{desc}</div>
    </div>""", unsafe_allow_html=True)


def mc(val: str, lbl: str, color: str = "") -> str:
    return f'<div class="m-card"><div class="m-val {color}">{val}</div><div class="m-lbl">{lbl}</div></div>'


def compression(orig: int, summ: int) -> int:
    return max(0, round((1 - summ / orig) * 100)) if orig else 0


def comp_ratio(orig: int, summ: int) -> float:
    return round(orig / summ, 1) if summ else 0.0


def flesch_score(text: str) -> float:
    """Approximate Flesch Reading Ease (0‒100)."""
    sentences = max(1, len(re.findall(r'[.!?]+', text)))
    words_list = re.findall(r'\b\w+\b', text)
    words = max(1, len(words_list))
    syllables = sum(_count_syllables(w) for w in words_list)
    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return round(max(0, min(100, score)), 1)


def _count_syllables(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r'[aeiou]+', word))
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def readability_label(score: float) -> tuple:
    if score >= 80:   return "Very Easy",     "emerald"
    if score >= 65:   return "Easy",          "emerald"
    if score >= 50:   return "Moderate",      "teal"
    if score >= 35:   return "Difficult",     "amber"
    return "Very Difficult", "rose"


def simple_sentiment(text: str) -> dict:
    """Lightweight keyword-based sentiment (no extra deps)."""
    pos_words = {"good","great","excellent","best","wonderful","fantastic","positive","success",
                 "happy","beneficial","improve","innovative","effective","efficient","achieve",
                 "advantage","helpful","important","significant","strong","leading"}
    neg_words = {"bad","poor","fail","failure","negative","problem","issue","difficult","hard",
                 "risk","threat","concern","loss","decline","reduce","lack","limit","weak","worse"}
    words = re.findall(r'\b\w+\b', text.lower())
    p = sum(1 for w in words if w in pos_words)
    n = sum(1 for w in words if w in neg_words)
    total = p + n
    if total == 0:
        return {"label": "Neutral", "pos": 0.5, "neg": 0.5, "color": "white-30"}
    pp = p / total
    if pp > 0.6:  return {"label": "Positive", "pos": pp, "neg": 1-pp, "color": "emerald"}
    if pp < 0.4:  return {"label": "Negative", "pos": pp, "neg": 1-pp, "color": "rose"}
    return {"label": "Neutral", "pos": pp, "neg": 1-pp, "color": "amber"}


def top_keywords(text: str, n: int = 12) -> list:
    """Return top-N (word, freq) pairs excluding stop words."""
    stop = {
        "the","a","an","is","are","was","were","be","been","have","has","had",
        "do","does","did","will","would","could","should","may","might","must",
        "to","of","in","for","on","with","at","by","from","as","it","its",
        "and","or","but","not","this","that","these","those","i","you","he",
        "she","we","they","their","our","your","its","also","can","all","more",
        "one","two","said","which","been","about","into","than","then","when",
        "there","where","who","what","how","if","so","just","up","out","no","my"
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    freq  = collections.Counter(w for w in words if w not in stop)
    return freq.most_common(n)


def avg_sentence_length(text: str) -> float:
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return round(sum(lengths) / len(lengths), 1)


def lsa_summarize(text: str, n: int = 3):
    t0      = time.time()
    parser  = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summ    = LsaSummarizer(stemmer)
    summ.stop_words = get_stop_words("english")
    result  = " ".join(str(s) for s in summ(parser.document, n))
    return result, round(time.time() - t0, 3)


@st.cache_resource(show_spinner=False)
def load_bart():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def bart_summarize(text: str):
    t0   = time.time()
    pipe = load_bart()
    out  = pipe(text[:1024], max_length=130, min_length=30, do_sample=False)
    return out[0]["summary_text"], round(time.time() - t0, 2)


def add_to_history(entry: dict) -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].insert(0, entry)
    st.session_state["history"] = st.session_state["history"][:5]   # keep last 5


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <span class="sb-brand">AI Text<br><strong>Summarizer</strong></span>
    <span class="sb-version">v2.0 &nbsp;<span>NEW</span></span>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigate",
        ["🏠  Home", "📄  Upload PDF", "⚡  Summarizer",
         "📊  Model Comparison", "🔬  Text Analysis", "🕘  History", "ℹ️  About"],
        label_visibility="visible",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model indicators
    st.markdown("""
    <div style="margin-bottom:16px">
        <div style="font-size:9px;font-weight:800;letter-spacing:0.26em;text-transform:uppercase;
                    color:rgba(248,250,252,0.25);margin-bottom:13px;font-family:'Manrope',sans-serif;">
            Active Models
        </div>
        <div style="display:flex;flex-direction:column;gap:11px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span class="live-dot dot-white"></span>
                <div>
                    <div style="font-size:13px;font-weight:700;color:#F8FAFC;font-family:'Manrope',sans-serif;">LSA</div>
                    <div style="font-size:10.5px;color:rgba(248,250,252,0.30);font-family:'Manrope',sans-serif;">Extractive · Sumy</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
                <span class="live-dot dot-teal"></span>
                <div>
                    <div style="font-size:13px;font-weight:700;color:#F8FAFC;font-family:'Manrope',sans-serif;">DistilBART</div>
                    <div style="font-size:10.5px;color:rgba(248,250,252,0.30);font-family:'Manrope',sans-serif;">Abstractive · HuggingFace</div>
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Session stats
    if "original" in st.session_state:
        wc    = len(st.session_state["original"].split())
        runs  = len(st.session_state.get("history", []))
        st.markdown(f"""
        <div style="padding:16px;
            background:linear-gradient(135deg,rgba(20,184,166,0.10),rgba(8,11,18,0.7));
            border:1px solid rgba(20,184,166,0.22);border-radius:13px;">
            <div style="font-size:9px;font-weight:800;letter-spacing:0.22em;text-transform:uppercase;
                        color:rgba(248,250,252,0.30);margin-bottom:12px;font-family:'Manrope',sans-serif;">Session</div>
            <div style="display:flex;gap:16px;">
                <div>
                    <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:2rem;
                                color:#2DD4BF;line-height:1;letter-spacing:-0.03em;">{wc:,}</div>
                    <div style="font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                                color:rgba(248,250,252,0.30);margin-top:3px;font-family:'Manrope',sans-serif;">words</div>
                </div>
                <div>
                    <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:2rem;
                                color:#2DD4BF;line-height:1;letter-spacing:-0.03em;">{runs}</div>
                    <div style="font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                                color:rgba(248,250,252,0.30);margin-top:3px;font-family:'Manrope',sans-serif;">runs</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# strip emoji prefix for routing
page = menu.split("  ", 1)[-1].strip() if "  " in menu else menu.strip()


# ══════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":
    page_header(
        "AI / NLP Portfolio  ·  v2.0",
        "AI Based Text",
        "Summarizer.",
        "Production-grade dual-model NLP system. Compare extractive and abstractive "
        "summaries with metrics, keyword analysis, readability scores, sentiment detection, "
        "and full session history — all in one interface.",
    )

    st.markdown("""
    <div class="feat-grid">
        <div class="feat-item">
            <div class="feat-icon">⚡</div>
            <div class="feat-title">LSA Extractive</div>
            <div class="feat-desc">Singular Value Decomposition selects the most
            semantically significant original sentences. Under 2 s. Zero hallucination.</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon">🤖</div>
            <div class="feat-title">DistilBART Abstractive</div>
            <div class="feat-desc">Distilled BART transformer generates entirely new
            fluent prose trained on CNN/DailyMail news corpora.</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon">🔑</div>
            <div class="feat-title">Keyword Extractor <span class="feat-new">NEW</span></div>
            <div class="feat-desc">Identifies the top 12 most significant terms
            from your text with frequency counts displayed as chips.</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon">📖</div>
            <div class="feat-title">Readability Score <span class="feat-new">NEW</span></div>
            <div class="feat-desc">Flesch Reading Ease analysis grades your text
            from Very Easy to Very Difficult with a visual progress bar.</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon">🌡️</div>
            <div class="feat-title">Sentiment Detector <span class="feat-new">NEW</span></div>
            <div class="feat-desc">Keyword-based positive/negative/neutral sentiment
            analysis with visual bar chart for both input and summaries.</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon">🕘</div>
            <div class="feat-title">Session History <span class="feat-new">NEW</span></div>
            <div class="feat-desc">Last 5 summarization runs are saved. Reload any
            previous result with one click directly from the History page.</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon">💾</div>
            <div class="feat-title">Download Results <span class="feat-new">NEW</span></div>
            <div class="feat-desc">Export both summaries plus all metrics as a
            clean .txt report file — ready to share or archive.</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon">🎚️</div>
            <div class="feat-title">Sentence Control <span class="feat-new">NEW</span></div>
            <div class="feat-desc">Slide to choose 1–8 sentences for the LSA
            extractive summary. More control, better results.</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon">📄</div>
            <div class="feat-title">PDF Upload</div>
            <div class="feat-desc">Upload any multi-page PDF. Text is extracted
            page-by-page via PyPDF2 and fed into both models.</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Quick start
    st.markdown("""
    <div class="glow-card">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="font-size:1.8rem;flex-shrink:0;">▶</div>
            <div>
                <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:800;
                            font-size:15px;color:#F8FAFC;margin-bottom:4px;letter-spacing:-0.01em;">
                    Quick Start
                </div>
                <div style="font-size:13.5px;color:rgba(248,250,252,0.45);font-family:'Manrope',sans-serif;">
                    Upload PDF &nbsp;→&nbsp; Summarizer &nbsp;→&nbsp; Text Analysis &nbsp;→&nbsp; Model Comparison
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <span style="font-size:9px;font-weight:800;letter-spacing:0.22em;text-transform:uppercase;
                     color:rgba(248,250,252,0.25);font-family:'Manrope',sans-serif;">Stack</span>
        <span class="tag-chip">Python 3</span>
        <span class="tag-chip">Streamlit</span>
        <span class="tag-chip">HuggingFace Transformers</span>
        <span class="tag-chip">Sumy NLP</span>
        <span class="tag-chip">NLTK</span>
        <span class="tag-chip">PyPDF2</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  UPLOAD PDF
# ══════════════════════════════════════════════════════════════
elif page == "Upload PDF":
    page_header(
        "Step 1 of 2",
        "Upload",
        "PDF.",
        "Extract text from any PDF document. Multi-page files fully supported. "
        "Content loads automatically into the Summarizer.",
    )

    uploaded = st.file_uploader("Drop your PDF here", type="pdf", label_visibility="collapsed")

    if uploaded:
        with st.spinner("Extracting text…"):
            reader   = PyPDF2.PdfReader(io.BytesIO(uploaded.read()))
            raw_text = "".join((p.extract_text() or "") for p in reader.pages)
        st.session_state["text"] = raw_text

        pages = len(reader.pages)
        words = len(raw_text.split())
        chars = len(raw_text)
        avg_sl = avg_sentence_length(raw_text)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(mc(str(pages),   "Pages"),             unsafe_allow_html=True)
        c2.markdown(mc(f"{words:,}", "Words",      "teal"),unsafe_allow_html=True)
        c3.markdown(mc(f"{chars:,}", "Characters"),        unsafe_allow_html=True)
        c4.markdown(mc(str(avg_sl),  "Avg Sent Len","cyan"),unsafe_allow_html=True)

        st.success("PDF loaded — head to **Summarizer** to run both models.")

        with st.expander("Preview extracted text"):
            st.markdown(f"""
            <div class="card" style="font-size:12.5px;line-height:1.82;color:rgba(248,250,252,0.55);
                white-space:pre-wrap;font-family:'IBM Plex Mono',monospace;
                max-height:280px;overflow-y:auto;">{raw_text[:1500]}…</div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:72px 32px;">
            <div style="font-size:3rem;margin-bottom:18px;">📂</div>
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;
                font-size:1.5rem;color:#F8FAFC;margin-bottom:10px;letter-spacing:-0.02em;">
                Drop a PDF above
            </div>
            <div style="font-size:13.5px;color:rgba(248,250,252,0.35);
                font-family:'Manrope',sans-serif;line-height:1.65;">
                Multi-page documents supported &nbsp;·&nbsp;
                Text extracted page-by-page automatically
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SUMMARIZER
# ══════════════════════════════════════════════════════════════
elif page == "Summarizer":
    page_header(
        "Step 2 of 2",
        "Text",
        "Summarizer.",
        "Paste text or use your loaded PDF. Control the sentence count, run both "
        "models, then download your results.",
    )

    default = st.session_state.get("text", "")
    txt_in  = st.text_area(
        "Input Text",
        value            = default,
        height           = 220,
        placeholder      = "Paste any article, research paper, report, or long-form text here…",
        label_visibility = "visible",
    )

    wc = len(txt_in.split()) if txt_in.strip() else 0
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin-top:-10px;margin-bottom:16px;">
        <span class="stat-pill"><em>{wc:,}</em>&nbsp;words</span>
    </div>""", unsafe_allow_html=True)

    col_sl, col_btn, _ = st.columns([2, 1.5, 2.5])
    with col_sl:
        n_sent = st.slider("LSA Sentence Count", 1, 8, 3, label_visibility="visible")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Run Both Models")

    if go:
        if not txt_in.strip():
            st.warning("Please enter some text or upload a PDF first.")
        elif wc < 30:
            st.warning("Text too short — provide at least 30 words.")
        else:
            with st.spinner("Running LSA Extractive…"):
                ext_sum, ext_t = lsa_summarize(txt_in, n_sent)
            with st.spinner("Running DistilBART Abstractive (~30 s on CPU)…"):
                abs_sum, abs_t = bart_summarize(txt_in)

            entry = {
                "ts"       : datetime.datetime.now().strftime("%H:%M  %d %b"),
                "original" : txt_in,
                "extractive": ext_sum,
                "abstractive": abs_sum,
                "time_lsa" : ext_t,
                "time_bart": abs_t,
                "n_sent"   : n_sent,
            }
            st.session_state.update({**entry})
            add_to_history(entry)

            ext_wc  = len(ext_sum.split())
            abs_wc  = len(abs_sum.split())
            ext_red = compression(wc, ext_wc)
            abs_red = compression(wc, abs_wc)

            # ── result cards ──────────────
            st.markdown('<div class="div-label"><span>Results</span></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="result-card lsa">
                    <span class="pill pill-white">
                        <span class="live-dot dot-white" style="margin-right:2px;"></span>
                        Extractive &middot; LSA
                    </span>
                    <p style="font-size:14.5px;line-height:1.88;color:rgba(248,250,252,0.78);
                        font-family:'Manrope',sans-serif;margin:0;">{ext_sum}</p>
                    <div class="stat-row">
                        <div><div class="su-val" style="color:#F8FAFC;">{ext_t}s</div>
                             <div class="su-lbl">Time</div></div>
                        <div><div class="su-val" style="color:#F8FAFC;">{ext_wc}</div>
                             <div class="su-lbl">Words</div></div>
                        <div><div class="su-val" style="color:#F8FAFC;">{ext_red}%</div>
                             <div class="su-lbl">Reduced</div></div>
                        <div><div class="su-val" style="color:#F8FAFC;">{n_sent}</div>
                             <div class="su-lbl">Sentences</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="result-card bart">
                    <span class="pill pill-teal">
                        <span class="live-dot dot-teal" style="margin-right:2px;"></span>
                        Abstractive &middot; DistilBART
                    </span>
                    <p style="font-size:14.5px;line-height:1.88;color:rgba(248,250,252,0.78);
                        font-family:'Manrope',sans-serif;margin:0;">{abs_sum}</p>
                    <div class="stat-row">
                        <div><div class="su-val" style="color:#2DD4BF;">{abs_t}s</div>
                             <div class="su-lbl">Time</div></div>
                        <div><div class="su-val" style="color:#2DD4BF;">{abs_wc}</div>
                             <div class="su-lbl">Words</div></div>
                        <div><div class="su-val" style="color:#2DD4BF;">{abs_red}%</div>
                             <div class="su-lbl">Reduced</div></div>
                        <div><div class="su-val" style="color:#2DD4BF;">~</div>
                             <div class="su-lbl">Generated</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── download ──────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            report = f"""AI BASED TEXT SUMMARIZER  ·  Results Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

ORIGINAL TEXT ({wc} words)
{txt_in}

{'='*60}
LSA EXTRACTIVE SUMMARY  ({ext_wc} words · {ext_red}% reduction · {ext_t}s)
{ext_sum}

{'='*60}
DISTILBART ABSTRACTIVE SUMMARY  ({abs_wc} words · {abs_red}% reduction · {abs_t}s)
{abs_sum}

{'='*60}
METRICS
  Original Words   : {wc}
  LSA Words        : {ext_wc}  ({ext_red}% reduction)
  BART Words       : {abs_wc}  ({abs_red}% reduction)
  LSA Time         : {ext_t}s
  BART Time        : {abs_t}s
"""
            st.download_button(
                label     = "⬇  Download Full Report (.txt)",
                data      = report,
                file_name = f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime      = "text/plain",
            )

            st.success("Both models complete — check **Text Analysis** and **Model Comparison**.")


# ══════════════════════════════════════════════════════════════
#  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    page_header(
        "Deep Analysis",
        "Model",
        "Comparison.",
        "Every metric laid out side by side — compression, word count, "
        "processing time, and behaviour at a glance.",
    )

    if "original" not in st.session_state:
        st.markdown("""<div class="card" style="text-align:center;padding:72px 32px;">
            <div style="font-size:3rem;margin-bottom:18px;">📊</div>
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:1.5rem;
                color:#F8FAFC;letter-spacing:-0.02em;margin-bottom:10px;">No data yet</div>
            <div style="font-size:13.5px;color:rgba(248,250,252,0.35);font-family:'Manrope',sans-serif;">
                Run the <strong style="color:#14B8A6;">Summarizer</strong> first.
            </div></div>""", unsafe_allow_html=True)
    else:
        orig   = st.session_state["original"]
        ext    = st.session_state["extractive"]
        abst   = st.session_state["abstractive"]
        t_lsa  = st.session_state["time_lsa"]
        t_bart = st.session_state["time_bart"]

        ow  = len(orig.split());  ew  = len(ext.split());  aw  = len(abst.split())
        er  = compression(ow, ew); ar  = compression(ow, aw)
        eR  = comp_ratio(ow, ew);  aR  = comp_ratio(ow, aw)
        avg_orig = avg_sentence_length(orig)
        avg_ext  = avg_sentence_length(ext)
        avg_abs  = avg_sentence_length(abst)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(mc(f"{ow:,}", "Original Words"),       unsafe_allow_html=True)
        c2.markdown(mc(f"{er}%",  "LSA Reduction", "teal"),unsafe_allow_html=True)
        c3.markdown(mc(f"{ar}%",  "BART Reduction","cyan"),unsafe_allow_html=True)
        faster = f"{t_lsa}s" if t_lsa < t_bart else f"{t_bart}s"
        c4.markdown(mc(faster,    "Fastest Time",  "amber"),unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Bars
        st.markdown(f"""
        <div class="card">
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:800;font-size:1.1rem;
                color:#F8FAFC;margin-bottom:22px;letter-spacing:-0.02em;">Compression Ratio</div>
            <div style="margin-bottom:20px;">
                <div style="display:flex;justify-content:space-between;font-size:12.5px;
                    color:rgba(248,250,252,0.55);margin-bottom:7px;font-weight:700;font-family:'Manrope',sans-serif;">
                    <span style="display:flex;align-items:center;gap:8px;">
                        <span class="live-dot dot-white" style="animation:none;"></span>LSA Extractive
                    </span>
                    <span>{eR}x &nbsp;·&nbsp; {ew} words</span>
                </div>
                <div class="pbar-wrap"><div class="pbar-fill pbar-white" style="width:{min(er,100)}%;"></div></div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;font-size:12.5px;
                    color:rgba(248,250,252,0.55);margin-bottom:7px;font-weight:700;font-family:'Manrope',sans-serif;">
                    <span style="display:flex;align-items:center;gap:8px;">
                        <span class="live-dot dot-teal" style="animation:none;"></span>DistilBART Abstractive
                    </span>
                    <span>{aR}x &nbsp;·&nbsp; {aw} words</span>
                </div>
                <div class="pbar-wrap"><div class="pbar-fill pbar-teal" style="width:{min(ar,100)}%;"></div></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Full table
        e_t = '<span class="win-badge">Faster</span>'  if t_lsa  < t_bart else ''
        a_t = '<span class="win-badge">Faster</span>'  if t_bart < t_lsa  else ''
        e_w = '<span class="win-badge">Shorter</span>' if ew      < aw     else ''
        a_w = '<span class="win-badge">Shorter</span>' if aw      < ew     else ''
        e_r = '<span class="win-badge">Higher</span>'  if er      > ar     else ''
        a_r = '<span class="win-badge">Higher</span>'  if ar      > er     else ''

        st.markdown(f"""
        <table class="cmp-table">
            <thead><tr>
                <th>Metric</th>
                <th><span style="display:inline-flex;align-items:center;gap:7px;">
                    <span class="live-dot dot-white" style="animation:none;"></span>LSA Extractive</span></th>
                <th><span style="display:inline-flex;align-items:center;gap:7px;">
                    <span class="live-dot dot-teal" style="animation:none;"></span>DistilBART Abstractive</span></th>
            </tr></thead>
            <tbody>
                <tr><td>Original Words</td>         <td>{ow:,}</td>              <td>{ow:,}</td></tr>
                <tr><td>Summary Words</td>           <td>{ew} {e_w}</td>         <td>{aw} {a_w}</td></tr>
                <tr><td>Word Reduction</td>          <td>{er}% {e_r}</td>        <td>{ar}% {a_r}</td></tr>
                <tr><td>Compression Ratio</td>       <td>{eR}x</td>              <td>{aR}x</td></tr>
                <tr><td>Avg Sentence Length</td>     <td>{avg_ext} words</td>    <td>{avg_abs} words</td></tr>
                <tr><td>Processing Time</td>         <td>{t_lsa}s {e_t}</td>     <td>{t_bart}s {a_t}</td></tr>
                <tr><td>Output Type</td>             <td>Original sentences</td> <td>Generated text</td></tr>
                <tr><td>Hallucination Risk</td>      <td>None</td>               <td>Low</td></tr>
                <tr><td>GPU Required</td>            <td>No</td>                 <td>Optional</td></tr>
                <tr><td>Best Use Case</td>           <td>Technical / factual</td><td>Articles / narrative</td></tr>
            </tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="div-label"><span>Summary Outputs</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="card"><span class="pill pill-white">LSA Output</span>
                <p style="font-size:13.5px;line-height:1.85;color:rgba(248,250,252,0.70);
                font-family:'Manrope',sans-serif;margin:0;">{ext}</p></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card" style="border-color:rgba(20,184,166,0.25);">
                <span class="pill pill-teal">DistilBART Output</span>
                <p style="font-size:13.5px;line-height:1.85;color:rgba(248,250,252,0.70);
                font-family:'Manrope',sans-serif;margin:0;">{abst}</p></div>""",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TEXT ANALYSIS  (NEW PAGE)
# ══════════════════════════════════════════════════════════════
elif page == "Text Analysis":
    page_header(
        "NLP Insights",
        "Text",
        "Analysis.",
        "Deep-dive into your text — keyword frequency, readability score, "
        "sentiment detection, and structural statistics.",
    )

    if "original" not in st.session_state:
        st.markdown("""<div class="card" style="text-align:center;padding:72px 32px;">
            <div style="font-size:3rem;margin-bottom:18px;">🔬</div>
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:1.5rem;
                color:#F8FAFC;letter-spacing:-0.02em;margin-bottom:10px;">No text analysed yet</div>
            <div style="font-size:13.5px;color:rgba(248,250,252,0.35);font-family:'Manrope',sans-serif;">
                Run the <strong style="color:#14B8A6;">Summarizer</strong> first.
            </div></div>""", unsafe_allow_html=True)
    else:
        orig = st.session_state["original"]
        ext  = st.session_state.get("extractive", "")
        abst = st.session_state.get("abstractive", "")

        # ── Text stats ────────────────
        ow     = len(orig.split())
        sents  = len(re.findall(r'[.!?]+', orig))
        paras  = len([p for p in orig.split('\n\n') if p.strip()])
        avg_sl = avg_sentence_length(orig)
        fk     = flesch_score(orig)
        rl, rc = readability_label(fk)

        st.markdown('<div class="div-label"><span>Document Statistics</span></div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(mc(f"{ow:,}",    "Words",       "teal"),    unsafe_allow_html=True)
        c2.markdown(mc(str(sents),   "Sentences",   "cyan"),    unsafe_allow_html=True)
        c3.markdown(mc(str(paras),   "Paragraphs"),             unsafe_allow_html=True)
        c4.markdown(mc(str(avg_sl),  "Avg Sent Len","amber"),   unsafe_allow_html=True)
        c5.markdown(mc(str(fk),      "Flesch Score","emerald"), unsafe_allow_html=True)

        # ── Readability ───────────────
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            color_map = {
                "emerald":"#10B981","teal":"#14B8A6","amber":"#F59E0B","rose":"#F43F5E","white-30":"#888"
            }
            bar_color = color_map.get(rc, "#14B8A6")
            st.markdown(f"""
            <div class="card">
                <span class="pill pill-{rc}">Readability</span>
                <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:2rem;
                    color:{bar_color};letter-spacing:-0.02em;margin-bottom:4px;">{rl}</div>
                <div style="font-size:13px;color:rgba(248,250,252,0.35);font-family:'Manrope',sans-serif;
                    margin-bottom:12px;">Flesch Reading Ease: <strong style="color:{bar_color};">{fk}</strong> / 100</div>
                <div class="pbar-wrap" style="height:10px;">
                    <div class="pbar-fill" style="width:{fk}%;background:{bar_color};
                        box-shadow:0 0 14px {bar_color}55;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:6px;
                    font-size:9.5px;color:rgba(248,250,252,0.25);font-family:'Manrope',sans-serif;font-weight:700;">
                    <span>Very Difficult</span><span>Moderate</span><span>Very Easy</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_b:
            sent = simple_sentiment(orig)
            sc   = color_map.get(sent["color"], "#888")
            pos_pct = round(sent["pos"] * 100)
            neg_pct = 100 - pos_pct
            st.markdown(f"""
            <div class="card">
                <span class="pill pill-{sent['color']}">Sentiment</span>
                <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:2rem;
                    color:{sc};letter-spacing:-0.02em;margin-bottom:14px;">{sent['label']}</div>
                <div class="sent-row">
                    <span class="sent-label">Positive</span>
                    <div class="sent-bar-outer">
                        <div class="sent-bar-inner" style="width:{pos_pct}%;background:#10B981;"></div>
                    </div>
                    <span style="font-size:12px;font-weight:800;color:#10B981;min-width:36px;text-align:right;
                        font-family:'Cabinet Grotesk',sans-serif;">{pos_pct}%</span>
                </div>
                <div class="sent-row" style="margin-top:8px;">
                    <span class="sent-label">Negative</span>
                    <div class="sent-bar-outer">
                        <div class="sent-bar-inner" style="width:{neg_pct}%;background:#F43F5E;"></div>
                    </div>
                    <span style="font-size:12px;font-weight:800;color:#F43F5E;min-width:36px;text-align:right;
                        font-family:'Cabinet Grotesk',sans-serif;">{neg_pct}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Keywords ──────────────────
        st.markdown('<div class="div-label"><span>Top Keywords</span></div>', unsafe_allow_html=True)
        kws = top_keywords(orig, 12)
        if kws:
            chips = "".join(
                f'<div class="kw-chip">{w} <span class="kw-freq">{f}</span></div>'
                for w, f in kws
            )
            st.markdown(f'<div class="kw-grid">{chips}</div>', unsafe_allow_html=True)

        # ── Summary sentiment comparison ──
        if ext and abst:
            st.markdown('<div class="div-label"><span>Summary Sentiment Comparison</span></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            for col, text, label, pill in [
                (c1, ext,  "LSA Summary",         "pill-white"),
                (c2, abst, "DistilBART Summary",  "pill-teal"),
            ]:
                s  = simple_sentiment(text)
                sc = color_map.get(s["color"], "#888")
                pp = round(s["pos"] * 100)
                np_ = 100 - pp
                with col:
                    st.markdown(f"""
                    <div class="card">
                        <span class="pill {pill}">{label}</span>
                        <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;
                            font-size:1.5rem;color:{sc};letter-spacing:-0.02em;margin-bottom:12px;">
                            {s['label']}
                        </div>
                        <div class="sent-row">
                            <span class="sent-label">Positive</span>
                            <div class="sent-bar-outer"><div class="sent-bar-inner"
                                style="width:{pp}%;background:#10B981;"></div></div>
                            <span style="font-size:12px;font-weight:800;color:#10B981;min-width:36px;
                                text-align:right;font-family:'Cabinet Grotesk',sans-serif;">{pp}%</span>
                        </div>
                        <div class="sent-row" style="margin-top:8px;">
                            <span class="sent-label">Negative</span>
                            <div class="sent-bar-outer"><div class="sent-bar-inner"
                                style="width:{np_}%;background:#F43F5E;"></div></div>
                            <span style="font-size:12px;font-weight:800;color:#F43F5E;min-width:36px;
                                text-align:right;font-family:'Cabinet Grotesk',sans-serif;">{np_}%</span>
                        </div>
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HISTORY  (NEW PAGE)
# ══════════════════════════════════════════════════════════════
elif page == "History":
    page_header(
        "Session",
        "Run",
        "History.",
        "Your last 5 summarization runs. Click any entry to reload it into the current session.",
    )

    history = st.session_state.get("history", [])
    if not history:
        st.markdown("""<div class="card" style="text-align:center;padding:72px 32px;">
            <div style="font-size:3rem;margin-bottom:18px;">🕘</div>
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:1.5rem;
                color:#F8FAFC;letter-spacing:-0.02em;margin-bottom:10px;">No history yet</div>
            <div style="font-size:13.5px;color:rgba(248,250,252,0.35);font-family:'Manrope',sans-serif;">
                Each time you run the Summarizer, it appears here.
            </div></div>""", unsafe_allow_html=True)
    else:
        for i, h in enumerate(history):
            wc_h = len(h["original"].split())
            er_h = compression(wc_h, len(h["extractive"].split()))
            ar_h = compression(wc_h, len(h["abstractive"].split()))
            preview = h["original"][:120].replace('\n', ' ')

            col_info, col_btn = st.columns([5, 1])
            with col_info:
                st.markdown(f"""
                <div class="hist-item">
                    <div class="hist-ts">#{i+1} &nbsp;·&nbsp; {h['ts']} &nbsp;·&nbsp;
                        {wc_h:,} words &nbsp;·&nbsp; LSA {er_h}% reduced &nbsp;·&nbsp; BART {ar_h}% reduced
                    </div>
                    <div class="hist-preview">"{preview}…"</div>
                </div>""", unsafe_allow_html=True)
            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button(f"Load", key=f"hist_{i}"):
                    st.session_state.update({
                        "original":    h["original"],
                        "extractive":  h["extractive"],
                        "abstractive": h["abstractive"],
                        "time_lsa":    h["time_lsa"],
                        "time_bart":   h["time_bart"],
                    })
                    st.success(f"Run #{i+1} loaded — check Summarizer or Model Comparison.")

        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":
    page_header(
        "Project Info",
        "About This",
        "Project.",
        "Architecture, model details, new features in v2.0, and the full technology stack.",
    )

    # What's new
    st.markdown("""
    <div class="glow-card" style="margin-bottom:20px;">
        <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:800;font-size:1.1rem;
            color:#F8FAFC;margin-bottom:16px;letter-spacing:-0.02em;">
            ✨ &nbsp; What's New in v2.0
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="display:flex;align-items:center;gap:8px;font-size:13px;
                color:rgba(248,250,252,0.65);font-family:'Manrope',sans-serif;">
                <span style="color:#14B8A6;font-weight:800;">+</span> LSA sentence count slider
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:13px;
                color:rgba(248,250,252,0.65);font-family:'Manrope',sans-serif;">
                <span style="color:#14B8A6;font-weight:800;">+</span> Download report as .txt
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:13px;
                color:rgba(248,250,252,0.65);font-family:'Manrope',sans-serif;">
                <span style="color:#14B8A6;font-weight:800;">+</span> Keyword frequency extractor
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:13px;
                color:rgba(248,250,252,0.65);font-family:'Manrope',sans-serif;">
                <span style="color:#14B8A6;font-weight:800;">+</span> Flesch readability scoring
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:13px;
                color:rgba(248,250,252,0.65);font-family:'Manrope',sans-serif;">
                <span style="color:#14B8A6;font-weight:800;">+</span> Sentiment detector (input + summaries)
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:13px;
                color:rgba(248,250,252,0.65);font-family:'Manrope',sans-serif;">
                <span style="color:#14B8A6;font-weight:800;">+</span> Session history (last 5 runs)
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:13px;
                color:rgba(248,250,252,0.65);font-family:'Manrope',sans-serif;">
                <span style="color:#14B8A6;font-weight:800;">+</span> Avg sentence length metric
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:13px;
                color:rgba(248,250,252,0.65);font-family:'Manrope',sans-serif;">
                <span style="color:#14B8A6;font-weight:800;">+</span> New teal / cyan / obsidian theme
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Model cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card" style="border-top:2px solid rgba(248,250,252,0.25);">
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:1.3rem;
                color:#F8FAFC;letter-spacing:-0.02em;margin-bottom:3px;">LSA Extractive</div>
            <span style="font-size:9.5px;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;
                color:rgba(248,250,252,0.30);margin-bottom:18px;display:block;font-family:'Manrope',sans-serif;">
                Sumy NLP · Latent Semantic Analysis
            </span>
            <div style="font-size:13.5px;color:rgba(248,250,252,0.55);line-height:2;font-family:'Manrope',sans-serif;">
                <strong style="color:#F8FAFC;font-weight:700;">Algorithm</strong><br>
                Singular Value Decomposition on TF-IDF matrix<br><br>
                <strong style="color:#F8FAFC;font-weight:700;">Approach</strong><br>
                Ranks sentences by latent semantic weight, extracts originals intact<br><br>
                <strong style="color:#F8FAFC;font-weight:700;">Strengths</strong><br>
                &lt;2s · No hallucination · Factually faithful<br><br>
                <strong style="color:#F8FAFC;font-weight:700;">Best For</strong><br>
                Research papers · Legal docs · Technical reports
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card" style="border-top:2px solid rgba(20,184,166,0.60);">
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:900;font-size:1.3rem;
                color:#2DD4BF;letter-spacing:-0.02em;margin-bottom:3px;">DistilBART Abstractive</div>
            <span style="font-size:9.5px;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;
                color:rgba(248,250,252,0.30);margin-bottom:18px;display:block;font-family:'Manrope',sans-serif;">
                HuggingFace · sshleifer/distilbart-cnn-12-6
            </span>
            <div style="font-size:13.5px;color:rgba(248,250,252,0.55);line-height:2;font-family:'Manrope',sans-serif;">
                <strong style="color:#F8FAFC;font-weight:700;">Architecture</strong><br>
                Distilled BART — bidirectional encoder + autoregressive decoder<br><br>
                <strong style="color:#F8FAFC;font-weight:700;">Approach</strong><br>
                Encodes full context, decodes new tokens via beam search<br><br>
                <strong style="color:#F8FAFC;font-weight:700;">Strengths</strong><br>
                Fluent · Coherent · Human-sounding output<br><br>
                <strong style="color:#F8FAFC;font-weight:700;">Best For</strong><br>
                News articles · Blog posts · Narrative content
            </div>
        </div>""", unsafe_allow_html=True)

    # Pipeline
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:800;font-size:1.1rem;
            color:#F8FAFC;margin-bottom:22px;letter-spacing:-0.02em;">Pipeline Architecture</div>
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;">
            <div class="pipe-step"><div class="pipe-step-icon">📥</div>
                <div class="pipe-step-name">Input</div><div class="pipe-step-sub">Text / PDF</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step"><div class="pipe-step-icon">🔤</div>
                <div class="pipe-step-name">Tokenizer</div><div class="pipe-step-sub">NLTK punkt</div></div>
            <div class="pipe-arrow">→</div>
            <div style="display:flex;flex-direction:column;gap:8px;">
                <div class="pipe-step" style="border-color:rgba(248,250,252,0.22);">
                    <div class="pipe-step-name">LSA</div><div class="pipe-step-sub">Extractive</div></div>
                <div class="pipe-step" style="border-color:rgba(20,184,166,0.45);">
                    <div class="pipe-step-name" style="color:#2DD4BF;">DistilBART</div>
                    <div class="pipe-step-sub">Abstractive</div></div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step"><div class="pipe-step-icon">📊</div>
                <div class="pipe-step-name">Metrics</div><div class="pipe-step-sub">Compression · Time</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step"><div class="pipe-step-icon">🔬</div>
                <div class="pipe-step-name">Analysis</div><div class="pipe-step-sub">Keywords · Sentiment</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step" style="border-color:rgba(20,184,166,0.40);">
                <div class="pipe-step-icon">🎨</div>
                <div class="pipe-step-name">UI</div><div class="pipe-step-sub">Streamlit</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Stack
    st.markdown("""
    <div class="card" style="margin-top:8px;">
        <div style="font-family:'Cabinet Grotesk',sans-serif;font-weight:800;font-size:1.1rem;
            color:#F8FAFC;margin-bottom:16px;letter-spacing:-0.02em;">Technology Stack</div>
        <div style="display:flex;flex-wrap:wrap;gap:9px;margin-bottom:18px;">
            <span class="tag-chip">Python 3.x</span><span class="tag-chip">Streamlit</span>
            <span class="tag-chip">HuggingFace Transformers</span><span class="tag-chip">DistilBART CNN-12-6</span>
            <span class="tag-chip">Sumy NLP</span><span class="tag-chip">NLTK</span>
            <span class="tag-chip">PyPDF2</span><span class="tag-chip">Collections</span>
            <span class="tag-chip">RegEx</span>
        </div>
        <div style="font-size:12.5px;color:rgba(248,250,252,0.35);
            font-family:'IBM Plex Mono',monospace;line-height:2;
            background:rgba(20,184,166,0.04);padding:14px 16px;border-radius:10px;
            border:1px solid rgba(20,184,166,0.10);">
            <span style="color:#2DD4BF;">model_1</span>  sshleifer/distilbart-cnn-12-6 · HuggingFace<br>
            <span style="color:#2DD4BF;">model_2</span>  LSA via Sumy · Snowball stemmer · EN stop-words<br>
            <span style="color:#2DD4BF;">analysis</span> Flesch-Kincaid · keyword TF · keyword sentiment<br>
            <span style="color:#2DD4BF;">input</span>    Raw text paste · PDF (PyPDF2 extraction)<br>
            <span style="color:#2DD4BF;">export</span>   .txt report with full metrics
        </div>
    </div>""", unsafe_allow_html=True)
