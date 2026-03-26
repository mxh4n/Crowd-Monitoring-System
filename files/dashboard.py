"""
dashboard.py
------------
Streamlit-powered live dashboard for the AI Crowd Monitoring System.
Styled after a tactical surveillance HUD — dark theme, cyan accents,
live charts, animated alert ticker, zone density bars, and real-time
YOLO video feed.

Run:
    streamlit run dashboard.py
"""

import time
import threading
import queue
import base64
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as st_html

from detector import CrowdDetector
from utils    import (get_youtube_stream_url, open_video_capture,
                      read_frame, resize_frame, blend_heatmap,
                      encode_frame_to_jpeg, send_telegram_alert, RateLimiter)

# ─────────────────────────────────────────────────────────────────────────────
# Page config MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrowdSense — AI Crowd Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  (dark tactical HUD aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root palette ─────────────────────────────────────────────────────────── */
:root {
  --bg-base:     #080d13;
  --bg-panel:    #0d1520;
  --bg-card:     #101c2b;
  --bg-hover:    #152333;
  --cyan:        #00e5ff;
  --cyan-dim:    #00b8cc;
  --green:       #00ff6e;
  --yellow:      #ffc107;
  --red:         #ff2d55;
  --orange:      #ff6b35;
  --text-bright: #e8f4f8;
  --text-muted:  #5a7a8a;
  --border:      #1a3040;
  --border-cyan: #00e5ff33;
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--bg-base) !important;
  color: var(--text-bright);
  font-family: 'Rajdhani', sans-serif;
}
/* 🔥 FIX SIDEBAR WIDTH & POSITION */
[data-testid="stSidebar"] {
  min-width: 320px !important;
  max-width: 320px !important;
  width: 320px !important;
  position: relative !important;
  z-index: 1000 !important;
}

/* prevent main content overlap */
[data-testid="stAppViewContainer"] {
  display: flex;
}

/* ensure main area respects sidebar */
.block-container {
  margin-left: 0 !important;
}
[data-testid="stSidebar"] * { color: var(--text-bright) !important; }

/* hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 1.5rem 2rem !important; max-width: 100% !important; }

/* ── Headings & mono text ────────────────────────────────────────────────── */
.orbitron   { font-family: 'Orbitron', sans-serif; }
.mono       { font-family: 'Share Tech Mono', monospace; }
.rajdhani   { font-family: 'Rajdhani', sans-serif; }

/* ── Top nav bar ─────────────────────────────────────────────────────────── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #060b10;
  border-bottom: 1px solid var(--border);
  padding: 0 1.5rem;
  height: 52px;
  position: sticky;
  top: 0;
  z-index: 999;
}
.topbar-logo {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.25rem;
  font-weight: 900;
  color: var(--cyan);
  letter-spacing: 2px;
  display: flex; align-items: center; gap: 10px;
}
.topbar-logo .pin { color: var(--cyan); font-size: 1.1rem; }
.topbar-nav {
  display: flex; gap: 8px;
}
.nav-item {
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--text-muted);
  padding: 6px 16px;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.2s;
}
.nav-item.active {
  color: var(--cyan);
  background: var(--border-cyan);
  border-bottom: 2px solid var(--cyan);
}
.topbar-right {
  display: flex; align-items: center; gap: 20px;
}
.live-indicator {
  display: flex; align-items: center; gap: 6px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem;
  color: var(--green);
}
.live-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--green);
  animation: blink 1.2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Alert ticker ─────────────────────────────────────────────────────────── */
.alert-ticker {
  background: #0a1520;
  border-bottom: 1px solid var(--border);
  border-top: 1px solid var(--border);
  height: 36px;
  overflow: hidden;
  display: flex;
  align-items: center;
  padding: 0 1rem;
}
.ticker-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--yellow);
  letter-spacing: 2px;
  white-space: nowrap;
  margin-right: 20px;
  border: 1px solid var(--yellow);
  padding: 2px 8px;
  border-radius: 2px;
}
.ticker-label.red { color: var(--red); border-color: var(--red); }
.ticker-scroll {
  flex: 1;
  overflow: hidden;
  position: relative;
}
.ticker-inner {
  display: inline-block;
  white-space: nowrap;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-bright);
  animation: scroll-left 28s linear infinite;
}
@keyframes scroll-left {
  0%   { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}

/* ── Zone filter tabs ─────────────────────────────────────────────────────── */
.zone-tabs {
  display: flex; gap: 6px; flex-wrap: wrap;
  padding: 10px 0 4px;
}
.zone-tab {
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  padding: 5px 14px;
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--text-muted);
  background: transparent;
  cursor: pointer;
}
.zone-tab.active {
  color: var(--bg-base);
  background: var(--cyan);
  border-color: var(--cyan);
}

/* ── Stat cards ───────────────────────────────────────────────────────────── */
.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: 2px solid var(--cyan-dim);
  border-radius: 4px;
  padding: 14px 16px 12px;
  position: relative;
  overflow: hidden;
  transition: border-top-color 0.3s;
}
.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  opacity: 0.4;
}
.stat-card.red   { border-top-color: var(--red); }
.stat-card.yellow{ border-top-color: var(--yellow); }
.stat-card.green { border-top-color: var(--green); }

.stat-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 2px;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 6px;
}
.stat-value {
  font-family: 'Orbitron', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--text-bright);
  line-height: 1;
}
.stat-value .unit {
  font-size: 0.7rem;
  color: var(--text-muted);
  font-weight: 400;
}
.stat-sub {
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.78rem;
  margin-top: 6px;
}
.stat-sub.green  { color: var(--green); }
.stat-sub.yellow { color: var(--yellow); }
.stat-sub.red    { color: var(--red); }
.stat-bar-bg {
  background: #1a2535;
  border-radius: 2px;
  height: 4px;
  margin-top: 8px;
  overflow: hidden;
}
.stat-bar-fill {
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--cyan-dim), var(--cyan));
  transition: width 0.8s ease;
}

/* ── Section headers ─────────────────────────────────────────────────────── */
.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.section-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-bright);
}
.section-badge {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem;
  padding: 2px 8px;
  border-radius: 2px;
  letter-spacing: 1px;
}
.badge-live  { background: #00255a; color: var(--cyan); border: 1px solid var(--cyan); }
.badge-alert { background: #2d1a00; color: var(--yellow); border: 1px solid var(--yellow); }
.badge-crit  { background: #2d0014; color: var(--red);    border: 1px solid var(--red); animation: blink 1s infinite; }

/* ── Video panel ─────────────────────────────────────────────────────────── */
.video-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}
.video-panel img {
  width: 100%;
  display: block;
}
.video-overlay-bar {
  background: rgba(6,11,16,0.92);
  padding: 8px 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-top: 1px solid var(--border);
}

/* ── Zone density bar rows ────────────────────────────────────────────────── */
.zone-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 9px;
}
.zone-row-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.68rem;
  color: var(--text-muted);
  width: 70px;
  flex-shrink: 0;
}
.zone-bar-bg {
  flex: 1;
  background: #111e2c;
  border-radius: 2px;
  height: 8px;
  overflow: hidden;
}
.zone-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.6s ease, background 0.4s;
}
.zone-count {
  font-family: 'Orbitron', sans-serif;
  font-size: 0.65rem;
  width: 24px;
  text-align: right;
  color: var(--text-bright);
}
.zone-badge {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem;
  padding: 1px 6px;
  border-radius: 2px;
  width: 46px;
  text-align: center;
}
.zone-badge.GREEN  { background:#002a15; color:var(--green);  border:1px solid var(--green); }
.zone-badge.YELLOW { background:#2a1e00; color:var(--yellow); border:1px solid var(--yellow); }
.zone-badge.RED    { background:#2a0010; color:var(--red);    border:1px solid var(--red); animation:blink 0.8s infinite; }

/* ── Alert log ────────────────────────────────────────────────────────────── */
.alert-log {
  background: #080e14;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px;
  max-height: 180px;
  overflow-y: auto;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.68rem;
}
.alert-log::-webkit-scrollbar { width: 4px; }
.alert-log::-webkit-scrollbar-thumb { background: var(--border); }
.alert-row { padding: 3px 0; border-bottom: 1px solid #121c28; }
.alert-row.red    { color: var(--red); }
.alert-row.yellow { color: var(--yellow); }
.alert-row.normal { color: var(--green); }

/* ── Streamlit widget overrides ───────────────────────────────────────────── */
div[data-baseweb="slider"] div { background: var(--cyan) !important; }
div[data-testid="stSlider"] label { color: var(--text-muted) !important; font-family: 'Share Tech Mono',monospace; font-size:0.7rem; }
div[data-testid="stCheckbox"] label { color: var(--text-bright) !important; font-family: 'Rajdhani',sans-serif; }
div[data-baseweb="input"] input {
  background: var(--bg-card) !important;
  border-color: var(--border) !important;
  color: var(--text-bright) !important;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem !important;
}
div[data-testid="stButton"] > button {
  background: transparent !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 2px !important;
  font-size: 0.75rem !important;
  text-transform: uppercase;
  border-radius: 3px !important;
  transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
  background: var(--cyan) !important;
  color: var(--bg-base) !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
  border-color: var(--red) !important;
  color: var(--red) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: var(--red) !important;
  color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "running":       False,
        "frame_bytes":   None,
        "heatmap_bytes": None,
        "result":        None,
        "history":       deque(maxlen=60),   # (timestamp, total)
        "alert_log":     deque(maxlen=50),
        "frame_q":       queue.Queue(maxsize=2),
        "worker_thread": None,
        "stop_event":    threading.Event(),   # ✅ FIX: stop flag for worker thread
        "start_time":    None,
        "total_peak":    0,
        "frames_done":   0,
        "show_heatmap":  False,
        "show_zones":    True,
        "show_boxes":    True,
        "interval":      3.0,
        "youtube_url":   "https://www.youtube.com/watch?v=1EiC9bvVGnk",
        "model_path":    "yolov8n.pt",
        "tg_token":      "",
        "tg_chat_id":    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
S = st.session_state   # shorthand


# ─────────────────────────────────────────────────────────────────────────────
# Background worker thread
# ─────────────────────────────────────────────────────────────────────────────

def _worker(youtube_url: str, interval: float, model_path: str,
            show_heatmap: bool, show_zones: bool, show_boxes: bool,
            tg_token: str, tg_chat_id: str,
            q: queue.Queue, stop_event: threading.Event):  # ✅ FIX: accept Event

    try:
        # handle YouTube vs local file
        if youtube_url.startswith("http"):
            stream_url = get_youtube_stream_url(youtube_url)
        else:
            stream_url = youtube_url

        print("STREAM URL:", stream_url)

    except Exception as e:
        q.put({"error": str(e)})
        return

    try:
        cap      = open_video_capture(stream_url)
        detector = CrowdDetector(model_path=model_path)
        limiter  = RateLimiter(interval)
        alerted  = set()

        while not stop_event.is_set():   # ✅ FIX: use Event.is_set()

            ok, frame = read_frame(cap)
            if not ok:
                time.sleep(1)
                continue

            if not limiter.ready():
                time.sleep(0.05)
                continue

            frame  = resize_frame(frame, max_width=960)
            result = detector.detect(frame)

            display = result["annotated"].copy()
            if show_heatmap:
                display = blend_heatmap(display, result["heatmap_img"], alpha=0.45)

            frame_bytes   = encode_frame_to_jpeg(display)
            heatmap_bytes = encode_frame_to_jpeg(result["heatmap_img"])

            # Telegram alerts
            for msg in result["alerts"]:
                key = msg[:12]
                if key not in alerted:
                    send_telegram_alert(tg_token, tg_chat_id, msg)
                    alerted.add(key)

            for z, lvl in result["density"].items():
                if lvl != "RED":
                    alerted.discard(f"🔴 Zone {z}")

            payload = {
                "result":        result,
                "frame_bytes":   frame_bytes,
                "heatmap_bytes": heatmap_bytes,
                "ts":            datetime.now().strftime("%H:%M:%S"),
            }

            if not q.full():
                q.put(payload)

        cap.release()

    except Exception as e:
        q.put({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Helper: HTML stat card
# ─────────────────────────────────────────────────────────────────────────────

def stat_card(label: str, value: str, sub: str = "", sub_class: str = "green",
              card_class: str = "", bar_pct: float = 0.0, unit: str = "") -> str:
    bar_html = ""
    if bar_pct > 0:
        bar_html = f"""
        <div class="stat-bar-bg">
          <div class="stat-bar-fill" style="width:{min(bar_pct,100):.1f}%"></div>
        </div>"""
    return f"""
    <div class="stat-card {card_class}">
      <div class="stat-label">{label}</div>
      <div class="stat-value">{value}<span class="unit">{unit}</span></div>
      <div class="stat-sub {sub_class}">{sub}</div>
      {bar_html}
    </div>"""


def zone_bar_row(zone: str, count: int, density: str, max_count: int = 30) -> str:
    pct   = min(count / max(max_count, 1) * 100, 100)
    colour_map = {"GREEN": "#00ff6e", "YELLOW": "#ffc107", "RED": "#ff2d55"}
    bg_map     = {"GREEN": "#00ff6e33", "YELLOW": "#ffc10733", "RED": "#ff2d5533"}
    col = colour_map.get(density, "#00e5ff")
    bg  = bg_map.get(density, "#00e5ff33")
    names = {"TL": "Zone TL", "TR": "Zone TR", "BL": "Zone BL", "BR": "Zone BR"}
    return f"""
    <div class="zone-row">
      <div class="zone-row-label">{names.get(zone, zone)}</div>
      <div class="zone-bar-bg">
        <div class="zone-bar-fill" style="width:{pct:.1f}%; background:{col}; box-shadow:0 0 6px {bg};"></div>
      </div>
      <div class="zone-count">{count}</div>
      <div class="zone-badge {density}">{density}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',sans-serif;font-size:0.9rem;font-weight:700;
                color:#00e5ff;letter-spacing:3px;padding:14px 0 4px;">
      ⚙ CONFIGURATION
    </div>
    <div style="height:1px;background:#1a3040;margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)

    S.youtube_url = st.text_input("YouTube URL / Video Path", S.youtube_url)
    S.interval    = st.slider("Processing Interval (sec)", 1.0, 10.0, S.interval, 0.5)
    S.model_path  = st.selectbox("YOLOv8 Model",
                                  ["yolov8n.pt","yolov8s.pt","yolov8m.pt"],
                                  index=["yolov8n.pt","yolov8s.pt","yolov8m.pt"].index(S.model_path))

    st.markdown("""<div style="height:1px;background:#1a3040;margin:14px 0 10px;"></div>""",
                unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                   color:#5a7a8a;letter-spacing:2px;margin-bottom:8px;">OVERLAY OPTIONS</div>""",
                unsafe_allow_html=True)
    S.show_heatmap = st.checkbox("🌡️  Show Heatmap Overlay",  S.show_heatmap)
    S.show_zones   = st.checkbox("🔲  Show Zone Grid",         S.show_zones)
    S.show_boxes   = st.checkbox("📦  Show Bounding Boxes",    S.show_boxes)

    st.markdown("""<div style="height:1px;background:#1a3040;margin:14px 0 10px;"></div>""",
                unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                   color:#5a7a8a;letter-spacing:2px;margin-bottom:8px;">DENSITY THRESHOLDS</div>""",
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;font-size:0.8rem;line-height:2;">
      <span style="color:#00ff6e;">● Green</span>  : &lt; 10 people<br>
      <span style="color:#ffc107;">● Yellow</span> : 10–25 people<br>
      <span style="color:#ff2d55;">● Red</span>    : &gt; 25 people
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="height:1px;background:#1a3040;margin:14px 0 10px;"></div>""",
                unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                   color:#5a7a8a;letter-spacing:2px;margin-bottom:8px;">TELEGRAM ALERTS (optional)</div>""",
                unsafe_allow_html=True)
    S.tg_token   = st.text_input("Bot Token",   S.tg_token,   type="password")
    S.tg_chat_id = st.text_input("Chat ID",     S.tg_chat_id)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶  START", disabled=S.running):
            S.running    = True
            S.start_time = time.time()
            S.frame_q    = queue.Queue(maxsize=2)
            S.stop_event = threading.Event()   # ✅ FIX: fresh event each start
            t = threading.Thread(
                target=_worker,
                args=(S.youtube_url, S.interval, S.model_path,
                      S.show_heatmap, S.show_zones, S.show_boxes,
                      S.tg_token, S.tg_chat_id, S.frame_q,
                      S.stop_event),           # ✅ FIX: pass event to worker
                daemon=True
            )
            t.start()
            S.worker_thread = t
    with col_b:
        if st.button("⏹  STOP", type="primary", disabled=not S.running):
            S.running = False
            S.stop_event.set()   # ✅ FIX: signal worker to stop cleanly


# ─────────────────────────────────────────────────────────────────────────────
# Pull latest frame from queue
# ─────────────────────────────────────────────────────────────────────────────

if S.running:
    try:
        payload = S.frame_q.get_nowait()
        if "error" in payload:
            S.running = False
            st.error(f"Stream error: {payload['error']}")
        else:
            S.result        = payload["result"]
            S.frame_bytes   = payload["frame_bytes"]
            S.heatmap_bytes = payload["heatmap_bytes"]
            S.frames_done  += 1
            S.history.append((payload["ts"], payload["result"]["total"]))
            if payload["result"]["total"] > S.total_peak:
                S.total_peak = payload["result"]["total"]
            for a in payload["result"]["alerts"]:
                S.alert_log.appendleft({"ts": payload["ts"], "msg": a, "level": "red"})
    except queue.Empty:
        pass

# shorthand current result
R       = S.result or {}
zones   = R.get("zones",   {z: 0 for z in ["TL","TR","BL","BR"]})
density = R.get("density", {z: "GREEN" for z in ["TL","TR","BL","BR"]})
total   = R.get("total",   0)
alerts  = R.get("alerts",  [])

any_yellow = any(v == "YELLOW" for v in density.values())
any_red    = any(v == "RED"    for v in density.values())
if   any_red:    overall_status, status_cls = "CRITICAL", "red"
elif any_yellow: overall_status, status_cls = "WARNING",  "yellow"
else:            overall_status, status_cls = "NORMAL",   "green"

critical_zones = sum(1 for v in density.values() if v == "RED")

# uptime
uptime_str = "00:00"
if S.start_time and S.running:
    elapsed = int(time.time() - S.start_time)
    uptime_str = f"{elapsed//60:02d}:{elapsed%60:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV BAR
# ─────────────────────────────────────────────────────────────────────────────

clock_str = datetime.now().strftime("%H:%M:%S")
live_txt  = "LIVE FEED" if S.running else "OFFLINE"
live_col  = "#00ff6e" if S.running else "#5a7a8a"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-logo">
    <span class="pin">📍</span>
    CROWD<span style="color:#e8f4f8;font-weight:400;">SENSE</span>
  </div>
  <div class="topbar-nav">
    <div class="nav-item active">⬛ Overview</div>
    <div class="nav-item">📈 Analytics</div>
    <div class="nav-item">🔔 Alerts</div>
    <div class="nav-item">⚙ Settings</div>
  </div>
  <div class="topbar-right">
    <div class="live-indicator">
      <div class="live-dot" style="background:{live_col};"></div>
      <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                   color:{live_col};">{live_txt}</span>
    </div>
    <div style="font-family:'Orbitron',sans-serif;font-size:0.95rem;
                color:#00e5ff;letter-spacing:2px;">{clock_str}</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                color:#5a7a8a;">👤 admin</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:0.7rem;font-weight:700;
                letter-spacing:1px;border:1px solid #ff2d55;color:#ff2d55;
                padding:3px 10px;border-radius:2px;cursor:pointer;">LOGOUT</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ALERT TICKER
# ─────────────────────────────────────────────────────────────────────────────

if alerts:
    ticker_label_class = "red"
    ticker_label_text  = "🔴 CRITICAL"
    ticker_msgs = "  ●  ".join(alerts)
elif any_yellow:
    ticker_label_class = ""
    ticker_label_text  = "⚠ ALERT"
    ticker_msgs = f"Zone density elevated — monitor closely  ●  Uptime: {uptime_str}"
else:
    ticker_label_class = ""
    ticker_label_text  = "⚠ ALERTS"
    ticker_msgs = ("All zones within safe parameters  ●  "
                   f"System running normally  ●  Uptime: {uptime_str}  ●  "
                   f"Frames processed: {S.frames_done}")

st.markdown(f"""
<div class="alert-ticker">
  <div class="ticker-label {ticker_label_class}">{ticker_label_text}</div>
  <div class="ticker-scroll">
    <div class="ticker-inner">{ticker_msgs}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ZONE FILTER TABS  (cosmetic)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="zone-tabs">
  <div class="zone-tab active">All Zones</div>
  <div class="zone-tab">Zone TL</div>
  <div class="zone-tab">Zone TR</div>
  <div class="zone-tab">Zone BL</div>
  <div class="zone-tab">Zone BR</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAT CARDS ROW
# ─────────────────────────────────────────────────────────────────────────────

capacity   = 100
crowd_pct  = total / capacity * 100
avg_density = round(total / 4, 1) if total else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    sub_cls = "red" if total > 70 else ("yellow" if total > 40 else "green")
    st.markdown(stat_card(
        "TOTAL CROWD", f"{total:,}", f"↑ Peak: {S.total_peak:,}",
        sub_class=sub_cls, bar_pct=crowd_pct,
        card_class=("red" if total > 70 else "")
    ), unsafe_allow_html=True)

with c2:
    st.markdown(stat_card(
        "AVG DENSITY", f"{avg_density}", sub=f"→ {overall_status}",
        sub_class=status_cls, unit=" p/zone",
        bar_pct=avg_density / 30 * 100
    ), unsafe_allow_html=True)

with c3:
    c_cls = "red" if critical_zones > 0 else "green"
    st.markdown(stat_card(
        "CRITICAL ZONES", f"{critical_zones}", f"/4 active zones",
        sub_class=c_cls, card_class=("red" if critical_zones > 0 else "green")
    ), unsafe_allow_html=True)

with c4:
    st.markdown(stat_card(
        "FRAMES DONE", f"{S.frames_done:,}", f"Every {S.interval:.0f}s interval",
        sub_class="green", bar_pct=min(S.frames_done / 100 * 100, 100)
    ), unsafe_allow_html=True)

with c5:
    st.markdown(stat_card(
        "UPTIME", uptime_str, "Session active" if S.running else "Stopped",
        sub_class="green" if S.running else "red"
    ), unsafe_allow_html=True)

with c6:
    status_icon = "🔴" if any_red else ("🟡" if any_yellow else "🟢")
    st.markdown(stat_card(
        "SYSTEM STATUS", overall_status, f"{status_icon} All sensors OK",
        sub_class=status_cls,
        card_class=status_cls
    ), unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT ROW  |  video  |  charts  |  zone density + alert log  |
# ─────────────────────────────────────────────────────────────────────────────

left, mid, right = st.columns([3, 2.2, 1.8])

# ── LEFT: Live Video ──────────────────────────────────────────────────────────
with left:
    # section header
    st.markdown(f"""
    <div class="section-header">
      <div class="section-title">📹 Live Video Feed</div>
      <div class="section-badge {'badge-live' if S.running else ''}" >
        {'● LIVE' if S.running else '◌ PAUSED'}
      </div>
    </div>""", unsafe_allow_html=True)

    video_placeholder = st.empty()

    if S.frame_bytes:
        b64 = base64.b64encode(S.frame_bytes).decode()
        video_placeholder.markdown(f"""
        <div class="video-panel">
          <img src="data:image/jpeg;base64,{b64}" />
          <div class="video-overlay-bar">
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#00e5ff;">
              TOTAL: <b style="font-family:'Orbitron',sans-serif;font-size:0.9rem;">{total}</b> PEOPLE
            </span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#5a7a8a;">
              {datetime.now().strftime("%H:%M:%S")}
            </span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                         color:var(--{'red' if any_red else 'yellow' if any_yellow else 'green'});">
              STATUS: {overall_status}
            </span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        video_placeholder.markdown("""
        <div class="video-panel" style="height:360px;display:flex;align-items:center;
             justify-content:center;flex-direction:column;gap:14px;">
          <div style="font-family:'Orbitron',sans-serif;font-size:2rem;color:#1a3040;">
            ◌
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                      color:#2a4050;letter-spacing:3px;">
            NO SIGNAL — START MONITORING
          </div>
        </div>""", unsafe_allow_html=True)

    # Heatmap (if toggle on)
    if S.show_heatmap and S.heatmap_bytes:
        st.markdown("""
        <div style="margin-top:8px;">
          <div class="section-header">
            <div class="section-title">🌡️ Density Heatmap</div>
          </div>
        </div>""", unsafe_allow_html=True)
        b64h = base64.b64encode(S.heatmap_bytes).decode()
        st.markdown(f"""
        <div class="video-panel">
          <img src="data:image/jpeg;base64,{b64h}" />
        </div>""", unsafe_allow_html=True)


# ── MIDDLE: Crowd flow chart (sparkline built in HTML/JS) ─────────────────────
with mid:
    st.markdown(f"""
    <div class="section-header">
      <div class="section-title">📊 Crowd Flow — Live</div>
      <div class="section-badge badge-live">● LIVE</div>
    </div>""", unsafe_allow_html=True)

    # Build chart data
    hist_labels = [h[0] for h in S.history]
    hist_values = [h[1] for h in S.history]

    # Encode as JSON-safe strings
    labels_js = str(hist_labels).replace("'", '"')
    values_js = str(hist_values)

    chart_html = f"""
    <div style="background:#0d1520;border:1px solid #1a3040;border-radius:4px;padding:14px;">
      <canvas id="crowdChart" height="180"></canvas>
    </div>

    <!-- Zone Density mini-bars below chart -->
    <div style="margin-top:12px;">
      <div class="section-header" style="margin-bottom:8px;">
        <div class="section-title">🗺️ Zone Density</div>
        <div class="section-badge {'badge-crit' if critical_zones > 0 else 'badge-alert'}">
          {'🔴 ' + str(critical_zones) + ' CRITICAL' if critical_zones > 0 else '✓ NORMAL'}
        </div>
      </div>
      {''.join(zone_bar_row(z, zones.get(z,0), density.get(z,'GREEN'), max(total,1)) for z in ["TL","TR","BL","BR"])}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script>
    (function() {{
      const canvas = document.getElementById('crowdChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');

      const labels = {labels_js};
      const data   = {values_js};

      // destroy old instance
      if (window._crowdChart) {{ window._crowdChart.destroy(); }}

      window._crowdChart = new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: labels,
          datasets: [{{
            label: 'People',
            data: data,
            borderColor: '#00e5ff',
            backgroundColor: 'rgba(0,229,255,0.08)',
            pointBackgroundColor: '#00e5ff',
            pointRadius: 3,
            pointHoverRadius: 6,
            borderWidth: 2,
            fill: true,
            tension: 0.4,
          }}]
        }},
        options: {{
          responsive: true,
          animation: {{ duration: 400 }},
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: '#0d1520',
              borderColor: '#00e5ff',
              borderWidth: 1,
              titleColor: '#00e5ff',
              bodyColor: '#e8f4f8',
              titleFont: {{ family: 'Share Tech Mono', size: 11 }},
              bodyFont:  {{ family: 'Rajdhani', size: 12 }},
            }}
          }},
          scales: {{
            x: {{
              ticks: {{ color: '#3a5a6a', font: {{ family: 'Share Tech Mono', size: 9 }},
                        maxTicksLimit: 8 }},
              grid:  {{ color: '#111e2c' }}
            }},
            y: {{
              beginAtZero: true,
              ticks: {{ color: '#3a5a6a', font: {{ family: 'Share Tech Mono', size: 9 }} }},
              grid:  {{ color: '#111e2c' }}
            }}
          }}
        }}
      }});
    }})();
    </script>
    """
    st_html(chart_html, height=560)


# ── RIGHT: Alert log + zone status indicators ──────────────────────────────────
with right:
    st.markdown(f"""
    <div class="section-header">
      <div class="section-title">🔔 Alert Log</div>
      <div class="section-badge {'badge-crit' if alerts else ''}">
        {'ACTIVE' if alerts else 'CLEAR'}
      </div>
    </div>""", unsafe_allow_html=True)

    # Build alert log HTML
    log_rows = ""
    for entry in list(S.alert_log)[:20]:
        lvl = entry.get("level","normal")
        log_rows += f"""<div class="alert-row {lvl}">[{entry['ts']}] {entry['msg']}</div>"""
    if not log_rows:
        log_rows = '<div class="alert-row normal">[SYSTEM] All zones nominal.</div>'

    st.markdown(f'<div class="alert-log">{log_rows}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Zone status indicator grid ─────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="section-title">⬛ Zone Grid Status</div>
    </div>""", unsafe_allow_html=True)

    def zone_indicator(z, count, den):
        col_map = {"GREEN": "#00ff6e", "YELLOW": "#ffc107", "RED": "#ff2d55"}
        bg_map  = {"GREEN": "#001a0d", "YELLOW": "#1a1000", "RED": "#1a000a"}
        col = col_map.get(den, "#00e5ff")
        bg  = bg_map.get(den, "#0d1520")
        anim = "animation:blink 0.8s infinite;" if den=="RED" else ""
        return f"""
        <div style="background:{bg};border:1px solid {col};border-radius:4px;
                    padding:12px;text-align:center;{anim}">
          <div style="font-family:'Orbitron',sans-serif;font-size:0.6rem;
                      color:#5a7a8a;letter-spacing:2px;margin-bottom:4px;">ZONE {z}</div>
          <div style="font-family:'Orbitron',sans-serif;font-size:1.6rem;
                      font-weight:700;color:{col};line-height:1;">{count}</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
                      color:{col};margin-top:4px;letter-spacing:1px;">{den}</div>
        </div>"""

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(zone_indicator("TL", zones.get("TL",0), density.get("TL","GREEN")),
                    unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(zone_indicator("BL", zones.get("BL",0), density.get("BL","GREEN")),
                    unsafe_allow_html=True)
    with g2:
        st.markdown(zone_indicator("TR", zones.get("TR",0), density.get("TR","GREEN")),
                    unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(zone_indicator("BR", zones.get("BR",0), density.get("BR","GREEN")),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Session summary ────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header" style="margin-top:4px;">
      <div class="section-title">📋 Session Summary</div>
    </div>""", unsafe_allow_html=True)

    total_alerts = len(S.alert_log)
    st.markdown(f"""
    <div style="background:#0d1520;border:1px solid #1a3040;border-radius:4px;padding:12px;">
      <table style="width:100%;font-family:'Share Tech Mono',monospace;
                    font-size:0.68rem;border-collapse:collapse;">
        <tr style="border-bottom:1px solid #1a3040;">
          <td style="color:#5a7a8a;padding:5px 0;">Peak Count</td>
          <td style="color:#00e5ff;text-align:right;font-family:'Orbitron',sans-serif;">
            {S.total_peak}
          </td>
        </tr>
        <tr style="border-bottom:1px solid #1a3040;">
          <td style="color:#5a7a8a;padding:5px 0;">Frames Done</td>
          <td style="color:#e8f4f8;text-align:right;">{S.frames_done}</td>
        </tr>
        <tr style="border-bottom:1px solid #1a3040;">
          <td style="color:#5a7a8a;padding:5px 0;">Total Alerts</td>
          <td style="color:{'#ff2d55' if total_alerts else '#00ff6e'};text-align:right;">
            {total_alerts}
          </td>
        </tr>
        <tr>
          <td style="color:#5a7a8a;padding:5px 0;">Model</td>
          <td style="color:#e8f4f8;text-align:right;">{S.model_path}</td>
        </tr>
      </table>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh while running
# ─────────────────────────────────────────────────────────────────────────────

if S.running:
    time.sleep(0.8)
    st.rerun()