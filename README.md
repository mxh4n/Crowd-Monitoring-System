# 🎯 CrowdSense — AI-Based Crowd Monitoring System

A real-time crowd monitoring dashboard powered by **YOLOv8** detection and a
**CrowdSense-style tactical dark UI** built with Streamlit + HTML/CSS/JS.

---

## 📁 Project Structure

```
crowd_monitor/
├── detector.py      # YOLOv8 detection engine + heatmap + zone logic
├── utils.py         # YouTube stream extraction, Telegram alerts, helpers
├── main.py          # Headless CLI runner (no browser needed)
├── dashboard.py     # Full Streamlit dashboard (recommended)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Install

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. (Windows only) install yt-dlp binary
pip install yt-dlp

# 4. First run auto-downloads yolov8n.pt (~6 MB) from ultralytics
```

---

## 🚀 Run the Dashboard

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

### Steps in the UI:
1. Paste a **YouTube URL** (live stream or regular video) in the sidebar
2. Set **processing interval** (3 sec recommended for CPU)
3. Toggle overlays (heatmap, zones, bounding boxes)
4. Click **▶ START**

---

## 🖥️ Run Headless (CLI)

```bash
# YouTube live stream
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Local video file
python main.py --file "your_video.mp4"

# With OpenCV window
python main.py --url "..." --window

# Custom interval & model
python main.py --url "..." --interval 5 --model yolov8s.pt
```

---

## 📊 Dashboard Features

| Feature | Description |
|---|---|
| **Topbar** | Live clock, nav tabs, live/offline indicator |
| **Alert ticker** | Scrolling real-time alerts across the top |
| **6 stat cards** | Total crowd, avg density, critical zones, frames, uptime, status |
| **Live video** | YOLOv8 annotated feed with bounding boxes + zone overlays |
| **Heatmap** | Accumulated density heatmap (toggle in sidebar) |
| **Crowd flow chart** | Chart.js line chart — live history of people count |
| **Zone density bars** | Animated per-zone bars with GREEN/YELLOW/RED badges |
| **Zone grid** | 2×2 visual zone indicators with colour coding |
| **Alert log** | Timestamped log of all RED-zone events |
| **Session summary** | Peak count, frames done, total alerts |

---

## 🎨 Density Levels

| Colour | People per Zone | Meaning |
|---|---|---|
| 🟢 GREEN  | < 10  | Safe |
| 🟡 YELLOW | 10–25 | Monitor |
| 🔴 RED    | > 25  | Critical — alert fired |

---

## 📲 Telegram Alerts (Optional)

1. Message [@BotFather](https://t.me/botfather) → `/newbot` → copy token
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Enter both in the **sidebar** of the dashboard

---

## 🧠 Models

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | ~6 MB | Fastest | Good (default) |
| `yolov8s.pt` | ~22 MB | Fast | Better |
| `yolov8m.pt` | ~52 MB | Moderate | Best for CPU |

---

## 📦 All pip commands (manual)

```bash
pip install ultralytics
pip install opencv-python
pip install streamlit
pip install yt-dlp
pip install numpy
pip install requests
```

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| `yt-dlp` not found | `pip install yt-dlp` |
| Stream won't open | Try a different YouTube video/stream |
| Slow on CPU | Increase interval to 5–10s, use `yolov8n.pt` |
| No people detected | Lower confidence: edit `conf_threshold=0.3` in `detector.py` |
| Black heatmap | Needs a few frames to accumulate — wait 10–15 sec |
