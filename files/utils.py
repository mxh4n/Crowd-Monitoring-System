"""
utils.py
--------
Utility functions:
  - YouTube stream URL extraction via yt-dlp
  - Telegram alert sender (optional)
  - Frame resize / FPS helpers
"""

import subprocess
import time
import requests
import cv2
import numpy as np


# ── YouTube stream extraction ─────────────────────────────────────────────────

def get_youtube_stream_url(youtube_url: str) -> str:
    print(f"[utils] Extracting stream URL from: {youtube_url}")
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best", "-g", youtube_url],
            capture_output=True,
            text=True,
            timeout=10   # 🔥 reduce from 30 → 10
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        url = result.stdout.strip().split("\n")[0]
        print("[utils] Stream OK")
        return url

    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp timeout (YouTube too slow or blocked)")

    except Exception as e:
        raise RuntimeError(f"Failed to extract stream: {e}")


# ── OpenCV video capture helper ───────────────────────────────────────────────

def open_video_capture(url: str) -> cv2.VideoCapture:
    """Open an OpenCV VideoCapture from a URL or local path."""
    import os

    # FIX: resolve relative file paths against the script's own directory
    if not url.startswith("http"):
        if not os.path.isabs(url):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate1 = os.path.join(script_dir, url)
            candidate2 = os.path.join(os.getcwd(), url)
            if os.path.exists(candidate1):
                url = candidate1
                print(f"[utils] Resolved path: {url}")
            elif os.path.exists(candidate2):
                url = candidate2
                print(f"[utils] Resolved path (cwd): {url}")
            else:
                raise RuntimeError(
                    f"Video file not found: '{url}'\n"
                    f"Tried:\n  {candidate1}\n  {candidate2}\n"
                    f"Tip: use the full path e.g. C:/Users/Mohan/Downloads/files/test_crowd.mp4"
                )
        elif not os.path.exists(url):
            raise RuntimeError(f"Video file not found: '{url}'")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    time.sleep(2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {url[:80]}")

    print("[utils] VideoCapture opened successfully.")
    return cap


def read_frame(cap: cv2.VideoCapture):
    ret, frame = cap.read()

    # 🔥 Retry logic for YouTube streams
    if not ret or frame is None:
        time.sleep(0.5)

        # try again
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[utils] Frame read failed")
            return False, None

    return True, frame


def resize_frame(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    """Downscale frame to max_width while keeping aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale  = max_width / w
    new_wh = (max_width, int(h * scale))
    return cv2.resize(frame, new_wh, interpolation=cv2.INTER_AREA)


# ── Telegram alert (optional) ─────────────────────────────────────────────────

def send_telegram_alert(token: str, chat_id: str, message: str) -> bool:
    """
    Send a text message via Telegram Bot API.

    Args:
        token:   Bot token from @BotFather.
        chat_id: Recipient chat / group ID.
        message: Alert text to send.

    Returns:
        True on success, False on failure.
    """
    if not token or not chat_id:
        print("[Telegram] Token or chat_id not set — skipping alert.")
        return False
    try:
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        resp    = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"[Telegram] Alert sent: {message}")
            return True
        else:
            print(f"[Telegram] Failed ({resp.status_code}): {resp.text}")
            return False
    except Exception as e:
        print(f"[Telegram] Exception: {e}")
        return False


# ── Misc helpers ──────────────────────────────────────────────────────────────

def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR frame to JPEG bytes for Streamlit display."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def blend_heatmap(frame: np.ndarray, heatmap_img: np.ndarray,
                  alpha: float = 0.45) -> np.ndarray:
    """
    Alpha-blend a JET colourmap heatmap over a BGR frame.
    Both must be the same size.
    """
    if frame.shape[:2] != heatmap_img.shape[:2]:
        heatmap_img = cv2.resize(heatmap_img, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - alpha, heatmap_img, alpha, 0)


class RateLimiter:
    """Allow at most one action every `interval` seconds."""

    def __init__(self, interval: float):
        self.interval  = interval
        self._last     = 0.0

    def ready(self) -> bool:
        now = time.time()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False
