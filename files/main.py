"""
main.py
-------
Headless CLI version of the crowd monitoring system.
Reads from a YouTube stream (or local video file), runs YOLO detection,
and prints results to the console every N seconds.

Usage:
    python main.py --url "https://www.youtube.com/watch?v=YOUR_ID"
    python main.py --file "local_video.mp4"
"""

import argparse
import time
import cv2

from detector import CrowdDetector
from utils    import (get_youtube_stream_url, open_video_capture,
                      read_frame, resize_frame, RateLimiter,
                      send_telegram_alert)

# ── Telegram config (fill in if you want alerts) ──────────────────────────────
TELEGRAM_TOKEN   = ""   # e.g. "7123456789:AAF..."
TELEGRAM_CHAT_ID = ""   # e.g. "-100123456789"


def run(source_url: str, interval: float = 3.0,
        model: str = "yolov8n.pt", show_window: bool = False):
    """
    Main processing loop.

    Args:
        source_url:  Direct video URL or local path.
        interval:    Seconds between processed frames.
        model:       YOLOv8 weights path.
        show_window: If True, open an OpenCV display window.
    """
    detector = CrowdDetector(model_path=model)
    cap      = open_video_capture(source_url)
    timer    = RateLimiter(interval)
    alerted  = set()   # track which zones already triggered Telegram

    print("\n[main] Starting crowd monitoring loop. Press Ctrl-C to stop.\n")

    try:
        while True:
            ok, frame = read_frame(cap)
            if not ok:
                print("[main] Stream ended or frame read failed. Retrying...")
                time.sleep(2)
                continue

            if not timer.ready():
                time.sleep(0.1)
                continue

            frame = resize_frame(frame, max_width=960)
            result = detector.detect(frame)

            # ── Console output ────────────────────────────────────────────────
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  Total people: {result['total']:>4}  |  "
                  f"TL={result['zones']['TL']}({result['density']['TL']})  "
                  f"TR={result['zones']['TR']}({result['density']['TR']})  "
                  f"BL={result['zones']['BL']}({result['density']['BL']})  "
                  f"BR={result['zones']['BR']}({result['density']['BR']})")

            # ── Alerts ────────────────────────────────────────────────────────
            for alert_msg in result["alerts"]:
                print(f"  ⚠️  ALERT: {alert_msg}")
                zone_key = alert_msg[:8]
                if zone_key not in alerted:
                    send_telegram_alert(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, alert_msg)
                    alerted.add(zone_key)

            # Reset seen alerts when zone clears
            for z, lvl in result["density"].items():
                if lvl != "RED":
                    alerted.discard(f"🔴 Zone {z}")

            # ── Optional OpenCV window ─────────────────────────────────────────
            if show_window:
                cv2.imshow("Crowd Monitor", result["annotated"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[main] Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Crowd Monitoring System")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="YouTube URL")
    group.add_argument("--file", help="Local video file path")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between processed frames (default 3)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 weights (default yolov8n.pt)")
    parser.add_argument("--window", action="store_true",
                        help="Show OpenCV display window")
    args = parser.parse_args()

    if args.url:
        stream_url = get_youtube_stream_url(args.url)
    else:
        stream_url = args.file

    run(stream_url, interval=args.interval,
        model=args.model, show_window=args.window)
