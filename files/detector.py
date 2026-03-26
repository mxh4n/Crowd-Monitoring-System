"""
detector.py
-----------
YOLOv8-based person detection engine.
Handles detection, zone classification, density levels, and heatmap generation.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ── Density thresholds ────────────────────────────────────────────────────────
DENSITY_GREEN  = 10    # 0-9   → GREEN  (safe)
DENSITY_YELLOW = 25    # 10-25 → YELLOW (moderate)
                       # >25   → RED    (critical)

ZONE_LABELS = ["TL", "TR", "BL", "BR"]


def get_density_level(count: int) -> str:
    if count < DENSITY_GREEN:
        return "GREEN"
    elif count <= DENSITY_YELLOW:
        return "YELLOW"
    else:
        return "RED"


class CrowdDetector:
    """
    YOLOv8 wrapper that adds:
      - Person-only filtering (COCO class 0)
      - 2×2 grid zone assignment
      - Per-zone density level
      - Heatmap accumulation
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.4):
        print(f"[CrowdDetector] Loading model: {model_path}")
        self.model        = YOLO(model_path)
        self.conf         = conf_threshold
        self.heatmap      = None
        self.PERSON_CLASS = 0   # COCO index for 'person'

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ensure_heatmap(self, h, w):
        if self.heatmap is None or self.heatmap.shape[:2] != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)

    def _assign_zone(self, cx, cy, w, h):
        col = "L" if cx < w // 2 else "R"
        row = "T" if cy < h // 2 else "B"
        return row + col

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run inference on frame. Returns:
          total, zones, density, boxes, annotated, heatmap_img, alerts
        """
        h, w = frame.shape[:2]
        self._ensure_heatmap(h, w)

        results   = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes_raw = []
        zone_counts = {z: 0 for z in ZONE_LABELS}

        for box in results.boxes:
            if int(box.cls[0]) != self.PERSON_CLASS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_val        = float(box.conf[0])
            cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2
            boxes_raw.append((x1, y1, x2, y2, conf_val))
            zone_counts[self._assign_zone(cx, cy, w, h)] += 1
            # accumulate heatmap
            cv2.circle(self.heatmap, (cx, cy), radius=50, color=2.0, thickness=-1)

        total   = sum(zone_counts.values())
        density = {z: get_density_level(n) for z, n in zone_counts.items()}
        alerts  = [f"🔴 Zone {z} CRITICAL — {n} people!"
                   for z, n in zone_counts.items() if density[z] == "RED"]

        annotated   = self._draw_overlays(frame.copy(), boxes_raw, zone_counts, density)
        heatmap_img = self._render_heatmap(h, w)

        return dict(total=total, zones=zone_counts, density=density,
                    boxes=boxes_raw, annotated=annotated,
                    heatmap_img=heatmap_img, alerts=alerts)

    def reset_heatmap(self):
        if self.heatmap is not None:
            self.heatmap[:] = 0

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_overlays(self, frame, boxes, zone_counts, density):
        h, w   = frame.shape[:2]
        mid_x, mid_y = w // 2, h // 2
        COLOUR = {"GREEN": (0, 255, 80), "YELLOW": (0, 210, 255), "RED": (40, 40, 255)}

        # zone grid
        cv2.line(frame, (mid_x, 0), (mid_x, h), (60, 60, 60), 1)
        cv2.line(frame, (0, mid_y), (w, mid_y), (60, 60, 60), 1)

        corners = {"TL": (0,0,mid_x,mid_y), "TR": (mid_x,0,w,mid_y),
                   "BL": (0,mid_y,mid_x,h), "BR": (mid_x,mid_y,w,h)}

        for z, (zx1,zy1,zx2,zy2) in corners.items():
            col   = COLOUR[density[z]]
            count = zone_counts[z]
            ov    = frame.copy()
            cv2.rectangle(ov, (zx1,zy1), (zx2,zy2), col, -1)
            cv2.addWeighted(ov, 0.06, frame, 0.94, 0, frame)
            cv2.rectangle(frame, (zx1+2,zy1+2), (zx2-2,zy2-2), col, 2)
            cv2.putText(frame, f"{z}: {count} ({density[z]})",
                        (zx1+8, zy2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        for (x1,y1,x2,y2,cf) in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 180), 2)
            cv2.putText(frame, f"Person", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1)
        return frame

    def _render_heatmap(self, h, w):
        if self.heatmap is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        norm     = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return coloured
