"""
Real-Time Multi-Object Detection HUD (AI Vision System)

Installation:
    pip install opencv-python ultralytics numpy
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = 'yolov8n.pt'  # Downloaded automatically by Ultralytics
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
HUD_COLOR = (0, 255, 255)
CROSSHAIR_COLOR = (180, 255, 255)
BORDER_COLOR = (60, 60, 60)
FPS_SMOOTHING = 0.9

# Class color mapping
CLASS_COLORS = {
    'person': (255, 120, 0),      # Blue
    'bottle': (180, 0, 180),      # Purple
    'cell phone': (0, 200, 0),    # Green
}
DEFAULT_BOX_COLOR = (200, 200, 200)

# --- FUNCTIONS ---
def draw_hud(frame):
    h, w = frame.shape[:2]
    # Border/frame
    cv2.rectangle(frame, (8, 8), (w-8, h-8), BORDER_COLOR, 2, cv2.LINE_AA)
    # Scanning line
    scan_y = int((time.time() * 80) % (h-32) + 16)
    cv2.line(frame, (16, scan_y), (w-16, scan_y), HUD_COLOR, 1, cv2.LINE_AA)
    # Crosshair
    cx, cy = w // 2, h // 2
    cv2.drawMarker(frame, (cx, cy), CROSSHAIR_COLOR, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1, line_type=cv2.LINE_AA)

def get_box_color(cls_name):
    return CLASS_COLORS.get(cls_name, DEFAULT_BOX_COLOR)

def draw_label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(frame, (x, y-th-6), (x+tw+6, y), (0,0,0), -1)
    cv2.putText(frame, text, (x+3, y-3), FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

def smooth_boxes(prev_boxes, curr_boxes, alpha=0.5):
    if prev_boxes is None or len(prev_boxes) != len(curr_boxes):
        return curr_boxes
    smoothed = []
    for pb, cb in zip(prev_boxes, curr_boxes):
        x1 = int(pb[0]*alpha + cb[0]*(1-alpha))
        y1 = int(pb[1]*alpha + cb[1]*(1-alpha))
        x2 = int(pb[2]*alpha + cb[2]*(1-alpha))
        y2 = int(pb[3]*alpha + cb[3]*(1-alpha))
        smoothed.append((x1, y1, x2, y2, cb[4], cb[5], cb[6]))
    return smoothed

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    model = YOLO(MODEL_PATH)
    prev_boxes = None
    prev_time = time.time()
    fps = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        t0 = time.time()
        results = model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            boxes.append((x1, y1, x2, y2, conf, cls_name, cls_id))
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        boxes = [b for b in boxes if b[4] > 0.4]
        boxes = smooth_boxes(prev_boxes, boxes, alpha=0.7) if prev_boxes else boxes
        prev_boxes = boxes
        for x1, y1, x2, y2, conf, cls_name, _ in boxes:
            color = get_box_color(cls_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{cls_name} {conf:.2f}"
            draw_label(frame, label, x1, y1, color)
        draw_hud(frame)
        t1 = time.time()
        dt = t1 - t0
        fps = fps * FPS_SMOOTHING + (1.0/dt) * (1-FPS_SMOOTHING) if fps else 1.0/dt
        # HUD text
        cv2.putText(frame, f"FPS: {fps:.1f}", (16, 32), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Inference: {dt*1000:.0f} ms", (16, 60), FONT, 0.6, (200,255,200), 1, cv2.LINE_AA)
        cv2.putText(frame, "AI VISION ACTIVE", (16, 90), FONT, 0.7, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow('AI Vision HUD', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
