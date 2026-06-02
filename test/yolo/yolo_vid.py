import cv2 as cv
import time
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH      = "traffic_vd.mp4"
WINDOW_SIZE     = (1920, 1080)          # (width, height)
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]
CONFIDENCE      = 0.5

# ── Class colours (BGR) ───────────────────────────────────────────────────────
CLASS_COLORS = {
    "car":        (0, 255,   0),   # green
    "truck":      (0, 165, 255),   # orange
    "bus":        (255,   0,   0), # blue
    "motorcycle": (0,   0, 255),   # red
}
DEFAULT_COLOR = (200, 200, 200)

# ── Init ──────────────────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")
cap   = cv.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: video not found")
    exit()

prev_time    = 0
paused       = False
frame_number = 0          # ← NEW: frame counter
fps          = 0
frame        = None

print("Video is playing")
print("  SPACE = Pause / Resume")
print("  S     = Screenshot  (saves screenshot.jpg)")
print("  Q     = Quit")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:

    # ── Read frame ────────────────────────────────────────────────────────────
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break
        frame = cv.resize(frame, WINDOW_SIZE)
        frame_number += 1          # increment only on new frames, not while paused

    if frame is None:
        continue

    # ── YOLO inference ────────────────────────────────────────────────────────
    results = model(frame, conf=CONFIDENCE, verbose=False)[0]

    # ── Draw detections ───────────────────────────────────────────────────────
    vehicle_count = 0

    for box in results.boxes:
        cls_id     = int(box.cls[0].item())
        class_name = model.names[cls_id]
        confidence = box.conf[0].item()

        if class_name not in VEHICLE_CLASSES:
            continue

        vehicle_count += 1
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # ── Box ───────────────────────────────────────────────────────────────
        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # ── Label background ──────────────────────────────────────────────────
        label      = f"{class_name}  {confidence:.2f}"
        font       = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 2
        (text_w, text_h), baseline = cv.getTextSize(label, font, font_scale, thickness)

        cv.rectangle(frame,
                     (x1, y1 - text_h - baseline - 4),
                     (x1 + text_w + 4, y1),
                     color, cv.FILLED)

        # ── Label text ────────────────────────────────────────────────────────
        cv.putText(frame, label,
                   (x1 + 2, y1 - baseline - 2),
                   font, font_scale,
                   (0, 0, 0),
                   thickness)

    # ── FPS calc ──────────────────────────────────────────────────────────────
    curr_time = time.time()
    if not paused and (curr_time - prev_time) > 0:
        fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # =========================================================================
    # ── HUD ──────────────────────────────────────────────────────────────────
    # =========================================================================

    # ── 1. Blue title bar (full width, top) ───────────────────────────────────
    #   RGB(20, 90, 190)  →  BGR(190, 90, 20)  — rich royal blue
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (WINDOW_SIZE[0], 55), (190, 90, 20), cv.FILLED)
    cv.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

    title      = "TRAFFIC DETECTOR"
    title_font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv.getTextSize(title, title_font, 1.1, 2)
    title_x    = (WINDOW_SIZE[0] - tw) // 2   # horizontally centered
    cv.putText(frame, title, (title_x, 38), title_font, 1.1, (255, 255, 255), 2)

    # ── 2. Info panel (below title bar) ───────────────────────────────────────
    overlay2 = frame.copy()
    cv.rectangle(overlay2, (0, 55), (300, 125), (0, 0, 0), cv.FILLED)
    cv.addWeighted(overlay2, 0.50, frame, 0.50, 0, frame)

    cv.putText(frame, f"FPS      : {fps:.1f}",
               (10, 82), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # ── Vehicle count: green if ≤5, red if >5 ─────────────────────────────────
    v_color = (0, 255, 0) if vehicle_count <= 5 else (0, 0, 255)
    cv.putText(frame, f"Vehicles : {vehicle_count}",
               (10, 114), cv.FONT_HERSHEY_SIMPLEX, 0.65, v_color, 2)

    # ── 3. Frame number — bottom right ────────────────────────────────────────
    fn_label = f"Frame: {frame_number}"
    (fnw, fnh), _ = cv.getTextSize(fn_label, cv.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    fx = WINDOW_SIZE[0] - fnw - 15
    fy = WINDOW_SIZE[1] - 15
    # small dark background so text is readable over any content
    cv.rectangle(frame,
                 (fx - 8,     fy - fnh - 6),
                 (fx + fnw + 8, fy + 6),
                 (0, 0, 0), cv.FILLED)
    cv.putText(frame, fn_label, (fx, fy),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    # ── PAUSED banner ─────────────────────────────────────────────────────────
    if paused:
        cv.putText(frame, "PAUSED",
                   (WINDOW_SIZE[0] // 2 - 90, WINDOW_SIZE[1] // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    # ── Show ──────────────────────────────────────────────────────────────────
    cv.imshow("Vehicle Detection", frame)

    # ── Key handling ──────────────────────────────────────────────────────────
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit.")
        break

    elif key == ord(' '):
        paused = not paused
        print("Paused" if paused else "Resumed")

    elif key == ord('s'):
        # ── 4. Screenshot always saves as screenshot.jpg ──────────────────────
        cv.imwrite("screenshot.jpg", frame)
        print("Screenshot saved: screenshot.jpg")

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv.destroyAllWindows()