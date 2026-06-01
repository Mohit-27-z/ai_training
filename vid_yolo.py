import cv2 as cv
import time
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH      = "/home/mohit/Downloads/traffic_vd.mp4"
WINDOW_SIZE     = (1920, 1080)          # (width, height) — you had it swapped
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]
CONFIDENCE      = 0.4

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

if not cap.isOpened():          # fix: isOPened → isOpened
    print("Error: video not found")
    exit()

prev_time        = 0
paused           = False
screenshot_count = 0
fps              = 0
frame            = None

print("Video is playing")
print("  SPACE = Pause / Resume")
print("  S     = Screenshot")
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
            continue                        # skip non-vehicle detections

        vehicle_count += 1
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        # Bounding box coords
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
                   (0, 0, 0),          # black text on coloured bg
                   thickness)

    # ── HUD overlay ───────────────────────────────────────────────────────────
    curr_time = time.time()
    if not paused and (curr_time - prev_time) > 0:
        fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Semi-transparent dark bar at the top
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (340, 75), (0, 0, 0), cv.FILLED)
    cv.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv.putText(frame, f"FPS      : {fps:.1f}",
               (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv.putText(frame, f"Vehicles : {vehicle_count}",
               (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    if paused:
        cv.putText(frame, "PAUSED",
                   (WINDOW_SIZE[0] // 2 - 70, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

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
        filename = f"screenshot_{screenshot_count:03d}.png"
        cv.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        screenshot_count += 1

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv.destroyAllWindows()