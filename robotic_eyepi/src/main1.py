import cv2
import torch
import timm
import numpy as np
import math
from ultralytics import YOLO

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cpu")

# ==================================================
# LOAD YOLO (LIGHTWEIGHT)
# ==================================================
yolo_model = YOLO("models/yolo26n.pt")

# ==================================================
# LOAD DEPTH MODEL (STABLE, NO HUB, NO MIDAS REPO)
# ==================================================
depth_model = timm.create_model(
    "dpt_hybrid_384",
    pretrained=True
)
depth_model.to(device)
depth_model.eval()

# ==================================================
# DEPTH PREPROCESS FUNCTION
# ==================================================
def depth_transform(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

# ==================================================
# CAMERA PARAMETERS
# ==================================================
W, H = 640, 360
FOV = 65  # approximate webcam HFOV
focal_length = W / (2 * math.tan(math.radians(FOV / 2)))

# ==================================================
# CAMERA SETUP (LOW LATENCY)
# ==================================================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

print("✅ Robotic Eye Started (Fast & Stable)")
print("Press Q to quit")

# ==================================================
# STATE VARIABLES
# ==================================================
frame_id = 0
depth_map = None

MIN_DIST = 30     # cm
MAX_DIST = 300    # cm

# ==================================================
# MAIN LOOP
# ==================================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_id += 1

    # ----------------------------------------------
    # YOLO OBJECT DETECTION (EVERY FRAME)
    # ----------------------------------------------
    results = yolo_model(frame, conf=0.4, verbose=False)[0]

    # ----------------------------------------------
    # DEPTH ESTIMATION (EVERY 12 FRAMES)
    # ----------------------------------------------
    if frame_id % 12 == 0 or depth_map is None:
        with torch.no_grad():
            inp = depth_transform(frame).to(device)
            depth = depth_model(inp)

            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = depth.cpu().numpy()
        depth_map = np.nan_to_num(depth_map)
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_map = 1.0 - depth_map  # near = high value

    # ----------------------------------------------
    # FUSION: YOLO + DEPTH + SIZE
    # ----------------------------------------------
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        depth_val = np.median(roi)

        # Distance estimation
        distance_cm = MIN_DIST + (1 - depth_val) * (MAX_DIST - MIN_DIST)

        # Size estimation
        pixel_h = y2 - y1
        pixel_w = x2 - x1

        real_h = (pixel_h * distance_cm) / focal_length
        real_w = (pixel_w * distance_cm) / focal_length

        # Draw results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = (
            f"{label} | "
            f"D:{distance_cm:.0f}cm "
            f"H:{real_h:.1f}cm "
            f"W:{real_w:.1f}cm"
        )

        cv2.putText(
            frame,
            text,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Robotic Eye (Fast & Stable)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==================================================
# CLEANUP
# ==================================================
cap.release()
cv2.destroyAllWindows()
