import torch
import cv2
import numpy as np
import math
import sys
from ultralytics import YOLO

# ----------------------------
# ADD MIDAS PATH
# ----------------------------
sys.path.append("/home/uav1/MiDaS")

from midas.model_loader import load_model

# ----------------------------
# DEVICE (CPU for Pi)
# ----------------------------
device = torch.device("cpu")

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
yolo_model = YOLO("models/yolo26n.pt")

# ----------------------------
# LOAD MIDAS MODEL
# ----------------------------
midas, transform, net_w, net_h = load_model(
    device,
    model_path=None,
    model_type="dpt_levit_224",
    optimize=False
)

midas.eval()

# ----------------------------
# CAMERA PARAMETERS
# ----------------------------
image_width = 640
image_height = 480
fov = 65  # Approx webcam FOV

focal_length = image_width / (2 * math.tan(math.radians(fov / 2)))

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, image_width)
cap.set(4, image_height)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("Robotic Eye Started... Press Q to Quit")

frame_count = 0
depth_norm = None

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ----------------------------
    # YOLO DETECTION
    # ----------------------------
    results = yolo_model(frame)[0]

    # ----------------------------
    # DEPTH ESTIMATION (Every 3 Frames)
    # ----------------------------
    if frame_count % 3 == 1 or depth_norm is None:

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = transform({"image": img_rgb})["image"]
        input_batch = torch.from_numpy(input_batch).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy().astype(np.float32)
        depth_map = np.nan_to_num(depth_map)

        min_val = np.min(depth_map)
        max_val = np.max(depth_map)

        if max_val - min_val > 1e-6:
            depth_norm = (depth_map - min_val) / (max_val - min_val)
        else:
            depth_norm = np.zeros_like(depth_map)

        # Invert so near = high
        depth_norm = 1.0 - depth_norm

    # ----------------------------
    # COMBINE YOLO + DEPTH + SIZE
    # ----------------------------
    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_norm.shape[1], x2)
        y2 = min(depth_norm.shape[0], y2)

        roi = depth_norm[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        depth_value = np.median(roi)

        # ----------------------------
        # DISTANCE ESTIMATION (Approximate)
        # ----------------------------
        min_cm = 30
        max_cm = 300
        distance_cm = min_cm + (1 - depth_value) * (max_cm - min_cm)

        # ----------------------------
        # SIZE ESTIMATION
        # ----------------------------
        pixel_height = y2 - y1
        pixel_width = x2 - x1

        real_height_cm = (pixel_height * distance_cm) / focal_length
        real_width_cm = (pixel_width * distance_cm) / focal_length

        # ----------------------------
        # DRAW OUTPUT
        # ----------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = (
            f"{label} | "
            f"D:{distance_cm:.1f}cm "
            f"H:{real_height_cm:.1f}cm "
            f"W:{real_width_cm:.1f}cm"
        )

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Robotic Eye - Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
