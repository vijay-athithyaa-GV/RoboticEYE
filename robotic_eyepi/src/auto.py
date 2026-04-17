import torch
import cv2
import numpy as np
import math
import sys
import time




from ultralytics import YOLO

import sys
sys.path.append("/home/uav1")   # or your actual username
import carcontrol as car
# ----------------------------
# ADD MIDAS PATH
# ----------------------------
sys.path.append("/home/uav1/MiDaS")
from midas.model_loader import load_model

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cpu")

# ----------------------------
# LOAD MODELS
# ----------------------------
yolo_model = YOLO("models/yolo26n.pt")

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
fov = 65

focal_length = image_width / (2 * math.tan(math.radians(fov / 2)))

# ----------------------------
# CONTROL PARAMETERS
# ----------------------------
CENTER_THRESHOLD = 80
SAFE_DISTANCE = 40     # cm
STOP_DISTANCE = 25     # cm

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, image_width)
cap.set(4, image_height)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("🔥 Autonomous Phone Following Started... Press Q to Quit")

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
    phone_detected = False

    # ----------------------------
    # YOLO DETECTION
    # ----------------------------
    results = yolo_model(frame)[0]

    # ----------------------------
    # DEPTH ESTIMATION
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

        depth_norm = 1.0 - depth_norm

    frame_center_x = image_width // 2

    # ----------------------------
    # PROCESS DETECTIONS
    # ----------------------------
    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls]

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_norm.shape[1], x2)
        y2 = min(depth_norm.shape[0], y2)

        roi = depth_norm[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        depth_value = np.median(roi)

        # Distance estimation
        min_cm = 30
        max_cm = 300
        distance_cm = min_cm + (1 - depth_value) * (max_cm - min_cm)

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} {distance_cm:.1f}cm"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ----------------------------
        # PHONE FOLLOW LOGIC
        # ----------------------------
        if label == "cell phone":

            phone_detected = True

            object_center_x = (x1 + x2) // 2
            error = object_center_x - frame_center_x

            # DEBUG
            print(f"Phone | Dist: {distance_cm:.1f} | Error: {error}")

            # ----------------------------
            # DECISION
            # ----------------------------
            if distance_cm < STOP_DISTANCE:
                print("STOP (Too close)")
                car.stop()

            elif abs(error) < CENTER_THRESHOLD:
                print("FORWARD")
                car.forward()

            elif error > 0:
                print("RIGHT")
                car.right()

            elif error < 0:
                print("LEFT")
                car.left()

            break   # only track one phone

    # ----------------------------
    # NO PHONE DETECTED
    # ----------------------------
    if not phone_detected:
        print("No phone → STOP")
        car.stop()

    cv2.imshow("Autonomous Follow", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.1)

# ----------------------------
# CLEANUP
# ----------------------------
car.stop()
cap.release()
cv2.destroyAllWindows()
