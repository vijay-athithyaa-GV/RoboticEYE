import torch
import cv2
import numpy as np
import math
from ultralytics import YOLO

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
yolo_model = YOLO("yolo26n.pt")

# ----------------------------
# LOAD MIDAS MODEL
# ----------------------------
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# ----------------------------
# CAMERA PARAMETERS (Laptop Cam Approximation)
# ----------------------------
image_width = 640
fov = 65  # Approx horizontal FOV of laptop webcam
focal_length = image_width / (2 * math.tan(math.radians(fov / 2)))

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, image_width)
cap.set(4, 480)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("Robotic Eye Started... Press Q to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # YOLO OBJECT DETECTION
    # ----------------------------
    results = yolo_model(frame)[0]

    # ----------------------------
    # DEPTH ESTIMATION
    # ----------------------------
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # -------- Stable Normalization (0–1) --------
    depth_map = depth_map.astype(np.float32)
    depth_map = np.nan_to_num(depth_map)

    min_val = np.min(depth_map)
    max_val = np.max(depth_map)

    if max_val - min_val > 1e-6:
        depth_norm = (depth_map - min_val) / (max_val - min_val)
    else:
        depth_norm = np.zeros_like(depth_map)

    # Invert so near = high value
    depth_norm = 1.0 - depth_norm

    # Colored depth visualization
    depth_display = (depth_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

    # ----------------------------
    # COMBINE YOLO + DEPTH + SIZE ESTIMATION
    # ----------------------------
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls]

        # Clamp safely
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_norm.shape[1], x2)
        y2 = min(depth_norm.shape[0], y2)

        roi = depth_norm[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Stable depth value
        depth_value = np.median(roi)

        # -------- Convert to CM (Approximate Distance) --------
        min_cm = 30
        max_cm = 300
        distance_cm = min_cm + (1 - depth_value) * (max_cm - min_cm)

        # -------- REAL-WORLD SIZE ESTIMATION --------
        pixel_height = y2 - y1
        pixel_width = x2 - x1

        real_height_cm = (pixel_height * distance_cm) / focal_length
        real_width_cm = (pixel_width * distance_cm) / focal_length

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        text = f"{label} | D:{distance_cm:.1f}cm H:{real_height_cm:.1f}cm W:{real_width_cm:.1f}cm"
        cv2.putText(frame, text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0), 2)

    cv2.imshow("Robotic Eye - Detection", frame)
    cv2.imshow("Depth Map", depth_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
