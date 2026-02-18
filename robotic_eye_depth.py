import torch
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
yolo_model = YOLO("yolo26n.pt")  # Replace with your trained model

# ----------------------------
# LOAD MIDAS MODEL
# ----------------------------
model_type = "MiDaS_small"  # Lightweight
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
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
    # DEPTH ESTIMATION (MiDaS)
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


    # Normalize for display
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_norm = depth_map_norm.astype(np.uint8)

    # ----------------------------
    # COMBINE YOLO + DEPTH
    # ----------------------------
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls]

        # Get center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Get depth value
        depth_value = depth_map[cy, cx]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Display label + depth
        text = f"{label} | Depth: {depth_value:.2f}"
        cv2.putText(frame, text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0), 2)

    cv2.imshow("Robotic Eye - Detection", frame)
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)

    cv2.imshow("Depth Map", depth_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
