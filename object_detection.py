import argparse
import os
import cv2
from ultralytics import YOLO

# Load YOLOv8 Nano model
model = YOLO("ActualModel/my_model.pt")


def draw_boxes_and_labels(frame, results):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names.get(cls_id, str(cls_id))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label text
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )


def run_on_image(image_path, save_output=True):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model(frame, conf=0.4, device="cpu")
    draw_boxes_and_labels(frame, results)

    window_title = "Robotic EYE - Object Detection (Image)"
    cv2.imshow(window_title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_output:
        out_path = os.path.splitext(image_path)[0] + "_detected.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Saved result to: {out_path}")


def run_on_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, conf=0.4, device="cpu")
        draw_boxes_and_labels(frame, results)

        cv2.imshow("Robotic EYE - Object Detection (Linux)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 object detection (image or camera)")
    parser.add_argument("--image", "-i", help="Path to input image. If omitted, camera is used.")
    args = parser.parse_args()

    if args.image:
        run_on_image(args.image)
    else:
        run_on_camera()
