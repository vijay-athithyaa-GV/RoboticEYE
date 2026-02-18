# Robotic Eye – YOLO Object Detection with Depth Estimation

This project is a **real-time "Robotic Eye"** that uses:
- A webcam as input
- A custom **YOLOv8** object detection model
- The **MiDaS** depth-estimation network

It provides:
- Simple camera testing
- Real-time object detection (webcam or image)
- Real-time object detection **with approximate depth** at each detected object

The code is tested on Linux and uses CPU by default (GPU is optional if available).

---

## Project Structure

Key files and folders:

- `camera_test.py` – Minimal script to verify that the webcam works under Linux.
- `object_detection.py` – Runs YOLO object detection
  - On the webcam (default)
  - On a single image (`--image` argument)
  - Uses custom model weights: `ActualModel/my_model.pt`.
- `robotic_eye_depth.py` – Main **Robotic Eye** script:
  - Loads YOLO model: `yolo26n.pt` (you can replace with your own trained model)
  - Loads MiDaS (`MiDaS_small`) via `torch.hub`
  - Performs object detection + per-object depth estimation in real time.
- `ActualModel/`
  - `my_model.pt` – Custom YOLO model weights (large file, ignored by git)
  - `train/` – Training outputs (plots, confusion matrix, batch images, metrics, weights).
- `my_model/`
  - `my_model.pt` – Another copy/checkpoint of the trained model (ignored by git)
  - `train2/` – Training outputs for another run.
- `Dataset-Leaf/` – Original dataset images used for training (large, ignored by git).
- `yolo26n.pt`, `yolov8n.pt` – YOLO weights (large, ignored by git).
- `venv/` – Python virtual environment (ignored by git).

Git is configured via `.gitignore` to avoid committing large, generated, or environment files.

---

## How It Works

### 1. Camera Test (`camera_test.py`)

- Opens your default webcam using OpenCV (`cv2.VideoCapture(0)`).
- Displays the raw camera feed in a window titled **"Camera Test - Linux"**.
- Press `q` to exit.

### 2. Object Detection (`object_detection.py`)

- Loads YOLO model: `ActualModel/my_model.pt` using `ultralytics.YOLO`.
- For each frame (from webcam or from a static image):
  - Runs YOLO inference with `conf=0.4` on CPU.
  - Draws bounding boxes and class labels with confidence scores.
- The drawing logic is in `draw_boxes_and_labels(frame, results)`.

You can run it:
- On webcam: live continuous detection
- On image: single image detection + optional save of the result

### 3. Depth-Enabled Robotic Eye (`robotic_eye_depth.py`)

- Loads YOLO model: `YOLO("yolo26n.pt")`.
- Loads MiDaS depth model: `MiDaS_small` from `intel-isl/MiDaS` via `torch.hub`.
- For each frame from the webcam:
  1. Runs YOLO to obtain `results.boxes`.
  2. Converts BGR frame to RGB and applies MiDaS transform.
  3. Runs MiDaS to obtain a **relative depth map**.
  4. For every detected box:
     - Computes the center `(cx, cy)` of the box.
     - Reads the depth value from `depth_map[cy, cx]`.
     - Draws the bounding box and overlays text: `"<label> | Depth: <value>"`.
  5. Displays two windows:
     - **"Robotic Eye - Detection"** – original frame with boxes + depth text
     - **"Depth Map"** – colorized depth map using `COLORMAP_JET`

> Note: MiDaS gives **relative depth**, not absolute meters. Higher/lower values correspond to nearer/farther objects (depending on model normalization), but you should treat this as relative distance for ranking objects by depth rather than accurate physical distance.

---

## Installation

These steps assume Linux and Python 3.8+.

### 1. Clone the Repository

```bash
cd /path/to/where/you/want/the/project
# If this is already a local folder, you can skip cloning
# git clone <your-repo-url> "Robotic Eye"
cd "Robotic Eye"
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Create a `requirements.txt` if you like, but the minimal packages are:

```bash
pip install ultralytics torch torchvision torchaudio opencv-python numpy
```

Depending on your hardware, you may instead install a GPU-enabled PyTorch build from https://pytorch.org/.

---

## Model Weights and Large Files (Not in Git)

The `.gitignore` is configured to **exclude large/binary artifacts** from version control, including:

- All `*.pt`, `*.pth`, `*.onnx`, `*.engine` files
- Dataset folder: `Dataset-Leaf/`
- Training outputs: `ActualModel/train/`, `my_model/train2/`
- Zipped archives: `*.zip`
- Virtual environment: `venv/`

This keeps the repository lightweight and focused on **code and configs only**.

### Where to Place Your Weights

You should manually place or download your model weights into:

- `ActualModel/my_model.pt` – custom YOLO model for `object_detection.py`
- `yolo26n.pt` – YOLO model used by `robotic_eye_depth.py`
- (Optional) `yolov8n.pt` – additional YOLO weights for experimentation

Since weights are ignored by git, they will not be pushed to remote repositories. If you need to share them, use an external storage service and document the download link.

---

## Usage

Make sure your virtual environment is activated:

```bash
cd "/home/vijaylinux/Desktop/Robotic Eye"
source venv/bin/activate
```

### 1. Test the Camera

```bash
python camera_test.py
```

- A window **"Camera Test - Linux"** should open.
- Press `q` to quit.

### 2. Run Object Detection on Webcam

```bash
python object_detection.py
```

- Uses `ActualModel/my_model.pt`.
- Window title: **"Robotic EYE - Object Detection (Linux)"**.
- Press `q` to quit.

### 3. Run Object Detection on an Image

```bash
python object_detection.py --image path/to/your_image.jpg
```

- Shows detection result in a window.
- Saves an output image next to the input (same name with `_detected.jpg` suffix).

### 4. Run Robotic Eye with Depth Estimation

```bash
python robotic_eye_depth.py
```

- Uses YOLO (`yolo26n.pt`) and MiDaS (`MiDaS_small`).
- Shows two windows:
  - **"Robotic Eye - Detection"** – detections + depth info.
  - **"Depth Map"** – colorized depth map.
- Press `q` to quit.

If you want to use your own YOLO model, update this line inside `robotic_eye_depth.py`:

```python
yolo_model = YOLO("yolo26n.pt")  # Replace with your trained model
```

For example:

```python
yolo_model = YOLO("ActualModel/my_model.pt")
```

Just ensure the corresponding `.pt` file exists locally.

---

## Performance and Device Settings

- **Current configuration** runs YOLO on CPU in `object_detection.py` (`device="cpu"`).
- To use GPU (if available and supported by your PyTorch install), change:

```python
results = model(frame, conf=0.4, device="cpu")
```

to:

```python
results = model(frame, conf=0.4, device="0")  # GPU 0
```

- MiDaS is also executed on CPU by default. If you want GPU acceleration, you can move the model and tensors to CUDA (requires a compatible GPU and CUDA-enabled PyTorch).

---

## Training Artifacts

Folders like `ActualModel/train/` and `my_model/train2/` contain:

- `args.yaml` – training configuration used by Ultralytics.
- `results.csv` / `results.png` – metrics over epochs (mAP, precision, recall, etc.).
- `train_batch*.jpg`, `val_batch*.jpg` – visualizations of training/validation batches.
- `confusion_matrix*.png` – confusion matrices.
- `weights/best.pt`, `weights/last.pt` – best and last checkpoints.

These are **ignored by git** to keep the repo small but are useful locally for analyzing training quality.

---

## Acknowledgements

- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **MiDaS** depth estimation: https://github.com/isl-org/MiDaS
- **PyTorch**: https://pytorch.org/
- **OpenCV**: https://opencv.org/

---

## Notes and Future Improvements

- Calibrate depth values: currently the depth is relative; you could pair MiDaS output with known distances to derive an approximate mapping to real-world units.
- Multi-camera support: allow selecting the camera index via CLI argument.
- Simple UI controls: toggle depth overlay, change confidence threshold, switch models from the command line.
- Packaging: add `requirements.txt` or `pyproject.toml` and possibly a launcher script.

If you want, we can next add a `requirements.txt` and small CLI improvements (e.g., model path and device as arguments) to make the project even more reusable.