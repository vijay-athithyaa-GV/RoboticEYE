# Robotic Eye - Raspberry Pi 5 Deployment

A production-ready object detection and depth estimation system for Raspberry Pi 5 with Logitech USB camera.

## Features

- YOLO object detection using yolo26n.pt
- MiDaS depth estimation
- Distance estimation (30cm–300cm)
- Real-world size estimation (height & width in cm)
- Optimized for CPU-only operation
- Frame skipping for performance
- Single display window

## Hardware Requirements

- Raspberry Pi 5 (4GB RAM)
- 64-bit Raspberry Pi OS
- Logitech 720p USB camera
- Python 3.11

## Installation

1. Clone or copy the project to your Raspberry Pi:

   ```bash
   cd ~
   git clone <your-repo> robotic_eye
   cd robotic_eye
   ```

2. Place the YOLO model file:

   Copy `yolo26n.pt` to `models/yolo26n.pt`

3. Run the installation script:

   ```bash
   ./install.sh
   ```

   This will:
   - Update the system
   - Install python3-venv
   - Create a virtual environment
   - Install Python dependencies
   - Make scripts executable

## Camera Setup

1. Plug in the Logitech USB camera.

2. Test camera detection:

   ```bash
   v4l2-ctl --list-devices
   ls /dev/video*
   ```

   The camera should appear as `/dev/video0` (default).

## Running the Application

```bash
./run.sh
```

This will:
- Activate the virtual environment
- Start the Robotic Eye application

Press `Q` to quit.

## Performance Expectations

- Resolution: 320x240
- MiDaS runs every 3 frames
- CPU-only inference
- Expected FPS: 5-10 (depending on scene complexity)

## Troubleshooting

### Camera not found
- Check USB connection
- Verify camera with `v4l2-ctl --list-devices`
- Change camera index in code if needed (default is 0)

### Low performance
- Ensure no other heavy processes running
- Close unnecessary applications
- Monitor CPU usage with `top`

### Model loading errors
- Ensure `models/yolo26n.pt` exists
- Check internet for MiDaS download on first run

### Memory issues
- Restart Pi if needed
- Monitor with `free -h`

## Project Structure

```
robotic_eye/
├── models/
│   ├── yolo26n.pt          # YOLO model file
├── src/
│   ├── main.py             # Main application
├── requirements.txt        # Python dependencies
├── install.sh              # Installation script
├── run.sh                  # Run script
└── README.md               # This file
```

## Customization

- Adjust camera resolution in `src/main.py`
- Modify distance mapping (min_cm, max_cm)
- Change frame skipping interval
- Add more YOLO models

## License

[Add your license here]