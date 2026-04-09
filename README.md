# Finding--lanes_JU_TRAFFIQ_NEURODRIVE
# 🚗 TRAFFIQ — NEURO DRIVE

**Team:** Archideb Chatterjee · Sumedha Basu · Ayan Nandi · Sayan Das  
**Event:** TRAFFIQ 2025  
**Hardware:** Raspberry Pi 4B + Pi Camera V2

---

## Overview

NEURO DRIVE is a real-time autonomous lane-following AI pipeline built for the TRAFFIQ 2025 competition. It processes live camera frames to detect lane lines, identify obstacles, and handle adverse conditions like fog or low-light — outputting normalised speed and direction signals at every frame.

---

## Pipeline

```
Camera Frame (320×240)
    → Preprocess        (HSV white mask + CLAHE)
    → Canny             (edge detection)
    → ROI Crop          (lower 60% of frame)
    → Hough Lines       (lane line detection)
    → Obstacle Detection (contour analysis)
    → Fog Detection     (contrast std-dev)
    → Decision Logic    (priority: Fog → Obstacle → Normal)
    → [speed, direction] ∈ [-1.0, 1.0]
```

---

## Project Structure

```
FINDING--LANES/
├── lanes2.py        # Main AI pipeline
├── test_image.jpg   # Static test image
└── test2.mp4        # Test video clip
```

---

## Requirements

### Python Dependencies

```bash
pip install opencv-python numpy
```

### Python Version

Python 3.10+ recommended (uses `np.ndarray | None` union type hint syntax).

---

## Usage

### Run on test image

```bash
python lanes2.py image
```

Loads `test_image.jpg`, runs the full pipeline, and displays the annotated output. Press any key to close.

### Run on test video (default)

```bash
python lanes2.py
```

Processes `test2.mp4` frame by frame. Press `q` to quit.

### Run on live webcam

```bash
python lanes2.py 0
```

Replace `0` with the appropriate webcam index if needed.

---

## Output

Each frame produces a `[speed, direction]` pair:

| Value | Range | Meaning |
|-------|-------|---------|
| `speed` | `0.0 – 1.0` | Forward speed (0 = stop, 1 = full) |
| `direction` | `-1.0 – 1.0` | Steering (-1 = hard left, 0 = straight, +1 = hard right) |

---

## Decision Logic

The system uses a **priority-based** decision model:

| Priority | Condition | Behaviour |
|----------|-----------|-----------|
| 1 | **Fog / Low-light** | Speed = 0.3, steering damped to 50% |
| 2 | **Obstacle detected** | Speed = 0.3, steer away from obstacle |
| 3 | **Normal driving** | Speed scales with turn sharpness (max 1.0) |

Hysteresis counters prevent rapid flickering between states:
- Obstacle must persist for **3 frames** before triggering avoidance
- Fog must persist for **5 frames** before triggering fog mode

---

## Key Techniques

### Preprocessing
- BGR → HSV conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization) on the Value channel to handle uneven / low lighting
- White HSV range mask to isolate lane lines
- Morphological closing to fill line gaps

### Lane Detection
- Canny edge detection (thresholds: 50 / 150)
- Triangular ROI mask (lower 60% of frame)
- Probabilistic Hough Transform (`HoughLinesP`)
- Left/right line separation by slope sign
- Slope filtering: rejects lines with `|slope| < 0.3` or `|slope| > 5.0`

### Obstacle Detection
- Adaptive thresholding on the lower half of the frame
- Contour area filtering (minimum 800 px²)
- Aspect-ratio check to reject lane-line fragments

### Fog Detection
- Grayscale std-dev < 30 → fog mode active

### Steering Smoothing
- Exponential moving average: `α = 0.7` (weighted toward previous direction)

---

## HUD Overlay (Live / Video Mode)

The display window shows:

```
Direction: STRAIGHT / LEFT / RIGHT / AVOID / FOG
Speed: 0.850   Dir: 0.120
Fog: False   Obs: False
Output: [0.85, 0.12]
FPS: 28.4
```

Detected obstacles are highlighted with a **red bounding box**.  
Lane lines are drawn in **green**.

---

## Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OBSTACLE_HYSTERESIS` | `3` | Frames before obstacle avoidance triggers |
| `FOG_HYSTERESIS` | `5` | Frames before fog mode triggers |
| `STEERING_ALPHA` | `0.7` | Steering low-pass filter weight |
| Fog threshold | `30` | Grayscale std-dev below which fog is detected |
| Min obstacle area | `800 px²` | Contour area filter for obstacle detection |

---

## Notes

- The pipeline internally resizes all input to **320×240** for real-time performance on the Pi 4B.
- The `process_frame(frame)` function serves as the public API — pass any BGR image and receive `[speed, direction]`.
- No motor/serial output is included in this version; the pipeline runs in pure vision + decision mode.
