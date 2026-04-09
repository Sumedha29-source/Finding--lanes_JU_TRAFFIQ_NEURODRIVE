"""
TRAFFIQ AI Model — Team NEURO DRIVE
Members: Archideb Chatterjee, Sumedha Basu, Ayan Nandi, Sayan Das
Hardware: Raspberry Pi 4B + Pi Camera V2
Event: TRAFFIQ 2025

Pipeline:
  Camera Frame (320x240)
    → Preprocess (HSV white mask + CLAHE)
    → Canny Edge Detection
    → ROI Crop
    → Hough Lane Lines
    → Obstacle Detection (contour)
    → Fog / Low-light Detection
    → Decision Logic (priority: fog → obstacle → normal drive)
    → [speed, direction]  ∈ [-1, 1]
"""

import cv2
import numpy as np
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
log = logging.getLogger("TRAFFIQ")

# ─────────────────────────────────────────────
# STATE  
# ─────────────────────────────────────────────
_state = {
    "prev_direction": 0.0,
    "obstacle_counter": 0,   # hysteresis: avoid flickering AVOID↔NORMAL
    "fog_counter": 0,
}

OBSTACLE_HYSTERESIS = 3   # frames obstacle must persist before triggering
FOG_HYSTERESIS      = 5   # frames of low contrast before entering fog mode
STEERING_ALPHA      = 0.7  # low-pass filter weight for previous direction


# ═══════════════════════════════════════════════
# 1. PREPROCESS  (lighting handling)
#     "CLAHE + HSV white masking"
# ═══════════════════════════════════════════════
def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask isolating white lane lines.

    Steps:
      1. Convert BGR → HSV
      2. White range mask  (H: any, S: low, V: high)
      3. CLAHE on V-channel to handle low-light / fog
         (PDF page 8: 'CLAHE — Contrast Limited Adaptive Histogram Equalization')
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # CLAHE on the Value channel to compensate for dim/uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    h, s, v = cv2.split(hsv)
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge([h, s, v_eq])

    lower_white = np.array([0,   0, 180])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv_eq, lower_white, upper_white)

    # Morphological close to fill small gaps in lane lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ═══════════════════════════════════════════════
# 2. EDGE DETECTION
# ═══════════════════════════════════════════════
def canny(image: np.ndarray) -> np.ndarray:
    """Gaussian blur then Canny edge detection."""
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


# ═══════════════════════════════════════════════
# 3. REGION OF INTEREST 
# ═══════════════════════════════════════════════
def region_of_interest(image: np.ndarray) -> np.ndarray:
    """
    Triangular ROI mask covering the lower 60% of the frame.
    Eliminates sky / upper background that would confuse Hough.
    """
    height, width = image.shape[:2]
    polygons = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.5), int(height * 0.4))
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)


# ═══════════════════════════════════════════════
# 4. LANE PROCESSING
# ═══════════════════════════════════════════════
def make_coordinates(image: np.ndarray, line_parameters) -> np.ndarray:
    slope, intercept = line_parameters

    # Guard against near-zero slope (horizontal line)
    if abs(slope) < 0.01:
        slope = 0.01

    y1 = image.shape[0]
    y2 = int(y1 * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # Clamp to frame width to avoid wild extrapolation
    w  = image.shape[1]
    x1 = int(np.clip(x1, 0, w))
    x2 = int(np.clip(x2, 0, w))

    return np.array([x1, y1, x2, y2])


#  "Average slope and intercept for stability"
def average_slope_intercept(image: np.ndarray, lines) -> np.ndarray | None:
    """
    Separates detected Hough lines into left (negative slope) and right
    (positive slope) groups, averages each group, and returns up to two
    representative lane lines.
    """
    if lines is None:
        return None

    left_fit  = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # Skip near-horizontal lines (likely noise)
        if abs(x2 - x1) < 1:
            continue

        parameters        = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept  = parameters

        # Filter extreme slopes — likely noise
        if abs(slope) < 0.3 or abs(slope) > 5.0:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lines_out = []
    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        lines_out.append(make_coordinates(image, left_avg))
    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        lines_out.append(make_coordinates(image, right_avg))

    return np.array(lines_out) if lines_out else None


# ═══════════════════════════════════════════════
# 5. DISPLAY LINES  (debug overlay)
# ═══════════════════════════════════════════════
def display_lines(image: np.ndarray, lines) -> np.ndarray:
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image


# ═══════════════════════════════════════════════
# 6. STEERING  (normalised to [-1, 1])
# ═══════════════════════════════════════════════
def get_steering_normalised(frame: np.ndarray, lines) -> float:
    """
    Computes steering signal from lane line positions.

    Returns float in [-1, 1]:
      -1 = hard left,  0 = straight,  +1 = hard right

    Strategy: average the top x-coords of detected lane lines.
    Deviation of that average from frame centre → normalised steering.
    """
    height, width = frame.shape[:2]

    if lines is None or len(lines) == 0:
        return 0.0

    x_tops = [line[2] for line in lines]
    avg_x  = float(np.mean(x_tops))

    center        = width / 2.0
    deviation     = avg_x - center
    max_deviation = width / 2.0

    steering = deviation / max_deviation
    return float(np.clip(steering, -1.0, 1.0))


# ═══════════════════════════════════════════════
# 7. OBSTACLE DETECTION
# ═══════════════════════════════════════════════
def detect_obstacle(frame: np.ndarray):
    """
    Scans the lower half of the frame for dark blobs (obstacles on black track).

    Returns (bool, bounding_box_or_None)

    Improvements vs original:
      • Adaptive threshold instead of fixed 60
      • Minimum aspect-ratio check to ignore lane-line fragments
      • Returns largest obstacle (most imminent threat)
    """
    height, width = frame.shape[:2]
    roi = frame[height // 2:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold handles varying surface brightness better
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=4
    )

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_box  = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0

        # Reject very thin vertical fragments (likely lane lines leaking in)
        if aspect < 0.3:
            continue

        if area > best_area:
            best_area = area
            best_box  = (x, y + height // 2, w, h)

    return (best_box is not None), best_box


# ═══════════════════════════════════════════════
# 8. FOG / LOW-LIGHT DETECTION
# ═══════════════════════════════════════════════
def detect_fog(frame: np.ndarray) -> bool:
    """
    Detects fog or very low-light conditions by measuring global contrast
    (std-dev of grayscale values).  Threshold = 30 (tunable).
    """
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))
    return contrast < 30


# ═══════════════════════════════════════════════
# 9. DECISION MAKING
#    Output: [speed, direction]  both in [-1, 1]
# ═══════════════════════════════════════════════
def decide_movement(frame: np.ndarray, lines, obstacle: bool, fog: bool):
    """
    Returns ([speed, direction], label_str).

    Priority:
      1. Fog        → slow, damped steering
      2. Obstacle   → slow + steer away  (hysteresis prevents oscillation)
      3. Normal     → full speed, lane-centred steering
    """
    raw_direction = get_steering_normalised(frame, lines)

    # Low-pass filter for smooth steering 
    direction = (STEERING_ALPHA * _state["prev_direction"]
                 + (1 - STEERING_ALPHA) * raw_direction)
    _state["prev_direction"] = direction

    # ── Hysteresis counters ──────────────────────
    if obstacle:
        _state["obstacle_counter"] = min(_state["obstacle_counter"] + 1,
                                         OBSTACLE_HYSTERESIS + 1)
    else:
        _state["obstacle_counter"] = max(_state["obstacle_counter"] - 1, 0)

    if fog:
        _state["fog_counter"] = min(_state["fog_counter"] + 1,
                                    FOG_HYSTERESIS + 1)
    else:
        _state["fog_counter"] = max(_state["fog_counter"] - 1, 0)

    obstacle_active = _state["obstacle_counter"] >= OBSTACLE_HYSTERESIS
    fog_active      = _state["fog_counter"]      >= FOG_HYSTERESIS

    # ── Priority 1: Fog ──────────────────────────
    if fog_active:
        speed     = 0.3
        direction = direction * 0.5   # damp steering in low visibility
        label     = "FOG"
        return [round(speed, 3), round(direction, 3)], label

    # ── Priority 2: Obstacle ─────────────────────
    if obstacle_active:
        avoidance   = -np.sign(direction) * min(0.6 + abs(direction) * 0.2, 0.8)
        direction   = float(np.clip(avoidance, -1.0, 1.0))
        speed       = 0.3
        label       = "AVOID"
        return [round(speed, 3), round(direction, 3)], label

    # ── Priority 3: Normal driving ───────────────
    # Reduce speed proportionally when turning 
    speed = 1.0 - 0.4 * abs(direction)
    speed = float(np.clip(speed, 0.2, 1.0))

    if   direction >  0.2: label = "RIGHT"
    elif direction < -0.2: label = "LEFT"
    else:                  label = "STRAIGHT"

    return [round(speed, 3), round(direction, 3)], label


# ═══════════════════════════════════════════════
# 10. PROCESS SINGLE FRAME  (main API)
# ═══════════════════════════════════════════════
def process_frame(frame: np.ndarray) -> list:
    """
    Public API — takes a BGR image (any size ≥ 640×480),
    returns [speed, direction] both in [-1, 1].

    Internally resizes to 320×240 for real-time performance on Pi 4B.
    """
    frame          = cv2.resize(frame, (320, 240))
    mask           = preprocess(frame)
    edges          = canny(mask)
    cropped        = region_of_interest(edges)

    lines          = cv2.HoughLinesP(cropped, 2, np.pi / 180, 80,
                                     minLineLength=30, maxLineGap=10)
    averaged_lines = average_slope_intercept(frame, lines)

    obstacle, _    = detect_obstacle(frame)
    fog            = detect_fog(frame)

    output, _      = decide_movement(frame, averaged_lines, obstacle, fog)
    return output  # [speed, direction]


# ═══════════════════════════════════════════════
# 11. TEST ON SINGLE IMAGE  (python lanes2.py image)
# ═══════════════════════════════════════════════
def run_image_test():
    """
    Runs the full pipeline on test_image.jpg and displays result.
    Press any key to close.
    """
    frame = cv2.imread("test_image.jpg")  # input image: test_image.jpg
    if frame is None:
        log.error("Could not load 'test_image.jpg'. Place it in the same folder.")
        return

    frame = cv2.resize(frame, (320, 240))

    mask           = preprocess(frame)
    edges          = canny(mask)
    cropped        = region_of_interest(edges)

    lines          = cv2.HoughLinesP(cropped, 2, np.pi / 180, 80,
                                     minLineLength=30, maxLineGap=10)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image     = display_lines(frame, averaged_lines)

    obstacle, box  = detect_obstacle(frame)
    fog            = detect_fog(frame)

    output, label  = decide_movement(frame, averaged_lines, obstacle, fog)
    speed_val, dir_val = output

    if obstacle and box:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.putText(frame, f"Direction: {label}",
                (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Speed: {speed_val:.3f}  Dir: {dir_val:.3f}",
                (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Fog: {fog}  Obstacle: {obstacle}",
                (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Output: {output}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    log.info(f"[test_image.jpg] Output: {output} | Direction: {label}")

    cv2.imshow("TRAFFIQ — NEURO DRIVE", combo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════
# 12. MAIN LOOP
# ═══════════════════════════════════════════════
if __name__ == "__main__":

    # Usage:
    #   python lanes2.py image          → test on test_image.jpg
    #   python lanes2.py                → run on test2.mp4 (default)
    #   python lanes2.py /dev/video0    → run on live webcam

    if len(sys.argv) > 1 and sys.argv[1] == "image":
        run_image_test()
        sys.exit(0)

    source = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    try:
        source = int(source)   # numeric → webcam index
    except ValueError:
        pass                   # keep as filename string

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open video source: {source}")
        sys.exit(1)

    log.info(f"Starting TRAFFIQ pipeline on source: {source}")
    frame_count  = 0
    t_start      = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            log.info("End of stream.")
            break

        frame = cv2.resize(frame, (320, 240))

        # ── Core pipeline ──────────────────────
        mask           = preprocess(frame)
        edges          = canny(mask)
        cropped        = region_of_interest(edges)

        lines          = cv2.HoughLinesP(cropped, 2, np.pi / 180, 80,
                                         minLineLength=30, maxLineGap=10)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image     = display_lines(frame, averaged_lines)

        obstacle, box  = detect_obstacle(frame)
        fog            = detect_fog(frame)

        output, label  = decide_movement(frame, averaged_lines, obstacle, fog)
        speed_val, dir_val = output

        # ── FPS counter ────────────────────────
        frame_count += 1
        elapsed = time.time() - t_start
        fps     = frame_count / elapsed if elapsed > 0 else 0

        # ── Draw obstacle bounding box ─────────
        if obstacle and box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # ── HUD overlay ───────────────────────
        cv2.putText(frame, f"Direction: {label}",
                    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Speed: {speed_val:.3f}  Dir: {dir_val:.3f}",
                    (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Fog: {fog}  Obs: {obstacle}",
                    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Output: {output}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("TRAFFIQ — NEURO DRIVE", combo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            log.info("User quit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info(f"Done. Processed {frame_count} frames in {elapsed:.1f}s "
             f"(avg {fps:.1f} FPS)")