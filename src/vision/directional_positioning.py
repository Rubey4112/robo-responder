"""
Directional Positioning & Path Following Module
-----------------------------------------------
Uses OpenCV computer vision to detect a path/line and compute steering guidance
to keep the robot centered on the path without jittering or steering randomly.

Features:
1. Region of Interest (ROI) filtering focusing on the ground plane in front of the robot.
2. Multi-mode path detection:
   - Adaptive thresholding (auto-adjusts to room/lighting changes)
   - Dark line on light surface
   - Bright line on dark surface
   - Custom HSV color tape (e.g. blue, red, yellow)
3. Anti-Jitter & Stability:
   - Exponential Moving Average (EMA) smoothing on lateral error
   - Deadband / hysteresis zone to prevent oscillation around center
   - Lost-path hold (retains last known trajectory during brief frame drops)
   - PD (Proportional-Derivative) dampening to avoid overshooting
4. Control Output:
   - Normalized lateral offset: -1.0 (far left) to +1.0 (far right)
   - Steering direction: 'CENTERED', 'STEER_LEFT', 'STEER_RIGHT', 'LOST'
   - Recommended differential motor speeds (left_speed, right_speed)
5. Visual HUD:
   - Guidance corridor, path centroid, steering vector arrow, and dashboard gauge.
"""

from dataclasses import dataclass
from enum import Enum
import math
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class DetectionMode(Enum):
    ADAPTIVE = "adaptive"      # Dynamic thresholding based on local lighting
    DARK_LINE = "dark_line"    # Dark path/tape on lighter floor (e.g. black tape)
    BRIGHT_LINE = "bright_line"# Light path/tape on darker floor (e.g. white tape)
    COLOR_TAPE = "color_tape"  # Specific HSV color band (default: blue/yellow/red)


class SteeringState(Enum):
    CENTERED = "CENTERED"
    STEER_LEFT = "STEER_LEFT"
    STEER_RIGHT = "STEER_RIGHT"
    LOST = "LOST"


@dataclass
class PositioningResult:
    """Holds the computed positioning telemetry and steering guidance for a frame."""
    offset: float                     # Normalized offset: -1.0 (left) to 1.0 (right), 0.0 = centered
    offset_pixels: int                # Pixel distance from frame center
    angle_deg: float                  # Heading angle in degrees (-90 to +90)
    steering_state: SteeringState     # Current categorical guidance
    steering_effort: float            # Smoothed control effort: -1.0 (hard left) to +1.0 (hard right)
    motor_speeds: Tuple[float, float] # (left_motor_speed, right_motor_speed) normalized 0.0 to 1.0
    path_detected: bool               # Whether a valid path was found in this frame
    confidence: float                 # Confidence score (0.0 to 1.0)
    annotated_frame: np.ndarray       # Frame with visual tracking overlay and HUD


class PathTracker:
    """
    Analyzes camera frames to track a path and calculate stable steering guidance.
    """

    def __init__(
        self,
        mode: DetectionMode = DetectionMode.ADAPTIVE,
        roi_top_ratio: float = 0.50,       # Use bottom 50% of the frame (ground plane)
        deadband: float = 0.08,             # Offset within ±8% is considered centered (prevents jitter)
        smoothing_alpha: float = 0.35,      # EMA smoothing factor (0.0 to 1.0, lower = smoother)
        kp: float = 1.0,                    # Proportional steering gain
        kd: float = 0.3,                    # Derivative dampening gain (prevents overshooting)
        base_speed: float = 0.6,            # Nominal forward motor speed (0.0 to 1.0)
        min_contour_area: float = 500.0,    # Minimum pixel area to be considered a path
        max_lost_frames: int = 15,          # Number of frames to hold last known position before declaring LOST
        hsv_lower: Tuple[int, int, int] = (100, 80, 50),   # Default HSV for color mode (e.g. blue tape)
        hsv_upper: Tuple[int, int, int] = (130, 255, 255),
    ):
        self.mode = mode
        self.roi_top_ratio = roi_top_ratio
        self.deadband = deadband
        self.alpha = smoothing_alpha
        self.kp = kp
        self.kd = kd
        self.base_speed = base_speed
        self.min_contour_area = min_contour_area
        self.max_lost_frames = max_lost_frames
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

        # Internal state for anti-jitter smoothing & derivative calculation
        self.smoothed_offset: float = 0.0
        self.prev_offset: float = 0.0
        self.prev_time: float = time.time()
        self.lost_frame_count: int = 0
        self.last_valid_cx: Optional[int] = None
        self.last_valid_cy: Optional[int] = None

    def set_color_range(self, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> None:
        """Sets custom HSV range when using DetectionMode.COLOR_TAPE."""
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)
        self.mode = DetectionMode.COLOR_TAPE

    def process_frame(self, frame: np.ndarray) -> PositioningResult:
        """
        Analyzes a single camera frame and returns a PositioningResult with steering guidance.
        """
        if frame is None or frame.size == 0:
            return self._empty_result(frame)

        h, w = frame.shape[:2]
        roi_y = int(h * self.roi_top_ratio)
        roi = frame[roi_y:h, 0:w]
        roi_h, roi_w = roi.shape[:2]
        frame_center_x = w // 2

        # 1. Segment the path from ROI
        mask = self._segment_path(roi)

        # 2. Extract contours and find the most relevant path contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

        annotated = frame.copy()
        current_time = time.time()
        dt = max(1e-4, current_time - self.prev_time)
        self.prev_time = current_time

        if not valid_contours:
            self.lost_frame_count += 1
            if self.lost_frame_count > self.max_lost_frames:
                # Path is fully lost
                return self._build_lost_result(annotated, w, h, roi_y)
            else:
                # Coast / hold last known position to prevent sudden jerky movements
                offset = self.smoothed_offset
                angle_deg = 0.0
                confidence = max(0.1, 1.0 - (self.lost_frame_count / self.max_lost_frames))
                cx = self.last_valid_cx if self.last_valid_cx is not None else frame_center_x
                cy = self.last_valid_cy if self.last_valid_cy is not None else (roi_y + roi_h // 2)
        else:
            self.lost_frame_count = 0
            # Choose the contour closest to the previous target or largest
            if self.last_valid_cx is not None:
                def distance_to_last(c):
                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        return float("inf")
                    cx_c = int(M["m10"] / M["m00"])
                    return abs(cx_c - self.last_valid_cx)
                best_contour = min(valid_contours, key=distance_to_last)
            else:
                best_contour = max(valid_contours, key=cv2.contourArea)

            # Compute centroid of path contour
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx_local = int(M["m10"] / M["m00"])
                cy_local = int(M["m01"] / M["m00"])
                cx = cx_local
                cy = roi_y + cy_local
                self.last_valid_cx = cx
                self.last_valid_cy = cy

                # Raw normalized offset: -1.0 (left edge) to +1.0 (right edge)
                raw_offset = (cx - frame_center_x) / (w / 2.0)
                raw_offset = max(-1.0, min(1.0, raw_offset))

                # Calculate path angle (fit line to contour)
                angle_deg = self._compute_contour_angle(best_contour)

                # Confidence based on contour area vs frame size
                area_ratio = cv2.contourArea(best_contour) / (roi_w * roi_h)
                confidence = min(1.0, area_ratio * 5.0 + 0.5)

                offset = raw_offset
            else:
                offset = self.smoothed_offset
                angle_deg = 0.0
                confidence = 0.3
                cx = frame_center_x
                cy = roi_y + roi_h // 2

        # 3. Apply Anti-Jitter Smoothing (Exponential Moving Average)
        self.smoothed_offset = (self.alpha * offset) + ((1.0 - self.alpha) * self.smoothed_offset)

        # 4. Compute derivative and PD control effort
        d_error = (self.smoothed_offset - self.prev_offset) / dt
        self.prev_offset = self.smoothed_offset

        # Steering effort: positive means steer right, negative means steer left
        control_effort = (self.kp * self.smoothed_offset) + (self.kd * d_error)
        control_effort = max(-1.0, min(1.0, control_effort))

        # 5. Apply Deadband / Hysteresis
        if abs(self.smoothed_offset) <= self.deadband:
            steering_state = SteeringState.CENTERED
            # When centered, soften the control effort to 0 to prevent micro-twitching
            steering_effort = 0.0
        elif self.smoothed_offset > self.deadband:
            steering_state = SteeringState.STEER_RIGHT
            steering_effort = control_effort
        else:
            steering_state = SteeringState.STEER_LEFT
            steering_effort = control_effort

        # 6. Compute Recommended Differential Motor Speeds
        left_speed, right_speed = self._compute_motor_speeds(steering_effort)

        # 7. Render Visual HUD Overlay
        self._render_hud(
            annotated=annotated,
            w=w,
            h=h,
            roi_y=roi_y,
            cx=cx,
            cy=cy,
            offset=self.smoothed_offset,
            angle_deg=angle_deg,
            state=steering_state,
            effort=steering_effort,
            left_speed=left_speed,
            right_speed=right_speed,
            confidence=confidence,
        )

        offset_pixels = int(self.smoothed_offset * (w / 2.0))

        return PositioningResult(
            offset=round(float(self.smoothed_offset), 3),
            offset_pixels=offset_pixels,
            angle_deg=round(float(angle_deg), 1),
            steering_state=steering_state,
            steering_effort=round(float(steering_effort), 3),
            motor_speeds=(round(left_speed, 2), round(right_speed, 2)),
            path_detected=True,
            confidence=round(confidence, 2),
            annotated_frame=annotated,
        )

    def _segment_path(self, roi: np.ndarray) -> np.ndarray:
        """Extracts binary mask of the path based on selected detection mode."""
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        if self.mode == DetectionMode.ADAPTIVE:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            # Otsu's thresholding creates clean binary mask under varied lighting
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        elif self.mode == DetectionMode.DARK_LINE:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        elif self.mode == DetectionMode.BRIGHT_LINE:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        elif self.mode == DetectionMode.COLOR_TAPE:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        else:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological opening and closing to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _compute_contour_angle(self, contour: np.ndarray) -> float:
        """Computes orientation angle in degrees (-90 to +90). 0 is straight ahead."""
        if len(contour) < 5:
            return 0.0
        try:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx), float(vy)
            # Angle relative to vertical (upwards is straight)
            angle_rad = math.atan2(vx, -vy)
            return math.degrees(angle_rad)
        except Exception:
            return 0.0

    def _compute_motor_speeds(self, steering_effort: float) -> Tuple[float, float]:
        """
        Computes normalized differential drive motor speeds based on steering effort.
        steering_effort: positive means steer right, negative means steer left.
        """
        turn_response = 0.45 * steering_effort
        left = self.base_speed + turn_response
        right = self.base_speed - turn_response

        # Clamp between 0.0 (stopped) and 1.0 (full speed)
        left = max(0.0, min(1.0, left))
        right = max(0.0, min(1.0, right))
        return left, right

    def _build_lost_result(self, annotated: np.ndarray, w: int, h: int, roi_y: int) -> PositioningResult:
        """Builds result when path is completely out of view."""
        self._render_hud(
            annotated=annotated,
            w=w,
            h=h,
            roi_y=roi_y,
            cx=w // 2,
            cy=roi_y + (h - roi_y) // 2,
            offset=0.0,
            angle_deg=0.0,
            state=SteeringState.LOST,
            effort=0.0,
            left_speed=0.0,
            right_speed=0.0,
            confidence=0.0,
        )
        return PositioningResult(
            offset=0.0,
            offset_pixels=0,
            angle_deg=0.0,
            steering_state=SteeringState.LOST,
            steering_effort=0.0,
            motor_speeds=(0.0, 0.0),
            path_detected=False,
            confidence=0.0,
            annotated_frame=annotated,
        )

    def _empty_result(self, frame: np.ndarray) -> PositioningResult:
        empty = np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame
        return PositioningResult(
            offset=0.0,
            offset_pixels=0,
            angle_deg=0.0,
            steering_state=SteeringState.LOST,
            steering_effort=0.0,
            motor_speeds=(0.0, 0.0),
            path_detected=False,
            confidence=0.0,
            annotated_frame=empty,
        )

    def _render_hud(
        self,
        annotated: np.ndarray,
        w: int,
        h: int,
        roi_y: int,
        cx: int,
        cy: int,
        offset: float,
        angle_deg: float,
        state: SteeringState,
        effort: float,
        left_speed: float,
        right_speed: float,
        confidence: float,
    ) -> None:
        """Draws rich augmented-reality visual HUD on top of the camera frame."""
        frame_center_x = w // 2

        # 1. Draw ROI boundary line
        cv2.line(annotated, (0, roi_y), (w, roi_y), (255, 215, 0), 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            "GROUND ROI",
            (10, roi_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 215, 0),
            1,
            cv2.LINE_AA,
        )

        # 2. Draw Target Center Line & Deadband Corridor
        db_pixel = int(self.deadband * (w / 2.0))
        # Deadband corridor (safe zone)
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (frame_center_x - db_pixel, roi_y),
            (frame_center_x + db_pixel, h),
            (0, 255, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)

        # Robot center vertical line
        cv2.line(annotated, (frame_center_x, roi_y), (frame_center_x, h), (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Path centroid & Trajectory Vector
        if state != SteeringState.LOST:
            # Color based on state: Green for centered, Orange/Red for steering
            color = (0, 255, 0) if state == SteeringState.CENTERED else (0, 165, 255)

            # Target centroid circle
            cv2.circle(annotated, (cx, cy), 12, color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 18, (255, 255, 255), 2, cv2.LINE_AA)

            # Line connecting robot center to detected path
            cv2.line(annotated, (frame_center_x, cy), (cx, cy), color, 2, cv2.LINE_AA)

            # Steering vector arrow from bottom center towards target
            start_pt = (frame_center_x, h - 20)
            end_pt = (int(frame_center_x + (effort * 120)), h - 80)
            cv2.arrowedLine(annotated, start_pt, end_pt, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)

        # 4. Top Telemetry Dashboard Bar
        cv2.rectangle(annotated, (0, 0), (w, 55), (25, 25, 25), -1)

        # Status Badge
        badge_colors = {
            SteeringState.CENTERED: (46, 139, 87),     # Green
            SteeringState.STEER_LEFT: (0, 140, 255),   # Orange
            SteeringState.STEER_RIGHT: (0, 140, 255),  # Orange
            SteeringState.LOST: (0, 0, 200),           # Red
        }
        badge_color = badge_colors.get(state, (100, 100, 100))
        cv2.rectangle(annotated, (12, 10), (160, 45), badge_color, -1)
        cv2.putText(
            annotated,
            state.value,
            (22, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Offset & Motors Telemetry
        offset_pct = int(offset * 100)
        telemetry_text = f"Offset: {offset_pct:+3d}% | Angle: {angle_deg:+.1f} deg | Motors: L={left_speed:.2f} R={right_speed:.2f}"
        cv2.putText(
            annotated,
            telemetry_text,
            (175, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        # 5. Bottom Steering Slider / Gauge
        slider_y = h - 25
        slider_w = 260
        slider_x1 = frame_center_x - slider_w // 2
        slider_x2 = frame_center_x + slider_w // 2

        # Background bar
        cv2.rectangle(annotated, (slider_x1, slider_y - 6), (slider_x2, slider_y + 6), (50, 50, 50), -1)
        # Center marker
        cv2.line(annotated, (frame_center_x, slider_y - 10), (frame_center_x, slider_y + 10), (255, 255, 255), 2)
        # Slider indicator pill
        indicator_x = int(frame_center_x + (offset * (slider_w / 2.0)))
        indicator_x = max(slider_x1, min(slider_x2, indicator_x))
        indicator_color = (0, 255, 0) if state == SteeringState.CENTERED else (0, 140, 255)
        cv2.circle(annotated, (indicator_x, slider_y), 8, indicator_color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (indicator_x, slider_y), 9, (255, 255, 255), 1, cv2.LINE_AA)


def run_path_following_preview(
    camera_index: int = 0,
    mode: DetectionMode = DetectionMode.ADAPTIVE,
) -> None:
    """
    Launches an interactive camera viewfinder demonstrating real-time path following and centering.
    
    Controls:
      - [1] : Adaptive thresholding mode
      - [2] : Dark line on light surface mode
      - [3] : Bright line on dark surface mode
      - [4] : Color tape mode (Blue)
      - [Q] or [ESC] : Quit
    """
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}")
        return

    tracker = PathTracker(mode=mode)
    window_name = "Robot Path Follower & Centering - [Q] Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("=" * 65)
    print(" Robot Path Following & Centering Vision Active")
    print(f" Mode: {tracker.mode.value}")
    print(" Controls:")
    print("   [1] Adaptive Mode (Dynamic lighting)")
    print("   [2] Dark Line Mode (Black tape on light floor)")
    print("   [3] Bright Line Mode (White/reflective tape on dark floor)")
    print("   [4] Color Tape Mode (Blue tape)")
    print("   [Q] / [ESC] Quit")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            result = tracker.process_frame(frame)
            cv2.imshow(window_name, result.annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("1"):
                tracker.mode = DetectionMode.ADAPTIVE
                print("[MODE] Switched to Adaptive Thresholding")
            elif key == ord("2"):
                tracker.mode = DetectionMode.DARK_LINE
                print("[MODE] Switched to Dark Line on Light Surface")
            elif key == ord("3"):
                tracker.mode = DetectionMode.BRIGHT_LINE
                print("[MODE] Switched to Bright Line on Dark Surface")
            elif key == ord("4"):
                tracker.set_color_range(lower=(100, 80, 50), upper=(130, 255, 255))
                print("[MODE] Switched to Color Tape (Blue)")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_path_following_preview()
