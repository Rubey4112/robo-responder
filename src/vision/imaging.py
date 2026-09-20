"""
Computer Vision Photo Capture Module (OpenCV)
---------------------------------------------
Captures photos from a webcam or connected camera and saves them as high-quality JPEG files.

Features:
1. Interactive GUI mode with live preview, HUD overlay, and single-key capture.
2. Programmatic single-shot capture (with camera sensor warm-up).
3. Burst/Interval capture for gathering image datasets.
4. Command-line interface with customizable resolution, camera index, and output directory.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def open_camera(camera_index: int = 1) -> cv2.VideoCapture:
    """
    Initializes and opens the camera capture device.
    Uses DirectShow (CAP_DSHOW) on Windows for fast startup, falling back to default.
    """
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    return cap


def capture_single_photo(
    output_path: Optional[str] = None,
    camera_index: int = 1,
    warmup_frames: int = 5,
    jpeg_quality: int = 95,
) -> Optional[str]:
    """
    Takes a single photo programmatically without an interactive window.
    
    Args:
        output_path: Target path (e.g. 'captures/photo.jpg'). If None, an auto-timestamped path is used.
        camera_index: Index of the camera (default: 0).
        warmup_frames: Number of warmup frames to discard for camera auto-exposure.
        jpeg_quality: JPEG compression quality (0-100, default: 95).
        
    Returns:
        The absolute path to the saved JPEG file if successful, or None on failure.
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("captures")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"photo_{timestamp}.jpg")
    else:
        output_dir = Path(output_path).parent
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with index {camera_index}.")
        return None

    try:
        # Camera sensor warmup (allows auto-exposure, auto-white-balance to stabilize)
        for _ in range(max(1, warmup_frames)):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to read frame from camera.")
            return None

        # Ensure .jpg or .jpeg extension
        if not (output_path.lower().endswith(".jpg") or output_path.lower().endswith(".jpeg")):
            output_path += ".jpg"

        # Save as JPEG with specified quality
        success = cv2.imwrite(
            output_path,
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(0, min(100, jpeg_quality))],
        )

        if success:
            abs_path = os.path.abspath(output_path)
            print(f"[SUCCESS] Photo saved to: {abs_path}")
            return abs_path
        else:
            print(f"[ERROR] Failed to write image file to {output_path}")
            return None

    finally:
        cap.release()


def capture_interactive(
    output_dir: str = "captures",
    camera_index: int = 1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    jpeg_quality: int = 95,
) -> None:
    """
    Runs an interactive camera feed window.
    
    Controls:
      - [SPACE] or [C] : Capture photo and save as JPEG
      - [Q] or [ESC]   : Exit application
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera at index {camera_index}.")
        return

    # Set custom resolution if requested
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_title = "Camera Capture - Press [SPACE] to Photo, [Q] to Quit"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    photo_count = 0
    feedback_message = ""
    feedback_until = 0.0

    print("=" * 60)
    print(f" Camera started (Device {camera_index}, {actual_w}x{actual_h})")
    print(" Controls:")
    print("   - Press [SPACE] or [C] to snap a photo")
    print("   - Press [Q] or [ESC] to quit")
    print(f" Saving photos to: {target_dir.resolve()}")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Could not read frame from camera. Retrying...")
                time.sleep(0.1)
                continue

            # Create display frame to draw UI overlays without altering the captured image
            display_frame = frame.copy()
            now = time.time()

            # Status bar overlay (bottom banner)
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, h - 40), (w, h), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                f"Res: {w}x{h} | Saved: {photo_count} | [SPACE]: Capture | [Q]: Quit",
                (15, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Draw temporary capture feedback banner if recently captured
            if now < feedback_until:
                cv2.rectangle(display_frame, (0, 0), (w, 50), (46, 139, 87), -1)
                cv2.putText(
                    display_frame,
                    feedback_message,
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit on 'q' or ESC
            if key == ord("q") or key == 27:
                print("[INFO] Exiting...")
                break

            # Capture on SPACE or 'c'
            elif key == ord(" ") or key == ord("c") or key == ord("C"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                filename = f"photo_{timestamp}.jpg"
                filepath = target_dir / filename

                saved = cv2.imwrite(
                    str(filepath),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), max(0, min(100, jpeg_quality))],
                )

                if saved:
                    photo_count += 1
                    feedback_message = f"Saved: {filename}"
                    feedback_until = now + 1.8
                    print(f"[{photo_count}] Saved photo: {filepath.resolve()}")
                else:
                    feedback_message = "Failed to save photo!"
                    feedback_until = now + 1.8
                    print(f"[ERROR] Failed to save {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def capture_interval(
    interval_seconds: float = 2.0,
    max_photos: int = 10,
    output_dir: str = "captures/dataset",
    camera_index: int = 1,
    jpeg_quality: int = 95,
) -> None:
    """
    Captures a series of photos at a regular time interval (great for dataset collection).
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        return

    print(f"[INFO] Burst capture mode: {max_photos} photos, every {interval_seconds}s.")
    try:
        # Warmup
        for _ in range(5):
            cap.read()

        for i in range(1, max_photos + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[WARNING] Skipping frame {i}: could not read from camera.")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"burst_{i:04d}_{timestamp}.jpg"
            filepath = target_dir / filename

            cv2.imwrite(
                str(filepath),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            print(f"[{i}/{max_photos}] Saved {filepath.name}")

            if i < max_photos:
                time.sleep(interval_seconds)

    finally:
        cap.release()
        print(f"[DONE] Finished burst capture. Files saved in: {target_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="OpenCV Photo Capture Utility (JPEG)")
    parser.add_argument(
        "--mode",
        choices=["interactive", "single", "interval"],
        default="interactive",
        help="Capture mode: 'interactive' (live preview GUI), 'single' (one shot), or 'interval' (time-lapse / burst)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="captures",
        help="Directory to save JPEG photos (default: 'captures')",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Filename or path for single photo mode (e.g., 'photo.jpg')",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (0-100, default: 95)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Requested camera frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Requested camera frame height",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between photos in interval mode",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of photos to capture in interval mode",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        capture_interactive(
            output_dir=args.output_dir,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            jpeg_quality=args.quality,
        )
    elif args.mode == "single":
        capture_single_photo(
            output_path=args.output_file or os.path.join(args.output_dir, "single_shot.jpg"),
            camera_index=args.camera,
            jpeg_quality=args.quality,
        )
    elif args.mode == "interval":
        capture_interval(
            interval_seconds=args.interval,
            max_photos=args.count,
            output_dir=args.output_dir,
            camera_index=args.camera,
            jpeg_quality=args.quality,
        )


if __name__ == "__main__":
    main()
