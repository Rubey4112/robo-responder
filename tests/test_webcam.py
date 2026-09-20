"""
Webcam Diagnostic & Verification Test Suite
-------------------------------------------
Tests and verifies camera inputs for OpenCV on Windows, specifically helping
to distinguish and switch between the internal laptop webcam and external USB webcams.

Hardware Mapping on this system:
  * Index 0: Integrated Webcam (Internal laptop camera)
  * Index 1: NexiGo N660P FHD Webcam (External USB camera)

Usage:
  # 1. Run automated unit tests:
  pytest tests/test_webcam.py

  # 2. Run diagnostic scan and test external webcam:
  python tests/test_webcam.py

  # 3. Open interactive live preview of external webcam (Index 1):
  python tests/test_webcam.py --preview --index 1

  # 4. List all detected hardware cameras and their OpenCV indices:
  python tests/test_webcam.py --list

  # 5. Capture a side-by-side comparison from both cameras:
  python tests/test_webcam.py --compare
"""

import argparse
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hardware Discovery & Camera Index Identification
# ---------------------------------------------------------------------------

def get_windows_camera_devices() -> List[Dict[str, str]]:
    """
    Queries Windows PnP to find all registered camera devices and friendly names.
    """
    if not sys.platform.startswith("win"):
        return []

    ps_script = """
    Get-CimInstance Win32_PnPEntity | 
        Where-Object { $_.PNPClass -eq 'Camera' -or $_.Service -like '*usbvideo*' } | 
        Select-Object FriendlyName, Name, DeviceID, Status | 
        ConvertTo-Json
    """
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            devices = []
            seen_ids = set()
            for item in data:
                dev_id = item.get("DeviceID", "")
                # Skip duplicate interface endpoints for the same physical camera
                base_id = dev_id.split("\\")[1] if "\\" in dev_id else dev_id
                name = item.get("FriendlyName") or item.get("Name", "Unknown Camera")
                if base_id not in seen_ids:
                    seen_ids.add(base_id)
                    devices.append({
                        "name": name,
                        "device_id": dev_id,
                        "status": item.get("Status", "OK"),
                    })
            return devices
    except Exception as exc:
        print(f"[WARN] Could not query Windows PnP camera list: {exc}")

    return []


def list_available_cameras(max_tested: int = 4) -> List[Dict]:
    """
    Probes camera indices using OpenCV (DirectShow and Media Foundation)
    and maps them to detected hardware camera names.
    
    Returns:
        List of dicts with keys: 'index', 'name', 'backend', 'width', 'height', 'fps', 'is_external'
    """
    hw_devices = get_windows_camera_devices()
    external_names = [d["name"] for d in hw_devices if "integrated" not in d["name"].lower() and "internal" not in d["name"].lower()]
    laptop_names = [d["name"] for d in hw_devices if "integrated" in d["name"].lower() or "internal" in d["name"].lower()]

    available = []
    preferred_backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, preferred_backend)
        if not cap.isOpened() and sys.platform.startswith("win"):
            # Fallback to Media Foundation if DSHOW fails
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)

        if cap.isOpened():
            # Grab a test frame
            ret, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend_name = cap.getBackendName()
            cap.release()

            if ret and frame is not None:
                # Infer friendly name from Windows device order
                # Typically: Index 0 is Laptop/Integrated, Index 1 is External USB (NexiGo)
                friendly_name = f"Camera {idx}"
                is_ext = False
                if idx == 0:
                    friendly_name = laptop_names[0] if laptop_names else "Integrated / Laptop Webcam"
                    is_ext = False
                elif idx == 1:
                    friendly_name = external_names[0] if external_names else "External USB Webcam (NexiGo)"
                    is_ext = True
                elif idx < len(hw_devices):
                    friendly_name = hw_devices[idx]["name"]
                    is_ext = "integrated" not in friendly_name.lower()

                available.append({
                    "index": idx,
                    "name": friendly_name,
                    "backend": backend_name,
                    "width": w,
                    "height": h,
                    "fps": fps if fps > 0 else 30.0,
                    "is_external": is_ext,
                })

    return available


def get_external_webcam_index() -> int:
    """
    Detects and returns the camera index for the external webcam.
    Defaults to 1 on this system, or the first detected external camera.
    """
    cameras = list_available_cameras()
    for cam in cameras:
        if cam["is_external"]:
            return cam["index"]
    # Fallback to Index 1 if at least two cameras exist, else Index 0
    return 1 if len(cameras) > 1 else 0


def open_camera_safe(camera_index: int, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """
    Safely opens a camera on Windows with proper backend priority (CAP_DSHOW -> CAP_MSMF -> CAP_ANY)
    and configures resolution.
    """
    if sys.platform.startswith("win"):
        # CAP_DSHOW provides fastest startup and reliable multi-camera indexing on Windows
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


# ---------------------------------------------------------------------------
# Core test_webcam function
# ---------------------------------------------------------------------------

def test_webcam(
    camera_index: Optional[int] = None,
    preview: bool = False,
    warmup_frames: int = 5,
    save_snapshot: bool = True,
    output_dir: str = "captures",
    width: int = 1280,
    height: int = 720,
) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Tests webcam capture from a given camera index or auto-detected external webcam.
    
    Args:
        camera_index: OpenCV camera index to test. If None, auto-selects the external webcam (Index 1).
        preview: If True, opens an interactive GUI window displaying live camera feed.
        warmup_frames: Number of initial frames to discard for auto-exposure stabilization.
        save_snapshot: If True, saves a test snapshot to disk.
        output_dir: Directory where test snapshot is saved.
        width: Desired capture width (default 1280).
        height: Desired capture height (default 720).
        
    Returns:
        Tuple of (success: bool, frame: Optional[np.ndarray], message: str)
    """
    if camera_index is None:
        camera_index = get_external_webcam_index()

    print(f"\n[INFO] Testing camera index {camera_index}...")

    cap = open_camera_safe(camera_index, width=width, height=height)

    if not cap.isOpened():
        err_msg = (
            f"Failed to open camera index {camera_index}! "
            f"On this machine, only indices 0 (Laptop) and 1 (External NexiGo) exist. "
            f"Index {camera_index} is unavailable or locked by another application."
        )
        print(f"[ERROR] {err_msg}")
        return False, None, err_msg

    backend_name = cap.getBackendName()
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SUCCESS] Camera {camera_index} opened with backend '{backend_name}' ({actual_w}x{actual_h})")

    # Warm up camera sensor
    last_frame = None
    for i in range(max(1, warmup_frames)):
        ret, frame = cap.read()
        if ret and frame is not None:
            last_frame = frame
        time.sleep(0.05)

    if last_frame is None:
        cap.release()
        err_msg = f"Camera {camera_index} opened but could not read valid frames."
        print(f"[ERROR] {err_msg}")
        return False, None, err_msg

    snapshot_path = ""
    if save_snapshot:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        snapshot_path = os.path.join(output_dir, f"test_webcam_idx{camera_index}.jpg")
        cv2.imwrite(snapshot_path, last_frame)
        print(f"[INFO] Saved test snapshot to: {snapshot_path}")

    # Optional interactive preview mode
    if preview:
        print("\n" + "=" * 60)
        print(f" LIVE PREVIEW: Camera Index {camera_index}")
        print(" Controls:")
        print("   [0] : Switch to Laptop Webcam (Index 0)")
        print("   [1] : Switch to External Webcam (Index 1)")
        print("   [S] : Save snapshot")
        print("   [Q] or [ESC] : Close Preview")
        print("=" * 60 + "\n")

        current_idx = camera_index
        window_name = f"Webcam Test Preview - Camera {current_idx} [Q to quit]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            fps_count = 0
            start_t = time.time()
            fps_display = 0.0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Frame grab failed in preview loop.")
                    time.sleep(0.1)
                    continue

                fps_count += 1
                elapsed = time.time() - start_t
                if elapsed >= 1.0:
                    fps_display = fps_count / elapsed
                    fps_count = 0
                    start_t = time.time()

                # Overlay HUD
                hud_frame = frame.copy()
                cam_label = "EXTERNAL (NexiGo)" if current_idx == 1 else "LAPTOP (Integrated)"
                status_text = f"Index {current_idx}: {cam_label} | {frame.shape[1]}x{frame.shape[0]} @ {fps_display:.1f} FPS"
                cv2.rectangle(hud_frame, (10, 10), (600, 48), (0, 0, 0), -1)
                cv2.putText(hud_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(hud_frame, "Keys: [0] Laptop | [1] External | [S] Save | [Q] Quit", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                    break
                elif key in [ord('0'), ord('1')]:
                    new_idx = int(chr(key))
                    if new_idx != current_idx:
                        print(f"[INFO] Switching from Camera {current_idx} -> Camera {new_idx}...")
                        cap.release()
                        current_idx = new_idx
                        cap = open_camera_safe(current_idx, width=width, height=height)
                        cv2.destroyAllWindows()
                        window_name = f"Webcam Test Preview - Camera {current_idx} [Q to quit]"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                elif key in [ord('s'), ord('S')]:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    snap_file = os.path.join(output_dir, f"capture_cam{current_idx}_{ts}.jpg")
                    cv2.imwrite(snap_file, frame)
                    print(f"[INFO] Manual snapshot saved: {snap_file}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        cap.release()

    msg = f"Camera {camera_index} verified successfully ({actual_w}x{actual_h} via {backend_name})"
    return True, last_frame, msg


def compare_both_cameras(output_path: str = "captures/camera_comparison.jpg") -> bool:
    """
    Captures one frame from both Camera 0 (Laptop) and Camera 1 (External),
    stitches them side-by-side, and saves the comparison image.
    """
    print("\n[INFO] Capturing comparison between Camera 0 (Laptop) and Camera 1 (External)...")
    frames = {}

    for idx, label in [(0, "Laptop Webcam (Index 0)"), (1, "External NexiGo Webcam (Index 1)")]:
        cap = open_camera_safe(idx, width=640, height=480)
        if cap.isOpened():
            for _ in range(5):  # warm up
                ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                # Add label overlay
                labeled = cv2.resize(frame, (640, 480))
                cv2.rectangle(labeled, (10, 10), (450, 45), (0, 0, 0), -1)
                cv2.putText(labeled, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                frames[idx] = labeled

    if len(frames) == 2:
        combined = np.hstack([frames[0], frames[1]])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, combined)
        print(f"[SUCCESS] Both cameras captured! Comparison saved to: {output_path}")
        return True
    elif len(frames) == 1:
        idx = list(frames.keys())[0]
        print(f"[WARN] Only Camera {idx} was available for comparison.")
        return False
    else:
        print("[ERROR] Neither camera could be opened.")
        return False


# ---------------------------------------------------------------------------
# Automated Unit Tests (pytest / unittest)
# ---------------------------------------------------------------------------

class TestWebcamSuite(unittest.TestCase):
    """Automated unit test suite for webcam hardware verification."""

    def test_list_cameras_detection(self):
        """Verifies that at least one camera is detected on the machine."""
        cameras = list_available_cameras()
        self.assertGreater(len(cameras), 0, "No cameras detected on this machine!")
        print(f"\n[Test] Detected {len(cameras)} active camera(s):")
        for c in cameras:
            print(f"  - Index {c['index']}: {c['name']} ({c['width']}x{c['height']} via {c['backend']})")

    def test_external_webcam_index_1(self):
        """Tests that the external webcam at Index 1 opens and provides valid image data."""
        success, frame, msg = test_webcam(camera_index=1, preview=False, save_snapshot=False)
        self.assertTrue(success, f"External webcam (Index 1) test failed: {msg}")
        self.assertIsNotNone(frame, "Frame returned from external webcam is None")
        self.assertEqual(len(frame.shape), 3, "Frame must be a 3-channel BGR image")
        self.assertGreater(frame.shape[0], 0, "Frame height must be > 0")
        self.assertGreater(frame.shape[1], 0, "Frame width must be > 0")
        # Ensure frame is not completely black/zeroes
        self.assertGreater(frame.mean(), 5.0, "Frame is completely black, camera sensor may be occluded")

    def test_laptop_webcam_index_0(self):
        """Tests that the internal laptop webcam at Index 0 opens and provides valid image data."""
        success, frame, msg = test_webcam(camera_index=0, preview=False, save_snapshot=False)
        self.assertTrue(success, f"Laptop webcam (Index 0) test failed: {msg}")
        self.assertIsNotNone(frame, "Frame returned from laptop webcam is None")
        self.assertEqual(len(frame.shape), 3)

    def test_distinguish_cameras(self):
        """Verifies that Index 0 and Index 1 point to distinct camera sensors."""
        _, frame0, _ = test_webcam(camera_index=0, preview=False, save_snapshot=False)
        _, frame1, _ = test_webcam(camera_index=1, preview=False, save_snapshot=False)

        if frame0 is not None and frame1 is not None:
            # Resize to match shapes for pixel difference
            f0_resized = cv2.resize(frame0, (320, 240))
            f1_resized = cv2.resize(frame1, (320, 240))
            diff = np.mean(np.abs(f0_resized.astype(float) - f1_resized.astype(float)))
            print(f"\n[Test] Pixel difference between Index 0 and Index 1: {diff:.2f}")
            self.assertGreater(diff, 1.0, "Index 0 and Index 1 appear to be identical streams!")


# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Webcam Diagnostic & Verification Test Tool")
    parser.add_argument("--index", type=int, default=None, help="Camera index to test (0=Laptop, 1=External NexiGo). Default: auto-detect external (1)")
    parser.add_argument("--preview", action="store_true", help="Launch live interactive camera preview window")
    parser.add_argument("--list", action="store_true", help="Scan and list all detected hardware cameras and OpenCV indices")
    parser.add_argument("--compare", action="store_true", help="Capture a side-by-side comparison from Camera 0 and Camera 1")
    parser.add_argument("--width", type=int, default=1280, help="Requested resolution width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested resolution height (default: 720)")
    args = parser.parse_args()

    if args.list:
        print("\n" + "=" * 60)
        print(" DETECTED CAMERAS ON THIS SYSTEM")
        print("=" * 60)
        cameras = list_available_cameras()
        if not cameras:
            print("  [!] No cameras detected.")
        for cam in cameras:
            ext_tag = " [EXTERNAL WEBCAM]" if cam["is_external"] else " [INTERNAL LAPTOP]"
            print(f"  * OpenCV Index {cam['index']}: {cam['name']}{ext_tag}")
            print(f"    Resolution: {cam['width']}x{cam['height']} | Backend: {cam['backend']}")
        print("=" * 60 + "\n")
        return

    if args.compare:
        compare_both_cameras()
        return

    # Run single test
    target_idx = args.index if args.index is not None else get_external_webcam_index()
    success, frame, msg = test_webcam(
        camera_index=target_idx,
        preview=args.preview,
        width=args.width,
        height=args.height,
        save_snapshot=True,
    )
    if success:
        print(f"\n[DONE] {msg}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    # If running with pytest or unittest test runner, let unittest handle it
    if len(sys.argv) > 1 and sys.argv[1] in ["-m", "discover"]:
        unittest.main()
    elif len(sys.argv) == 1:
        # Default behavior when running `python tests/test_webcam.py` without args:
        # Show detected devices and test the external camera
        print("\n" + "=" * 60)
        print(" WEBCAM DIAGNOSTIC & DISCOVERY TOOL")
        print("=" * 60)
        cams = list_available_cameras()
        print("\nFound Camera Devices:")
        for c in cams:
            tag = " <-- (EXTERNAL WEBCAM)" if c["is_external"] else " <-- (LAPTOP WEBCAM)"
            print(f"  * Index {c['index']}: {c['name']} ({c['width']}x{c['height']}){tag}")

        ext_idx = get_external_webcam_index()
        print(f"\nAuto-testing external camera at Index {ext_idx}...")
        test_webcam(camera_index=ext_idx, preview=False, save_snapshot=True)
        print("\nTip: To view live feed with HUD controls, run:")
        print("  python tests/test_webcam.py --preview --index 1")
        print("  python tests/test_webcam.py --compare\n")
    else:
        main()
