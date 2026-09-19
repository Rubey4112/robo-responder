"""
Webcam Multi-Sensor Module (Camera & Microphone)
-------------------------------------------------
Provides unified, synchronized access to both the optical image sensor (camera)
and acoustic sensor (microphone) of a webcam or connected hardware.

Features:
1. CameraSensor:
   - Captures live video stream using OpenCV (with DirectShow optimization on Windows).
   - Computes real-time optical sensor metrics: luminance/brightness, frame-difference motion score, and FPS.
2. MicrophoneSensor:
   - Captures live audio stream using SoundDevice.
   - Computes acoustic sensor metrics: RMS amplitude, dBFS, peak volume, and voice/sound activity trigger.
   - Provides live waveform buffer for visual oscilloscope rendering.
   - Automatically detects webcam microphone device (e.g. NexiGo, USB Webcam).
3. WebcamSensor:
   - Coordinates both sensors into a synchronized telemetry stream (WebcamReading).
   - Can save synchronized snapshots (JPEG image + WAV audio clip).
4. Visual HUD Preview:
   - Renders live camera feed with an integrated audio VU meter, oscilloscope waveform,
     and sensory indicators (luminance, motion, decibels, sound trigger).
"""

from dataclasses import dataclass
import argparse
import collections
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import List, Optional, Tuple
import wave

import cv2
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


@dataclass
class WebcamReading:
    """Synchronized optical and acoustic sensor snapshot."""
    timestamp: float
    # Optical / Camera Sensor Metrics
    frame: Optional[np.ndarray]
    frame_width: int
    frame_height: int
    fps: float
    luminance: float         # Average brightness: 0.0 (pitch black) to 255.0 (bright white)
    motion_score: float      # Motion energy: 0.0 (static) to 1.0 (rapid motion)
    camera_active: bool

    # Acoustic / Microphone Sensor Metrics
    audio_chunk: Optional[np.ndarray]
    sample_rate: int
    rms_volume: float        # Normalized RMS amplitude: 0.0 to 1.0
    dbfs: float              # Decibels relative to Full Scale: approx -80 dBFS to 0 dBFS
    peak_volume: float       # Peak audio sample amplitude: 0.0 to 1.0
    sound_detected: bool     # True if audio exceeds sound_threshold_db
    mic_active: bool


def list_audio_devices() -> List[dict]:
    """Returns a list of available audio input devices."""
    if sd is None:
        return []
    devices = []
    try:
        hostapis = sd.query_hostapis()
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                api_name = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else "Unknown"
                devices.append({
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "api": api_name,
                })
    except Exception as e:
        print(f"[WARNING] Error querying audio devices: {e}")
    return devices


def find_webcam_audio_device(preferred_keywords: Optional[List[str]] = None) -> Optional[int]:
    """
    Attempts to auto-detect a webcam microphone by matching known keywords
    in the audio device name (e.g., 'NexiGo', 'Webcam', 'USB', 'Camera').
    Falls back to default if no specific webcam device name matches.
    """
    if sd is None:
        return None

    if preferred_keywords is None:
        preferred_keywords = ["nexigo", "webcam", "camera", "fhd", "usb audio", "usb"]

    devices = list_audio_devices()
    for kw in preferred_keywords:
        for dev in devices:
            if kw.lower() in dev["name"].lower():
                return dev["index"]

    return None  # None uses system default input device


class CameraSensor:
    """
    Interfaces with the webcam's optical CMOS/CCD image sensor using OpenCV.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_frames: int = 5,
    ):
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.warmup_frames = warmup_frames

        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_width = 0
        self.actual_height = 0

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_frame_time = time.time()
        self.fps = 0.0

    def start(self) -> bool:
        """Opens the camera video capture device."""
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera device at index {self.camera_index}.")
            return False

        if self.requested_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sensor warm-up
        for _ in range(max(1, self.warmup_frames)):
            self.cap.read()

        self.prev_frame_time = time.time()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float, float]:
        """
        Reads one frame and calculates optical sensor metrics.

        Returns:
            (success, frame, luminance, motion_score)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, 0.0, 0.0

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, 0.0, 0.0

        # Calculate FPS
        now = time.time()
        dt = now - self.prev_frame_time
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = 0.8 * self.fps + 0.2 * current_fps if self.fps > 0 else current_fps
        self.prev_frame_time = now

        # Convert to grayscale for metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Luminance (Average scene brightness)
        luminance = float(np.mean(gray))

        # 2. Optical Motion Energy (Frame difference)
        motion_score = 0.0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self.prev_gray)
            motion_score = float(np.mean(diff) / 255.0)  # Normalized 0.0 to 1.0
            motion_score = min(1.0, motion_score * 5.0)  # Scale up for responsiveness
        self.prev_gray = gray

        return True, frame, luminance, motion_score

    def stop(self) -> None:
        """Releases the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.prev_gray = None


class MicrophoneSensor:
    """
    Interfaces with the webcam's acoustic microphone sensor using SoundDevice.
    Captures live audio samples and computes acoustic levels and sound events.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_size: int = 1024,
        sound_threshold_db: float = -38.0,
        waveform_history: int = 1024,
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.sound_threshold_db = sound_threshold_db

        self.stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

        # Telemetry state
        self.latest_chunk: Optional[np.ndarray] = None
        self.rms_volume: float = 0.0
        self.dbfs: float = -80.0
        self.peak_volume: float = 0.0
        self.sound_detected: bool = False

        # Live waveform buffer for HUD oscilloscope
        self.waveform_buffer = collections.deque(maxlen=waveform_history)
        for _ in range(waveform_history):
            self.waveform_buffer.append(0.0)

        # Recording buffer for snapshots
        self.recent_samples = collections.deque(maxlen=sample_rate * 5)  # 5 seconds rolling buffer

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback invoked by sounddevice for each new audio buffer."""
        if status:
            pass  # Overflow / underflow warning ignored for non-critical stream

        # Flatten to mono 1D float array
        audio = indata[:, 0].astype(np.float32)

        # Compute RMS volume
        square_sum = np.sum(audio ** 2)
        rms = math.sqrt(square_sum / len(audio)) if len(audio) > 0 else 0.0
        rms = min(1.0, rms)

        # Compute dBFS (Decibels relative to Full Scale)
        dbfs = 20.0 * math.log10(max(rms, 1e-4))
        dbfs = max(-80.0, min(0.0, dbfs))

        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        detected = dbfs > self.sound_threshold_db

        with self._lock:
            self.latest_chunk = audio.copy()
            self.rms_volume = rms
            self.dbfs = dbfs
            self.peak_volume = peak
            self.sound_detected = detected

            # Store in rolling buffers
            for sample in audio:
                self.waveform_buffer.append(float(sample))
                self.recent_samples.append(float(sample))

    def start(self) -> bool:
        """Starts the audio recording stream."""
        if sd is None:
            print("[WARNING] sounddevice library is not installed. Audio capture disabled.")
            return False

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            )
            self.stream.start()
            dev_name = "Default Device"
            if self.device_index is not None:
                try:
                    dev_info = sd.query_devices(self.device_index)
                    dev_name = dev_info["name"]
                except Exception:
                    dev_name = f"Index {self.device_index}"
            print(f"[INFO] Acoustic Microphone Sensor started on: {dev_name} ({self.sample_rate} Hz)")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start microphone sensor: {e}")
            self.stream = None
            return False

    def read_metrics(self) -> Tuple[Optional[np.ndarray], float, float, float, bool, np.ndarray]:
        """
        Thread-safe read of latest acoustic sensor values.

        Returns:
            (chunk, rms, dbfs, peak, sound_detected, waveform_array)
        """
        with self._lock:
            chunk = self.latest_chunk.copy() if self.latest_chunk is not None else None
            rms = self.rms_volume
            dbfs = self.dbfs
            peak = self.peak_volume
            detected = self.sound_detected
            waveform = np.array(self.waveform_buffer, dtype=np.float32)

        return chunk, rms, dbfs, peak, detected, waveform

    def get_recent_audio_seconds(self, seconds: float = 2.0) -> np.ndarray:
        """Returns the most recent N seconds of recorded audio."""
        num_samples = int(self.sample_rate * seconds)
        with self._lock:
            samples = list(self.recent_samples)
        if len(samples) > num_samples:
            samples = samples[-num_samples:]
        return np.array(samples, dtype=np.float32)

    def stop(self) -> None:
        """Stops the audio stream."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class WebcamSensor:
    """
    Unified Multi-Sensor Controller for Webcam Video & Microphone.
    Coordinates the camera sensor and microphone sensor, provides synchronized
    telemetry, snapshots, and interactive HUD visualization.
    """

    def __init__(
        self,
        camera_index: int = 0,
        audio_device_index: Optional[int] = None,
        auto_match_microphone: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sample_rate: int = 44100,
        sound_threshold_db: float = -38.0,
    ):
        self.camera_index = camera_index
        self.auto_match_microphone = auto_match_microphone

        # Auto-match webcam microphone if requested
        if audio_device_index is None and auto_match_microphone:
            detected_mic = find_webcam_audio_device()
            if detected_mic is not None:
                print(f"[INFO] Auto-detected webcam microphone at device index {detected_mic}")
                audio_device_index = detected_mic

        self.audio_device_index = audio_device_index
        self.camera = CameraSensor(camera_index=camera_index, width=width, height=height)
        self.microphone = MicrophoneSensor(
            device_index=audio_device_index,
            sample_rate=sample_rate,
            sound_threshold_db=sound_threshold_db,
        )
        self.is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self) -> bool:
        """Starts both optical and acoustic sensors."""
        print("[INFO] Initializing Webcam Multi-Sensor System...")
        cam_ok = self.camera.start()
        mic_ok = self.microphone.start()
        self.is_running = cam_ok or mic_ok
        return self.is_running

    def get_sensor_reading(self) -> WebcamReading:
        """
        Polls both sensors to return a synchronized multimodal sensor reading.
        """
        now = time.time()
        cam_ok, frame, luminance, motion_score = self.camera.read()
        audio_chunk, rms, dbfs, peak, sound_detected, waveform = self.microphone.read_metrics()

        return WebcamReading(
            timestamp=now,
            frame=frame,
            frame_width=self.camera.actual_width,
            frame_height=self.camera.actual_height,
            fps=self.camera.fps,
            luminance=luminance,
            motion_score=motion_score,
            camera_active=cam_ok,
            audio_chunk=audio_chunk,
            sample_rate=self.microphone.sample_rate,
            rms_volume=rms,
            dbfs=dbfs,
            peak_volume=peak,
            sound_detected=sound_detected,
            mic_active=(self.microphone.stream is not None and self.microphone.stream.active),
        )

    def save_snapshot(
        self,
        output_dir: str = "captures",
        file_prefix: str = "sensor_capture",
        audio_seconds: float = 2.0,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Captures a synchronized snapshot:
        - Current camera frame saved as JPEG
        - Last N seconds of microphone audio saved as WAV
        """
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Read current frame
        reading = self.get_sensor_reading()
        image_path = None
        audio_path = None

        if reading.frame is not None:
            image_path = str(target_dir / f"{file_prefix}_{timestamp}.jpg")
            cv2.imwrite(image_path, reading.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        audio_data = self.microphone.get_recent_audio_seconds(audio_seconds)
        if len(audio_data) > 0:
            audio_path = str(target_dir / f"{file_prefix}_{timestamp}.wav")
            int_data = (audio_data * 32767).astype(np.int16)
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.microphone.sample_rate)
                wf.writeframes(int_data.tobytes())

        return image_path, audio_path

    def stop(self) -> None:
        """Stops and releases both camera and microphone sensors."""
        print("[INFO] Releasing Webcam Multi-Sensor resources...")
        self.camera.stop()
        self.microphone.stop()
        self.is_running = False


def render_sensor_hud(
    frame: np.ndarray,
    reading: WebcamReading,
    waveform: np.ndarray,
    status_message: str = "",
    status_time_remaining: float = 0.0,
) -> np.ndarray:
    """
    Draws a visual sensor HUD over the camera frame:
    - Top telemetry bar (FPS, Luminance, Motion score)
    - Bottom audio visualizer:
        - Live Volume VU meter (Green -> Yellow -> Red)
        - Mini Oscilloscope waveform
        - Decibel (dBFS) readout & Sound Trigger indicator
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # --- TOP TELEMETRY BAR ---
    top_bar_h = 36
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Status indicators
    cam_status_color = (80, 220, 80) if reading.camera_active else (80, 80, 220)
    mic_status_color = (80, 220, 80) if reading.mic_active else (80, 80, 220)

    cv2.circle(canvas, (18, 18), 6, cam_status_color, -1)
    cv2.putText(canvas, "CAM", (28, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.circle(canvas, (80, 18), 6, mic_status_color, -1)
    cv2.putText(canvas, "MIC", (90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    telemetry_str = (
        f"FPS: {reading.fps:4.1f} | "
        f"Light: {reading.luminance:3.0f}/255 | "
        f"Motion: {int(reading.motion_score * 100)}%"
    )
    cv2.putText(canvas, telemetry_str, (150, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    # --- BOTTOM AUDIO SENSOR DASHBOARD ---
    bottom_h = 100
    bottom_y = h - bottom_h
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, bottom_y), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

    # 1. VU Meter (Audio level bar)
    meter_x = 20
    meter_y = bottom_y + 20
    meter_w = int(w * 0.38)
    meter_h = 16
    cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (40, 40, 40), -1)
    cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (80, 80, 80), 1)

    # Fill percentage based on RMS (scaled logarithmically for natural VU response)
    fill_ratio = min(1.0, max(0.0, (reading.dbfs + 60.0) / 60.0))
    fill_w = int(meter_w * fill_ratio)

    if fill_w > 0:
        # Dynamic color: Green -> Amber -> Red
        if fill_ratio < 0.6:
            vu_color = (60, 200, 60)
        elif fill_ratio < 0.85:
            vu_color = (40, 190, 255)
        else:
            vu_color = (40, 40, 240)
        cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + fill_w, meter_y + meter_h), vu_color, -1)

    # VU Label & dBFS value
    vu_label = f"Audio Level: {reading.dbfs:4.1f} dBFS"
    cv2.putText(canvas, vu_label, (meter_x, meter_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Sound Activity Trigger Alert
    if reading.sound_detected:
        cv2.rectangle(canvas, (meter_x, meter_y + 24), (meter_x + 150, meter_y + 44), (20, 20, 180), -1)
        cv2.putText(
            canvas,
            "! SOUND TRIGGER !",
            (meter_x + 8, meter_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "Ambient / Quiet",
            (meter_x, meter_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

    # 2. Oscilloscope / Waveform Visualizer
    scope_x = int(w * 0.45)
    scope_y = bottom_y + 12
    scope_w = int(w * 0.52)
    scope_h = 56
    cv2.rectangle(canvas, (scope_x, scope_y), (scope_x + scope_w, scope_y + scope_h), (25, 25, 25), -1)
    cv2.rectangle(canvas, (scope_x, scope_y), (scope_x + scope_w, scope_y + scope_h), (60, 60, 60), 1)

    # Centerline
    mid_y = scope_y + scope_h // 2
    cv2.line(canvas, (scope_x, mid_y), (scope_x + scope_w, mid_y), (45, 45, 45), 1)

    # Draw waveform points
    if len(waveform) > 1:
        step = max(1, len(waveform) // scope_w)
        sampled = waveform[::step][:scope_w]
        pts = []
        for i, val in enumerate(sampled):
            px = scope_x + i
            # Scale sample amplitude into scope box
            py = int(mid_y - (val * (scope_h // 2 - 2)))
            py = max(scope_y + 1, min(scope_y + scope_h - 1, py))
            pts.append((px, py))

        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], (0, 235, 235), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Live Acoustic Waveform (Acoustic Sensor)", (scope_x, scope_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # Controls Helper on bottom edge
    controls_text = "[SPACE]: Capture Synced Audio+Image | [Q]: Quit"
    cv2.putText(canvas, controls_text, (meter_x, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)

    # Temporary feedback notification banner
    if status_time_remaining > 0 and status_message:
        cv2.rectangle(canvas, (0, top_bar_h), (w, top_bar_h + 36), (40, 120, 60), -1)
        cv2.putText(canvas, status_message, (20, top_bar_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def run_sensor_hud(
    camera_index: int = 0,
    audio_device_index: Optional[int] = None,
    output_dir: str = "captures",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Runs the interactive visual HUD for monitoring both webcam camera and microphone sensors.
    """
    sensor = WebcamSensor(
        camera_index=camera_index,
        audio_device_index=audio_device_index,
        auto_match_microphone=True,
        width=width,
        height=height,
    )

    if not sensor.start():
        print("[ERROR] Failed to start Webcam Sensor system.")
        return

    window_name = "Webcam Multi-Sensor Stream (Camera + Microphone)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    status_msg = ""
    status_expiry = 0.0

    print("=" * 65)
    print(" Webcam Multi-Sensor Monitor (Optical + Acoustic Sensors)")
    print(" Controls:")
    print("   - [SPACE]  : Capture synchronized snapshot (JPEG image + WAV audio)")
    print("   - [Q]/[ESC]: Exit")
    print("=" * 65)

    try:
        while True:
            reading = sensor.get_sensor_reading()
            if not reading.camera_active or reading.frame is None:
                time.sleep(0.05)
                continue

            _, _, _, _, _, waveform = sensor.microphone.read_metrics()

            now = time.time()
            time_remaining = max(0.0, status_expiry - now)

            hud_frame = render_sensor_hud(
                reading.frame,
                reading,
                waveform,
                status_message=status_msg,
                status_time_remaining=time_remaining,
            )

            cv2.imshow(window_name, hud_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("[INFO] Quitting sensor monitor...")
                break
            elif key == ord(" ") or key == ord("c") or key == ord("C"):
                img_path, aud_path = sensor.save_snapshot(output_dir=output_dir)
                status_msg = f"Saved Synced Snapshot! Image & Audio in {output_dir}/"
                status_expiry = now + 2.5
                print(f"[SNAPSHOT] Image: {img_path}")
                print(f"[SNAPSHOT] Audio: {aud_path}")

    finally:
        sensor.stop()
        cv2.destroyAllWindows()


def run_sensor_monitor_cli(
    camera_index: int = 0,
    audio_device_index: Optional[int] = None,
    duration_seconds: float = 10.0,
    interval: float = 0.5,
) -> None:
    """
    Runs a headless console telemetry monitor logging live readings from both sensors.
    """
    with WebcamSensor(
        camera_index=camera_index,
        audio_device_index=audio_device_index,
        auto_match_microphone=True,
    ) as sensor:
        print(f"\n[INFO] Streaming sensor telemetry for {duration_seconds}s (interval: {interval}s)...")
        print(f"{'Time (s)':<10} | {'FPS':<6} | {'Light (0-255)':<14} | {'Motion':<8} | {'Audio (dBFS)':<13} | {'Sound Detected':<14}")
        print("-" * 75)

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            reading = sensor.get_sensor_reading()
            elapsed = time.time() - start_time
            print(
                f"{elapsed:<10.1f} | "
                f"{reading.fps:<6.1f} | "
                f"{reading.luminance:<14.1f} | "
                f"{int(reading.motion_score * 100):>3}%   | "
                f"{reading.dbfs:<13.1f} | "
                f"{str(reading.sound_detected):<14}"
            )
            time.sleep(interval)


def print_available_sensors():
    """Prints all detected camera and audio sensors on the host machine."""
    print("\n" + "=" * 60)
    print(" DETECTED HARDWARE SENSORS")
    print("=" * 60)

    # 1. Optical Sensors (Cameras)
    print("\n[Optical / Camera Sensors]")
    for idx in range(4):
        if sys.platform.startswith("win"):
            test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not test_cap.isOpened():
                test_cap = cv2.VideoCapture(idx)
        else:
            test_cap = cv2.VideoCapture(idx)

        if test_cap.isOpened():
            w = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  * Camera Index {idx}: Available (Native {w}x{h})")
            test_cap.release()

    # 2. Acoustic Sensors (Microphones)
    print("\n[Acoustic / Microphone Sensors]")
    devices = list_audio_devices()
    matched_idx = find_webcam_audio_device()
    for dev in devices:
        is_webcam = (dev["index"] == matched_idx)
        tag = "  <-- (Auto-detected Webcam Microphone)" if is_webcam else ""
        print(f"  * Device {dev['index']:2d}: {dev['name']} ({dev['channels']} ch, API: {dev['api']}){tag}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Webcam Multi-Sensor Streamer (Camera & Microphone)")
    parser.add_argument(
        "--mode",
        choices=["preview", "monitor", "snapshot", "list"],
        default="preview",
        help="Operation mode: 'preview' (Interactive HUD), 'monitor' (Headless CLI Telemetry), 'snapshot' (Single capture), or 'list' (List sensors)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--audio-device", type=int, default=None, help="Audio input device index (default: auto-detected)")
    parser.add_argument("--output-dir", type=str, default="captures", help="Output directory for snapshots")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration for monitor mode in seconds")
    parser.add_argument("--width", type=int, default=None, help="Target frame width")
    parser.add_argument("--height", type=int, default=None, help="Target frame height")

    args = parser.parse_args()

    if args.mode == "list":
        print_available_sensors()
    elif args.mode == "preview":
        run_sensor_hud(
            camera_index=args.camera,
            audio_device_index=args.audio_device,
            output_dir=args.output_dir,
            width=args.width,
            height=args.height,
        )
    elif args.mode == "monitor":
        run_sensor_monitor_cli(
            camera_index=args.camera,
            audio_device_index=args.audio_device,
            duration_seconds=args.duration,
        )
    elif args.mode == "snapshot":
        with WebcamSensor(
            camera_index=args.camera,
            audio_device_index=args.audio_device,
            auto_match_microphone=True,
            width=args.width,
            height=args.height,
        ) as sensor:
            img, aud = sensor.save_snapshot(output_dir=args.output_dir)
            print(f"[SUCCESS] Saved snapshot image: {img}")
            print(f"[SUCCESS] Saved snapshot audio: {aud}")


if __name__ == "__main__":
    main()
