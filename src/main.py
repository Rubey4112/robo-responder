#!/usr/bin/env python3
"""
Roboguide: Autonomous Robotic Evacuation Responder
===================================================
Main demonstration program integrating live OpenCV camera vision,
Google Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) spatial
reasoning, and physical XRP robot motor control over USB-Serial (`COM4`).

Designed for live hackathon presentations and judge evaluations:
  - Live video feed with real-time HUD telemetry & spatial decision overlays.
  - Multi-turn spatial memory allowing Gemini to track movement across frames.
  - Automatic hardware tool execution on the physical XRP rover.
  - High-visibility terminal logging formatted for clear stage demonstrations.
  - Fail-safe emergency stops ([S] key, [Q] key, or Ctrl+C).

Usage Examples:
  # 1. Run standard autonomous navigation demo with GUI HUD:
  python src/main.py

  # 2. Run in step-by-step mode (press SPACE to advance each turn):
  python src/main.py --mode step

  # 3. Specify external webcam index (e.g., USB camera on index 1):
  python src/main.py --camera 1

  # 4. Run headless (terminal-only, no OpenCV window):
  python src/main.py --headless

  # 5. Limit to 5 turns for a quick demo:
  python src/main.py --turns 5
"""

import argparse
import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.agent.driving_functions import (
    DRIVING_TOOLS,
    AsyncRobotBridge,
    detect_xrp_port,
    execute_async_tool,
    get_robot_bridge,
)
from src.agent.llm_client import (
    EMERGENCY_SYSTEM_INSTRUCTIONS,
    ROBOGUIDE_SYSTEM_INSTRUCTION,
    RoboguideClient,
)
from src.audio.live_audio import (
    AudioPlayer,
    RoboguideLiveAudioSession,
    get_audio_session,
)

# Configure logger
logger = logging.getLogger("RoboguideMain")

# Terminal ANSI Color Codes for Rich Formatting
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"


# Configure stdout for safe UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================================
# Rich Terminal Telemetry Formatters
# ============================================================================

def print_banner(model: str, audio_model: str, port: str, is_simulated: bool, camera_idx: int, mode: str, hazard: str, state: str):
    """Displays the executive hackathon header banner."""
    hw_status = f"{C_YELLOW}SIMULATION (No Hardware){C_RESET}" if is_simulated else f"{C_GREEN}{port} (Connected){C_RESET}"
    banner = f"""
{C_CYAN}{C_BOLD}+==============================================================================+
|             ROBOGUIDE: DUAL-AGENT AUTONOMOUS EVACUATION ROVER                |
|      Powered by Gemini 3.8 Live (Audio) & Gemini Robotics ER 2 (Spatial)      |
+==============================================================================+{C_RESET}
  {C_BOLD}[*] Spatial Reasoning Model :{C_RESET} {C_MAGENTA}{model}{C_RESET}
  {C_BOLD}[*] Live Audio Voice Model  :{C_RESET} {C_CYAN}{audio_model}{C_RESET}
  {C_BOLD}[*] XRP Rover Hardware Bus  :{C_RESET} {hw_status}
  {C_BOLD}[*] Live Camera Optical Feed:{C_RESET} {C_WHITE}Device {camera_idx} (DirectShow / OpenCV){C_RESET}
  {C_BOLD}[*] Initial System State    :{C_RESET} {C_YELLOW}{state.upper()}{C_RESET} | {C_BOLD}Active Protocol:{C_RESET} {C_GREEN}{hazard.upper()}{C_RESET}
  {C_BOLD}[*] Operational Mode        :{C_RESET} {C_YELLOW}{mode.upper()}{C_RESET}
  {C_BOLD}[*] Demo Stage Controls     :{C_RESET} {C_WHITE}[E] Emergency Triage | [SPACE] Step | [S] STOP | [Q] Exit{C_RESET}
{C_CYAN}--------------------------------------------------------------------------------{C_RESET}
"""
    print(banner)


def print_turn_start(turn: int, max_turns: int, hazard: str):
    """Prints turn start telemetry."""
    turns_info = f"{turn}/{max_turns}" if max_turns > 0 else f"{turn}"
    print(f"\n{C_BLUE}{C_BOLD}==================== [TURN {turns_info} - PROTOCOL: {hazard.upper()}] ===================={C_RESET}")
    print(f"{C_CYAN}[CAMERA]{C_RESET} Flushing optical sensor buffer and capturing fresh environment frame...")


def print_gemini_start():
    """Prints Gemini reasoning status."""
    print(f"{C_MAGENTA}[GEMINI ER 2]{C_RESET} Streaming frame to Gemini API... Analyzing spatial geometry & path safety...")


def print_action_executed(action_name: str, duration: float, port: str, status: str):
    """Prints the physical hardware action dispatch."""
    dur_str = f"for {duration:.1f}s" if duration > 0 else "immediately"
    print(
        f"  {C_BOLD}{C_GREEN}>>> [ROBOT MOTOR DISPATCH] >>>{C_RESET} "
        f"{C_BOLD}{C_YELLOW}{action_name.upper()}{C_RESET} {dur_str} on {port} "
        f"({C_GREEN}STATUS: {status.upper()}{C_RESET})"
    )


def print_gemini_decision(reasoning: str, last_action: str):
    """Prints formatted summary of the model's spatial reasoning and evacuee announcement."""
    print(f"\n{C_MAGENTA}{C_BOLD}+-- [GEMINI ER 2 SPATIAL REASONING & DECISION] ----------------------------+{C_RESET}")
    for line in reasoning.strip().splitlines():
        print(f"{C_MAGENTA}|{C_RESET}  {line}")
    print(f"{C_MAGENTA}+--------------------------------------------------------------------------+{C_RESET}")
    print(f"{C_CYAN}[ACTION COMMITTED]:{C_RESET} {C_BOLD}{C_GREEN}{last_action}{C_RESET}")


# ============================================================================
# OpenCV Graphical HUD Renderer
# ============================================================================

def render_hud(
    frame: np.ndarray,
    turn: int,
    status_text: str,
    last_action: str,
    evacuee_text: str,
    port: str,
    is_simulated: bool,
    hazard: str = "fire",
    state: str = "EVACUATION",
    motion_count: int = 0,
) -> np.ndarray:
    """
    Renders a broadcast-quality semi-transparent HUD overlay on top of the
    live OpenCV video frame for judge observation.
    """
    hud = frame.copy()
    h, w = hud.shape[:2]

    # 1. Semi-transparent Top Header Bar
    top_bar_height = 68
    top_overlay = hud[0:top_bar_height, 0:w].copy()
    cv2.rectangle(top_overlay, (0, 0), (w, top_bar_height), (20, 20, 25), -1)
    hud[0:top_bar_height, 0:w] = cv2.addWeighted(top_overlay, 0.75, hud[0:top_bar_height, 0:w], 0.25, 0)

    # Title & Subtitle
    cv2.putText(
        hud,
        "ROBOGUIDE  |  DUAL-AGENT AI EVACUATION ROVER (LIVE AUDIO + ER 2)",
        (16, 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (0, 235, 255),  # Cyan
        1,
        cv2.LINE_AA,
    )

    hw_label = f"ROBOT: {port}" if not is_simulated else "ROBOT: SIMULATED"
    state_color = (0, 255, 120) if state == "EVACUATION" else (0, 220, 255)
    cv2.putText(
        hud,
        f"STATE: {state}  |  HAZARD: {hazard.upper()}  |  {hw_label}  |  STATUS: {status_text}",
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        state_color,
        1,
        cv2.LINE_AA,
    )

    # 2. Semi-transparent Bottom Action Banner
    bottom_bar_height = 68
    y_start = h - bottom_bar_height
    bottom_overlay = hud[y_start:h, 0:w].copy()
    cv2.rectangle(bottom_overlay, (0, 0), (w, bottom_bar_height), (15, 18, 28), -1)
    hud[y_start:h, 0:w] = cv2.addWeighted(bottom_overlay, 0.80, hud[y_start:h, 0:w], 0.20, 0)

    # Motor Action Line
    cv2.putText(
        hud,
        f"ACTION: {last_action}   |   MOTIONS: {motion_count}",
        (16, y_start + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (0, 255, 255),  # Yellow
        1,
        cv2.LINE_AA,
    )

    # Evacuee Guidance Line (truncated to fit)
    display_directive = evacuee_text.replace("\n", " ")[:85]
    if len(evacuee_text) > 85:
        display_directive += "..."
    cv2.putText(
        hud,
        f"VOICE: \"{display_directive}\"",
        (16, y_start + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),  # Crisp White
        1,
        cv2.LINE_AA,
    )

    # Center spatial crosshair
    cx, cy = w // 2, h // 2
    cv2.drawMarker(hud, (cx, cy), (0, 255, 200), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

    return hud


# ============================================================================
# Camera Vision Manager
# ============================================================================

class VisionSensor:
    """Manages OpenCV webcam capture, sensor warmup, and fresh frame retrieval."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Opens camera using standard backend or DirectShow on Windows."""
        # Try direct open first
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened() and sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            # Try alternate fallback index 1 if 0 fails
            alt_idx = 1 if self.camera_index == 0 else 0
            logger.warning(f"Could not open camera {self.camera_index}. Trying alternate index {alt_idx}...")
            self.cap = cv2.VideoCapture(alt_idx)
            if self.cap.isOpened():
                self.camera_index = alt_idx

        if not self.cap.isOpened():
            return False

        # Set requested resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Warm up optical sensor (let auto-exposure settle)
        for _ in range(5):
            self.cap.read()
            time.sleep(0.02)

        return True

    def get_fresh_frame(self, flush_buffers: int = 4) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Flushes the internal OS video capture buffer so that the returned frame
        reflects the current real-time environment after robot movement.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        # Discard stale buffered frames from the driver queue
        for _ in range(flush_buffers):
            self.cap.grab()

        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Releases camera resources cleanly."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None


# ============================================================================
# Main Roboguide Application Controller
# ============================================================================

class RoboguideDemoApp:
    """
    Master coordination loop for the Roboguide dual-agent autonomous demonstration:
      - Gemini 3.8 Live (`gemini-3.8-live`): Spoken conversation, triage, and evacuee voice updates.
      - Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`): Camera spatial reasoning and XRP motor driving.
    """

    def __init__(
        self,
        camera_index: int = 0,
        mode: str = "auto",
        interval: float = 1.5,
        max_turns: int = 0,
        headless: bool = False,
        save_dir: str = "captures/demo",
        model: str = "gemini-robotics-er-2-preview",
        audio_model: str = "gemini-3.8-live",
        thinking_level: str = "LOW",
        max_retained_frames: int = 4,
        hazard: str = "prompt",
        start_idle: bool = True,
    ):
        self.camera_index = camera_index
        self.mode = mode.lower()
        self.interval = interval
        self.max_turns = max_turns
        self.headless = headless
        self.save_dir = Path(save_dir)
        self.model = model
        self.audio_model = audio_model
        self.thinking_level = thinking_level
        self.max_retained_frames = max_retained_frames

        # Dual-Agent State Machine
        self.state: str = "IDLE" if start_idle else "EVACUATION"
        self.hazard: str = hazard if hazard != "prompt" else "fire"
        self.audio_session: RoboguideLiveAudioSession = get_audio_session()

        self.motion_count: int = 0
        self.next_voice_motion_threshold: int = 3

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = get_robot_bridge()
        self.vision = VisionSensor(camera_index=self.camera_index)
        self.client: Optional[RoboguideClient] = None
        self.chat_session: Any = None

        self.running = False
        self.turn_count = 0
        self.last_action_desc = "STANDBY"
        self.last_evacuee_text = "System in standby. Press [E] to trigger emergency."
        self.window_name = "Roboguide - Live Hackathon Demo (Press Q to Exit)"

        # Intercept bridge movements for live telemetry reporting and motion counting
        self._setup_hardware_interceptor()

    def _setup_hardware_interceptor(self):
        """Wraps bridge.execute_motion to feed live HUD status, count motions, and trigger voice."""
        orig_execute = self.bridge.execute_motion

        async def intercepted_execute_motion(cmd: str, duration_seconds: float = 0.0) -> Dict[str, Any]:
            cmd_upper = cmd.strip().upper()
            dur_str = f"{duration_seconds:.1f}s" if duration_seconds > 0 else "instant"
            self.last_action_desc = f"{cmd_upper} ({dur_str})"

            if cmd_upper not in ("STOP", "STOP_OBSTACLE", "INITIAL_STATE", "NONE"):
                self.motion_count += 1

            print_action_executed(
                action_name=cmd_upper,
                duration=duration_seconds,
                port=self.bridge.port or "COM4",
                status="DISPATCHED",
            )
            result = await orig_execute(cmd, duration_seconds)
            return result

        self.bridge.execute_motion = intercepted_execute_motion

    async def initialize(self) -> bool:
        """Connects to hardware, camera, audio, and sets up initial telemetry."""
        # 1. Connect XRP Robot
        await self.bridge.connect()

        # 2. Open Camera
        cam_ok = self.vision.open()
        if not cam_ok:
            print(f"{C_RED}[ERROR] Failed to open camera device {self.camera_index}.{C_RESET}")
            return False

        # 3. Initialize Gemini Robotics Client
        try:
            self.client = RoboguideClient(
                model=self.model,
                thinking_level=self.thinking_level,
            )
            self.chat_session = self.client.start_spatial_chat(
                model=self.model, hazard_type=self.hazard
            )
        except Exception as e:
            print(f"{C_RED}[ERROR] Failed to initialize Gemini client: {e}{C_RESET}")
            return False

        # 4. Display Welcome Telemetry
        print_banner(
            model=self.model,
            audio_model=self.audio_model,
            port=self.bridge.port or "COM4",
            is_simulated=self.bridge.is_simulated,
            camera_idx=self.vision.camera_index,
            mode=self.mode,
            hazard=self.hazard,
            state=self.state,
        )
        return True

    def _handle_gui_events(self, frame: np.ndarray, status: str) -> Optional[str]:
        """Renders HUD frame and processes keypresses."""
        if self.headless:
            return None

        hud_frame = render_hud(
            frame=frame,
            turn=self.turn_count,
            status_text=status,
            last_action=self.last_action_desc,
            evacuee_text=self.last_evacuee_text,
            port=self.bridge.port or "COM4",
            is_simulated=self.bridge.is_simulated,
            hazard=self.hazard,
            state=self.state,
            motion_count=self.motion_count,
        )
        cv2.imshow(self.window_name, hud_frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
            return "exit"
        elif key in (ord("s"), ord("S")):  # Emergency stop
            return "stop"
        elif key in (ord("e"), ord("E")):  # Emergency triage trigger
            return "emergency"
        elif key == ord("1"):
            return "1"
        elif key == ord("2"):
            return "2"
        elif key == ord("3"):
            return "3"
        elif key == 32:  # SPACE bar
            return "step"
        return None

    async def trigger_emergency_triage(self):
        """Conducts spoken emergency triage via Gemini 3.8 Live."""
        print(f"\n{C_YELLOW}{C_BOLD}[EMERGENCY TRIAGE ACTIVATED]{C_RESET}")
        self.state = "TRIAGE"
        self.last_action_desc = "TRIAGE IN PROGRESS"

        # 1. Gemini 3.8 Live asks what the emergency is
        triage_question = await self.audio_session.ask_emergency_triage()
        self.last_evacuee_text = triage_question
        print(f"{C_CYAN}[ROBOGUIDE VOICE (Gemini 3.8 Live)]:{C_RESET} \"{triage_question}\"")
        print(f"  {C_BOLD}Options:{C_RESET} [1] Fire  |  [2] Earthquake  |  [3] Tornado")

        # 2. Capture user selection (GUI keypress or console input)
        user_choice = ""
        if self.headless:
            user_choice = await asyncio.to_thread(input, ">>> State Emergency [1: Fire, 2: Earthquake, 3: Tornado]: ")
        else:
            print("  >>> Press [1], [2], or [3] in the video window (or type in console) >>>")
            start_wait = time.time()
            while self.running and not user_choice and (time.time() - start_wait) < 12.0:
                ret, frame = self.vision.get_fresh_frame(flush_buffers=1)
                if ret and frame is not None:
                    event = self._handle_gui_events(frame, status="WAITING USER RESPONSE (1:Fire, 2:Quake, 3:Tornado)")
                    if event in ("1", "2", "3"):
                        user_choice = event
                        break
                    elif event == "exit":
                        self.running = False
                        return
                    elif event == "stop":
                        await self.bridge.execute_motion("STOP", 0.0)
                await asyncio.sleep(0.04)

            if not user_choice and self.running:
                user_choice = self.hazard or "fire"

        # 3. Process emergency response with Gemini 3.8 Live
        print(f"{C_GREEN}[PROCESSING TRIAGE]:{C_RESET} User declared: '{user_choice}'")
        self.hazard = await self.audio_session.process_emergency_response(user_choice)
        print(f"{C_MAGENTA}[ACTIVATED PROTOCOL]:{C_RESET} {C_BOLD}{self.hazard.upper()}{C_RESET}")

        # 4. Initialize Gemini Robotics ER 2 spatial session with this specific hazard
        self.chat_session = self.client.start_spatial_chat(model=self.model, hazard_type=self.hazard)
        self.state = "EVACUATION"
        self.next_voice_motion_threshold = self.motion_count + 3

    async def run_turn(self, turn_idx: int) -> bool:
        """
        Executes a complete sense-plan-act turn for the active hazard:
          1. Grabs fresh frame from camera
          2. Renders HUD & saves frame
          3. Prompts Gemini ER 2 with hazard-specific spatial reasoning
          4. Automatically executes invoked motor tools on XRP hardware
          5. Checks periodic voice updates via Gemini 3.8 Live (every 3-4 moves)
        """
        self.turn_count = turn_idx
        print_turn_start(turn_idx, self.max_turns, self.hazard)

        # 1. Capture fresh camera frame
        ret, frame = self.vision.get_fresh_frame(flush_buffers=4)
        if not ret or frame is None:
            print(f"{C_RED}[ERROR] Failed to read frame from optical sensor.{C_RESET}")
            return False

        # 2. Save capture to disk for post-run audit & judge review
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"turn_{turn_idx:03d}_{self.hazard}_{timestamp_str}.jpg"
        cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"  {C_CYAN}[SAVED]{C_RESET} Frame archived: {save_path.name} ({frame.shape[1]}x{frame.shape[0]})")

        # 3. Update GUI HUD with REASONING state
        event = self._handle_gui_events(frame, status=f"ANALYZING SCENE ({self.hazard.upper()})...")
        if event == "exit":
            return False
        elif event == "stop":
            await self.bridge.execute_motion("STOP", 0.0)

        # 4. Formulate prompt tailored to active hazard
        if self.hazard == "earthquake":
            hazard_guidance = (
                "EARTHQUAKE PROTOCOL: Look for sturdy tables, heavy desks, or structural cover. "
                "Steer toward cover and call stop_robot() once adjacent so evacuees can take cover underneath. "
                "Avoid glass windows, mirrors, and overhead light fixtures."
            )
        elif self.hazard == "tornado":
            hazard_guidance = (
                "TORNADO PROTOCOL: Look for interior hallways and windowless rooms. "
                "Steer into deep interior corridors. Avoid exterior walls, glass windows, and tipping shelves."
            )
        else:
            hazard_guidance = (
                "FIRE PROTOCOL: Look for illuminated EXIT signs, double exit doors, and clear corridors. "
                "Steer toward the exit door and avoid smoke or flames."
            )

        if turn_idx == 1:
            instruction = (
                f"You are Roboguide responding to a {self.hazard.upper()} emergency. {hazard_guidance} "
                "Inspect the camera view, state what you observe, declare your chosen move, and call the "
                "appropriate driving tool (move_forward, steer_slight_left, turn_hard_right, stop_robot) to guide evacuees."
            )
        else:
            instruction = (
                f"Camera Feed [Turn {turn_idx}]. Action completed: {self.last_action_desc}. {hazard_guidance} "
                "SPATIAL MEMORY: Compare this view with previous frames: Did your position or heading change? "
                "Is your route clear or blocked? State your spatial reasoning and call the next driving tool."
            )

        print_gemini_start()

        # 5. Send frame to Gemini ER 2 & execute tools via Automatic Function Calling
        try:
            reasoning = await self.client.send_spatial_frame(
                chat_session=self.chat_session,
                frame=frame,
                frame_index=turn_idx,
                instruction=instruction,
                max_retained_frames=self.max_retained_frames,
            )
        except Exception as e:
            print(f"{C_RED}[ERROR] Gemini API inference exception: {e}{C_RESET}")
            reasoning = f"Sensor communication error: {e}"

        # 6. Extract evacuee directive and display decision
        self.last_evacuee_text = reasoning.strip()
        print_gemini_decision(reasoning, self.last_action_desc)

        # 7. Check periodic voice reassurance (every 3-4 movements)
        if self.motion_count >= self.next_voice_motion_threshold:
            print(f"\n{C_CYAN}{C_BOLD}[GEMINI 3.8 LIVE AUDIO]{C_RESET} Broadcasting spoken update to evacuees (Motion #{self.motion_count})...")
            asyncio.create_task(
                self.audio_session.announce_spatial_progress(
                    current_action=self.last_action_desc,
                    spatial_reasoning=reasoning,
                    motion_count=self.motion_count,
                )
            )
            self.next_voice_motion_threshold = self.motion_count + 3

        # 8. Update HUD with completed state
        event = self._handle_gui_events(frame, status=f"ACTIVE: {self.last_action_desc}")
        if event == "exit":
            return False

        return True

    async def run(self):
        """Main operational execution loop with State Machine."""
        self.running = True
        turn = 1

        try:
            # Phase 1: If starting in IDLE, run standby loop waiting for trigger
            if self.state == "IDLE":
                print(f"\n{C_YELLOW}[SYSTEM IDLE]{C_RESET} Robot parked in standby. Press {C_BOLD}[E]{C_RESET} or {C_BOLD}[SPACE]{C_RESET} in GUI (or [ENTER] in console) to trigger emergency triage...")
                if self.headless:
                    await asyncio.to_thread(input, ">>> Press ENTER to trigger Emergency Mode >>> ")
                    await self.trigger_emergency_triage()
                else:
                    while self.running and self.state == "IDLE":
                        ret, frame = self.vision.get_fresh_frame(flush_buffers=1)
                        if ret and frame is not None:
                            event = self._handle_gui_events(frame, status="STANDBY: PRESS [E] OR [SPACE] TO TRIGGER")
                            if event in ("emergency", "step"):
                                await self.trigger_emergency_triage()
                                break
                            elif event == "exit":
                                self.running = False
                                return
                            elif event == "stop":
                                await self.bridge.execute_motion("STOP", 0.0)
                        await asyncio.sleep(0.03)

            # Phase 2: Active Evacuation Loop
            while self.running and self.state == "EVACUATION":
                if self.max_turns > 0 and turn > self.max_turns:
                    print(f"\n{C_GREEN}[COMPLETED] Reached requested turn limit ({self.max_turns}).{C_RESET}")
                    break

                # Execute one full navigation turn
                success = await self.run_turn(turn)
                if not success:
                    break

                # Step vs Auto Mode Handling
                if self.mode == "step":
                    print(f"\n{C_YELLOW}[STEP MODE]{C_RESET} Turn {turn} finished. Press {C_BOLD}[SPACE]{C_RESET} in GUI or {C_BOLD}[ENTER]{C_RESET} in console for next turn (or [Q] to quit)...")
                    if self.headless:
                        await asyncio.to_thread(input, ">>> Press ENTER to continue >>> ")
                    else:
                        waiting_step = True
                        while waiting_step and self.running:
                            ret, preview_frame = self.vision.get_fresh_frame(flush_buffers=1)
                            if ret and preview_frame is not None:
                                ev = self._handle_gui_events(preview_frame, status="WAITING [SPACE]")
                                if ev in ("step", "emergency"):
                                    waiting_step = False
                                elif ev == "exit":
                                    self.running = False
                                elif ev == "stop":
                                    await self.bridge.execute_motion("STOP", 0.0)
                            await asyncio.sleep(0.03)
                else:
                    # Auto mode: settle camera and wait interval
                    print(f"{C_CYAN}[SETTLING]{C_RESET} Waiting {self.interval:.1f}s for rover motion to settle before next observation...")
                    start_wait = time.time()
                    while (time.time() - start_wait) < self.interval and self.running:
                        if not self.headless:
                            ret, preview_frame = self.vision.get_fresh_frame(flush_buffers=1)
                            if ret and preview_frame is not None:
                                ev = self._handle_gui_events(preview_frame, status=f"SETTLING ({self.interval:.1f}s)")
                                if ev == "exit":
                                    self.running = False
                                elif ev == "stop":
                                    await self.bridge.execute_motion("STOP", 0.0)
                        await asyncio.sleep(0.05)

                turn += 1

        except (KeyboardInterrupt, asyncio.CancelledError):
            print(f"\n{C_YELLOW}[INTERRUPT] Received user interrupt signal (Ctrl+C).{C_RESET}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Guarantees physical rover motors are halted and all resources freed."""
        print(f"\n{C_CYAN}[SHUTDOWN] Safely cutting motor power and releasing resources...{C_RESET}")
        try:
            self.audio_session.player.stop()
            if self.bridge.is_connected:
                await self.bridge.execute_motion("STOP", 0.0)
                await asyncio.sleep(0.3)
                await self.bridge.disconnect()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        self.vision.release()
        if not self.headless:
            cv2.destroyAllWindows()

        print(f"{C_GREEN}{C_BOLD}[OK] Roboguide shutdown safely completed. Total turns: {self.turn_count}, Total motions: {self.motion_count}.{C_RESET}\n")


# ============================================================================
# CLI Entrypoint
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roboguide: Dual-Agent Autonomous Evacuation Rover (Gemini 3.8 Live + Gemini Robotics ER 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (0 for integrated webcam, 1 for external USB camera)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "step"],
        default="auto",
        help="Navigation loop mode: 'auto' runs continuously; 'step' pauses each turn for judge inspection",
    )
    parser.add_argument(
        "--hazard",
        choices=["prompt", "fire", "earthquake", "tornado"],
        default="prompt",
        help="Emergency hazard type ('prompt' launches interactive voice triage)",
    )
    parser.add_argument(
        "--no-idle",
        action="store_true",
        help="Skip IDLE standby and launch evacuation immediately",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Inter-turn settling delay in seconds (ensures camera frame is not blurred during drive)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=0,
        help="Maximum turns to execute before pausing (0 = run indefinitely until quit)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless terminal mode without displaying OpenCV GUI window",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="captures/demo",
        help="Directory where captured demo frames are archived",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-robotics-er-2-preview",
        help="Gemini Robotics spatial model name",
    )
    parser.add_argument(
        "--audio-model",
        type=str,
        default="gemini-3.8-live",
        help="Gemini Live audio model name",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="LOW",
        help="Gemini ER 2 thinking depth level",
    )
    parser.add_argument(
        "--max-retained-frames",
        type=int,
        default=4,
        help="Number of historical visual image payloads to retain in Gemini chat history (default: 4). Older image bytes are automatically pruned to save free tier tokens.",
    )
    return parser.parse_args()


async def main():
    args = parse_arguments()
    app = RoboguideDemoApp(
        camera_index=args.camera,
        mode=args.mode,
        interval=args.interval,
        max_turns=args.turns,
        headless=args.headless,
        save_dir=args.save_dir,
        model=args.model,
        audio_model=args.audio_model,
        thinking_level=args.thinking_level,
        max_retained_frames=args.max_retained_frames,
        hazard=args.hazard,
        start_idle=not args.no_idle,
    )

    initialized = await app.initialize()
    if not initialized:
        sys.exit(1)

    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
