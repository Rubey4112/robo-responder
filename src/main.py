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
    ROBOGUIDE_SYSTEM_INSTRUCTION,
    RoboguideClient,
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

def print_banner(model: str, port: str, is_simulated: bool, camera_idx: int, mode: str):
    """Displays the executive hackathon header banner."""
    hw_status = f"{C_YELLOW}SIMULATION (No Hardware){C_RESET}" if is_simulated else f"{C_GREEN}{port} (Connected){C_RESET}"
    banner = f"""
{C_CYAN}{C_BOLD}+==============================================================================+
|                   ROBOGUIDE: AUTONOMOUS EVACUATION ROVER                     |
|         Powered by Google Gemini Robotics ER 2 & Physical XRP Hardware        |
+==============================================================================+{C_RESET}
  {C_BOLD}[*] Target AI Model  :{C_RESET} {C_MAGENTA}{model}{C_RESET}
  {C_BOLD}[*] XRP Rover Bus    :{C_RESET} {hw_status}
  {C_BOLD}[*] Live Camera Feed :{C_RESET} {C_WHITE}Device {camera_idx} (DirectShow / OpenCV){C_RESET}
  {C_BOLD}[*] Operational Mode :{C_RESET} {C_YELLOW}{mode.upper()}{C_RESET}
  {C_BOLD}[*] Demo Controls    :{C_RESET} {C_WHITE}[SPACE] Next Step | [S] Emergency STOP | [Q] Safe Exit{C_RESET}
{C_CYAN}--------------------------------------------------------------------------------{C_RESET}
"""
    print(banner)


def print_turn_start(turn: int, max_turns: int):
    """Prints turn start telemetry."""
    turns_info = f"{turn}/{max_turns}" if max_turns > 0 else f"{turn}"
    print(f"\n{C_BLUE}{C_BOLD}============================== [TURN {turns_info}] =============================={C_RESET}")
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
) -> np.ndarray:
    """
    Renders a broadcast-quality semi-transparent HUD overlay on top of the
    live OpenCV video frame for judge observation.
    """
    hud = frame.copy()
    h, w = hud.shape[:2]

    # 1. Semi-transparent Top Header Bar
    top_bar_height = 64
    top_overlay = hud[0:top_bar_height, 0:w].copy()
    cv2.rectangle(top_overlay, (0, 0), (w, top_bar_height), (20, 20, 25), -1)
    hud[0:top_bar_height, 0:w] = cv2.addWeighted(top_overlay, 0.75, hud[0:top_bar_height, 0:w], 0.25, 0)

    # Title & Subtitle
    cv2.putText(
        hud,
        "ROBOGUIDE  |  GEMINI ROBOTICS ER 2 EVACUATION ROVER",
        (16, 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        (0, 235, 255),  # Cyan
        1,
        cv2.LINE_AA,
    )

    hw_label = f"ROBOT: {port} (LIVE)" if not is_simulated else "ROBOT: SIMULATED"
    hw_color = (0, 255, 120) if not is_simulated else (0, 180, 255)
    cv2.putText(
        hud,
        f"TURN #{turn}   |   {hw_label}   |   STATUS: {status_text}",
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        hw_color,
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
        f"COMMAND: {last_action}",
        (16, y_start + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
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
        f"DIRECTIVE: \"{display_directive}\"",
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
    Master coordination loop for the Roboguide autonomous demonstration.
    Connects camera vision, Gemini spatial chat, and XRP motor bridge.
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
        thinking_level: str = "LOW",
    ):
        self.camera_index = camera_index
        self.mode = mode.lower()
        self.interval = interval
        self.max_turns = max_turns
        self.headless = headless
        self.save_dir = Path(save_dir)
        self.model = model
        self.thinking_level = thinking_level

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = get_robot_bridge()
        self.vision = VisionSensor(camera_index=self.camera_index)
        self.client: Optional[RoboguideClient] = None
        self.chat_session: Any = None

        self.running = False
        self.turn_count = 0
        self.last_action_desc = "STANDBY"
        self.last_evacuee_text = "System initialized. Preparing to navigate."
        self.window_name = "Roboguide - Live Hackathon Demo (Press Q to Exit)"

        # Intercept bridge movements for live telemetry reporting
        self._setup_hardware_interceptor()

    def _setup_hardware_interceptor(self):
        """Wraps bridge.execute_motion to feed live HUD status and terminal logs."""
        orig_execute = self.bridge.execute_motion

        async def intercepted_execute_motion(cmd: str, duration_seconds: float = 0.0) -> Dict[str, Any]:
            cmd_upper = cmd.strip().upper()
            dur_str = f"{duration_seconds:.1f}s" if duration_seconds > 0 else "instant"
            self.last_action_desc = f"{cmd_upper} ({dur_str})"

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
        """Connects to hardware, camera, and sets up Gemini spatial chat."""
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
            self.chat_session = self.client.start_spatial_chat(model=self.model)
        except Exception as e:
            print(f"{C_RED}[ERROR] Failed to initialize Gemini client: {e}{C_RESET}")
            return False

        # 4. Display Welcome Telemetry
        print_banner(
            model=self.model,
            port=self.bridge.port or "COM4",
            is_simulated=self.bridge.is_simulated,
            camera_idx=self.vision.camera_index,
            mode=self.mode,
        )
        return True

    def _handle_gui_events(self, frame: np.ndarray, status: str) -> Optional[str]:
        """Renders HUD frame and processes keypresses. Returns 'exit', 'step', 'stop', or None."""
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
        )
        cv2.imshow(self.window_name, hud_frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
            return "exit"
        elif key in (ord("s"), ord("S")):  # Emergency stop
            return "stop"
        elif key == 32:  # SPACE bar
            return "step"
        return None

    async def run_turn(self, turn_idx: int) -> bool:
        """
        Executes a complete sense-plan-act turn:
          1. Grabs fresh frame from camera
          2. Renders HUD & saves frame
          3. Prompts Gemini ER 2 with spatial reasoning context
          4. Automatically executes invoked motor tools on XRP hardware
          5. Logs reasoning and updates guidance announcement
        """
        self.turn_count = turn_idx
        print_turn_start(turn_idx, self.max_turns)

        # 1. Capture fresh camera frame
        ret, frame = self.vision.get_fresh_frame(flush_buffers=4)
        if not ret or frame is None:
            print(f"{C_RED}[ERROR] Failed to read frame from optical sensor.{C_RESET}")
            return False

        # 2. Save capture to disk for post-run audit & judge review
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"turn_{turn_idx:03d}_{timestamp_str}.jpg"
        cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"  {C_CYAN}[SAVED]{C_RESET} Frame archived: {save_path.name} ({frame.shape[1]}x{frame.shape[0]})")

        # 3. Update GUI HUD with REASONING state
        event = self._handle_gui_events(frame, status="ANALYZING SCENE...")
        if event == "exit":
            return False
        elif event == "stop":
            await self.bridge.execute_motion("STOP", 0.0)

        # 4. Construct prompt enforcing spatial awareness across turns
        if turn_idx == 1:
            instruction = (
                "You are Roboguide at initial deployment. Inspect the camera view: "
                "Identify any emergency exit signs ('EXIT', green signs), exit doors, open corridors, "
                "or obstacles. State what you observe, declare your chosen navigation move, and call the "
                "appropriate driving tool (e.g. move_forward, steer_slight_left, turn_hard_right, stop_robot) "
                "to guide evacuees to safety."
            )
        else:
            instruction = (
                f"Camera Feed [Turn {turn_idx}]. You just completed action: {self.last_action_desc}. "
                "SPATIAL MEMORY COMPARISON: Compare this view with your previous observation: "
                "1. Did your position or heading change? "
                "2. Are you closer to the exit, doorway, or obstacle? "
                "3. Is the route directly ahead clear? "
                "State your spatial reasoning, tell the evacuees what to do, and call the next driving tool."
            )

        print_gemini_start()

        # 5. Send frame to Gemini ER 2 & execute tools via Automatic Function Calling
        try:
            reasoning = await self.client.send_spatial_frame(
                chat_session=self.chat_session,
                frame=frame,
                frame_index=turn_idx,
                instruction=instruction,
            )
        except Exception as e:
            print(f"{C_RED}[ERROR] Gemini API inference exception: {e}{C_RESET}")
            reasoning = f"Sensor communication error: {e}"

        # 6. Extract evacuee directive and display decision
        self.last_evacuee_text = reasoning.strip()
        print_gemini_decision(reasoning, self.last_action_desc)

        # 7. Update HUD with completed state
        event = self._handle_gui_events(frame, status=f"ACTIVE: {self.last_action_desc}")
        if event == "exit":
            return False

        return True

    async def run(self):
        """Main operational execution loop."""
        self.running = True
        turn = 1

        try:
            while self.running:
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
                        # Wait for console input in thread
                        await asyncio.to_thread(input, ">>> Press ENTER to continue >>> ")
                    else:
                        waiting_step = True
                        while waiting_step and self.running:
                            ret, preview_frame = self.vision.get_fresh_frame(flush_buffers=1)
                            if ret and preview_frame is not None:
                                ev = self._handle_gui_events(preview_frame, status="WAITING [SPACE]")
                                if ev == "step":
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
            if self.bridge.is_connected:
                # Emergency zero-duration STOP
                await self.bridge.execute_motion("STOP", 0.0)
                await asyncio.sleep(0.3)
                await self.bridge.disconnect()
        except Exception as e:
            logger.error(f"Error during bridge disconnect: {e}")

        self.vision.release()
        if not self.headless:
            cv2.destroyAllWindows()

        print(f"{C_GREEN}{C_BOLD}[OK] Roboguide shutdown safely completed. Total turns executed: {self.turn_count}.{C_RESET}\n")


# ============================================================================
# CLI Entrypoint
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roboguide: Autonomous Emergency Evacuation Rover (Gemini ER 2 + XRP Hardware)",
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
        help="Gemini Robotics model name",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="LOW",
        help="Gemini ER 2 thinking depth level",
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
        thinking_level=args.thinking_level,
    )

    initialized = await app.initialize()
    if not initialized:
        sys.exit(1)

    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
