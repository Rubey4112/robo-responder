"""
Vision Streaming Pipeline
-------------------------
Coordinates the real-time camera capture, bidirectional Gemini Robotics ER 2 Live session,
WebSocket streaming server, and interactive OpenCV HUD display.

Architecture:
  [Camera / WebcamSensor]
           | (30 FPS raw frames)
           v
  [VisionStreamPipeline]
      |---> [WebSocket Broadcast] (30 FPS Base64 JPEG to web/remote clients)
      |---> [Gemini Live Client]  (1 FPS JPEG to gemini-robotics-er-2-streaming-preview)
      |---> [OpenCV Display HUD]  (Live annotated video with AI reasoning & directives)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from .gemini_live import (
    GeminiRoboticsLiveClient,
    GeminiLiveEvent,
    RobotDirective,
    DEFAULT_MODEL,
)
from .websocket_streamer import VisionWebSocketServer, InboundCommand
from .imaging import open_camera

logger = logging.getLogger("vision.stream_pipeline")


@dataclass
class StreamPipelineConfig:
    """Configuration options for VisionStreamPipeline."""
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    jpeg_quality: int = 80
    stream_fps: float = 30.0
    gemini_fps: float = 1.0
    gemini_model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    mock_gemini: bool = False
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    enable_gui: bool = True
    output_dir: str = "captures"


class VisionStreamPipeline:
    """
    Main coordinator for Roboguide vision streaming, Gemini Live API, and WebSockets.
    """

    def __init__(
        self,
        config: Optional[StreamPipelineConfig] = None,
        on_directive: Optional[Callable[[RobotDirective], Any]] = None,
    ):
        self.config = config or StreamPipelineConfig()
        self.on_directive = on_directive

        # Initialize WebSocket Server
        self.ws_server = VisionWebSocketServer(
            host=self.config.ws_host,
            port=self.config.ws_port,
            on_command=self._handle_client_command,
            on_inbound_frame=self._handle_inbound_frame,
        )

        # Initialize Gemini Live Client
        self.gemini_client = GeminiRoboticsLiveClient(
            api_key=self.config.api_key,
            model=self.config.gemini_model,
            mock_mode=self.config.mock_gemini,
            frame_interval_sec=1.0 / max(0.1, self.config.gemini_fps),
            on_event=self._handle_gemini_event,
            on_directive=self._handle_gemini_directive,
        )

        # Internal state
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cam_worker")
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._fps_actual = 0.0
        self._last_directive: Optional[RobotDirective] = None
        self._directive_time: float = 0.0
        self._latest_evacuation_status: Dict[str, Any] = {}

    async def start(self):
        """Starts camera capture, Gemini Live session, and WebSocket server."""
        if self._running:
            return

        self._running = True
        logger.info("Initializing Roboguide Vision Stream Pipeline...")

        # 1. Start WebSocket Server
        await self.ws_server.start()

        # 2. Start Gemini Live Client
        await self.gemini_client.start()

        # 3. Open Camera (if in local capture mode)
        loop = asyncio.get_running_loop()
        self._cap = await loop.run_in_executor(
            self._executor, open_camera, self.config.camera_index
        )

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            logger.info(f"Camera device {self.config.camera_index} opened successfully.")
        else:
            logger.warning(f"Could not open physical camera {self.config.camera_index}. Pipeline awaiting stream frames.")

        # 4. Start concurrent pipeline loops
        task_capture = asyncio.create_task(self._capture_and_dispatch_loop())
        task_gui = asyncio.create_task(self._gui_render_loop()) if self.config.enable_gui else None

        try:
            await task_capture
            if task_gui:
                await task_gui
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    def request_stop(self):
        """Signals the pipeline to stop gracefully."""
        self._running = False

    async def stop(self):
        """Stops all running tasks and releases hardware devices."""
        if not self._running:
            return

        self._running = False
        logger.info("Shutting down Vision Stream Pipeline...")

        # Stop Gemini & WebSockets safely
        try:
            await self.gemini_client.stop()
        except Exception as e:
            logger.warning(f"Error stopping gemini client: {e}")

        try:
            await self.ws_server.stop()
        except Exception as e:
            logger.warning(f"Error stopping websocket server: {e}")

        # Release Camera
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None

        if self.config.enable_gui:
            cv2.destroyAllWindows()

        self._executor.shutdown(wait=False)
        logger.info("Vision Stream Pipeline terminated.")

    async def _capture_and_dispatch_loop(self):
        """Continuously reads frames from camera, encodes them, and dispatches them."""
        loop = asyncio.get_running_loop()
        target_interval = 1.0 / max(1.0, self.config.stream_fps)
        frame_count = 0
        fps_timer = time.time()

        while self._running:
            start_time = time.time()

            if self._cap and self._cap.isOpened():
                ret, frame = await loop.run_in_executor(self._executor, self._cap.read)
                if ret and frame is not None:
                    self._latest_frame = frame
                    frame_count += 1

                    # Measure FPS
                    now = time.time()
                    elapsed = now - fps_timer
                    if elapsed >= 1.0:
                        self._fps_actual = frame_count / elapsed
                        frame_count = 0
                        fps_timer = now

                    # Encode to JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
                    success, buffer = cv2.imencode(".jpg", frame, encode_param)
                    if success:
                        jpeg_bytes = buffer.tobytes()
                        self._latest_jpeg = jpeg_bytes
                        h, w = frame.shape[:2]

                        # Calculate basic optical metrics
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        luminance = float(np.mean(gray))

                        telemetry = {
                            "luminance": round(luminance, 1),
                            "model": self.config.gemini_model,
                            "gemini_status": self.gemini_client.latest_status,
                            "is_mock": self.gemini_client.is_mock,
                            "clients_connected": self.ws_server.client_count,
                        }

                        # 1. Broadcast frame to WebSocket clients (up to 30 FPS)
                        self.ws_server.broadcast_frame(
                            jpeg_bytes=jpeg_bytes,
                            width=w,
                            height=h,
                            fps=self._fps_actual,
                            telemetry=telemetry,
                        )

                        # 2. Stream frame to Gemini Live API (throttled to ~1 FPS)
                        await self.gemini_client.send_frame(jpeg_bytes)

            # Regulate frame rate
            cycle_time = time.time() - start_time
            sleep_needed = target_interval - cycle_time
            if sleep_needed > 0:
                await asyncio.sleep(sleep_needed)
            else:
                await asyncio.sleep(0.001)

    async def _gui_render_loop(self):
        """Renders local OpenCV preview window with Augmented Reality HUD overlays."""
        loop = asyncio.get_running_loop()
        window_name = "Roboguide ER-2 Live Stream HUD"

        while self._running:
            if self._latest_frame is not None:
                # Render HUD in executor to avoid stalling event loop
                display_frame = await loop.run_in_executor(
                    self._executor, self._render_hud_overlay, self._latest_frame.copy()
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    logger.info("User requested exit via GUI.")
                    self._running = False
                    break
                elif key in (ord("s"), ord("S")):  # 's' for snapshot
                    self.save_snapshot()
                elif key in (ord("p"), ord("P")):  # 'p' to trigger prompt
                    await self.gemini_client.send_text_prompt(
                        "Analyze scene: Detect exit door, exit signs, and determine immediate path."
                    )

            await asyncio.sleep(0.03)

        cv2.destroyAllWindows()

    def _render_hud_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draws visual HUD overlay with Gemini thoughts, status, and navigation directives."""
        h, w = frame.shape[:2]

        # Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.line(frame, (0, 55), (w, 55), (0, 200, 255), 2)

        # Title and Model Tag
        cv2.putText(
            frame,
            "ROBOGUIDE ER-2 STREAM",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        model_str = f"Model: {self.config.gemini_model}"
        if self.gemini_client.is_mock:
            model_str += " (MOCK)"
        cv2.putText(
            frame,
            model_str,
            (12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # Right-side telemetry badges
        ws_info = f"WS: {self.ws_server.client_count} client(s) | {self._fps_actual:.1f} FPS"
        cv2.putText(
            frame,
            ws_info,
            (w - 240, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        status_text = f"Gemini: {self.gemini_client.latest_status.upper()}"
        status_color = (0, 255, 0) if "connected" in self.gemini_client.latest_status.lower() or "active" in self.gemini_client.latest_status.lower() else (0, 165, 255)
        cv2.putText(
            frame,
            status_text,
            (w - 240, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Bottom HUD: AI Reasoning & Robot Directive
        cv2.rectangle(frame, (0, h - 85), (w, h), (15, 15, 15), -1)
        cv2.line(frame, (0, h - 85), (w, h - 85), (0, 150, 255), 1)

        # Reasoning Text
        reasoning = self.gemini_client.latest_reasoning or "Awaiting Gemini Live perception..."
        if len(reasoning) > 85:
            reasoning = reasoning[:82] + "..."
        cv2.putText(
            frame,
            f"AI Reasoning: {reasoning}",
            (12, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 240, 255),
            1,
            cv2.LINE_AA,
        )

        # Active Directive Banner
        if self._last_directive and (time.time() - self._directive_time) < 6.0:
            d_name = self._last_directive.tool_name
            d_args = self._last_directive.arguments
            if d_name == "navigate_robot":
                action = d_args.get("action", "")
                dist = d_args.get("distance_cm", 0)
                angle = d_args.get("angle_deg", 0)
                spd = d_args.get("speed_percent", 50)
                directive_str = f"DIRECTIVE: {action} {dist}cm (Angle: {angle} deg, Speed: {spd}%)"
                dir_color = (0, 255, 0)
            elif d_name == "emergency_stop":
                directive_str = f"DIRECTIVE: EMERGENCY STOP - {d_args.get('reason', '')}"
                dir_color = (0, 0, 255)
            elif d_name == "report_evacuation_status":
                direction = d_args.get("exit_direction", "")
                inst = d_args.get("instructions", "")
                directive_str = f"STATUS: EXIT {direction} - {inst}"
                dir_color = (255, 200, 0)
            else:
                directive_str = f"DIRECTIVE: {d_name}({d_args})"
                dir_color = (0, 255, 255)

            cv2.putText(
                frame,
                directive_str,
                (12, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                dir_color,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "DIRECTIVE: STANDBY / PERCEIVING",
                (12, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

        # Quick Key guide in bottom right
        cv2.putText(
            frame,
            "[Q] Quit | [S] Save | [P] Prompt",
            (w - 215, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

        return frame

    def save_snapshot(self) -> Optional[str]:
        """Saves current frame to disk as a high-quality JPEG."""
        if self._latest_frame is None:
            return None

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = out_dir / f"stream_capture_{timestamp}.jpg"

        success = cv2.imwrite(str(filename), self._latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            abs_path = str(filename.resolve())
            logger.info(f"Saved snapshot to: {abs_path}")
            return abs_path
        return None

    def _handle_gemini_event(self, event: GeminiLiveEvent):
        """Relays Gemini events to connected WebSocket clients."""
        if event.event_type == "reasoning":
            self.ws_server.broadcast_event("ai_reasoning", {"text": event.data})
        elif event.event_type == "directive":
            d: RobotDirective = event.data
            self._last_directive = d
            self._directive_time = time.time()
            self.ws_server.broadcast_event(
                "robot_directive",
                {"tool_name": d.tool_name, "arguments": d.arguments, "call_id": d.call_id},
            )
        elif event.event_type == "status":
            self.ws_server.broadcast_event("gemini_status", {"status": str(event.data)})

    def _handle_gemini_directive(self, directive: RobotDirective) -> Dict[str, Any]:
        """Invoked when Gemini ER 2 outputs a physical robot movement directive."""
        self._last_directive = directive
        self._directive_time = time.time()
        logger.info(f"Received robot directive: {directive.tool_name} -> {directive.arguments}")

        # If user registered external callback (e.g. Robot / Agent bridge), forward it
        if self.on_directive:
            try:
                res = self.on_directive(directive)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.error(f"Error executing directive callback: {e}")
                return {"status": "error", "message": str(e)}

        return {"status": "accepted", "tool": directive.tool_name}

    async def _handle_client_command(self, cmd: InboundCommand) -> Dict[str, Any]:
        """Handles inbound commands from WebSocket clients."""
        logger.info(f"WebSocket client command: {cmd.command} with params: {cmd.params}")

        if cmd.command in ("prompt", "PROMPT"):
            text = cmd.params.get("text", "")
            if text:
                await self.gemini_client.send_text_prompt(text)
                return {"status": "prompt_sent", "prompt": text}
            return {"status": "error", "message": "Missing 'text' in prompt"}

        elif cmd.command in ("snapshot", "SNAPSHOT"):
            path = self.save_snapshot()
            return {"status": "snapshot_saved", "path": path}

        elif cmd.command in ("ping", "PING"):
            return {"type": "pong", "timestamp": time.time()}

        elif cmd.command in ("status", "STATUS"):
            return {
                "gemini_status": self.gemini_client.latest_status,
                "reasoning": self.gemini_client.latest_reasoning,
                "fps": self._fps_actual,
                "clients": self.ws_server.client_count,
            }

        return {"status": "unknown_command", "command": cmd.command}

    def _handle_inbound_frame(self, frame_bytes: bytes):
        """Allows external client/camera to push video frames into Roboguide pipeline."""
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._latest_frame = frame
                self._latest_jpeg = frame_bytes
        except Exception as e:
            logger.error(f"Error handling inbound frame: {e}")


def run_vision_stream(
    camera_index: int = 0,
    ws_host: str = "0.0.0.0",
    ws_port: int = 8765,
    gemini_model: str = DEFAULT_MODEL,
    gemini_fps: float = 1.0,
    mock_gemini: bool = False,
    enable_gui: bool = True,
    output_dir: str = "captures",
):
    """
    Synchronous launcher for the Roboguide Vision Streaming Pipeline.
    """
    config = StreamPipelineConfig(
        camera_index=camera_index,
        ws_host=ws_host,
        ws_port=ws_port,
        gemini_model=gemini_model,
        gemini_fps=gemini_fps,
        mock_gemini=mock_gemini,
        enable_gui=enable_gui,
        output_dir=output_dir,
    )

    pipeline = VisionStreamPipeline(config=config)

    print("=" * 65)
    print(" Roboguide Vision Streaming Pipeline")
    print(f" WebSocket Server:  ws://{ws_host}:{ws_port}")
    print(f" Gemini Live Model: {gemini_model} (Live API)")
    print(f" Stream FPS:        30 FPS (WebSocket) / {gemini_fps} FPS (Gemini)")
    print(f" GUI Display:       {'Enabled' if enable_gui else 'Headless'}")
    print(f" Mode:              {'MOCK Simulation' if mock_gemini else 'Live API / Auto-Detect'}")
    print("=" * 65)

    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    finally:
        print("Roboguide streaming pipeline shutdown complete.")
