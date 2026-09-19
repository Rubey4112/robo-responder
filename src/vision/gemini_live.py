"""
Gemini Robotics Live Streaming Client
-------------------------------------
Provides real-time bidirectional streaming communication with Google's
`gemini-robotics-er-2-streaming-preview` vision-language model using the
Google GenAI Live API (WebSocket-based bidirectional stream).

Features:
1. Bidirectional streaming of 1 FPS video frames (JPEG) to Gemini Live.
2. Robust tool/function calling with `BLOCKING` behavior for robot actions:
   - navigate_robot(action, distance_cm, angle_deg, speed_percent, reason)
   - emergency_stop(reason, hazard_identified)
   - report_evacuation_status(exit_visible, exit_direction, hazards_detected, instructions)
3. Event callback architecture for:
   - Live AI reasoning text chunks
   - Robot tool execution directives
   - Session status changes (connecting, active, error, closed)
4. Offline / Mock Simulation Mode for development without API keys.
"""

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("vision.gemini_live")

# Attempt importing google-genai SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False


DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"

DEFAULT_SYSTEM_INSTRUCTION = """
You are Roboguide ER-2, an autonomous robotic emergency responder guiding evacuees
to safety during building disasters (fires, earthquakes, debris, structural hazards).
Your primary task is to continuously analyze the live camera stream to:
1. Identify emergency exit signs, egress doors, stairwells, and clear corridors.
2. Detect physical hazards, fire, smoke, collapsed obstacles, and impassable debris.
3. Determine safe navigation waypoints for the differential-drive robot.
4. Call physical navigation tools (navigate_robot, emergency_stop, report_evacuation_status)
   to steer the robot and lead people safely to the nearest exit.
5. Provide concise, clear, and calm situational reasoning and instructions.
Always issue BLOCKING movement directives so the robot can complete each movement safely.
"""


@dataclass
class RobotDirective:
    """Represents a physical navigation directive issued by Gemini."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class GeminiLiveEvent:
    """Event emitted during Gemini Live session."""
    event_type: str  # "reasoning", "directive", "status", "error"
    data: Any
    timestamp: float = field(default_factory=time.time)


def build_robotics_tools() -> List[Any]:
    """Builds function declarations with BLOCKING behavior for robotics control."""
    if not GENAI_AVAILABLE or types is None:
        return []

    navigate_declaration = types.FunctionDeclaration(
        name="navigate_robot",
        description=(
            "Commands the robot to navigate through the environment towards an exit, "
            "around obstacles, or down a corridor."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "action": types.Schema(
                    type=types.Type.STRING,
                    description="Movement command: FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, or STOP",
                    enum=["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"],
                ),
                "distance_cm": types.Schema(
                    type=types.Type.NUMBER,
                    description="Distance in centimeters to move (for FORWARD / BACKWARD). Recommended 20 to 100 cm.",
                ),
                "angle_deg": types.Schema(
                    type=types.Type.NUMBER,
                    description="Angle in degrees to turn (for TURN_LEFT / TURN_RIGHT). Range 5 to 180 degrees.",
                ),
                "speed_percent": types.Schema(
                    type=types.Type.NUMBER,
                    description="Drive speed percentage from 10 to 100. Default 50.",
                ),
                "reason": types.Schema(
                    type=types.Type.STRING,
                    description="Visual rationale based on the current scene (e.g. 'Heading towards green Exit sign').",
                ),
            },
            required=["action", "reason"],
        ),
        behavior=types.Behavior.BLOCKING,
    )

    emergency_stop_declaration = types.FunctionDeclaration(
        name="emergency_stop",
        description="Immediately halts all robot movement due to immediate hazard, blocked path, or safety risk.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "reason": types.Schema(
                    type=types.Type.STRING,
                    description="Detailed reason for the emergency halt.",
                ),
                "hazard_identified": types.Schema(
                    type=types.Type.STRING,
                    description="Description of detected hazard (fire, smoke, cliff/stairs, collapsed wall, human).",
                ),
            },
            required=["reason"],
        ),
        behavior=types.Behavior.BLOCKING,
    )

    evacuation_status_declaration = types.FunctionDeclaration(
        name="report_evacuation_status",
        description="Broadcasts an emergency situation update and guidance message for evacuees.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "exit_visible": types.Schema(
                    type=types.Type.BOOLEAN,
                    description="True if an emergency exit sign or door is currently visible in the camera view.",
                ),
                "exit_direction": types.Schema(
                    type=types.Type.STRING,
                    description="Direction to the exit: AHEAD, LEFT, RIGHT, BEHIND, or UNKNOWN",
                    enum=["AHEAD", "LEFT", "RIGHT", "BEHIND", "UNKNOWN"],
                ),
                "hazards_detected": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="List of detected environmental hazards.",
                ),
                "instructions": types.Schema(
                    type=types.Type.STRING,
                    description="Spoken or visual guidance instructions for evacuees following the robot.",
                ),
            },
            required=["exit_visible", "exit_direction", "instructions"],
        ),
        behavior=types.Behavior.BLOCKING,
    )

    return [
        types.Tool(
            function_declarations=[
                navigate_declaration,
                emergency_stop_declaration,
                evacuation_status_declaration,
            ]
        )
    ]


class GeminiRoboticsLiveClient:
    """
    Manages real-time bidirectional streaming with `gemini-robotics-er-2-streaming-preview`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        mock_mode: bool = False,
        frame_interval_sec: float = 1.0,
        on_event: Optional[Callable[[GeminiLiveEvent], Any]] = None,
        on_directive: Optional[Callable[[RobotDirective], Any]] = None,
    ):
        """
        Args:
            api_key: Google Gemini API key. If None, checks GEMINI_API_KEY / GOOGLE_API_KEY env vars.
            model: Gemini model identifier (default: 'gemini-robotics-er-2-streaming-preview').
            system_instruction: Prompt context governing robot reasoning.
            mock_mode: If True or if no API key is found, runs simulated reasoning.
            frame_interval_sec: Minimum seconds between video frames sent to Gemini (default 1.0s / 1 FPS).
            on_event: Callback invoked on live events.
            on_directive: Callback invoked when Gemini outputs a robot action directive.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.system_instruction = system_instruction
        self.frame_interval_sec = frame_interval_sec
        self.on_event = on_event
        self.on_directive = on_directive

        # Determine mode
        if mock_mode or not self.api_key or not GENAI_AVAILABLE:
            self.is_mock = True
            if not self.api_key:
                logger.info("No Gemini API key supplied. Running GeminiRoboticsLiveClient in MOCK simulation mode.")
            elif not GENAI_AVAILABLE:
                logger.warning("google-genai SDK not available. Falling back to MOCK mode.")
        else:
            self.is_mock = False

        self._session = None
        self._client = None
        self._running = False
        self._last_frame_sent_time = 0.0
        self._latest_reasoning = ""
        self._latest_status = "initialized"
        self._active_directives: List[RobotDirective] = []
        self._task_receive: Optional[asyncio.Task] = None
        self._task_mock: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_reasoning(self) -> str:
        return self._latest_reasoning

    @property
    def latest_status(self) -> str:
        return self._latest_status

    def _emit(self, event_type: str, data: Any):
        event = GeminiLiveEvent(event_type=event_type, data=data)
        if event_type == "status":
            self._latest_status = str(data)
        elif event_type == "reasoning":
            self._latest_reasoning = str(data)

        if self.on_event:
            try:
                res = self.on_event(event)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception as e:
                logger.error(f"Error in on_event callback: {e}")

    async def start(self):
        """Starts the Gemini Live streaming session."""
        if self._running:
            return

        self._running = True
        self._emit("status", "connecting")

        if self.is_mock:
            self._emit("status", "mock_active")
            self._task_mock = asyncio.create_task(self._run_mock_loop())
            return

        try:
            self._client = genai.Client(api_key=self.api_key)
            tools = build_robotics_tools()

            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.TEXT],
                system_instruction=types.Content(
                    parts=[types.Part.from_text(text=self.system_instruction)]
                ),
                tools=tools,
            )

            # Establish the async Live connection
            # We maintain the context manager alive via a worker task
            self._task_receive = asyncio.create_task(self._session_runner(config))
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Live session: {e}", exc_info=True)
            self._emit("error", str(e))
            self._emit("status", f"connection_failed: {e}")
            # Fall back to mock so pipeline keeps functioning
            self.is_mock = True
            self._task_mock = asyncio.create_task(self._run_mock_loop())

    async def _session_runner(self, config: Any):
        """Maintains the async live session connection and receive loop."""
        try:
            async with self._client.aio.live.connect(model=self.model, config=config) as session:
                self._session = session
                self._emit("status", "connected")
                logger.info(f"Connected to Gemini Live model: {self.model}")

                # Initial kickoff scene prompt
                await session.send(
                    input="System startup: Roboguide emergency responder active. Begin continuous scene perception and exit guidance.",
                    end_of_turn=True,
                )

                async for message in session.receive():
                    if not self._running:
                        break
                    await self._handle_server_message(message)

        except asyncio.CancelledError:
            logger.info("Gemini Live session runner cancelled.")
        except Exception as e:
            logger.error(f"Error in Gemini Live session: {e}", exc_info=True)
            self._emit("error", str(e))
            self._emit("status", f"disconnected: {e}")
        finally:
            self._session = None
            self._emit("status", "disconnected")

    async def _handle_server_message(self, message: Any):
        """Processes incoming messages from Gemini Live server."""
        # 1. Check for text reasoning chunks
        if message.server_content and message.server_content.model_turn:
            text_accum = []
            for part in message.server_content.model_turn.parts:
                if getattr(part, "text", None):
                    text_accum.append(part.text)
            if text_accum:
                full_text = "".join(text_accum).strip()
                if full_text:
                    self._latest_reasoning = full_text
                    self._emit("reasoning", full_text)

        # 2. Check for tool / function calls
        if message.tool_call and message.tool_call.function_calls:
            responses = []
            for fc in message.tool_call.function_calls:
                call_id = fc.id
                name = fc.name
                args = fc.args or {}
                logger.info(f"[Gemini Directive] FunctionCall {name}({args}) [id={call_id}]")

                directive = RobotDirective(
                    tool_name=name,
                    arguments=args,
                    call_id=call_id,
                )
                self._active_directives.append(directive)
                self._emit("directive", directive)

                # Dispatch directive to handler if registered
                result_payload = {"status": "success", "executed_at": time.time()}
                if self.on_directive:
                    try:
                        res = self.on_directive(directive)
                        if asyncio.iscoroutine(res):
                            res = await res
                        if isinstance(res, dict):
                            result_payload.update(res)
                    except Exception as e:
                        logger.error(f"Error executing directive {name}: {e}")
                        result_payload["status"] = "error"
                        result_payload["error"] = str(e)

                # Build tool response (BLOCKING calls require response before model continues)
                if types:
                    func_resp = types.FunctionResponse(
                        name=name,
                        id=call_id,
                        response=result_payload,
                    )
                    responses.append(func_resp)

            # Send tool responses back to the live session
            if responses and self._session and self._running:
                try:
                    await self._session.send_tool_response(function_responses=responses)
                except Exception as e:
                    logger.error(f"Failed to send tool response to Gemini Live: {e}")

    async def send_frame(self, jpeg_bytes: bytes, force: bool = False) -> bool:
        """
        Sends a JPEG video frame to the Gemini Live session.
        Throttled to self.frame_interval_sec (default 1 FPS) unless forced.
        """
        if not self._running or not jpeg_bytes:
            return False

        now = time.time()
        if not force and (now - self._last_frame_sent_time) < self.frame_interval_sec:
            return False

        self._last_frame_sent_time = now

        if self.is_mock:
            # Frame acknowledged in mock mode
            return True

        if self._session is None:
            return False

        try:
            blob = types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
            await self._session.send_realtime_input(media=blob)
            return True
        except Exception as e:
            logger.error(f"Failed to send frame to Gemini Live: {e}")
            self._emit("error", f"send_frame_error: {e}")
            return False

    async def send_text_prompt(self, text: str):
        """Sends a text message/prompt into the ongoing live session."""
        if not self._running or not text.strip():
            return

        if self.is_mock:
            self._latest_reasoning = f"[User Query Acknowledged]: {text}"
            self._emit("reasoning", self._latest_reasoning)
            return

        if self._session is None:
            return

        try:
            await self._session.send(input=text, end_of_turn=True)
        except Exception as e:
            logger.error(f"Failed to send text prompt: {e}")
            self._emit("error", f"send_prompt_error: {e}")

    async def _run_mock_loop(self):
        """Simulated reasoning loop when running without an API key or offline."""
        logger.info("Started Mock Gemini Robotics ER-2 Simulation Loop.")
        step = 0
        scenarios = [
            (
                "Analyzing corridor. Green emergency EXIT sign identified at 12 o'clock ahead.",
                "report_evacuation_status",
                {"exit_visible": True, "exit_direction": "AHEAD", "hazards_detected": [], "instructions": "Follow Roboguide forward toward the emergency exit."},
            ),
            (
                "Path clear for 1.5 meters. Aligning robot heading to center hallway.",
                "navigate_robot",
                {"action": "FORWARD", "distance_cm": 50.0, "speed_percent": 60.0, "reason": "Moving along unobstructed center path toward exit sign."},
            ),
            (
                "Obstacle detected on right (discarded debris). Veering slightly left.",
                "navigate_robot",
                {"action": "TURN_LEFT", "angle_deg": 15.0, "speed_percent": 40.0, "reason": "Bypassing debris hazard while maintaining corridor tracking."},
            ),
            (
                "Exit door within 2 meters. Instructing evacuees to push emergency egress bar.",
                "report_evacuation_status",
                {"exit_visible": True, "exit_direction": "AHEAD", "hazards_detected": [], "instructions": "Exit door directly ahead. Push door bar to evacuate building."},
            ),
        ]

        try:
            while self._running:
                await asyncio.sleep(4.0)
                if not self._running:
                    break

                reasoning, tool_name, tool_args = scenarios[step % len(scenarios)]
                step += 1

                self._latest_reasoning = f"[MOCK ER-2] {reasoning}"
                self._emit("reasoning", self._latest_reasoning)

                call_id = f"mock_call_{step}_{int(time.time())}"
                directive = RobotDirective(
                    tool_name=tool_name,
                    arguments=tool_args,
                    call_id=call_id,
                )
                self._active_directives.append(directive)
                self._emit("directive", directive)

                if self.on_directive:
                    try:
                        res = self.on_directive(directive)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        logger.error(f"Error in mock directive handler: {e}")

        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Gracefully closes the Gemini Live session."""
        self._running = False
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if self._task_receive:
            self._task_receive.cancel()
            try:
                task_loop = getattr(self._task_receive, "get_loop", lambda: getattr(self._task_receive, "_loop", None))()
                if current_loop is not None and current_loop == task_loop:
                    await self._task_receive
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._task_receive = None

        if self._task_mock:
            self._task_mock.cancel()
            try:
                task_loop = getattr(self._task_mock, "get_loop", lambda: getattr(self._task_mock, "_loop", None))()
                if current_loop is not None and current_loop == task_loop:
                    await self._task_mock
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._task_mock = None

        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

        self._emit("status", "stopped")
        logger.info("GeminiRoboticsLiveClient stopped.")
