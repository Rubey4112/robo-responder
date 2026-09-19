#!/usr/bin/env python3
"""
Roboguide Gemini Robotics ER 2 LLM Client
-----------------------------------------
Asynchronous client interface to Google Gemini SDK (`google-genai`), tailored for
`gemini-robotics-er-2-streaming-preview` with real-time video/vision streaming and
autonomous motor navigation using driving tools from `src.agent.driving_functions`.

Key Features:
  - Native support for `gemini-robotics-er-2-streaming-preview` and Gemini Live API.
  - Automatic tool binding and async execution for all XRP driving tools:
      * move_forward
      * steer_slight_left
      * steer_slight_right
      * turn_hard_left
      * turn_hard_right
      * stop_robot
      * stop_for_obstacle
      * get_robot_status
  - Real-time video frame streaming via `LiveClientRealtimeInput` (OpenCV BGR frames / JPEG bytes).
  - Background live receiver loop with automatic tool execution and response dispatch.
  - Single-shot async frame analysis (`analyze_frame_and_navigate`) with tool invocation.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import sys
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import cv2
import numpy as np

# Google GenAI SDK (v2+)
from google import genai
from google.genai import types

# Support direct execution as well as package imports
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from src.agent.driving_functions import (
        DRIVING_TOOLS,
        TOOL_MAP,
        execute_async_tool,
        get_robot_bridge,
    )
except ImportError:
    from driving_functions import (
        DRIVING_TOOLS,
        TOOL_MAP,
        execute_async_tool,
        get_robot_bridge,
    )

# Logger setup
logger = logging.getLogger("RoboguideLLM")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"

ROBOGUIDE_SYSTEM_INSTRUCTION = """You are Roboguide, an autonomous robotic emergency evacuation responder guiding humans safely to the nearest emergency exit during hazards (fires, earthquakes, smoke, collapsed pathways).

Your responsibilities:
1. Scene Evaluation: Inspect real-time camera frames to identify exit signage ("EXIT", green running man, exit doors), open corridors, doorways, and clear paths.
2. Obstacle & Hazard Detection: Detect impassable rubble, smoke, flames, dead ends, walls, or humans in need of guidance.
3. Autonomous Navigation Decisions: Invoke driving tools to control the robot's physical movement:
   - Call `move_forward(duration_seconds)` when the corridor or path directly ahead is clear and you want to advance towards safety.
   - Call `steer_slight_left(duration_seconds)` or `steer_slight_right(duration_seconds)` (0.5s - 1.0s) for gentle curvature following, hallway centering, or drifting away from obstacles.
   - Call `turn_hard_left(duration_seconds)` or `turn_hard_right(duration_seconds)` (~0.6s for 90 degrees) to turn corners at intersections or pivot away from blocked corridors.
   - Call `stop_for_obstacle(reason)` IMMEDIATELY whenever an obstacle, wall, hazard, fire, or person blocks the immediate path.
   - Call `stop_robot()` when you reach the exit door, when pausing to confirm bearings, or when waiting for human evacuees.
   - Call `get_robot_status()` to inspect connectivity and telemetry when starting up.
4. Voice & Verbal Directives: Speak clearly, calmly, and authoritatively to evacuees following you (e.g., "Exit ahead, follow me.", "Debris detected, turning right.").
"""


class RoboguideLiveSession:
    """
    Manages an active bidirectional streaming session with `gemini-robotics-er-2-streaming-preview`.

    Handles streaming camera frames, receiving text/audio from the model, and automatically
    executing driving tools asynchronously without blocking the video stream.
    """

    def __init__(
        self,
        session: Any,
        on_text: Optional[Callable[[str], Any]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        on_tool_result: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], Any]] = None,
    ):
        self.session = session
        self.on_text = on_text
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result

        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._latest_model_text = ""

    async def start(self):
        """Starts the background receiver loop to process responses and tool calls."""
        if self._running:
            return
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Roboguide live session background receiver started.")

    async def _receive_loop(self):
        """Continuously consumes server messages, dispatches text, and handles tool calls."""
        try:
            async for response in self.session.receive():
                if not self._running:
                    break

                # 1. Process model turn content (text / thought)
                server_content = response.server_content
                if server_content and server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        if hasattr(part, "text") and part.text:
                            self._latest_model_text = part.text
                            logger.info(f"[GEMINI]: {part.text}")
                            if self.on_text:
                                try:
                                    if asyncio.iscoroutinefunction(self.on_text):
                                        await self.on_text(part.text)
                                    else:
                                        self.on_text(part.text)
                                except Exception as e:
                                    logger.error(f"Error in on_text callback: {e}")

                # 2. Process tool calls
                tool_call = response.tool_call
                if tool_call and tool_call.function_calls:
                    function_responses = []
                    for call in tool_call.function_calls:
                        call_name = call.name
                        call_args = call.args or {}
                        call_id = getattr(call, "id", None) or f"call_{int(time.time() * 1000)}"

                        logger.info(f"[TOOL REQUEST] Gemini ER 2 requested tool: {call_name}({call_args})")
                        if self.on_tool_call:
                            try:
                                if asyncio.iscoroutinefunction(self.on_tool_call):
                                    await self.on_tool_call(call_name, call_args)
                                else:
                                    self.on_tool_call(call_name, call_args)
                            except Exception as e:
                                logger.error(f"Error in on_tool_call callback: {e}")

                        # Execute tool asynchronously (with preemption & watchdog keepalive)
                        result = await execute_async_tool(call_name, call_args)
                        logger.info(f"[TOOL RESULT] {call_name} -> {result}")

                        if self.on_tool_result:
                            try:
                                if asyncio.iscoroutinefunction(self.on_tool_result):
                                    await self.on_tool_result(call_name, call_args, result)
                                else:
                                    self.on_tool_result(call_name, call_args, result)
                            except Exception as e:
                                logger.error(f"Error in on_tool_result callback: {e}")

                        function_responses.append(
                            types.FunctionResponse(
                                name=call_name,
                                id=call_id,
                                response=result,
                            )
                        )

                    # Transmit tool response back to the live session
                    try:
                        await self.session.send(
                            input=types.LiveClientToolResponse(
                                function_responses=function_responses
                            )
                        )
                        logger.debug(f"Dispatched {len(function_responses)} tool responses back to Gemini.")
                    except Exception as e:
                        logger.error(f"Failed to transmit tool response to live session: {e}")

                # 3. Handle tool cancellations
                if response.tool_call_cancellation:
                    logger.warning(
                        f"[TOOL CANCELLED] Gemini cancelled tool calls: {response.tool_call_cancellation.ids}"
                    )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                logger.error(f"Exception encountered in live receive loop: {e}")
        finally:
            self._running = False

    async def send_frame(
        self,
        frame: Union[np.ndarray, bytes],
        mime_type: str = "image/jpeg",
        jpeg_quality: int = 80,
    ) -> bool:
        """
        Sends a single camera frame in real time to the live session.

        Args:
            frame: Either an OpenCV BGR numpy array (from cv2.VideoCapture) or raw image bytes.
            mime_type: Image MIME type (default: 'image/jpeg').
            jpeg_quality: JPEG compression quality if frame is a numpy ndarray (1-100).

        Returns:
            bool: True if transmitted successfully, False otherwise.
        """
        if not self._running:
            logger.warning("Attempted to send frame to an inactive live session.")
            return False

        try:
            if isinstance(frame, np.ndarray):
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                success, encoded_img = cv2.imencode(".jpg", frame, encode_param)
                if not success:
                    logger.error("Failed to JPEG-encode image frame.")
                    return False
                frame_bytes = encoded_img.tobytes()
            else:
                frame_bytes = frame

            # Package as realtime input media chunk
            realtime_input = types.LiveClientRealtimeInput(
                media_chunks=[types.Blob(data=frame_bytes, mime_type=mime_type)]
            )
            await self.session.send(input=realtime_input)
            return True
        except Exception as e:
            logger.error(f"Error transmitting frame to live session: {e}")
            return False

    async def send_text(self, text: str, end_of_turn: bool = True) -> bool:
        """
        Sends an instruction or user prompt to the live session.

        Args:
            text: Text message or emergency command directive.
            end_of_turn: Whether this message ends the user's turn.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        try:
            await self.session.send(input=text, end_of_turn=end_of_turn)
            return True
        except Exception as e:
            logger.error(f"Error transmitting text prompt: {e}")
            return False

    async def close(self):
        """Terminates the live receiver loop and closes the session."""
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        logger.info("Roboguide live session closed.")


class RoboguideClient:
    """
    Main Roboguide Gemini Client wrapper.

    Provides high-level integration with `gemini-robotics-er-2-streaming-preview`
    for both live bidirectional sessions and one-shot frame navigation requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        system_instruction: str = ROBOGUIDE_SYSTEM_INSTRUCTION,
        tools: Optional[List[Callable[..., Any]]] = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.system_instruction = system_instruction
        self.tools = tools if tools is not None else DRIVING_TOOLS
        self._client: Optional[genai.Client] = None

        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)
        else:
            logger.warning(
                "GEMINI_API_KEY is not set. Please set the environment variable "
                "'GEMINI_API_KEY' or pass 'api_key' to RoboguideClient before making API calls."
            )

    @property
    def client(self) -> genai.Client:
        """Returns the active genai.Client or raises a descriptive error if unconfigured."""
        if self._client is None:
            if not self.api_key:
                self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise RuntimeError(
                    "No Gemini API key provided. Please set the 'GEMINI_API_KEY' environment variable "
                    "or pass 'api_key' when constructing RoboguideClient."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @asynccontextmanager
    async def connect_live(
        self,
        response_modalities: Optional[List[str]] = None,
        on_text: Optional[Callable[[str], Any]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        on_tool_result: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], Any]] = None,
    ) -> AsyncIterator[RoboguideLiveSession]:
        """
        Asynchronously connects to a Gemini Live WebSocket session with driving tools.

        Yields:
            RoboguideLiveSession: The managed live session instance.
        """
        modalities = [types.LiveModality.TEXT]
        if response_modalities:
            modalities = [types.LiveModality(m) for m in response_modalities]

        live_config = types.LiveConnectConfig(
            response_modalities=modalities,
            tools=self.tools,
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=self.system_instruction)]
            ),
        )

        logger.info(f"Connecting to Gemini Live API with model: '{self.model}'...")
        async with self.client.aio.live.connect(model=self.model, config=live_config) as raw_session:
            live_session = RoboguideLiveSession(
                session=raw_session,
                on_text=on_text,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
            )
            await live_session.start()
            try:
                yield live_session
            finally:
                await live_session.close()

    async def analyze_frame_and_navigate(
        self,
        frame: Union[np.ndarray, bytes],
        prompt: str = "Inspect this camera frame. If you spot an emergency exit or clear hall, navigate toward it. If blocked, steer or stop.",
        execute_tools: bool = True,
    ) -> Dict[str, Any]:
        """
        One-shot analysis: sends a single frame to the model and executes any requested driving tools.

        Args:
            frame: OpenCV BGR frame (numpy.ndarray) or raw JPEG image bytes.
            prompt: Text prompt accompanying the image frame.
            execute_tools: If True, automatically executes function calls returned by the model.

        Returns:
            Dict containing:
                - text: The model's reasoning/verbal response.
                - tool_calls: List of requested tool calls.
                - tool_results: Results from executed tools.
        """
        if isinstance(frame, np.ndarray):
            success, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                return {"error": "Failed to JPEG encode frame"}
            image_bytes = encoded_img.tobytes()
        else:
            image_bytes = frame

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        contents = [prompt, image_part]

        config = types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=self.system_instruction,
            temperature=0.2,
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            result_summary: Dict[str, Any] = {
                "text": response.text if hasattr(response, "text") else "",
                "tool_calls": [],
                "tool_results": [],
            }

            # Extract any tool calls from response candidates
            if response.function_calls:
                for call in response.function_calls:
                    call_info = {"name": call.name, "args": call.args}
                    result_summary["tool_calls"].append(call_info)
                    logger.info(f"Model requested tool: {call.name}({call.args})")

                    if execute_tools:
                        tool_res = await execute_async_tool(call.name, call.args)
                        result_summary["tool_results"].append(
                            {"name": call.name, "result": tool_res}
                        )

            return result_summary

        except Exception as e:
            logger.error(f"Error in analyze_frame_and_navigate: {e}")
            return {"error": str(e)}


# ============================================================================
# Self-Test and Example Demonstration
# ============================================================================

async def _self_test():
    """Demonstrates client configuration, offline tool checks, and synthetic frame testing."""
    print("\n" + "=" * 65)
    print(" ROBOGUIDE GEMINI ROBOTICS ER 2 CLIENT VERIFICATION ")
    print("=" * 65)

    client = RoboguideClient()
    print(f"[*] Target Model       : {client.model}")
    print(f"[*] Driving Tools Count: {len(client.tools)}")
    print(f"[*] Registered Tools   : {[f.__name__ for f in client.tools]}")

    api_key_present = bool(os.environ.get("GEMINI_API_KEY"))
    print(f"[*] GEMINI_API_KEY set : {api_key_present}")

    # Generate synthetic dummy frame (640x480 gray hallway mockup)
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        synthetic_frame,
        "EMERGENCY EXIT -->",
        (120, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    if api_key_present:
        print("\n[*] Testing analyze_frame_and_navigate with synthetic EXIT frame...")
        res = await client.analyze_frame_and_navigate(
            synthetic_frame,
            prompt="You see an exit sign. What navigation action should you take?",
            execute_tools=True,
        )
        print(f"[OK] Response Received:\n{res}")
    else:
        print("\n[!] Notice: GEMINI_API_KEY is not set in this environment.")
        print("[*] To test with live Gemini ER 2 API, set:")
        print("      $env:GEMINI_API_KEY='your_api_key'")
        print("[*] Verifying tool declarations format with GenerateContentConfig...")
        config = types.GenerateContentConfig(
            tools=client.tools,
            system_instruction=client.system_instruction,
        )
        print(f"[OK] GenerateContentConfig successfully verified with {len(config.tools)} tools!")

    print("\n[OK] Client self-test completed successfully.\n")


if __name__ == "__main__":
    asyncio.run(_self_test())
