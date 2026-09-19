"""
Hardware-Integrated Unit and Integration Tests for Roboguide LLM Client
------------------------------------------------------------------------
Tests the interaction between RoboguideClient, RoboguideLiveSession, and the
physical XRP robot hardware using mocked Gemini API responses (no real API keys
or cloud network calls required).

When the XRP robot is plugged in (e.g. COM4), tool calls physically transmit
real motor commands over USB-Serial to the MicroPython firmware on the robot.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from google.genai import types
from src.agent.driving_functions import (
    DRIVING_TOOLS,
    AsyncRobotBridge,
    execute_async_tool,
    get_robot_bridge,
)
from src.agent.llm_client import (
    DEFAULT_MODEL,
    ROBOGUIDE_SYSTEM_INSTRUCTION,
    RoboguideClient,
    RoboguideLiveSession,
)


class TestRoboguideClientInitialization(unittest.TestCase):
    """Verifies RoboguideClient configuration, model defaults, and API key handling."""

    def test_default_configuration(self):
        """Verifies default model identifier, tool count, and system instruction."""
        client = RoboguideClient(api_key="mock_api_key_test")
        self.assertEqual(client.model, DEFAULT_MODEL)
        self.assertEqual(client.model, "gemini-robotics-er-2-streaming-preview")
        self.assertEqual(len(client.tools), len(DRIVING_TOOLS))
        self.assertEqual(client.system_instruction, ROBOGUIDE_SYSTEM_INSTRUCTION)

        tool_names = [func.__name__ for func in client.tools]
        expected_tools = [
            "move_forward",
            "steer_slight_left",
            "steer_slight_right",
            "turn_hard_left",
            "turn_hard_right",
            "stop_robot",
            "stop_for_obstacle",
            "get_robot_status",
        ]
        for expected in expected_tools:
            self.assertIn(expected, tool_names)

    def test_missing_api_key_raises_runtime_error(self):
        """Accessing client without an API key or env var raises a helpful RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            client = RoboguideClient(api_key=None)
            with self.assertRaises(RuntimeError) as ctx:
                _ = client.client
            self.assertIn("GEMINI_API_KEY", str(ctx.exception))


class TestHardwareAnalyzeFrameAndNavigate(unittest.IsolatedAsyncioTestCase):
    """
    Tests one-shot frame analysis and autonomous robot command execution.
    Sends real motor commands over serial to the connected XRP robot.
    """

    async def asyncSetUp(self):
        self.bridge = get_robot_bridge()
        # Connect to actual hardware (auto-detects COM4 / RP2040 Pico)
        await self.bridge.connect()
        self.client = RoboguideClient(api_key="mock_test_key")
        self.mock_genai_client = MagicMock()
        self.client._client = self.mock_genai_client
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    async def asyncTearDown(self):
        # Always issue immediate safety stop after each test to ensure motors halt
        if self.bridge.is_connected:
            await self.bridge.execute_motion("STOP", 0.0)

    async def test_hardware_move_forward_directive(self):
        """Mocks Gemini ER 2 proposing move_forward and verifies hardware execution."""
        mock_call = MagicMock()
        mock_call.name = "move_forward"
        mock_call.args = {"duration_seconds": 0.2}

        mock_response = MagicMock()
        mock_response.text = "Path to exit is clear. Moving forward."
        mock_response.function_calls = [mock_call]

        self.mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await self.client.analyze_frame_and_navigate(
            frame=self.dummy_frame,
            prompt="Find the exit",
            execute_tools=True,
        )

        self.assertEqual(result["text"], "Path to exit is clear. Moving forward.")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "move_forward")
        self.assertEqual(len(result["tool_results"]), 1)

        tool_res = result["tool_results"][0]["result"]
        self.assertIn(tool_res["status"], ("success", "simulated"))
        self.assertEqual(tool_res["action"], "FORWARD")
        self.assertEqual(self.bridge.last_command, "STOP")

    async def test_hardware_emergency_stop_directive(self):
        """Mocks Gemini ER 2 detecting an obstacle and triggering emergency hardware stop."""
        mock_call = MagicMock()
        mock_call.name = "stop_for_obstacle"
        mock_call.args = {"reason": "Dense smoke and flames spotted in doorway"}

        mock_response = MagicMock()
        mock_response.text = "Fire detected! Halting immediately."
        mock_response.function_calls = [mock_call]

        self.mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await self.client.analyze_frame_and_navigate(
            frame=self.dummy_frame,
            prompt="Check for hazards",
            execute_tools=True,
        )

        self.assertEqual(len(result["tool_results"]), 1)
        tool_res = result["tool_results"][0]["result"]
        self.assertEqual(tool_res["action"], "STOP_OBSTACLE")
        self.assertEqual(tool_res["reason"], "Dense smoke and flames spotted in doorway")
        self.assertEqual(self.bridge.last_command, "STOP_OBSTACLE")

    async def test_hardware_pivot_turns(self):
        """Tests turn_hard_left and turn_hard_right motor execution on hardware."""
        # Test hard left
        mock_call_left = MagicMock()
        mock_call_left.name = "turn_hard_left"
        mock_call_left.args = {"duration_seconds": 0.2}

        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text="Turning left", function_calls=[mock_call_left])
        )

        res_left = await self.client.analyze_frame_and_navigate(self.dummy_frame)
        self.assertEqual(res_left["tool_results"][0]["result"]["action"], "HARD_LEFT")
        self.assertEqual(self.bridge.last_command, "STOP")

        # Test hard right
        mock_call_right = MagicMock()
        mock_call_right.name = "turn_hard_right"
        mock_call_right.args = {"duration_seconds": 0.2}

        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text="Turning right", function_calls=[mock_call_right])
        )

        res_right = await self.client.analyze_frame_and_navigate(self.dummy_frame)
        self.assertEqual(res_right["tool_results"][0]["result"]["action"], "HARD_RIGHT")
        self.assertEqual(self.bridge.last_command, "STOP")

    async def test_hardware_steer_curves(self):
        """Tests steer_slight_left and steer_slight_right gentle navigation on hardware."""
        mock_call_curve = MagicMock()
        mock_call_curve.name = "steer_slight_left"
        mock_call_curve.args = {"duration_seconds": 0.2}

        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text="Centering in hallway", function_calls=[mock_call_curve])
        )

        res = await self.client.analyze_frame_and_navigate(self.dummy_frame)
        self.assertEqual(res["tool_results"][0]["result"]["action"], "SLIGHT_LEFT")
        self.assertEqual(self.bridge.last_command, "STOP")

    async def test_execute_tools_false_flag(self):
        """When execute_tools=False, tool calls are recorded but not dispatched to hardware."""
        mock_call = MagicMock()
        mock_call.name = "move_forward"
        mock_call.args = {"duration_seconds": 0.5}

        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text="Planned move", function_calls=[mock_call])
        )

        self.bridge.last_command = "INITIAL_STATE"
        result = await self.client.analyze_frame_and_navigate(
            frame=self.dummy_frame,
            execute_tools=False,
        )

        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(len(result["tool_results"]), 0)
        self.assertEqual(self.bridge.last_command, "INITIAL_STATE")

    async def test_api_exception_handling(self):
        """Catches API exceptions and returns structured error dict without crashing."""
        self.mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Network timeout during live inference")
        )

        result = await self.client.analyze_frame_and_navigate(self.dummy_frame)
        self.assertIn("error", result)
        self.assertIn("Network timeout", result["error"])


class TestHardwareRoboguideLiveSession(unittest.IsolatedAsyncioTestCase):
    """
    Tests bidirectional live streaming session, frame transmission, and tool dispatch
    against physical robot hardware.
    """

    async def asyncSetUp(self):
        self.bridge = get_robot_bridge()
        await self.bridge.connect()
        self.raw_session = MagicMock()
        self.raw_session.send = AsyncMock()

    async def asyncTearDown(self):
        if self.bridge.is_connected:
            await self.bridge.execute_motion("STOP", 0.0)

    async def test_send_frame_formats_realtime_input(self):
        """Verifies sending OpenCV BGR frame encodes to JPEG and sends LiveClientRealtimeInput."""
        session = RoboguideLiveSession(session=self.raw_session)
        session._running = True

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        success = await session.send_frame(dummy_frame, jpeg_quality=75)
        self.assertTrue(success)

        self.raw_session.send.assert_called_once()
        call_kwargs = self.raw_session.send.call_args[1]
        sent_input = call_kwargs["input"]
        self.assertIsInstance(sent_input, types.LiveClientRealtimeInput)
        self.assertEqual(len(sent_input.media_chunks), 1)
        self.assertEqual(sent_input.media_chunks[0].mime_type, "image/jpeg")

    async def test_send_text_directive(self):
        """Verifies send_text forwards the user message prompt with end_of_turn=True."""
        session = RoboguideLiveSession(session=self.raw_session)
        success = await session.send_text("Where is the emergency exit?")
        self.assertTrue(success)

        self.raw_session.send.assert_called_once_with(
            input="Where is the emergency exit?",
            end_of_turn=True,
        )

    async def test_live_session_hardware_tool_dispatch(self):
        """Mocks live server message with tool call and verifies physical robot execution."""
        msg = MagicMock()

        # Part 1: Text reasoning
        part = MagicMock()
        part.text = "Exit sign identified. Steering slight right."
        msg.server_content = MagicMock()
        msg.server_content.model_turn.parts = [part]

        # Part 2: Tool call
        func_call = MagicMock()
        func_call.id = "call_hw_456"
        func_call.name = "steer_slight_right"
        func_call.args = {"duration_seconds": 0.2}
        msg.tool_call = MagicMock()
        msg.tool_call.function_calls = [func_call]
        msg.tool_call_cancellation = None

        async def mock_receive_stream():
            yield msg
            await asyncio.sleep(0.05)

        self.raw_session.receive = mock_receive_stream

        texts_received = []
        tools_called = []
        tools_results = []

        session = RoboguideLiveSession(
            session=self.raw_session,
            on_text=lambda t: texts_received.append(t),
            on_tool_call=lambda name, args: tools_called.append((name, args)),
            on_tool_result=lambda name, args, res: tools_results.append((name, res)),
        )

        await session.start()
        await asyncio.sleep(0.35)
        await session.close()

        # Verify callbacks
        self.assertEqual(texts_received, ["Exit sign identified. Steering slight right."])
        self.assertEqual(len(tools_called), 1)
        self.assertEqual(tools_called[0][0], "steer_slight_right")
        self.assertEqual(len(tools_results), 1)
        self.assertEqual(tools_results[0][1]["action"], "SLIGHT_RIGHT")
        self.assertEqual(self.bridge.last_command, "STOP")

        # Verify tool response was transmitted back to Gemini Live API
        self.raw_session.send.assert_called_once()
        send_kwargs = self.raw_session.send.call_args[1]
        tool_response = send_kwargs["input"]
        self.assertIsInstance(tool_response, types.LiveClientToolResponse)
        self.assertEqual(len(tool_response.function_responses), 1)
        resp = tool_response.function_responses[0]
        self.assertEqual(resp.name, "steer_slight_right")
        self.assertEqual(resp.id, "call_hw_456")
        self.assertEqual(resp.response["action"], "SLIGHT_RIGHT")

    async def test_unknown_tool_graceful_handling(self):
        """Unknown tool calls return an error payload back to Gemini without crashing."""
        msg = MagicMock()
        msg.server_content = None

        func_call = MagicMock()
        func_call.id = "call_err_789"
        func_call.name = "non_existent_robotic_arm_grab"
        func_call.args = {}
        msg.tool_call = MagicMock()
        msg.tool_call.function_calls = [func_call]
        msg.tool_call_cancellation = None

        async def mock_receive_stream():
            yield msg
            await asyncio.sleep(0.05)

        self.raw_session.receive = mock_receive_stream
        session = RoboguideLiveSession(session=self.raw_session)

        await session.start()
        await asyncio.sleep(0.15)
        await session.close()

        self.raw_session.send.assert_called_once()
        tool_response = self.raw_session.send.call_args[1]["input"]
        self.assertEqual(tool_response.function_responses[0].response["status"], "error")


class TestDirectHardwareToolExecution(unittest.IsolatedAsyncioTestCase):
    """Direct hardware execution of tools via execute_async_tool."""

    async def asyncSetUp(self):
        self.bridge = get_robot_bridge()
        await self.bridge.connect()

    async def asyncTearDown(self):
        if self.bridge.is_connected:
            await self.bridge.execute_motion("STOP", 0.0)

    async def test_direct_move_forward(self):
        res = await execute_async_tool("move_forward", {"duration_seconds": 0.2})
        self.assertEqual(res["action"], "FORWARD")
        self.assertIn(res["status"], ("success", "simulated"))

    async def test_direct_stop_for_obstacle(self):
        res = await execute_async_tool("stop_for_obstacle", {"reason": "Wall collision risk"})
        self.assertEqual(res["action"], "STOP_OBSTACLE")
        self.assertEqual(res["reason"], "Wall collision risk")

    async def test_direct_get_robot_status(self):
        res = await execute_async_tool("get_robot_status", {})
        self.assertTrue(res["is_connected"])
        self.assertIsNotNone(res["port"])
        self.assertIn("last_command", res)


def tearDownModule():
    """Disconnect robot bridge cleanly after entire test suite completes."""
    bridge = get_robot_bridge()
    asyncio.run(bridge.disconnect())


if __name__ == "__main__":
    unittest.main()
