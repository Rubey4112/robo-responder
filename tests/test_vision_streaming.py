"""
Unit and Integration Tests for Vision Streaming Subsystem
---------------------------------------------------------
Tests WebSocket streaming, frame encoding, Gemini Live config generation,
and mock event loop dispatching.
"""

import asyncio
import base64
import json
import unittest
import numpy as np
import cv2
import websockets

from src.vision.gemini_live import (
    GeminiRoboticsLiveClient,
    GeminiLiveEvent,
    RobotDirective,
    build_robotics_tools,
    DEFAULT_MODEL,
)
from src.vision.websocket_streamer import VisionWebSocketServer, InboundCommand
from src.vision.stream_pipeline import VisionStreamPipeline, StreamPipelineConfig


class TestGeminiLiveClient(unittest.TestCase):
    """Test suite for GeminiRoboticsLiveClient."""

    def test_build_robotics_tools(self):
        """Verifies function declarations and BLOCKING behavior."""
        tools = build_robotics_tools()
        self.assertEqual(len(tools), 1)
        declarations = tools[0].function_declarations
        names = [d.name for d in declarations]
        self.assertIn("navigate_robot", names)
        self.assertIn("emergency_stop", names)
        self.assertIn("report_evacuation_status", names)

        # Ensure all declarations have BLOCKING behavior
        for decl in declarations:
            self.assertEqual(decl.behavior.name, "BLOCKING")

    def test_mock_client_lifecycle(self):
        """Tests that mock mode generates events and handles directives."""
        events = []
        directives = []

        async def run_test():
            client = GeminiRoboticsLiveClient(
                mock_mode=True,
                on_event=lambda e: events.append(e),
                on_directive=lambda d: directives.append(d),
            )
            await client.start()
            self.assertTrue(client.is_running)
            self.assertTrue(client.is_mock)

            # Test frame sending in mock mode
            dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
            sent = await client.send_frame(dummy_jpeg, force=True)
            self.assertTrue(sent)

            # Test prompt sending in mock mode
            await client.send_text_prompt("Find exit door")
            self.assertIn("Find exit door", client.latest_reasoning)

            # Wait briefly for mock loop to tick
            await asyncio.sleep(0.5)

            await client.stop()
            self.assertFalse(client.is_running)

        asyncio.run(run_test())
        self.assertTrue(any(e.event_type == "status" for e in events))


class TestWebSocketServer(unittest.TestCase):
    """Test suite for VisionWebSocketServer."""

    def test_websocket_server_broadcast_and_commands(self):
        """Tests server start, client handshake, frame broadcast, and command dispatch."""
        test_port = 18765
        commands_received = []

        def on_cmd(cmd: InboundCommand):
            commands_received.append(cmd)
            return {"ack": True, "command": cmd.command}

        async def run_ws_test():
            server = VisionWebSocketServer(host="127.0.0.1", port=test_port, on_command=on_cmd)
            await server.start()
            self.assertTrue(server.is_running)

            # Connect client
            uri = f"ws://127.0.0.1:{test_port}"
            async with websockets.connect(uri) as client:
                # 1. Verify Welcome message
                welcome_raw = await client.recv()
                welcome = json.loads(welcome_raw)
                self.assertEqual(welcome.get("type"), "welcome")

                # 2. Test Ping / Pong
                await client.send(json.dumps({"command": "ping", "timestamp": 12345}))
                pong_raw = await client.recv()
                pong = json.loads(pong_raw)
                self.assertEqual(pong.get("type"), "pong")

                # 3. Test Frame Broadcast
                dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                _, buf = cv2.imencode(".jpg", dummy_frame)
                jpeg_bytes = buf.tobytes()

                server.broadcast_frame(
                    jpeg_bytes=jpeg_bytes,
                    width=100,
                    height=100,
                    fps=25.0,
                    telemetry={"test": 123},
                )

                frame_msg_raw = await asyncio.wait_for(client.recv(), timeout=2.0)
                frame_msg = json.loads(frame_msg_raw)
                self.assertEqual(frame_msg.get("type"), "frame")
                self.assertEqual(frame_msg.get("width"), 100)
                self.assertEqual(frame_msg.get("height"), 100)
                decoded_data = base64.b64decode(frame_msg["data"])
                self.assertEqual(decoded_data, jpeg_bytes)

                # 4. Test Inbound Custom Command
                await client.send(json.dumps({"command": "prompt", "text": "Locate exit"}))
                resp_raw = await asyncio.wait_for(client.recv(), timeout=2.0)
                resp = json.loads(resp_raw)
                self.assertEqual(resp.get("type"), "command_response")
                self.assertEqual(resp.get("command"), "prompt")

            await server.stop()
            self.assertFalse(server.is_running)

        asyncio.run(run_ws_test())
        self.assertEqual(len(commands_received), 1)
        self.assertEqual(commands_received[0].command, "prompt")


class TestStreamPipeline(unittest.TestCase):
    """Test suite for VisionStreamPipeline components."""

    def test_pipeline_config_and_hud(self):
        """Tests HUD rendering and pipeline initialization."""
        config = StreamPipelineConfig(
            camera_index=-1,  # Non-existent camera for test
            ws_port=18766,
            mock_gemini=True,
            enable_gui=False,
        )
        pipeline = VisionStreamPipeline(config=config)

        # Test HUD overlay rendering on synthetic frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.gemini_client._latest_reasoning = "Test evacuation reasoning"
        pipeline._fps_actual = 29.5

        rendered = pipeline._render_hud_overlay(test_frame)
        self.assertEqual(rendered.shape, (480, 640, 3))
        # Ensure frame has non-zero pixels from header/text drawn
        self.assertGreater(np.sum(rendered), 0)


if __name__ == "__main__":
    unittest.main()
