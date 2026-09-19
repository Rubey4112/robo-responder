"""
Vision WebSocket Streaming Server
---------------------------------
Asynchronous WebSocket server for real-time video streaming, telemetry
broadcasting, and bidirectional interaction with Roboguide vision and Gemini AI.

Features:
1. High-framerate JPEG video stream broadcasting (base64 or raw binary).
2. Live broadcasting of Gemini ER 2 reasoning, emergency directives, and sensor telemetry.
3. Inbound command handling:
   - Custom prompts to Gemini Live session.
   - Remote snapshot triggers.
   - Streaming pause/resume.
   - Ping/pong heartbeat.
   - Remote client frame ingestion (allowing phone or remote camera stream).
4. Multi-client support with non-blocking broadcast queues.
"""

import asyncio
import base64
from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional, Set

import websockets
from websockets.asyncio.server import ServerConnection, serve

logger = logging.getLogger("vision.websocket_streamer")


@dataclass
class InboundCommand:
    """Represents a command received from a connected WebSocket client."""
    command: str
    params: Dict[str, Any]
    client_id: str
    timestamp: float = time.time()


class VisionWebSocketServer:
    """
    WebSocket server for streaming live camera feeds and telemetry to remote viewers/agents.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_command: Optional[Callable[[InboundCommand], Any]] = None,
        on_inbound_frame: Optional[Callable[[bytes], Any]] = None,
        max_clients: int = 10,
    ):
        self.host = host
        self.port = port
        self.on_command = on_command
        self.on_inbound_frame = on_inbound_frame
        self.max_clients = max_clients

        self._clients: Set[ServerConnection] = set()
        self._server = None
        self._running = False
        self._broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        self._task_broadcast: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def start(self):
        """Starts the WebSocket server."""
        if self._running:
            return

        self._running = True
        self._task_broadcast = asyncio.create_task(self._broadcast_worker())

        try:
            self._server = await serve(self._handle_client, self.host, self.port)
            logger.info(f"Vision WebSocket server listening at ws://{self.host}:{self.port}")
        except Exception as e:
            self._running = False
            logger.error(f"Failed to bind WebSocket server on {self.host}:{self.port}: {e}")
            raise

    async def stop(self):
        """Stops the WebSocket server and disconnects clients."""
        self._running = False
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if self._task_broadcast:
            self._task_broadcast.cancel()
            try:
                task_loop = getattr(self._task_broadcast, "get_loop", lambda: getattr(self._task_broadcast, "_loop", None))()
                if current_loop is not None and current_loop == task_loop:
                    await self._task_broadcast
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._task_broadcast = None

        # Close all active connections
        for client in list(self._clients):
            try:
                await client.close(code=1000, reason="Server shutting down")
            except Exception:
                pass
        self._clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("Vision WebSocket server stopped.")

    async def _handle_client(self, websocket: ServerConnection):
        """Connection handler for an individual WebSocket client."""
        client_addr = getattr(websocket, "remote_address", "unknown")
        if len(self._clients) >= self.max_clients:
            logger.warning(f"Rejecting client {client_addr}: max client limit reached ({self.max_clients})")
            await websocket.close(code=1008, reason="Max clients reached")
            return

        self._clients.add(websocket)
        logger.info(f"WebSocket client connected: {client_addr} (Active clients: {len(self._clients)})")

        # Send greeting / handshake
        try:
            welcome_msg = json.dumps({
                "type": "welcome",
                "message": "Connected to Roboguide Vision Stream Server",
                "timestamp": time.time(),
                "server_version": "2.0.0",
                "active_clients": len(self._clients),
            })
            await websocket.send(welcome_msg)
        except Exception as e:
            logger.warning(f"Failed to send welcome message to {client_addr}: {e}")

        try:
            async for raw_message in websocket:
                await self._process_inbound_message(websocket, raw_message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected cleanly: {client_addr}")
        except Exception as e:
            logger.warning(f"Error handling WebSocket client {client_addr}: {e}")
        finally:
            self._clients.discard(websocket)
            logger.info(f"WebSocket client removed: {client_addr} (Remaining: {len(self._clients)})")

    async def _process_inbound_message(self, websocket: ServerConnection, message: Any):
        """Parses and executes an inbound command or frame from a client."""
        client_id = str(getattr(websocket, "remote_address", "client"))

        # Binary frame upload
        if isinstance(message, bytes):
            if self.on_inbound_frame:
                try:
                    res = self.on_inbound_frame(message)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    logger.error(f"Error in on_inbound_frame callback: {e}")
            return

        # Text / JSON message
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON format"}))
            return

        msg_type = payload.get("type") or payload.get("command")

        # Built-in Ping-Pong
        if msg_type in ("ping", "PING"):
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": time.time(),
                "client_timestamp": payload.get("timestamp"),
            }))
            return

        # Base64 encoded frame upload
        if msg_type == "frame" and "data" in payload:
            try:
                raw_bytes = base64.b64decode(payload["data"])
                if self.on_inbound_frame:
                    res = self.on_inbound_frame(raw_bytes)
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as e:
                logger.error(f"Error decoding inbound base64 frame: {e}")
            return

        # Application-level command dispatch
        if self.on_command:
            cmd = InboundCommand(
                command=str(msg_type or "unknown"),
                params=payload,
                client_id=client_id,
            )
            try:
                res = self.on_command(cmd)
                if asyncio.iscoroutine(res):
                    res = await res
                if isinstance(res, dict):
                    await websocket.send(json.dumps({
                        "type": "command_response",
                        "command": cmd.command,
                        "result": res,
                    }))
            except Exception as e:
                logger.error(f"Error executing inbound command {cmd.command}: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "command": cmd.command,
                    "message": str(e),
                }))

    async def _broadcast_worker(self):
        """Worker task that dequeues broadcast packets and sends them to all clients."""
        while self._running:
            try:
                packet = await self._broadcast_queue.get()
                if not self._clients:
                    self._broadcast_queue.task_done()
                    continue

                # Prepare dead client set
                dead_clients = set()
                # Broadcast concurrently
                coros = []
                for client in self._clients:
                    coros.append(self._safe_send(client, packet, dead_clients))

                if coros:
                    await asyncio.gather(*coros)

                for dead in dead_clients:
                    self._clients.discard(dead)

                self._broadcast_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket broadcast worker: {e}")

    async def _safe_send(self, client: ServerConnection, packet: str, dead_clients: Set[ServerConnection]):
        """Safely sends packet to client, catching connection failures."""
        try:
            await client.send(packet)
        except Exception:
            dead_clients.add(client)

    def broadcast_frame(
        self,
        jpeg_bytes: bytes,
        width: int,
        height: int,
        fps: float = 0.0,
        telemetry: Optional[Dict[str, Any]] = None,
    ):
        """
        Enqueues a live camera frame to be broadcast to all connected WebSocket clients.
        Drops oldest frame if queue is saturated to prevent latency lag.
        """
        if not self._running or not self._clients or not jpeg_bytes:
            return

        b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
        payload = {
            "type": "frame",
            "timestamp": time.time(),
            "width": width,
            "height": height,
            "fps": round(fps, 1),
            "data": b64_str,
            "telemetry": telemetry or {},
        }
        msg = json.dumps(payload)

        try:
            if self._broadcast_queue.full():
                try:
                    self._broadcast_queue.get_nowait()
                    self._broadcast_queue.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    pass
            self._broadcast_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def broadcast_event(self, event_type: str, data: Any):
        """Broadcasts an AI reasoning or status event to all WebSocket clients."""
        if not self._running or not self._clients:
            return

        payload = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        msg = json.dumps(payload)

        try:
            if self._broadcast_queue.full():
                try:
                    self._broadcast_queue.get_nowait()
                    self._broadcast_queue.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    pass
            self._broadcast_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass
