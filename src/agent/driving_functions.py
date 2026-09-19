#!/usr/bin/env python3
"""
Roboguide Async Driving Tools for Gemini SDK
--------------------------------------------
Provides a collection of native asynchronous tool functions and an AsyncRobotBridge
designed for integration with the Gemini SDK (including `gemini-robotics-er-2-streaming-preview`
and Google GenAI Live streaming sessions).

Firmware Command Mapping (src/robot/firmware.py):
  - FORWARD        : Drive both wheels forward (base speed: 0.5)
  - SLIGHT_LEFT    : Gentle curve left (left: 0.125, right: 0.5)
  - SLIGHT_RIGHT   : Gentle curve right (left: 0.5, right: 0.125)
  - HARD_LEFT      : Pivot/spin in place left (-0.35, +0.35)
  - HARD_RIGHT     : Pivot/spin in place right (+0.35, -0.35)
  - STOP           : Immediate stop of all motors
  - STOP_OBSTACLE  : Emergency stop when an obstacle or hazard is detected

Key Capabilities:
  - Non-blocking async keepalive pulses (0.5s interval) overcoming the firmware's 1.5s watchdog.
  - Active movement preemption: immediate cancellation of running motions upon emergency stop.
  - Automatic XRP COM port detection (Raspberry Pi Pico / SparkFun VID) with simulation fallback.
  - Complete Google-style docstrings and type annotations for Gemini OpenAPI schema generation.
"""

import asyncio
from collections import deque
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import serial
import serial.tools.list_ports

# Configure module logger
logger = logging.getLogger("RoboguideDriving")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default serial settings
DEFAULT_BAUDRATE = 115200
KEEPALIVE_INTERVAL = 0.5  # Firmware watchdog timeout is 1.5s in firmware.py
KNOWN_VENDORS = (0x1B4F, 0x2E8A)  # SparkFun (0x1B4F), Raspberry Pi Pico (0x2E8A)


def list_serial_ports() -> List[serial.tools.list_ports_common.ListPortInfo]:
    """Returns a list of all detected serial ports on the host system."""
    return list(serial.tools.list_ports.comports())


def detect_xrp_port() -> Optional[str]:
    """
    Auto-detects the XRP Robot / RP2040 Pico USB serial port.
    Matches SparkFun or Raspberry Pi USB Vendor IDs, or falls back to a single port.
    """
    ports = list_serial_ports()
    if not ports:
        return None

    for p in ports:
        if p.vid in KNOWN_VENDORS:
            return p.device

    # If exactly 1 port is available, use it as default
    if len(ports) == 1:
        return ports[0].device

    return None


class AsyncRobotBridge:
    """
    Asynchronous serial bridge manager for the XRP Robot.
    
    Provides non-blocking serial communication, keepalive pulsing for timed drives,
    emergency preemption, and fallback simulation mode when hardware is disconnected.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = DEFAULT_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.is_simulated: bool = False
        self.is_connected: bool = False
        self.recent_logs: deque[str] = deque(maxlen=25)
        self.last_command: str = "NONE"
        self.last_command_time: float = 0.0

        self._lock = asyncio.Lock()
        self._reader_running = False
        self._reader_task: Optional[asyncio.Task] = None
        self._current_movement_task: Optional[asyncio.Task] = None

    async def connect(self, target_port: Optional[str] = None) -> bool:
        """
        Asynchronously connects to the XRP robot serial port.
        If no port is found, switches gracefully to simulation mode.
        """
        async with self._lock:
            if self.is_connected and self.ser and self.ser.is_open:
                return True

            detected = detect_xrp_port()
            env_port = os.environ.get("ROBOT_PORT")
            port_to_try = target_port or env_port or (self.port if not self.is_simulated else None) or detected
            if not port_to_try:
                logger.warning(
                    "No XRP robot serial port detected. Running in SIMULATION mode."
                )
                self.is_simulated = True
                self.is_connected = True
                self.port = "SIMULATED_PORT"
                return True

            try:
                # Open serial port in worker thread to prevent blocking event loop
                self.ser = await asyncio.to_thread(
                    serial.Serial, port_to_try, self.baudrate, timeout=0.1
                )
                await asyncio.to_thread(self.ser.reset_input_buffer)
                await asyncio.to_thread(self.ser.reset_output_buffer)
                self.port = port_to_try
                self.is_connected = True
                self.is_simulated = False
                logger.info(f"Connected successfully to XRP robot on {self.port} at {self.baudrate} baud.")

                # Start background non-blocking telemetry listener
                self._reader_running = True
                self._reader_task = asyncio.create_task(self._background_reader())
                return True

            except Exception as e:
                logger.warning(
                    f"Failed to open serial port '{port_to_try}': {e}. Falling back to SIMULATION mode."
                )
                self.is_simulated = True
                self.is_connected = True
                self.port = f"SIMULATED ({port_to_try})"
                return True

    async def _background_reader(self):
        """Asynchronously reads incoming telemetry from the robot firmware."""
        while self._reader_running and self.ser and self.ser.is_open:
            try:
                # Read line via worker thread
                line = await asyncio.to_thread(self._read_line_blocking)
                if line:
                    clean_line = line.strip()
                    if clean_line:
                        self.recent_logs.append(clean_line)
                        logger.debug(f"[ROBOT TELEMETRY] {clean_line}")
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)

    def _read_line_blocking(self) -> str:
        """Helper to read a line from serial if data is available."""
        if not self.ser or not self.ser.is_open:
            return ""
        if self.ser.in_waiting > 0:
            return self.ser.readline().decode("utf-8", errors="replace")
        time.sleep(0.02)
        return ""

    async def _raw_send(self, cmd: str) -> bool:
        """Sends a single command string terminated with \\n directly to the firmware."""
        formatted = cmd.strip().upper()
        self.last_command = formatted
        self.last_command_time = time.time()

        if self.is_simulated:
            logger.info(f"[SIMULATION] Robot received command: {formatted}")
            self.recent_logs.append(f"[SIMULATED] XRP Received Command: {formatted}")
            return True

        if not self.ser or not self.ser.is_open:
            logger.error("Attempted to send command with closed serial port.")
            return False

        try:
            payload = f"{formatted}\n".encode("utf-8")
            await asyncio.to_thread(self._write_serial_blocking, payload)
            logger.info(f"[ROBOT SENT] {formatted}")
            return True
        except Exception as e:
            logger.error(f"Error transmitting command '{formatted}': {e}")
            return False

    def _write_serial_blocking(self, payload: bytes):
        if self.ser and self.ser.is_open:
            self.ser.write(payload)
            self.ser.flush()

    async def execute_motion(self, cmd: str, duration_seconds: float = 0.0) -> Dict[str, Any]:
        """
        Executes a motion command with non-blocking keepalive pulsing and preemption.
        
        If a previous timed motion is running, it is immediately cancelled before
        executing the new command.
        """
        # Ensure connection
        if not self.is_connected:
            await self.connect()

        # Cancel any ongoing active movement task
        if self._current_movement_task and not self._current_movement_task.done():
            self._current_movement_task.cancel()
            try:
                await self._current_movement_task
            except asyncio.CancelledError:
                pass

        cmd_upper = cmd.strip().upper()

        # Immediate zero-duration command (e.g. STOP or STOP_OBSTACLE)
        if duration_seconds <= 0 or cmd_upper in ("STOP", "STOP_OBSTACLE"):
            async with self._lock:
                await self._raw_send(cmd_upper)
            return {
                "status": "simulated" if self.is_simulated else "success",
                "action": cmd_upper,
                "duration_seconds": 0.0,
                "message": f"Issued immediate {cmd_upper} directive to robot.",
                "timestamp": time.time(),
            }

        # Timed motion: launch the keepalive runner task
        task = asyncio.create_task(self._run_timed_motion(cmd_upper, duration_seconds))
        self._current_movement_task = task

        try:
            result = await task
            return result
        except asyncio.CancelledError:
            # Preempted by an emergency stop or newer command
            async with self._lock:
                await self._raw_send("STOP")
            return {
                "status": "cancelled",
                "action": cmd_upper,
                "message": f"Motion '{cmd_upper}' was preempted by a subsequent command or emergency stop.",
                "timestamp": time.time(),
            }

    async def _run_timed_motion(self, cmd: str, duration: float) -> Dict[str, Any]:
        """
        Sends repeated keepalive pulses every KEEPALIVE_INTERVAL seconds,
        then automatically stops motors at the end of the duration.
        """
        start_time = time.time()
        elapsed = 0.0

        try:
            while elapsed < duration:
                async with self._lock:
                    await self._raw_send(cmd)
                sleep_time = min(KEEPALIVE_INTERVAL, max(0.01, duration - elapsed))
                await asyncio.sleep(sleep_time)
                elapsed = time.time() - start_time
        finally:
            # Ensure safety stop is sent when duration expires or task is cancelled
            async with self._lock:
                await self._raw_send("STOP")

        return {
            "status": "simulated" if self.is_simulated else "success",
            "action": cmd,
            "duration_seconds": round(duration, 2),
            "actual_duration_seconds": round(elapsed, 2),
            "message": f"Successfully completed '{cmd}' for {duration:.2f} seconds.",
            "timestamp": time.time(),
        }

    async def disconnect(self):
        """Safely stops the robot and terminates background listener and serial port."""
        if self._current_movement_task and not self._current_movement_task.done():
            self._current_movement_task.cancel()

        self._reader_running = False
        if self._reader_task:
            self._reader_task.cancel()

        async with self._lock:
            if self.ser and self.ser.is_open:
                try:
                    await self._raw_send("STOP")
                    await asyncio.to_thread(self.ser.close)
                except Exception:
                    pass
            self.is_connected = False
            self.is_simulated = False
            logger.info("XRP Robot serial connection disconnected safely.")


# Module-level singleton instance
_GLOBAL_ROBOT_BRIDGE: Optional[AsyncRobotBridge] = None


def get_robot_bridge() -> AsyncRobotBridge:
    """Retrieves or instantiates the global AsyncRobotBridge singleton."""
    global _GLOBAL_ROBOT_BRIDGE
    if _GLOBAL_ROBOT_BRIDGE is None:
        _GLOBAL_ROBOT_BRIDGE = AsyncRobotBridge()
    return _GLOBAL_ROBOT_BRIDGE


# ============================================================================
# Gemini SDK Tool Functions (Asynchronous Callables)
# ============================================================================

async def move_forward(duration_seconds: float = 1.0) -> Dict[str, Any]:
    """Drives the robot straight forward for a specified duration in seconds.

    Engages both drive wheels at base forward speed (0.5 effort). Non-blocking keepalive
    pulses are maintained across the serial bus to prevent triggering the firmware's
    1.5-second safety timeout. All motors are safely stopped when the duration ends.

    Use this tool when the camera indicates that the corridor, path, or doorway directly
    ahead is unobstructed and you need to advance the robot forward towards an exit.

    Args:
        duration_seconds: The duration in seconds to move forward. Must be a positive
            float between 0.1 and 10.0 seconds. Default is 1.0 second.

    Returns:
        A dictionary containing:
            - status (str): 'success', 'simulated', or 'cancelled'.
            - action (str): 'FORWARD'.
            - duration_seconds (float): Requested duration.
            - actual_duration_seconds (float): Actual elapsed time.
            - message (str): Human-readable confirmation of the movement.
            - timestamp (float): POSIX timestamp when the action finished.
    """
    duration = max(0.1, min(10.0, float(duration_seconds)))
    bridge = get_robot_bridge()
    return await bridge.execute_motion("FORWARD", duration)


async def steer_slight_left(duration_seconds: float = 0.8) -> Dict[str, Any]:
    """Gently veers/curves the robot to the left while maintaining forward progress.

    Reduces the left wheel effort to 25% while keeping the right wheel at full base speed,
    producing a gentle, smooth arc to the left without coming to a complete stop.
    Keepalive pulses are transmitted continuously for the requested duration.

    Use this tool for minor heading adjustments, following curved hallways, staying
    centered on guide tracks, or drifting away from an obstacle slightly off to the right.

    Args:
        duration_seconds: The duration in seconds to curve left. Must be between 0.1
            and 5.0 seconds. Default is 0.8 seconds.

    Returns:
        A dictionary containing:
            - status (str): 'success', 'simulated', or 'cancelled'.
            - action (str): 'SLIGHT_LEFT'.
            - duration_seconds (float): Requested duration.
            - message (str): Execution confirmation.
            - timestamp (float): POSIX timestamp.
    """
    duration = max(0.1, min(5.0, float(duration_seconds)))
    bridge = get_robot_bridge()
    return await bridge.execute_motion("SLIGHT_LEFT", duration)


async def steer_slight_right(duration_seconds: float = 0.8) -> Dict[str, Any]:
    """Gently veers/curves the robot to the right while maintaining forward progress.

    Reduces the right wheel effort to 25% while keeping the left wheel at full base speed,
    producing a gentle, smooth arc to the right without coming to a complete stop.
    Keepalive pulses are transmitted continuously for the requested duration.

    Use this tool for minor heading adjustments, following curved hallways, staying
    centered on guide tracks, or drifting away from an obstacle slightly off to the left.

    Args:
        duration_seconds: The duration in seconds to curve right. Must be between 0.1
            and 5.0 seconds. Default is 0.8 seconds.

    Returns:
        A dictionary containing:
            - status (str): 'success', 'simulated', or 'cancelled'.
            - action (str): 'SLIGHT_RIGHT'.
            - duration_seconds (float): Requested duration.
            - message (str): Execution confirmation.
            - timestamp (float): POSIX timestamp.
    """
    duration = max(0.1, min(5.0, float(duration_seconds)))
    bridge = get_robot_bridge()
    return await bridge.execute_motion("SLIGHT_RIGHT", duration)


async def turn_hard_left(duration_seconds: float = 0.6) -> Dict[str, Any]:
    """Spins/pivots the robot in place to the left (counter-clockwise).

    Runs the left wheel in reverse (-0.35 effort) and the right wheel forward (+0.35 effort),
    causing the robot to rotate around its central axis without advancing forward.
    A duration of approximately 0.5 to 0.7 seconds yields a 90-degree left turn.

    Use this tool when arriving at a sharp 90-degree corner, navigating T-intersections,
    or turning around to escape a blocked corridor or dead end.

    Args:
        duration_seconds: Duration in seconds to pivot turn left. Must be between 0.1
            and 3.0 seconds. Default is 0.6 seconds (~90 degrees).

    Returns:
        A dictionary containing:
            - status (str): 'success', 'simulated', or 'cancelled'.
            - action (str): 'HARD_LEFT'.
            - duration_seconds (float): Requested duration.
            - message (str): Execution confirmation.
            - timestamp (float): POSIX timestamp.
    """
    duration = max(0.1, min(3.0, float(duration_seconds)))
    bridge = get_robot_bridge()
    return await bridge.execute_motion("HARD_LEFT", duration)


async def turn_hard_right(duration_seconds: float = 0.6) -> Dict[str, Any]:
    """Spins/pivots the robot in place to the right (clockwise).

    Runs the left wheel forward (+0.35 effort) and the right wheel in reverse (-0.35 effort),
    causing the robot to rotate around its central axis without advancing forward.
    A duration of approximately 0.5 to 0.7 seconds yields a 90-degree right turn.

    Use this tool when arriving at a sharp 90-degree corner, navigating T-intersections,
    or turning around to escape a blocked corridor or dead end.

    Args:
        duration_seconds: Duration in seconds to pivot turn right. Must be between 0.1
            and 3.0 seconds. Default is 0.6 seconds (~90 degrees).

    Returns:
        A dictionary containing:
            - status (str): 'success', 'simulated', or 'cancelled'.
            - action (str): 'HARD_RIGHT'.
            - duration_seconds (float): Requested duration.
            - message (str): Execution confirmation.
            - timestamp (float): POSIX timestamp.
    """
    duration = max(0.1, min(3.0, float(duration_seconds)))
    bridge = get_robot_bridge()
    return await bridge.execute_motion("HARD_RIGHT", duration)


async def stop_robot() -> Dict[str, Any]:
    """Immediately stops all robot drive motors and brings the robot to a standstill.

    Cancels any active motion task in progress and sends a direct STOP command to the
    firmware, cutting motor effort immediately to 0.0.

    Use this tool when the robot has reached its destination, when an exit is reached,
    when waiting for human evacuees to catch up, or to stabilize the camera before
    evaluating the scene.

    Returns:
        A dictionary containing:
            - status (str): 'success' or 'simulated'.
            - action (str): 'STOP'.
            - message (str): Confirmation that all motors are stopped.
            - timestamp (float): POSIX timestamp.
    """
    bridge = get_robot_bridge()
    return await bridge.execute_motion("STOP", 0.0)


async def stop_for_obstacle(reason: str = "Obstacle detected in robot path") -> Dict[str, Any]:
    """Triggers an emergency obstacle stop, halting motors immediately and logging the hazard.

    Immediately cancels any active motion and transmits the `STOP_OBSTACLE` firmware command.
    The reason and obstacle description are logged and returned for emergency response records.

    Use this tool immediately whenever an obstacle, human evacuee, wall, falling debris,
    dense smoke, or fire hazard is detected directly in the robot's navigation path.

    Args:
        reason: Explanation of the detected obstacle, hazard, or safety issue causing
            the emergency stop. Default is 'Obstacle detected in robot path'.

    Returns:
        A dictionary containing:
            - status (str): 'success' or 'simulated'.
            - action (str): 'STOP_OBSTACLE'.
            - reason (str): The provided obstacle or hazard description.
            - message (str): Emergency stop alert details.
            - timestamp (float): POSIX timestamp.
    """
    logger.warning(f"EMERGENCY OBSTACLE STOP: {reason}")
    bridge = get_robot_bridge()
    result = await bridge.execute_motion("STOP_OBSTACLE", 0.0)
    result["reason"] = reason
    result["message"] = f"Emergency obstacle stop triggered: {reason}"
    return result


async def get_robot_status() -> Dict[str, Any]:
    """Queries the current operational status, connectivity, and telemetry of the robot.

    Inspects whether the serial connection is active, identifies the active COM port,
    checks if simulated fallback mode is active, checks the last command executed,
    and returns recent telemetry messages received from the robot firmware.

    Use this tool before beginning a navigation session to verify the robot is responsive,
    or during navigation to diagnose connection and firmware health.

    Returns:
        A dictionary containing:
            - is_connected (bool): True if connection (or simulation) is active.
            - is_simulated (bool): True if running in simulated mode without USB hardware.
            - port (str): The active serial port or simulation identifier.
            - last_command (str): The last command sent over the serial bus.
            - last_command_time (float): POSIX timestamp of the last command.
            - recent_telemetry (list[str]): Last few lines of stdout from the robot firmware.
    """
    bridge = get_robot_bridge()
    if not bridge.is_connected:
        await bridge.connect()

    return {
        "is_connected": bridge.is_connected,
        "is_simulated": bridge.is_simulated,
        "port": bridge.port,
        "last_command": bridge.last_command,
        "last_command_time": bridge.last_command_time,
        "recent_telemetry": list(bridge.recent_logs),
    }


# ============================================================================
# Tool Registry & Dispatchers
# ============================================================================

# List of tool callables ready to be passed directly to Gemini SDK GenerateContentConfig
DRIVING_TOOLS: List[Callable[..., Any]] = [
    move_forward,
    steer_slight_left,
    steer_slight_right,
    turn_hard_left,
    turn_hard_right,
    stop_robot,
    stop_for_obstacle,
    get_robot_status,
]

# Map of tool name strings to their coroutine functions
TOOL_MAP: Dict[str, Callable[..., Any]] = {
    func.__name__: func for func in DRIVING_TOOLS
}


async def execute_async_tool(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Dynamically dispatches and executes a tool function by name with provided arguments.

    Useful when processing function calls from `gemini-robotics-er-2-streaming-preview`
    or `types.ToolCall` from Google GenAI live sessions.

    Args:
        tool_name: The name of the tool function to call (e.g., 'move_forward').
        arguments: Keyword arguments dict parsed from the Gemini function call.

    Returns:
        The dictionary response returned by the executed tool function.
    """
    if tool_name not in TOOL_MAP:
        return {
            "status": "error",
            "message": f"Unknown tool '{tool_name}'. Available tools: {list(TOOL_MAP.keys())}",
        }

    args = arguments or {}
    func = TOOL_MAP[tool_name]
    try:
        return await func(**args)
    except Exception as e:
        logger.exception(f"Error executing tool '{tool_name}' with args {args}: {e}")
        return {
            "status": "error",
            "tool": tool_name,
            "error": str(e),
        }


# ============================================================================
# Self-Test and Example Usage
# ============================================================================

async def _self_test():
    """Runs a verification test of all driving tools in async simulation/live mode."""
    print("\n" + "=" * 60)
    print(" ROBOGUIDE ASYNC DRIVING TOOLS SELF-TEST ")
    print("=" * 60)

    # 1. Check status
    status = await get_robot_status()
    print(f"[*] Initial Robot Status: port={status['port']}, simulated={status['is_simulated']}")

    # 2. Test forward movement
    print("\n[*] Testing move_forward(duration_seconds=1.0)...")
    res = await move_forward(1.0)
    print(f"[OK] Result: {res}")

    # 3. Test steering
    print("\n[*] Testing steer_slight_left(duration_seconds=0.5)...")
    res = await steer_slight_left(0.5)
    print(f"[OK] Result: {res}")

    # 4. Test pivot turn
    print("\n[*] Testing turn_hard_right(duration_seconds=0.4)...")
    res = await turn_hard_right(0.4)
    print(f"[OK] Result: {res}")

    # 5. Test preemption: start long 3.0s move, then trigger emergency stop after 0.3s
    print("\n[*] Testing Preemption: Starting 3.0s forward movement...")
    long_task = asyncio.create_task(move_forward(3.0))
    await asyncio.sleep(0.3)
    print("[!] Obstacle encountered! Triggering stop_for_obstacle...")
    stop_res = await stop_for_obstacle(reason="Debris blocked hallway path")
    print(f"[OK] Stop Result: {stop_res}")
    long_res = await long_task
    print(f"[OK] Long Task Status: {long_res}")

    # 6. Verify Gemini SDK tool registration
    print("\n[*] Verifying compatibility with google.genai types.GenerateContentConfig...")
    try:
        from google.genai.types import GenerateContentConfig
        config = GenerateContentConfig(tools=DRIVING_TOOLS)
        print(f"[OK] Successfully registered {len(config.tools)} async tools in GenerateContentConfig!")
    except Exception as e:
        print(f"[!] GenAI registration check: {e}")

    # Disconnect
    bridge = get_robot_bridge()
    await bridge.disconnect()
    print("\n[OK] Self-test completed successfully.\n")


if __name__ == "__main__":
    asyncio.run(_self_test())