#!/usr/bin/env python3
"""
Roboguide Serial Controller CLI
--------------------------------
A CLI program to send navigation and motor control strings over serial
to the XRP Robot running the MicroPython firmware in src/robot/.md/firmware.py.

Supported Firmware Commands:
  - FORWARD        : Drive both wheels forward
  - SLIGHT_LEFT    : Steer slight left
  - SLIGHT_RIGHT   : Steer slight right
  - HARD_LEFT      : Pivot turn left
  - HARD_RIGHT     : Pivot turn right
  - STOP / STOP_OBSTACLE : Immediate emergency stop

Features:
  - Auto-detection of connected XRP / Raspberry Pi Pico COM ports
  - Interactive REPL with single-key shortcuts (W, A, S, D, Q, E)
  - Timed duration support (e.g., 'FORWARD 2.5' drives for 2.5s with keepalive pulses)
  - Non-blocking background listener printing robot serial output in real time
  - One-shot CLI mode (--cmd FORWARD) for scripted testing
"""

import argparse
import sys
import time
import threading
from typing import Optional, List
import serial
import serial.tools.list_ports

# Standard baud rate for USB CDC Serial communication
DEFAULT_BAUDRATE = 115200

# Firmware safety watchdog timeout is 1.5s in firmware.py
KEEPALIVE_INTERVAL = 0.5

# Recognized commands and handy CLI aliases
COMMAND_ALIASES = {
    "W": "FORWARD",
    "F": "FORWARD",
    "FORWARD": "FORWARD",
    "A": "SLIGHT_LEFT",
    "SLIGHT_LEFT": "SLIGHT_LEFT",
    "D": "SLIGHT_RIGHT",
    "SLIGHT_RIGHT": "SLIGHT_RIGHT",
    "Q": "HARD_LEFT",
    "HARD_LEFT": "HARD_LEFT",
    "E": "HARD_RIGHT",
    "HARD_RIGHT": "HARD_RIGHT",
    "S": "STOP",
    "X": "STOP",
    "STOP": "STOP",
    "STOP_OBSTACLE": "STOP_OBSTACLE",
}


def list_serial_ports() -> List[serial.tools.list_ports_common.ListPortInfo]:
    """Returns a list of all detected serial ports."""
    return list(serial.tools.list_ports.comports())


def detect_xrp_port() -> Optional[str]:
    """
    Attempt to auto-detect the XRP Robot / RP2040 Pico serial port.
    Matches SparkFun (VID 0x1B4F) or Raspberry Pi (VID 0x2E8A) or single active port.
    """
    ports = list_serial_ports()
    if not ports:
        return None

    # Known Vendor IDs: SparkFun = 0x1B4F, Raspberry Pi = 0x2E8A
    for p in ports:
        vid = p.vid
        if vid in (0x1B4F, 0x2E8A):
            return p.device

    # If only 1 port is available, use it as the default
    if len(ports) == 1:
        return ports[0].device

    return None


class RobotSerialBridge:
    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self._stop_listener = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None

    def connect(self):
        """Open the serial connection and start the background reader thread."""
        print(f"[*] Connecting to robot on {self.port} at {self.baudrate} baud...")
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        # Flush buffers
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        print(f"[✓] Connected successfully to {self.port}!\n")

        # Start background listener thread
        self._stop_listener.clear()
        self._listener_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._listener_thread.start()

    def _read_loop(self):
        """Continuously reads incoming messages from the robot's stdout."""
        while not self._stop_listener.is_set():
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        print(f"\r\033[92m[ROBOT]\033[0m {line}\n> ", end="", flush=True)
                else:
                    time.sleep(0.02)
            except Exception:
                break

    def send_command(self, cmd: str) -> bool:
        """Sends a single string line terminated with \\n to the robot."""
        if not self.ser or not self.ser.is_open:
            print("[!] Serial port is not open.")
            return False

        formatted = cmd.strip().upper()
        payload = f"{formatted}\n".encode("utf-8")
        self.ser.write(payload)
        self.ser.flush()
        print(f"[*] Sent: \033[1m{formatted}\033[0m")
        return True

    def run_timed_command(self, cmd: str, duration_sec: float):
        """
        Sends repeated keepalive commands for the given duration (sec),
        then automatically sends STOP. Essential for overcoming the 1.5s safety watchdog.
        """
        print(f"[*] Executing '{cmd}' for {duration_sec:.1f} seconds (keepalive pulse)...")
        start_time = time.time()
        try:
            while (time.time() - start_time) < duration_sec:
                self.send_command(cmd)
                time.sleep(KEEPALIVE_INTERVAL)
        except KeyboardInterrupt:
            print("\n[!] Timed command interrupted by user.")
        finally:
            self.send_command("STOP")

    def disconnect(self):
        """Stops the listener and closes the serial port safely."""
        self._stop_listener.set()
        if self.ser and self.ser.is_open:
            try:
                # Emergency stop robot before closing port
                self.ser.write(b"STOP\n")
                self.ser.flush()
            except Exception:
                pass
            self.ser.close()
        print("[*] Serial port closed.")


def print_help():
    print("""
=====================================================
            ROBOGUIDE SERIAL CONTROLLER
=====================================================
Commands:
  FORWARD       (shortcut: W or F) - Drive forward
  SLIGHT_LEFT   (shortcut: A)      - Steer slight left
  SLIGHT_RIGHT  (shortcut: D)      - Steer slight right
  HARD_LEFT     (shortcut: Q)      - Pivot turn left
  HARD_RIGHT    (shortcut: E)      - Pivot turn right
  STOP          (shortcut: S or X) - Emergency Stop

Timed Movements:
  You can specify duration in seconds after any command:
    > FORWARD 2        (Drives forward for 2.0 seconds, then stops)
    > HARD_LEFT 1.5    (Pivots left for 1.5 seconds, then stops)

Other Options:
  HELP or ?      - Show this menu
  EXIT or QUIT   - Disconnect and exit
=====================================================
""")


def interactive_session(bridge: RobotSerialBridge):
    """Run an interactive CLI session allowing direct command input."""
    print_help()
    try:
        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            parts = user_input.split()
            cmd_token = parts[0].upper()
            duration: Optional[float] = None

            if len(parts) > 1:
                try:
                    duration = float(parts[1])
                except ValueError:
                    print(f"[!] Invalid duration '{parts[1]}'. Enter a number in seconds.")
                    continue

            if cmd_token in ("EXIT", "QUIT"):
                break
            elif cmd_token in ("HELP", "?"):
                print_help()
                continue

            # Resolve shortcut or full name
            resolved_cmd = COMMAND_ALIASES.get(cmd_token, cmd_token)

            if duration is not None and duration > 0:
                bridge.run_timed_command(resolved_cmd, duration)
            else:
                bridge.send_command(resolved_cmd)

    except KeyboardInterrupt:
        print("\n[*] Exiting interactive mode...")
    finally:
        bridge.send_command("STOP")


def main():
    parser = argparse.ArgumentParser(
        description="CLI utility to send navigation strings over serial to the XRP robot firmware."
    )
    parser.add_argument(
        "-p", "--port",
        type=str,
        help="Serial port (e.g. COM4 on Windows, /dev/ttyACM0 on Linux/Mac). Auto-detects if omitted."
    )
    parser.add_argument(
        "-b", "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help=f"Baud rate (default: {DEFAULT_BAUDRATE})"
    )
    parser.add_argument(
        "-c", "--cmd",
        type=str,
        help="Send a single command or shortcut and exit (e.g., --cmd FORWARD or --cmd 'FORWARD 2.5')"
    )
    parser.add_argument(
        "-l", "--list-ports",
        action="store_true",
        help="List all detected serial ports and exit."
    )

    args = parser.parse_args()

    # List ports mode
    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports detected.")
        else:
            print("Detected serial ports:")
            for p in ports:
                hw_info = f"VID:PID={p.vid:04X}:{p.pid:04X}" if (p.vid and p.pid) else "N/A"
                print(f"  - {p.device}: {p.description} ({hw_info})")
        return

    # Determine serial port
    target_port = args.port
    if not target_port:
        target_port = detect_xrp_port()
        if target_port:
            print(f"[*] Auto-detected robot serial port: {target_port}")
        else:
            ports = list_serial_ports()
            if not ports:
                print("[!] Error: No serial ports detected! Please connect the XRP robot via USB.")
                sys.exit(1)
            print("Multiple ports found. Please specify one with -p <PORT>:")
            for i, p in enumerate(ports, 1):
                print(f"  [{i}] {p.device} - {p.description}")
            choice = input("Select port number: ").strip()
            try:
                target_port = ports[int(choice) - 1].device
            except (ValueError, IndexError):
                print("[!] Invalid selection.")
                sys.exit(1)

    bridge = RobotSerialBridge(port=target_port, baudrate=args.baudrate)

    try:
        bridge.connect()

        if args.cmd:
            parts = args.cmd.strip().split()
            cmd_token = parts[0].upper()
            duration = float(parts[1]) if len(parts) > 1 else None
            resolved_cmd = COMMAND_ALIASES.get(cmd_token, cmd_token)

            if duration and duration > 0:
                bridge.run_timed_command(resolved_cmd, duration)
            else:
                bridge.send_command(resolved_cmd)
                # Give the robot firmware a moment to respond
                time.sleep(0.3)
        else:
            interactive_session(bridge)

    except serial.SerialException as e:
        print(f"[!] Serial error: {e}")
    finally:
        bridge.disconnect()


if __name__ == "__main__":
    main()
