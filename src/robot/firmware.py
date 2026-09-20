import sys
import select
import time
from machine import UART, Pin
from XRPLib.defaults import *

# ==========================================
# 1. Initialize XRP Motors & Setup
# ==========================================
# Get default differential drive instance from XRPLib
drivetrain = DifferentialDrive.get_default_differential_drive()

# Optional: Set up physical UART pins if not using USB-Serial
# uart = UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1))

# Default movement speed in Centimeters per Second (cm/s)
BASE_SPEED = 30.0  # cm/s
TURN_SPEED = 20.0  # cm/s

def execute_command(cmd: str):
    """
    Translates high-level steering directives into XRPLib closed-loop motor speeds (cm/s).
    """
    cmd = cmd.strip().upper()
    print(f"XRP Received Command: {cmd}")

    if cmd == "FORWARD":
        # Both wheels forward at 15 cm/s
        drivetrain.set_speed(BASE_SPEED, BASE_SPEED)

    elif cmd == "SLIGHT_LEFT":
        # Slow left wheel, keep right wheel moving
        drivetrain.set_speed(BASE_SPEED * 0.25, BASE_SPEED)

    elif cmd == "SLIGHT_RIGHT":
        # Keep left wheel moving, slow right wheel
        drivetrain.set_speed(BASE_SPEED, BASE_SPEED * 0.25)

    elif cmd == "HARD_LEFT":
        # Spin in place to the left
        drivetrain.set_speed(-TURN_SPEED, TURN_SPEED)

    elif cmd == "HARD_RIGHT":
        # Spin in place to the right
        drivetrain.set_speed(TURN_SPEED, -TURN_SPEED)

    elif cmd == "STOP_OBSTACLE" or cmd == "STOP":
        # Emergency stop
        drivetrain.stop()

    else:
        # Unknown command safety stop
        drivetrain.stop()


# ==========================================
# 2. Non-Blocking Serial Listener Loop
# ==========================================
print("[XRP] MicroPython Navigation Firmware Ready...")
drivetrain.stop()

# Turn RGB LED Blue (R=0, G=0, B=255) to indicate firmware is running
try:
    board.set_rgb_led(0, 0, 255)
except Exception:
    board.led_on()  # Fallback to monochrome LED if board has no NeoPixel

# Use select.poll() on raw stdin.buffer to prevent Unicode decoding crashes from serial noise
poll_obj = select.poll()
poll_obj.register(sys.stdin.buffer, select.POLLIN)

# Safety timeout: Stop motors if no new command received in 1500ms (1.5s)
SAFETY_TIMEOUT_MS = 1500
last_command_time = time.ticks_ms()

while True:
    # Check if USB stdin serial data is available (Non-blocking, 0ms timeout)
    if poll_obj.poll(0):
        try:
            raw_bytes = sys.stdin.buffer.readline()
            if raw_bytes:
                try:
                    line = raw_bytes.decode("utf-8").strip()
                except UnicodeError:
                    # Fallback: extract printable ASCII characters if noise occurs
                    line = "".join(chr(b) for b in raw_bytes if 32 <= b < 127).strip()

                if line:
                    execute_command(line)
                    last_command_time = time.ticks_ms()
        except (UnicodeError, ValueError, Exception):
            pass

    # Safety watchdog check (time.ticks_diff handles tick rollover safely)
    if time.ticks_diff(time.ticks_ms(), last_command_time) > SAFETY_TIMEOUT_MS:
        drivetrain.stop()

    time.sleep_ms(10)