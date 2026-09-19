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

# Default movement effort speed (-1.0 to 1.0)
BASE_SPEED = 0.5
TURN_SPEED = 0.35

def execute_command(cmd: str):
    """
    Translates high-level steering directives into XRPLib motor efforts.
    """
    cmd = cmd.strip().upper()
    print(f"XRP Received Command: {cmd}")

    if cmd == "FORWARD":
        # Both wheels forward
        drivetrain.set_effort(BASE_SPEED, BASE_SPEED)

    elif cmd == "SLIGHT_LEFT":
        # Slow left wheel, keep right wheel moving
        drivetrain.set_effort(BASE_SPEED * 0.25, BASE_SPEED)

    elif cmd == "SLIGHT_RIGHT":
        # Keep left wheel moving, slow right wheel
        drivetrain.set_effort(BASE_SPEED, BASE_SPEED * 0.25)

    elif cmd == "HARD_LEFT":
        # Spin in place to the left
        drivetrain.set_effort(-TURN_SPEED, TURN_SPEED)

    elif cmd == "HARD_RIGHT":
        # Spin in place to the right
        drivetrain.set_effort(TURN_SPEED, -TURN_SPEED)

    elif cmd == "STOP_OBSTACLE" or cmd == "STOP":
        # Emergency stop
        drivetrain.stop()

    else:
        # Unknown command safety stop
        drivetrain.stop()


# ==========================================
# 2. Non-Blocking Serial Listener Loop
# ==========================================
print("🚀 XRP MicroPython Navigation Firmware Ready...")
drivetrain.stop()

# Safety timeout: Stop motors if no new command received in 1.5s
last_command_time = time.time()
SAFETY_TIMEOUT_SEC = 1.5

while True:
    # Check if USB stdin serial data is available (Non-blocking)
    if select.select([sys.stdin], [], [], 0.01):
        line = sys.stdin.readline().strip()
        if line:
            execute_command(line)
            last_command_time = time.time()

    # Safety watchdog check
    if time.time() - last_command_time > SAFETY_TIMEOUT_SEC:
        drivetrain.stop()

    time.sleep(0.01)