# Walkthrough: Async Driving Tools for Gemini Robotics ER 2

We adapted the serial motor control from [`src/agent/serial_controller_cli.py`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/serial_controller_cli.py) into a collection of asynchronous tool functions in [`src/agent/driving_functions.py`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py), ready for use with the Gemini SDK (`google-genai`) and streaming models such as `gemini-robotics-er-2-streaming-preview`.

## Key Features

1. **Native Asynchronous (`async def`) Implementation**:
   - Uses `await asyncio.sleep()` for non-blocking keepalive pulses (0.5s interval), preventing watchdog timeouts (1.5s in [`src/robot/firmware.py`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/robot/firmware.py)) without freezing WebSocket or camera streaming pipelines.
2. **Active Motion Preemption**:
   - If an emergency stop (`stop_for_obstacle` or `stop_robot`) or new command arrives while a motion is running, the running task is cancelled instantly and motors are halted in real time.
3. **Auto-Detection & Simulation Fallback**:
   - Detects the Raspberry Pi Pico / SparkFun COM port automatically (`0x2E8A`, `0x1B4F`). If no robot is connected, falls back to simulation mode without crashing.
4. **Google-Style Docstrings & Type Annotations**:
   - All tools include explicit descriptions, arguments, ranges, and return schemas, allowing Gemini SDK's `GenerateContentConfig(tools=DRIVING_TOOLS)` to automatically generate valid OpenAPI tool declarations.

---

## Tool Functions Overview

| Tool Name | Action / Motors | Duration Range | Primary Use Case |
|---|---|---|---|
| [`move_forward`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py#L286) | Both wheels forward (0.5 effort) | 0.1s - 10.0s | Advancing through open corridors or toward exit doors. |
| [`steer_slight_left`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py#L312) | Left: 0.125, Right: 0.5 | 0.1s - 5.0s | Minor course correction, lane centering, curved hallways. |
| [`steer_slight_right`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py#L338) | Left: 0.5, Right: 0.125 | 0.1s - 5.0s | Minor course correction, lane centering, curved hallways. |
| [`turn_hard_left`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py#L364) | Left: -0.35, Right: +0.35 | 0.1s - 3.0s | 90-degree left turns at intersections or escaping dead ends. |
| [`turn_hard_right`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py#L391) | Left: +0.35, Right: -0.35 | 0.1s - 3.0s | 90-degree right turns at intersections or escaping dead ends. |
| [`stop_robot`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py#L418) | Both wheels: 0.0 | Immediate | Pausing at destination, human evacuee wait, frame capture. |
| [`stop_for_obstacle`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py#L440) | Emergency `STOP_OBSTACLE` | Immediate | Immediate safety stop when hazards/obstacles are detected. |
| [`get_robot_status`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/driving_functions.py#L471) | Queries telemetry & port | Diagnostics | Pre-flight connectivity check and telemetry inspection. |

---

## Example Usage with `gemini-robotics-er-2-streaming-preview`

### 1. Registering Tools with the Gemini Client
```python
import asyncio
from google import genai
from google.genai import types
from src.agent.driving_functions import DRIVING_TOOLS, execute_async_tool

client = genai.Client()

# Pass the list of async callables directly into the configuration
config = types.GenerateContentConfig(
    model="gemini-robotics-er-2-streaming-preview",
    tools=DRIVING_TOOLS,
    temperature=0.2,
)
```

### 2. Handling Live Streaming Tool Calls
When processing streaming WebSocket responses (e.g., using `client.aio.live.connect`):
```python
async def handle_live_session(session):
    async for response in session.receive():
        # Check for tool call directives from Gemini ER 2
        if response.tool_call:
            function_responses = []
            for call in response.tool_call.function_calls:
                # Execute asynchronously without blocking the event loop
                result = await execute_async_tool(call.name, call.args)
                function_responses.append(
                    types.FunctionResponse(
                        name=call.name,
                        id=call.id,
                        response=result,
                    )
                )

            # Send tool response back to the live session
            await session.send(
                input=types.LiveClientToolResponse(
                    function_responses=function_responses
                )
            )
```

---

## LLM Client (`src/agent/llm_client.py`)

We built [`RoboguideClient`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/llm_client.py) and [`RoboguideLiveSession`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/src/agent/llm_client.py#L74) using the Google GenAI SDK (`google-genai`).

### Key Capabilities:
- **Default Model**: Configured for `gemini-robotics-er-2-streaming-preview`.
- **Integrated Driving Tools**: Pre-configured with all 8 async driving tools from `src.agent.driving_functions`.
- **Bidirectional Live Streaming**:
  - Streams OpenCV camera frames or JPEG bytes directly to Gemini via `LiveClientRealtimeInput`.
  - Automatically receives tool calls (`tool_call.function_calls`), dispatches them asynchronously through `execute_async_tool`, and sends back `LiveClientToolResponse`.
- **One-Shot Frame Analysis**:
  - `client.analyze_frame_and_navigate(frame, prompt)` accepts a camera frame, prompts Gemini ER 2, executes the chosen tool calls, and returns structured action telemetry.
- **Custom System Instruction**:
  - Injected emergency guidance prompt instructing the model on corridor navigation, obstacle avoidance, and calm evacuee communication.

### Verification Results
```powershell
& .venv/Scripts/python.exe src/agent/llm_client.py
```
- Verified initialization with `gemini-robotics-er-2-streaming-preview`.
- Verified 8 driving tools registered.
- Validated tool schema compilation with `types.GenerateContentConfig`.
- Successfully validated import from top-level package:
  ```python
  from src.agent import RoboguideClient, DRIVING_TOOLS, DEFAULT_MODEL
  ```

---

## Hardware-Integrated Unit & Mock Tests (`tests/test_llm_client.py`)

The test suite in [`tests/test_llm_client.py`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/tests/test_llm_client.py) connects directly to the **physical XRP robot** over serial (`COM4`), executing actual motor commands without making real Gemini cloud API calls (all API responses are mocked).

### Hardware Action Flow Verified:
1. **Physical COM Port Connection**: Auto-detected and connected to XRP Robot on `COM4` at 115200 baud.
2. **Move Forward**: Mocks Gemini ER 2 proposing `move_forward(0.2s)` $\to$ Transmits `FORWARD` over `COM4` $\to$ Robot wheels drive forward $\to$ Transmits `STOP` $\to$ Returns `status: 'success'`.
3. **Emergency Obstacle Stop**: Mocks Gemini ER 2 detecting smoke/flames $\to$ Transmits `STOP_OBSTACLE` over `COM4` $\to$ Motors cut effort immediately.
4. **Pivot Turns**: Mocks Gemini ER 2 cornering $\to$ Transmits `HARD_LEFT` and `HARD_RIGHT` $\to$ Wheels rotate in opposite directions $\to$ Stops.
5. **Hallway Centering Curves**: Mocks Gemini ER 2 adjusting heading $\to$ Transmits `SLIGHT_LEFT` and `SLIGHT_RIGHT` $\to$ Differential wheel speeds execute gentle arc.
6. **Live Streaming Tool Dispatch**: Feeds mock live stream messages to `RoboguideLiveSession` $\to$ Transmits `SLIGHT_RIGHT` to physical robot $\to$ Transmits `LiveClientToolResponse` back to the live session.
7. **Safety Teardown**: Automatically cuts motor power (`STOP`) after each test and safely closes the COM port when the suite finishes.

### Test Execution:
```powershell
& .venv/Scripts/python.exe -m unittest tests/test_llm_client.py
```
**Output Summary (Smoothed ~1.0s Hardware Movements)**:
```
[INFO] - Connected successfully to XRP robot on COM4 at 115200 baud.
[INFO] - [ROBOT SENT] FORWARD
[INFO] - [ROBOT SENT] FORWARD        # Keepalive pulse at 0.5s
[INFO] - [ROBOT SENT] STOP           # Halts cleanly after 1.0s
[INFO] - [ROBOT SENT] STOP_OBSTACLE  # Emergency halt
[INFO] - [ROBOT SENT] HARD_LEFT
[INFO] - [ROBOT SENT] HARD_LEFT      # Pivot left for 0.8s
[INFO] - [ROBOT SENT] STOP
[INFO] - [ROBOT SENT] HARD_RIGHT
[INFO] - [ROBOT SENT] HARD_RIGHT     # Pivot right for 0.8s
[INFO] - [ROBOT SENT] STOP
[INFO] - [ROBOT SENT] SLIGHT_LEFT
[INFO] - [ROBOT SENT] SLIGHT_LEFT    # Steer curve left for 1.0s
[INFO] - [ROBOT SENT] STOP
[INFO] - [ROBOT SENT] SLIGHT_RIGHT
[INFO] - [ROBOT SENT] SLIGHT_RIGHT   # Steer curve right for 1.0s
[INFO] - [ROBOT SENT] STOP
[INFO] - XRP Robot serial connection disconnected safely.
----------------------------------------------------------------------
Ran 15 tests in 11.035s
OK
---

## Live Gemini API Spatial Awareness Integration Test

The test suite in [`tests/test_llm_client.py`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/tests/test_llm_client.py) includes `TestLiveGeminiSpatialAwareness`, which tests the **actual Gemini Robotics ER 2 API** (`gemini-robotics-er-2-preview`) with mocked camera frames from [`tests/test_photos/`](file:///c:/Users/Rubey/Documents/GitHub/vthacks-2026/tests/test_photos/).

### Test Methodology
1. **Mocked Camera Input**: Reads photo frames incrementally in numerical order (`photo1.jpg`, `photo2.jpg`, `photo3.jpg`, `photo4.jpg`).
2. **1-Second Inter-Turn Delay**: Waits 1.0 second between frames (`await asyncio.sleep(1.0)`) as requested.
3. **Multi-Turn Spatial Memory**: Creates a continuous chat session via `RoboguideClient.start_spatial_chat()`, maintaining full visual and conversational history.
4. **Physical Robot Actuation**: Dispatches live Gemini function calls directly to the physical XRP robot over USB-Serial (`COM4` at 115200 baud).
5. **Spatial Reasoning Verification**: Asserts that Gemini ER 2 recognizes relative changes in the scene, demonstrates awareness of its prior position, and plans subsequent movements.

### Execution Command:
```powershell
& .venv/Scripts/python.exe -m unittest tests.test_llm_client.TestLiveGeminiSpatialAwareness
```

### Verified Multi-Turn Results:
```
======================================================================
 LIVE GEMINI ROBOTICS ER 2 SPATIAL AWARENESS TEST
[*] Target Model : gemini-robotics-er-2-streaming-preview / preview
[*] Robot Port   : COM4 (simulated: False)
[*] Total Photos : 8 found in test_photos
======================================================================

--- [Turn 1/4] Sending photo1.jpg (60108 bytes) ---
[INFO] - [ROBOT SENT] FORWARD (keepalive pulses) -> STOP
[Gemini ER 2 Reasoning - Frame 1]:
"I have identified the emergency exit doors directly ahead, marked by a clear,
illuminated red 'EXIT' sign above them. The path between the desk on the right
and the trash can on the left is open. I have initiated a forward movement
of 2.0 seconds to approach the exit doors."

[*] Waiting 1.0s before sending next photo frame...

--- [Turn 2/4] Sending photo2.jpg (59474 bytes) ---
[INFO] - [ROBOT SENT] FORWARD (keepalive pulses) -> STOP
[Gemini ER 2 Reasoning - Frame 2]:
"In Frame 2, we are noticeably closer to the double exit doors. The 'EXIT' sign
is larger and positioned higher in the frame, and the doors occupy a larger
portion of the field of view. The desk on the right and the trash can on the left
are now further to the sides, confirming our forward progress. I have executed
another forward movement of 1.5 seconds to continue approaching the exit doors."

[*] Waiting 1.0s before sending next photo frame...

--- [Turn 3/4] Sending photo3.jpg (48115 bytes) ---
[INFO] - [ROBOT SENT] HARD_RIGHT (keepalive pulses) -> STOP
[Gemini ER 2 Reasoning - Frame 3]:
"In Frame 3, the scene has changed completely. We are now facing a long wall
with a closed door on the right side, indicating we have transitioned into a hallway
or a different section of the building. The wall directly in front of us is a barrier
to forward movement. To re-orient and explore the hallway or the door on the right,
I have executed a 0.8-second hard right turn."

[*] Waiting 1.0s before sending next photo frame...

--- [Turn 4/4] Sending photo4.jpg (54227 bytes) ---
[INFO] - [ROBOT SENT] FORWARD (keepalive pulses) -> STOP
[Gemini ER 2 Reasoning - Frame 4]:
"In Frame 4, the hard right turn has successfully revealed a long, clear hallway
with an 'EXIT' door and sign visible at the far end. The path directly ahead is
completely unobstructed. I have executed a forward movement of 2.0 seconds
to advance down the hallway toward the exit."

[OK] Live spatial awareness test completed successfully.
```

