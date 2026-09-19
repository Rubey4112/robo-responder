# Vision Streaming Subsystem

Real-time computer vision streaming and embodied reasoning architecture for **Roboguide ER-2**, integrating:
1. **Google Gemini Robotics ER 2 Streaming Preview** (`gemini-robotics-er-2-streaming-preview`) via the Gemini Live API.
2. **Asynchronous WebSocket Streaming Server** (`ws://0.0.0.0:8765`) for distributing video, sensor telemetry, and AI navigation directives.
3. **Augmented Reality HUD & Frame Decoupling Pipeline** serving 30 FPS video locally/over WebSocket while downsampling to 1 FPS for Gemini Live multimodal perception.

---

## Architecture Overview

```
                                  +--------------------------------------------------+
                                  |     Google Gemini Live API (Bidirectional WSS)   |
                                  |      Model: gemini-robotics-er-2-streaming-preview|
                                  +--------------------------------------------------+
                                              ^                         |
                               1 FPS Frames   |                         | Reasoning &
                             & Scene Prompts  |                         | Blocking Tool Calls
                                              |                         v
+------------------+             +---------------------------------------------------+
|  Camera Source   |             |           GeminiRoboticsLiveClient                |
|  (OpenCV Webcam  |  30 FPS     |  - Async bidirectional Live session               |
|  / WebcamSensor) | ----------> |  - Multimodal frame encoding (Blob/JPEG)          |
+------------------+             |  - Robotics emergency guidance prompts            |
         |                       |  - Tool call dispatch & response feedback         |
         |                       +---------------------------------------------------+
         |                                                 |
         v                                                 | Events / Directives
+------------------------------------------------------------------------------------+
|                               VisionStreamPipeline                                 |
|  - Frame rate decoupling (1 FPS for Gemini, 30 FPS for WebSockets / HUD)          |
|  - Augmented Reality HUD overlay (Gemini thoughts, exit guidance, status)          |
+------------------------------------------------------------------------------------+
         |                                                 |
         v                                                 v
+------------------------------------+        +--------------------------------------+
|       VisionWebSocketServer        |        |         Local HUD Window             |
|  (ws://0.0.0.0:8765)               |        |  (OpenCV interactive display)        |
|  - Live video stream (Base64/JPEG) |        +--------------------------------------+
|  - Realtime telemetry & AI events  |
|  - Client command injection        |
+------------------------------------+
```

---

## 1. Google Gemini Robotics ER 2 Live API

The model endpoint **`gemini-robotics-er-2-streaming-preview`** provides low-latency embodied reasoning over persistent WebSocket sessions.

### Tool Declarations (with `BLOCKING` behavior)
Robotic actions must use blocking execution to ensure physical movements are synchronized with model reasoning:

- **`navigate_robot(action, distance_cm, angle_deg, speed_percent, reason)`**:
  - `action`: `FORWARD`, `BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`
  - `distance_cm`: Movement distance (cm)
  - `angle_deg`: Turn angle (degrees)
  - `speed_percent`: Motor power percentage (10-100)
  - `reason`: Rationale derived from visual scene understanding
- **`emergency_stop(reason, hazard_identified)`**:
  - Immediately stops the XRP robot if smoke, fire, or impassable obstacles are sighted.
- **`report_evacuation_status(exit_visible, exit_direction, hazards_detected, instructions)`**:
  - Broadcasts guidance messages to human evacuees following the robot.

---

## 2. WebSocket Protocol (`ws://<host>:<port>`)

### Outgoing Messages (Server -> Client)

#### A. Video Frame (`type: "frame"`)
```json
{
  "type": "frame",
  "timestamp": 1726771000.123,
  "width": 640,
  "height": 480,
  "fps": 29.8,
  "data": "<base64_encoded_jpeg>",
  "telemetry": {
    "luminance": 128.4,
    "model": "gemini-robotics-er-2-streaming-preview",
    "gemini_status": "connected",
    "is_mock": false,
    "clients_connected": 1
  }
}
```

#### B. AI Reasoning Stream (`type: "ai_reasoning"`)
```json
{
  "type": "ai_reasoning",
  "timestamp": 1726771001.456,
  "data": {
    "text": "Exit sign identified 15 degrees right. Hallway is clear of debris."
  }
}
```

#### C. Robot Navigation Directive (`type: "robot_directive"`)
```json
{
  "type": "robot_directive",
  "timestamp": 1726771002.789,
  "data": {
    "tool_name": "navigate_robot",
    "arguments": {
      "action": "FORWARD",
      "distance_cm": 60.0,
      "speed_percent": 50.0,
      "reason": "Traversing corridor towards illuminated Exit sign"
    },
    "call_id": "call_987463"
  }
}
```

### Inbound Messages (Client -> Server)

#### Prompt Injection
```json
{
  "command": "prompt",
  "text": "Locate the nearest fire extinguisher or emergency door."
}
```

#### Save Snapshot
```json
{
  "command": "snapshot"
}
```

#### Ping / Heartbeat
```json
{
  "command": "ping",
  "timestamp": 1726771005.0
}
```

---

## 3. Command Line Interface

```bash
# Launch live streaming with camera and Gemini Robotics ER 2 Live API
python -m src.vision --app stream

# Launch in offline simulation / mock mode (no API key required)
python -m src.vision --app stream --mock-gemini

# Run headless on Raspberry Pi 4 host (no OpenCV window, WebSocket server active)
python -m src.vision --app stream --no-gui --ws-port 8765

# Custom camera index and Gemini frame rate
python -m src.vision --app stream --camera 1 --gemini-fps 1.0 --ws-port 9000
```

---

## 4. Python WebSocket Client Example

```python
import asyncio
import base64
import json
import websockets

async def watch_stream():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as ws:
        print("Connected to Roboguide Vision Stream Server!")
        async for message in ws:
            packet = json.loads(message)
            msg_type = packet.get("type")
            if msg_type == "frame":
                print(f"Received frame: {packet['width']}x{packet['height']} @ {packet['fps']} FPS")
            elif msg_type == "robot_directive":
                print(f"Robot action directive: {packet['data']}")
            elif msg_type == "ai_reasoning":
                print(f"Gemini ER 2 Thought: {packet['data']['text']}")

if __name__ == "__main__":
    asyncio.run(watch_stream())
```
