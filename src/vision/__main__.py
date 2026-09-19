"""
Main entry point for running the vision package directly:
python -m src.vision
python src/vision
"""

import argparse
import sys

from .imaging import capture_interactive
from .directional_positioning import run_path_following_preview
from .gemini_live import DEFAULT_MODEL
from .stream_pipeline import run_vision_stream
from ..sensors.webcam_sensor import run_sensor_hud


def main():
    parser = argparse.ArgumentParser(
        description="Roboguide Computer Vision Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.vision                    # Launches camera capture GUI (default)
  python -m src.vision --app stream       # Launches WebSocket stream + Gemini Robotics ER 2 Live API
  python -m src.vision --app stream --mock-gemini  # Launches stream pipeline in offline mock mode
  python -m src.vision --app imaging      # Launches camera photo capture HUD
  python -m src.vision --app tracking     # Launches line & path tracking HUD
  python -m src.vision --app sensor       # Launches multi-sensor (camera + mic) HUD
        """,
    )
    parser.add_argument(
        "--app",
        choices=["imaging", "tracking", "sensor", "stream"],
        default="imaging",
        help="Vision application to run (default: 'imaging')",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="captures",
        help="Directory to save photos or captures (default: 'captures')",
    )
    parser.add_argument(
        "--ws-host",
        type=str,
        default="0.0.0.0",
        help="WebSocket streaming server host (default: '0.0.0.0')",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket streaming server port (default: 8765)",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini Live model ID (default: '{DEFAULT_MODEL}')",
    )
    parser.add_argument(
        "--gemini-fps",
        type=float,
        default=1.0,
        help="Rate at which frames are streamed to Gemini Live API (default: 1.0 FPS)",
    )
    parser.add_argument(
        "--mock-gemini",
        action="store_true",
        help="Force Gemini Robotics ER 2 simulation/mock mode without requiring an API key",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless mode without local OpenCV GUI window (ideal for Raspberry Pi SBC)",
    )

    args, unknown = parser.parse_known_args()

    print("=" * 60)
    print(" Roboguide Vision Suite")
    print(f" Launching Application: {args.app.upper()} (Camera {args.camera})")
    print("=" * 60)

    if args.app == "imaging":
        capture_interactive(output_dir=args.output_dir, camera_index=args.camera)
    elif args.app == "tracking":
        run_path_following_preview(camera_index=args.camera)
    elif args.app == "sensor":
        run_sensor_hud(camera_index=args.camera, output_dir=args.output_dir)
    elif args.app == "stream":
        run_vision_stream(
            camera_index=args.camera,
            ws_host=args.ws_host,
            ws_port=args.ws_port,
            gemini_model=args.gemini_model,
            gemini_fps=args.gemini_fps,
            mock_gemini=args.mock_gemini,
            enable_gui=not args.no_gui,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
