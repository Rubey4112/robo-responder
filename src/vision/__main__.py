"""
Main entry point for running the vision package directly:
python -m src.vision
python src/vision
"""

import argparse
import sys

from .imaging import capture_interactive
from .directional_positioning import run_path_following_preview
from ..sensors.webcam_sensor import run_sensor_hud


def main():
    parser = argparse.ArgumentParser(
        description="Roboguide Computer Vision Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.vision                    # Launches camera capture GUI (default)
  python -m src.vision --app imaging      # Launches camera photo capture HUD
  python -m src.vision --app tracking     # Launches line & path tracking HUD
  python -m src.vision --app sensor       # Launches multi-sensor (camera + mic) HUD
        """,
    )
    parser.add_argument(
        "--app",
        choices=["imaging", "tracking", "sensor"],
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


if __name__ == "__main__":
    main()
