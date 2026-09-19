# Robotics Emergency Responder - Roboguide

## Overview

A robotic emergency responder that helps guide people to the nearest exit during an emergency. The system uses computer vision to detect exits and obstacles, and a Raspberry Pi to control the robot's movement. The system is designed to be used in a variety of emergency situations, such as fires, earthquakes, and other disasters.

## Hardware

- Raspberry Pi 4 Model B
- USB C Webcam
- XRP Robotics Kit

## Software

- Python 3.11
- Google Gemini API Robotics ER 2
- OpenCV
- XRPLib
- MicroPython

## System Architecture

The system is composed of three main components:

1. **Computer Vision System** - Captures images from the camera and processes them to detect exits and obstacles.
2. **LLM System** - Analyzes the processed images and determines the best path to the nearest exit.
3. **Robot Control System** - Controls the robot's movement based on the LLM's output.

## Detailed Design

### Computer Vision System

The computer vision system is responsible for capturing images from the camera and processing them to detect exits and obstacles. The system uses OpenCV to capture images from the camera and process them to detect exits and obstacles. The system is designed to be used in a variety of emergency situations, such as fires, earthquakes, and other disasters.

### LLM System

The LLM system is responsible for analyzing the processed images and determining the best path to the nearest exit. The system uses Gemini Robotics ER 2 to analyze the processed images and determine the best path to the nearest exit. The system is designed to be used in a variety of emergency situations, such as fires, earthquakes, and other disasters.

### Robot Control System

The robot control system is responsible for controlling the robot's movement based on the LLM's output. The system uses XRPLib on a Pi Pico to control the robot's movement based on the function calls output by the LLM. The system is designed to be used in a variety of emergency situations, such as fires, earthquakes, and other disasters.

### Communication Bus

The robot control system communicates with the LLM system via a USB serial connection. The LLM system sends motor commands to the robot control system via the serial connection. The robot control system uses the USB-Serial connection to update the MicroPython firmware on the Pi Pico.  The robot control system uses the USB-Serial connection to send sensor data to the LLM system.

## Usage

1. Power on the robot.
2. Ensure the camera is properly positioned.
3. The robot will automatically start scanning for exits and obstacles.
4. The robot will guide people to the nearest exit during an emergency.