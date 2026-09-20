"""
Roboguide Audio Subsystem
-------------------------
Provides live bidirectional audio streaming via Gemini 3.8 Live (`gemini-3.8-live`),
real-time 24kHz PCM playback through sounddevice, and offline TTS fallback via pyttsx3.
"""

from src.audio.live_audio import (
    AudioPlayer,
    OfflineTTSPlayer,
    RoboguideLiveAudioSession,
    get_audio_session,
)

__all__ = [
    "AudioPlayer",
    "OfflineTTSPlayer",
    "RoboguideLiveAudioSession",
    "get_audio_session",
]
