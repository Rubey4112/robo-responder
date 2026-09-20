#!/usr/bin/env python3
"""
Roboguide Live Audio Session & Voice Synthesizer
=================================================
Manages real-time audio interaction with evacuees using Google Gemini 3.8 Live
(`gemini-3.8-live`), streaming 24kHz PCM audio directly to laptop or Bluetooth
speakers via `sounddevice`. Includes offline fallback via `pyttsx3`.

Key Capabilities:
  1. Conversational Triage: Inquires about the emergency and identifies hazard type (Fire, Earthquake, Tornado).
  2. Voice Announcements: Broadcasts calming, authoritative evacuation directives.
  3. Spatial Coordination: Periodically speaks the robot's spatial progress and goals to evacuees.
  4. Non-Blocking Playback: Plays audio in worker threads without stalling camera frames or robot motors.
"""

import asyncio
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import wave

import numpy as np

# Audio hardware output
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

# Offline TTS fallback
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

# Google GenAI SDK
from google import genai
from google.genai import types

logger = logging.getLogger("RoboguideAudio")

ROBOGUIDE_LIVE_AUDIO_MODEL = "gemini-3.8-live"
DEFAULT_VOICE_NAME = "Aoede"  # Consistent persona: "Aoede", "Puck", "Charon", "Kore", "Fenrir"

ROBOGUIDE_AUDIO_SYSTEM_INSTRUCTION = """You are the reassuring, calm, and authoritative voice of Roboguide, an autonomous robotic emergency evacuation responder.
You communicate directly with humans caught in extreme building emergencies (fires, earthquakes, tornadoes) to guide them safely.

Core Directives:
1. Voice & Demeanor: Speak with a calm, composed, confident vocal presence. Never shout or sound frightened. Keep your sentences concise, direct, and loud enough to be heard clearly in noisy emergency conditions.
2. Emergency Triage: When prompted to initiate triage, announce that an emergency alert has activated and ask the user to declare whether the hazard is a fire, an earthquake, or a tornado.
3. Hazard Instructions:
   - FIRE: "Attention everyone, a fire emergency is active. Please stay calm, stay low beneath any smoke, and follow me immediately toward the emergency exit."
   - EARTHQUAKE: "Earthquake detected! Drop, cover, and hold on! Move under sturdy desks or tables immediately and protect your head and neck."
   - TORNADO: "Tornado warning in effect! Move immediately into an interior hallway away from all windows, exterior walls, and tall shelving."
4. Periodic Navigation Announcements: When provided with the robot's movement or spatial observation, announce your current intention to evacuees (e.g., "Advancing down the main hallway toward the exit doors. Please stay close together.").
"""


# ============================================================================
# Audio Hardware Player (sounddevice / PCM 24kHz)
# ============================================================================

class AudioPlayer:
    """Manages audio output via sounddevice for raw PCM 24kHz mono streams."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.is_playing = False
        self._playback_lock = asyncio.Lock()

    def play_pcm(self, pcm_bytes: bytes, wait: bool = False):
        """Plays raw 16-bit PCM little-endian audio bytes through default speaker."""
        if not HAS_SOUNDDEVICE or not pcm_bytes:
            return

        try:
            # Convert raw bytes to int16 numpy array
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            self.is_playing = True
            sd.play(audio_array, samplerate=self.sample_rate)
            if wait:
                sd.wait()
                self.is_playing = False
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            self.is_playing = False

    async def play_pcm_async(self, pcm_bytes: bytes):
        """Asynchronously plays audio bytes without blocking the event loop."""
        if not HAS_SOUNDDEVICE or not pcm_bytes:
            return

        async with self._playback_lock:
            await asyncio.to_thread(self.play_pcm, pcm_bytes, True)

    def stop(self):
        """Instantly stops any ongoing audio playback."""
        if HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass
        self.is_playing = False


# ============================================================================
# Offline TTS Engine (pyttsx3 Fallback)
# ============================================================================

class OfflineTTSPlayer:
    """Zero-network Text-to-Speech fallback using native Windows SAPI."""

    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._engine: Optional[Any] = None
        if HAS_PYTTSX3:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._engine.setProperty("volume", self.volume)
            except Exception as e:
                logger.warning(f"Could not initialize pyttsx3: {e}")

    def speak(self, text: str):
        """Synchronously speaks the given text."""
        if not self._engine or not text:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in offline TTS speak: {e}")

    async def speak_async(self, text: str):
        """Asynchronously speaks text in a worker thread."""
        await asyncio.to_thread(self.speak, text)


# ============================================================================
# Gemini 3.8 Live Audio Session
# ============================================================================

class RoboguideLiveAudioSession:
    """
    Manages real-time bidirectional audio streaming with `gemini-3.8-live`.
    Handles conversational emergency triage, evacuee guidance, and spoken status updates.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = ROBOGUIDE_LIVE_AUDIO_MODEL,
        voice_name: str = DEFAULT_VOICE_NAME,
        system_instruction: str = ROBOGUIDE_AUDIO_SYSTEM_INSTRUCTION,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if "GEMINI_API_KEY=" in line and not line.strip().startswith("#"):
                        self.api_key = line.split("=", 1)[1].strip()
                        os.environ["GEMINI_API_KEY"] = self.api_key
                        break

        self.model = model
        self.voice_name = voice_name
        self.system_instruction = system_instruction
        self.player = AudioPlayer(sample_rate=24000)
        self.offline_tts = OfflineTTSPlayer()

        self._client: Optional[genai.Client] = None
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)

        self.current_hazard: str = "fire"  # default hazard
        self.is_connected = False

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY not found for audio live session.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def generate_speech_audio(self, prompt_text: str) -> bytes:
        """
        Sends a prompt to `gemini-3.8-live` and streams back the generated 24kHz PCM audio.
        Uses a consistent prebuilt voice (e.g. Aoede) across all invocations.
        """
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=self.system_instruction)]
            ),
        )

        audio_chunks: List[bytes] = []
        try:
            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                await session.send_client_content(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prompt_text)],
                        )
                    ],
                    turn_complete=True,
                )

                async for resp in session.receive():
                    if resp.server_content and resp.server_content.model_turn:
                        for part in resp.server_content.model_turn.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                audio_chunks.append(part.inline_data.data)
                    if resp.server_content and resp.server_content.turn_complete:
                        break

            return b"".join(audio_chunks)

        except Exception as e:
            logger.warning(f"Live audio generation via {self.model} encountered an issue: {e}. Using offline voice fallback.")
            return b""

    async def speak(self, directive_text: str, wait: bool = False):
        """
        Synthesizes speech using Gemini 3.8 Live audio and plays it through the speakers.
        """
        pcm_bytes = await self.generate_speech_audio(
            f"Speak this announcement to the evacuees clearly, calmly, and authoritatively: {directive_text}"
        )
        if pcm_bytes:
            if wait:
                await self.player.play_pcm_async(pcm_bytes)
            else:
                asyncio.create_task(self.player.play_pcm_async(pcm_bytes))
        else:
            # Fallback was triggered
            if wait:
                await self.offline_tts.speak_async(directive_text)
            else:
                asyncio.create_task(self.offline_tts.speak_async(directive_text))

    def start_recording(self, duration: float = 4.0, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Starts non-blocking background microphone recording using sounddevice.
        Returns numpy array buffer being populated asynchronously by sounddevice.
        """
        if not HAS_SOUNDDEVICE:
            return None
        try:
            return sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        except Exception as e:
            logger.error(f"Failed to start sounddevice microphone recording: {e}")
            return None

    def stop_recording(self):
        """Immediately stops sounddevice recording."""
        if HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass

    async def classify_audio_samples(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[str, str]:
        """
        Takes raw int16 microphone audio samples, encodes them as in-memory WAV,
        and uses Gemini to transcribe the speech and classify the emergency hazard.

        Returns:
            Tuple[str, str]: (emergency_type, transcript)
            where emergency_type is one of: 'fire', 'earthquake', 'tornado', or 'unknown'.
        """
        if audio_samples is None or len(audio_samples) == 0:
            return "unknown", ""

        max_amp = float(np.max(np.abs(audio_samples)))
        if max_amp < 600:
            logger.info(f"Audio below vocal energy threshold (max amp: {max_amp:.0f}). Treating as silence.")
            return "unknown", ""

        # Encode int16 audio array to in-memory WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_samples.tobytes())
        wav_bytes = buf.getvalue()

        audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
        prompt = (
            "You are an emergency triage classifier for an autonomous evacuation rover. "
            "Listen carefully to this audio recording of a person answering what emergency has occurred. "
            "Classify which emergency they declared: 'fire', 'earthquake', or 'tornado'. "
            "If they spoke words related to earthquake (e.g. earthquake, quake, shaking, trembling), classify as 'earthquake'. "
            "If they spoke words related to fire (e.g. fire, burning, smoke, flames), classify as 'fire'. "
            "If they spoke words related to tornado (e.g. tornado, twister, storm, cyclone, wind), classify as 'tornado'. "
            "If the recording is silent, static, or unintelligible noise, classify as 'unknown'. "
            "Return JSON: {\"emergency\": \"fire\" | \"earthquake\" | \"tornado\" | \"unknown\", \"transcript\": \"<exact words spoken>\"}"
        )

        try:
            resp = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.5-flash",
                contents=[audio_part, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            raw_json = resp.text.strip() if resp.text else "{}"
            parsed = json.loads(raw_json)
            emergency = str(parsed.get("emergency", "unknown")).lower().strip()
            transcript = str(parsed.get("transcript", "")).strip()

            clean_t = transcript.lower()
            if emergency not in ("fire", "earthquake", "tornado"):
                if any(w in clean_t for w in ("earthquake", "quake", "shake", "shaking")):
                    emergency = "earthquake"
                elif any(w in clean_t for w in ("tornado", "storm", "twister", "cyclone", "wind")):
                    emergency = "tornado"
                elif any(w in clean_t for w in ("fire", "smoke", "flame", "burning")):
                    emergency = "fire"

            logger.info(f"Vocal triage classification: emergency='{emergency}', transcript='{transcript}'")
            return emergency, transcript
        except Exception as e:
            logger.error(f"Error classifying vocal audio with Gemini: {e}")
            return "unknown", ""

    async def ask_emergency_triage(self) -> str:
        """
        Vocalizes the triage question to the room asking what emergency has occurred.
        """
        triage_prompt = (
            "Speak to the room: Emergency alert detected. What is the emergency? "
            "Please state if there is a fire, an earthquake, or a tornado."
        )
        await self.speak(triage_prompt, wait=True)
        return "Emergency alert detected. What is the emergency? Fire, earthquake, or tornado?"

    async def process_emergency_response(self, user_text: str) -> str:
        """
        Processes human response, determines hazard type (fire, earthquake, tornado),
        and vocalizes the initial safety directive.
        """
        clean_text = user_text.lower().strip()
        if "earthquake" in clean_text or "quake" in clean_text or "shake" in clean_text or clean_text == "2":
            self.current_hazard = "earthquake"
            directive = (
                "Earthquake protocol activated! Drop, cover, and hold on! "
                "Move under a sturdy desk or table immediately and protect your head and neck."
            )
        elif "tornado" in clean_text or "storm" in clean_text or "wind" in clean_text or clean_text == "3":
            self.current_hazard = "tornado"
            directive = (
                "Tornado protocol activated! Move immediately to an interior hallway or room. "
                "Stay away from all windows, exterior walls, and heavy shelving."
            )
        else:
            # Default to fire
            self.current_hazard = "fire"
            directive = (
                "Fire emergency protocol activated! Please stay calm and follow me. "
                "Stay low beneath any smoke and move toward the nearest emergency exit doors."
            )

        # Vocalize the chosen emergency instructions
        await self.speak(directive, wait=True)
        return self.current_hazard

    async def announce_spatial_progress(self, current_action: str, spatial_reasoning: str, motion_count: int):
        """
        Every 3-4 movements, vocalizes the robot's spatial progress and goals to keep evacuees calm.
        """
        context_prompt = (
            f"You are navigating during a {self.current_hazard.upper()} emergency. "
            f"The robot has just executed movement #{motion_count}: {current_action}. "
            f"Robot spatial observation: {spatial_reasoning[:120]}. "
            f"In 1 to 2 short sentences, tell the evacuees following you where the robot is heading and keep them calm."
        )
        await self.speak(context_prompt, wait=False)


# Global singleton helper
_global_audio_session: Optional[RoboguideLiveAudioSession] = None

def get_audio_session(
    api_key: Optional[str] = None,
    model: str = ROBOGUIDE_LIVE_AUDIO_MODEL,
    voice_name: str = DEFAULT_VOICE_NAME,
) -> RoboguideLiveAudioSession:
    """Returns the singleton RoboguideLiveAudioSession instance."""
    global _global_audio_session
    if _global_audio_session is None:
        _global_audio_session = RoboguideLiveAudioSession(
            api_key=api_key,
            model=model,
            voice_name=voice_name,
        )
    return _global_audio_session


if __name__ == "__main__":
    # Standalone verification
    async def _test_audio():
        print("Testing Roboguide Live Audio with Gemini 3.8 Live...")
        session = get_audio_session()
        print(f"Target model: {session.model}")
        print("1. Vocalizing triage question...")
        await session.ask_emergency_triage()
        print("2. Processing response: 'There is a fire!'...")
        hazard = await session.process_emergency_response("fire")
        print(f"Activated hazard: {hazard}")
        print("3. Vocalizing spatial update...")
        await session.announce_spatial_progress(
            current_action="FORWARD (2.0s)",
            spatial_reasoning="Exit doors spotted directly ahead, path clear.",
            motion_count=3,
        )
        await asyncio.sleep(4.0)
        print("Audio verification complete.")

    asyncio.run(_test_audio())
