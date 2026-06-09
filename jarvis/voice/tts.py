"""Text-to-speech: ElevenLabs primary, pyttsx3 fallback."""
from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path

from jarvis.config import settings


class TTSEngine:
    """Unified TTS interface.

    Priority:
    1. ElevenLabs (high quality, requires API key)
    2. pyttsx3    (offline, no API key)
    """

    async def speak(self, text: str) -> bytes | None:
        """Return raw audio bytes (mp3 or wav), or None if unavailable."""
        if settings.tts_provider == "elevenlabs" and settings.elevenlabs_api_key:
            return await self._elevenlabs(text)
        return await self._pyttsx3(text)

    async def speak_and_save(self, text: str, path: str) -> str:
        """Save audio to path and return the path."""
        audio = await self.speak(text)
        if audio is None:
            return ""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(audio)
        return str(p)

    # ── ElevenLabs ─────────────────────────────────────────────────────────

    async def _elevenlabs(self, text: str) -> bytes | None:
        try:
            from elevenlabs import VoiceSettings
            from elevenlabs.client import AsyncElevenLabs

            client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
            audio_gen = await client.text_to_speech.convert(
                text=text,
                voice_id=settings.elevenlabs_voice_id,
                model_id="eleven_turbo_v2",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.8,
                    style=0.0,
                    use_speaker_boost=True,
                ),
                output_format="mp3_44100_128",
            )
            # Collect async generator
            chunks = []
            async for chunk in audio_gen:
                chunks.append(chunk)
            return b"".join(chunks)
        except Exception as e:
            print(f"[TTS][ElevenLabs] {e} — falling back to pyttsx3")
            return await self._pyttsx3(text)

    # ── pyttsx3 (offline) ──────────────────────────────────────────────────

    async def _pyttsx3(self, text: str) -> bytes | None:
        try:
            import pyttsx3
            import tempfile
            import wave

            loop = asyncio.get_event_loop()

            def _synth():
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.setProperty("volume", 1.0)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp = f.name
                engine.save_to_file(text, tmp)
                engine.runAndWait()
                data = Path(tmp).read_bytes()
                Path(tmp).unlink(missing_ok=True)
                return data

            return await loop.run_in_executor(None, _synth)
        except Exception as e:
            print(f"[TTS][pyttsx3] {e}")
            return None
