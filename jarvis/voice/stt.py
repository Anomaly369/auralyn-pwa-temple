"""Speech-to-text: OpenAI Whisper (local model) + microphone capture."""
from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional

from jarvis.config import settings


class STTEngine:
    """Transcribe audio bytes or record from microphone using Whisper."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(settings.whisper_model)
        return self._model

    async def transcribe_bytes(self, audio_bytes: bytes, ext: str = "wav") -> str:
        """Transcribe raw audio bytes (wav/mp3/webm/ogg) to text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_transcribe_bytes, audio_bytes, ext)

    def _sync_transcribe_bytes(self, audio_bytes: bytes, ext: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            model = self._load_model()
            result = model.transcribe(tmp, fp16=False)
            return result.get("text", "").strip()
        except Exception as e:
            return f"Transcription error: {e}"
        finally:
            Path(tmp).unlink(missing_ok=True)

    async def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file by path."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_transcribe_file, path)

    def _sync_transcribe_file(self, path: str) -> str:
        try:
            model = self._load_model()
            result = model.transcribe(path, fp16=False)
            return result.get("text", "").strip()
        except Exception as e:
            return f"Transcription error: {e}"

    async def record_and_transcribe(
        self, duration_seconds: float = 5.0, sample_rate: int = 16000
    ) -> str:
        """Record from microphone for `duration_seconds`, then transcribe."""
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None, self._record_mic, duration_seconds, sample_rate
        )
        if audio_bytes is None:
            return "Microphone not available."
        return await self.transcribe_bytes(audio_bytes, ext="wav")

    @staticmethod
    def _record_mic(duration: float, sample_rate: int) -> Optional[bytes]:
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np

            print(f"[STT] Recording for {duration}s …")
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            print("[STT] Done recording.")

            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        except Exception as e:
            print(f"[STT][mic] {e}")
            return None
