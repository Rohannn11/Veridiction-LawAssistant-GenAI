"""Step 4: Audio transcription using faster-whisper distil-large-v3.

Supports:
- File transcription (wav/mp3/m4a)
- Optional microphone capture to wav
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriberConfig:
    """Runtime configuration for faster-whisper transcription."""

    model_name: str = "distil-large-v3"
    model_dir: str = "data/models/faster-whisper"
    local_files_only: bool = False
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True


class AudioTranscriber:
    """Transcribes user audio into text for downstream legal pipeline stages."""

    def __init__(self, config: TranscriberConfig | None = None) -> None:
        self.config = config or TranscriberConfig()
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        LOGGER.info(
            "Loading faster-whisper model=%s on device=%s compute_type=%s model_dir=%s local_only=%s",
            self.config.model_name,
            device,
            compute_type,
            self.config.model_dir,
            self.config.local_files_only,
        )
        LOGGER.info("This may take 1-5 minutes on first run (downloading ~600MB model)...")
        
        try:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
            start = time.time()
            self._model = WhisperModel(
                self.config.model_name,
                device=device,
                compute_type=compute_type,
                download_root=self.config.model_dir,
                local_files_only=self.config.local_files_only,
            )
            elapsed = time.time() - start
            LOGGER.info("Model loaded successfully in %.1f seconds", elapsed)
        except KeyboardInterrupt as exc:
            raise RuntimeError(
                "Model download/loading was interrupted. Re-run and allow first download to complete (1-5 min). "
                "Do NOT press Ctrl+C."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load faster-whisper model: {exc}. "
                "Check internet/HF access or use --audio-local-files-only after a successful first download."
            ) from exc

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str | None = None,
        beam_size: int | None = None,
        vad_filter: bool | None = None,
    ) -> dict[str, Any]:
        """Transcribe a local audio file to plain text and segment metadata."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        self._ensure_model()

        language_to_use = language or self.config.language
        beam = beam_size if beam_size is not None else self.config.beam_size
        vad = vad_filter if vad_filter is not None else self.config.vad_filter

        segments_iter, info = self._model.transcribe(
            str(path),
            language=language_to_use,
            beam_size=beam,
            vad_filter=vad,
            condition_on_previous_text=True,
            word_timestamps=False,
            task="transcribe",
        )

        segments: list[dict[str, Any]] = []
        texts: list[str] = []
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue
            texts.append(text)
            segments.append(
                {
                    "id": int(segment.id),
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": text,
                    "avg_logprob": round(float(segment.avg_logprob), 4),
                    "no_speech_prob": round(float(segment.no_speech_prob), 4),
                }
            )

        transcript = " ".join(texts).strip()
        return {
            "text": transcript,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "duration": round(float(info.duration), 3),
            "segments": segments,
        }


def record_microphone_to_wav(
    output_wav: str | Path,
    duration_seconds: int,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Record microphone audio to a PCM16 WAV file for transcription."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")

    import sounddevice as sd

    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Recording microphone for %s seconds at %s Hz to %s",
        duration_seconds,
        sample_rate,
        output_path,
    )

    recording = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
    )
    sd.wait()

    audio_array = np.asarray(recording, dtype=np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())

    return output_path


def record_microphone_live_to_wav(
    output_wav: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    max_seconds: int = 30,
    silence_threshold: float = 0.01,
    silence_seconds: float = 2.0,
    enable_enter_to_stop: bool = True,
) -> Path:
    """Record live microphone input and stop via Enter, silence, or max duration."""
    import sounddevice as sd

    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    stop_event = threading.Event()
    start_time = time.time()
    silence_start: float | None = None
    blocksize = 1024

    def _wait_for_enter() -> None:
        try:
            input("Recording... press Enter to stop.\n")
            stop_event.set()
        except EOFError:
            # Non-interactive shells may not support input(); rely on max/silence stop.
            return

    def _callback(indata: np.ndarray, _frames: int, _time_info, status) -> None:
        nonlocal silence_start
        if status:
            LOGGER.warning("Microphone stream status: %s", status)

        chunk = np.asarray(indata, dtype=np.int16).copy()
        frames.append(chunk)

        # Silence-based stop to make voice capture feel real-time and hands-free.
        signal = chunk.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0

        if rms < silence_threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= silence_seconds and time.time() - start_time > 1.5:
                stop_event.set()
        else:
            silence_start = None

    if enable_enter_to_stop:
        threading.Thread(target=_wait_for_enter, daemon=True).start()

    LOGGER.info(
        "Live mic capture started (max=%ss, silence_stop=%ss, threshold=%s)",
        max_seconds,
        silence_seconds,
        silence_threshold,
    )
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        blocksize=blocksize,
        callback=_callback,
    ):
        while not stop_event.is_set():
            if max_seconds > 0 and time.time() - start_time >= max_seconds:
                stop_event.set()
                break
            sd.sleep(100)

    if not frames:
        raise RuntimeError("No microphone audio captured. Check microphone permissions/device.")

    audio_array = np.concatenate(frames, axis=0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())

    LOGGER.info("Live mic capture saved to %s", output_path)
    return output_path


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 4 audio transcription with faster-whisper")
    parser.add_argument("--model-name", type=str, default="distil-large-v3")
    parser.add_argument("--model-dir", type=str, default="data/models/faster-whisper")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Download/load model and exit")
    parser.add_argument("--audio-file", type=str, default=None, help="Path to input audio file")
    parser.add_argument("--live-mic", action="store_true", help="Capture real-time microphone input")
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=0,
        help="If > 0, record from microphone for N seconds and transcribe",
    )
    parser.add_argument("--record-out", type=str, default="data/audio_recorded.wav")
    parser.add_argument("--max-seconds", type=int, default=30)
    parser.add_argument("--silence-threshold", type=float, default=0.01)
    parser.add_argument("--silence-seconds", type=float, default=2.0)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    parser.add_argument("--json-out", type=str, default="data/transcription_result.json")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_cli().parse_args()

    transcriber = AudioTranscriber(
        config=TranscriberConfig(
            model_name=args.model_name,
            model_dir=args.model_dir,
            local_files_only=args.local_files_only,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
        )
    )

    if args.download_only:
        print(f"\n{'='*80}")
        print("STEP 4: Audio Model Warmup - Download Mode")
        print(f"{'='*80}")
        print(f"Model: {args.model_name}")
        print(f"Cache directory: {args.model_dir}")
        print(f"Local files only: {args.local_files_only}")
        print("Status: DOWNLOADING... (this may take 1-5 minutes, please wait)")
        print("Do NOT press Ctrl+C or close this window.")
        print(f"{'='*80}\n")
        
        try:
            transcriber._ensure_model()
            print(f"\n{'='*80}")
            print("✅ SUCCESS: Model downloaded and cached successfully!")
            print(f"{'='*80}\n")
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "model_name": args.model_name,
                        "model_dir": args.model_dir,
                        "local_files_only": args.local_files_only,
                    },
                    ensure_ascii=True,
                    indent=2,
                )
            )
        except Exception as exc:
            print(f"\n❌ ERROR: {exc}\n")
            raise
        return

    input_audio: Path | None = None
    if args.live_mic:
        input_audio = record_microphone_live_to_wav(
            output_wav=args.record_out,
            sample_rate=16000,
            channels=1,
            max_seconds=args.max_seconds,
            silence_threshold=args.silence_threshold,
            silence_seconds=args.silence_seconds,
            enable_enter_to_stop=True,
        )
    elif args.record_seconds > 0:
        input_audio = record_microphone_to_wav(
            output_wav=args.record_out,
            duration_seconds=args.record_seconds,
            sample_rate=16000,
            channels=1,
        )
    elif args.audio_file:
        input_audio = Path(args.audio_file)

    if input_audio is None:
        raise ValueError("Provide --live-mic, --audio-file, or --record-seconds > 0")

    result = transcriber.transcribe_file(
        audio_path=input_audio,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
    )

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
