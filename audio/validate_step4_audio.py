"""Step 4 validation script for audio transcription and E2E pipeline integration."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from audio.transcriber import AudioTranscriber, TranscriberConfig, record_microphone_to_wav
from agents.langgraph_flow import VeridictionGraph

LOGGER = logging.getLogger(__name__)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Step 4 audio transcription")
    parser.add_argument("--model-name", type=str, default="distil-large-v3")
    parser.add_argument("--model-dir", type=str, default="data/models/faster-whisper")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--audio-file", type=str, default=None)
    parser.add_argument("--record-seconds", type=int, default=0)
    parser.add_argument("--record-out", type=str, default="data/audio_validation.wav")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report-out", type=str, default="data/step4_validation_report.json")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_cli().parse_args()

    input_audio: Path | None = None
    if args.record_seconds > 0:
        input_audio = record_microphone_to_wav(
            output_wav=args.record_out,
            duration_seconds=args.record_seconds,
            sample_rate=16000,
            channels=1,
        )
    elif args.audio_file:
        input_audio = Path(args.audio_file)

    if input_audio is None:
        raise ValueError("Provide --audio-file or --record-seconds > 0")
    if not input_audio.exists():
        raise FileNotFoundError(f"Audio file not found: {input_audio}")

    transcriber = AudioTranscriber(
        config=TranscriberConfig(
            model_name=args.model_name,
            model_dir=args.model_dir,
            local_files_only=args.local_files_only,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=True,
        )
    )

    start = time.perf_counter()
    transcription = transcriber.transcribe_file(
        audio_path=input_audio,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=True,
    )
    transcribe_ms = round((time.perf_counter() - start) * 1000, 2)

    transcript = transcription.get("text", "").strip()
    transcription_checks = {
        "text_non_empty": bool(transcript),
        "language_detected": bool(transcription.get("language")),
        "duration_positive": float(transcription.get("duration", 0.0)) > 0.0,
        "segments_non_empty": len(transcription.get("segments", [])) > 0,
    }

    e2e_output: dict = {}
    e2e_ms = 0.0
    if transcript:
        flow = VeridictionGraph(top_k=args.top_k)
        e2e_start = time.perf_counter()
        e2e_output = flow.run(transcript)
        e2e_ms = round((time.perf_counter() - e2e_start) * 1000, 2)

    pipeline_checks = {
        "claim_type_present": bool(e2e_output.get("claim_type")),
        "retrieved_passages_present": len(e2e_output.get("retrieved_passages", [])) > 0,
        "advisor_present": bool(e2e_output.get("advisor")),
        "safety_present": bool(e2e_output.get("safety")),
    }

    overall_pass = all(transcription_checks.values()) and all(pipeline_checks.values())

    report = {
        "step": "Step 4 Audio Transcription",
        "audio_file": str(input_audio),
        "transcription": {
            "text": transcript,
            "language": transcription.get("language"),
            "language_probability": transcription.get("language_probability"),
            "duration": transcription.get("duration"),
            "segment_count": len(transcription.get("segments", [])),
            "time_ms": transcribe_ms,
            "checks": transcription_checks,
        },
        "pipeline": {
            "time_ms": e2e_ms,
            "checks": pipeline_checks,
            "output_summary": {
                "claim_type": e2e_output.get("claim_type"),
                "urgency": e2e_output.get("urgency"),
                "confidence": e2e_output.get("confidence"),
                "retrieved_count": len(e2e_output.get("retrieved_passages", [])),
            },
        },
        "overall_status": "PASS" if overall_pass else "FAIL",
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
