"""Step 6: Gradio system interface for end-to-end testing.

Flow:
- Text or microphone/upload audio input
- Step 4 transcription (when audio provided)
- Steps 1-3 legal pipeline (classifier + retriever + advisor + safety)
- Step 5 TTS output generation
"""

from __future__ import annotations

import json
import os
import socket
import time
import traceback
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

from agents.langgraph_flow import VeridictionGraph
from audio.transcriber import AudioTranscriber, TranscriberConfig
from tts.speak import TTSConfig, TTSGenerator


APP_CSS = """
:root {
  --bg-soft: #f5f7fb;
  --card-bg: #ffffff;
  --ink: #12203a;
  --muted: #5f6f8a;
  --brand: #0b66ff;
  --brand-2: #0c8f7a;
  --accent: #f3f8ff;
  --border: #d9e2f2;
}

body {
  background: radial-gradient(circle at 0% 0%, #eef5ff 0%, #f8fbff 40%, #f5f7fb 100%);
}

#app-shell {
  max-width: 1380px;
  margin: 0 auto;
}

.hero {
  border: 1px solid var(--border);
  background: linear-gradient(110deg, #0b66ff 0%, #0c8f7a 100%);
  color: #ffffff;
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 14px;
  box-shadow: 0 10px 26px rgba(8, 33, 74, 0.14);
}

.hero h1 {
  margin: 0;
  font-size: 28px;
  line-height: 1.2;
}

.hero p {
  margin: 10px 0 0 0;
  opacity: 0.95;
  font-size: 14px;
}

.card {
  border: 1px solid var(--border);
  background: var(--card-bg);
  border-radius: 14px;
  padding: 10px;
  box-shadow: 0 6px 16px rgba(17, 39, 83, 0.06);
}

.section-title {
  color: var(--ink);
  font-size: 16px;
  font-weight: 700;
  margin: 4px 2px 10px 2px;
}

.subtle {
  color: var(--muted);
  font-size: 13px;
}

.btn-primary {
  background: linear-gradient(90deg, #0b66ff 0%, #004ec9 100%) !important;
  color: #ffffff !important;
  border: none !important;
}

.btn-secondary {
  background: #ffffff !important;
  color: #0b66ff !important;
  border: 1px solid #0b66ff !important;
}

.badge-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(120px, 1fr));
  gap: 8px;
}

.badge-item {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 10px;
  background: var(--accent);
  color: var(--ink);
  font-size: 13px;
}

.flowchart-wrap {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: #fbfdff;
    padding: 10px;
}

.flowchart-title {
    color: var(--ink);
    font-weight: 700;
    margin-bottom: 8px;
}

.flowchart-note {
    color: var(--muted);
    font-size: 12px;
    margin-top: 8px;
}

.mermaid svg {
    max-width: 100%;
    height: auto;
}
"""


class AppServices:
    """Lazy-loaded services for UI actions."""

    def __init__(self) -> None:
        self._flow: VeridictionGraph | None = None
        self._flow_provider: str = "auto"
        self._transcriber_cache: dict[tuple[str, str, bool], AudioTranscriber] = {}
        self._tts_cache: dict[tuple[str, str], TTSGenerator] = {}

    def get_flow(self, top_k: int, advisor_provider: str) -> VeridictionGraph:
        if self._flow is None or self._flow_provider != advisor_provider:
            self._flow = VeridictionGraph(top_k=top_k, advisor_provider=advisor_provider)
            self._flow_provider = advisor_provider
        self._flow.top_k = top_k
        return self._flow

    def get_transcriber(self, model_name: str, model_dir: str, local_only: bool) -> AudioTranscriber:
        key = (model_name, model_dir, local_only)
        if key not in self._transcriber_cache:
            self._transcriber_cache[key] = AudioTranscriber(
                config=TranscriberConfig(
                    model_name=model_name,
                    model_dir=model_dir,
                    local_files_only=local_only,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                )
            )
        return self._transcriber_cache[key]

    def get_tts(self, engine: str, fallback_engine: str) -> TTSGenerator:
        key = (engine, fallback_engine)
        if key not in self._tts_cache:
            self._tts_cache[key] = TTSGenerator(
                config=TTSConfig(
                    preferred_engine=engine,
                    fallback_engine=fallback_engine,
                    output_dir="data/tts",
                )
            )
        return self._tts_cache[key]


SERVICES = AppServices()


def _format_passages_for_table(passages: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for idx, item in enumerate(passages, start=1):
        score = float(item.get("score") or 0.0)
        text = str(item.get("passage", "")).strip().replace("\n", " ")
        preview = text[:260].rstrip() + (" ..." if len(text) > 260 else "")
        source = str(item.get("metadata", {}).get("dataset", ""))
        rows.append([idx, round(score, 4), source, preview])
    return rows


def _ui_confidence(output: dict[str, Any]) -> float:
    classifier_conf = float(output.get("confidence", 0.0) or 0.0)
    passages = output.get("retrieved_passages", []) or []
    retrieval_top = max((float(item.get("score", 0.0) or 0.0) for item in passages), default=0.0)
    return round(max(classifier_conf, retrieval_top), 4)


def _build_structured_json(output: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    safety = output.get("safety", {}) or {}
    passages = output.get("retrieved_passages", []) or []
    structured_response = output.get("structured_response", {}) or {}
    raw_classifier_confidence = float(output.get("confidence", 0.0) or 0.0)
    display_confidence = _ui_confidence(output)

    structured = {
        "meta": {
            "input_mode": output.get("input_mode", "text"),
            "latency_ms": round(elapsed_ms, 2),
            "top_k_retrieved": len(passages),
        },
        "input": {
            "transcript_or_query": output.get("transcript", ""),
            "audio_metadata": output.get("audio_metadata", {}),
        },
        "classification": {
            "claim_type": output.get("claim_type", "other"),
            "normalized_query": output.get("normalized_query", ""),
            "lemmatized_query": output.get("lemmatized_query", ""),
            "named_entities": output.get("named_entities", {}),
            "urgency": output.get("urgency", "low"),
            "confidence": display_confidence,
            "classifier_confidence_raw": raw_classifier_confidence,
        },
        "retrieval": {
            "passages": passages,
        },
        "structured_response": structured_response,
        "safety": {
            "risk_flags": safety.get("risk_flags", []),
            "safe_next_steps": safety.get("safe_next_steps", []),
            "disclaimer": safety.get("disclaimer", ""),
        },
        "tts": output.get("tts", {}),
        "tts_summary": output.get("tts_summary", ""),
        "final_text": output.get("final_text", ""),
    }
    return structured


def _section_to_text(title: str, lines: list[str]) -> str:
    if not lines:
        return f"{title}:\n- Not available"
    out = [f"{title}:"]
    for item in lines:
        out.append(f"- {item}")
    return "\n".join(out)


def _extract_structured_sections(structured: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
    scenario = structured.get("case_scenario", {}) or {}
    steps = structured.get("possible_steps", {}) or {}
    docs = structured.get("required_documentation", {}) or {}
    filing = structured.get("courts_and_filing_process", {}) or {}
    severity = structured.get("severity_assessment", {}) or {}
    helplines = structured.get("helplines_india", []) or []
    flowchart = structured.get("flowchart", []) or []

    scenario_lines = [scenario.get("summary", "")] + list(scenario.get("key_facts", []))
    if scenario.get("missing_details"):
        scenario_lines.append("Missing details: " + "; ".join(scenario.get("missing_details", [])))

    step_lines = (
        [f"Immediate: {x}" for x in steps.get("immediate_actions", [])]
        + [f"Legal: {x}" for x in steps.get("legal_actions", [])]
        + [f"Next 48 Hours: {x}" for x in steps.get("next_48_hours", [])]
    )

    doc_lines = (
        [f"Mandatory: {x}" for x in docs.get("mandatory", [])]
        + [f"Supporting: {x}" for x in docs.get("supporting", [])]
        + [f"Optional: {x}" for x in docs.get("optional", [])]
    )

    court_lines = (
        [f"State: {filing.get('state', '')}"]
        + [f"Forum: {x}" for x in filing.get("courts_forum", [])]
        + [f"Process: {x}" for x in filing.get("application_process", [])]
        + [f"Jurisdiction: {filing.get('jurisdiction_note', '')}"]
    )

    severity_lines = [
        f"Level: {severity.get('level', '')}",
        f"Rationale: {severity.get('rationale', '')}",
        f"Time Sensitivity: {severity.get('time_sensitivity', '')}",
    ]

    helpline_lines = [
        f"{h.get('name', '')}: {h.get('number', '')} | {h.get('availability', '')} | {h.get('applicability', '')}"
        for h in helplines
    ]

    flow_lines = [f"Step {s.get('step', '')}: {s.get('title', '')} - {s.get('details', '')}" for s in flowchart]

    return (
        _section_to_text("Case Scenario", [x for x in scenario_lines if x]),
        _section_to_text("Possible Steps", [x for x in step_lines if x]),
        _section_to_text("Required Documentation", [x for x in doc_lines if x]),
        _section_to_text("Courts and Filing Process", [x for x in court_lines if x]),
        _section_to_text("Severity", [x for x in severity_lines if x]),
        _section_to_text("India Helplines", [x for x in helpline_lines if x]),
        _section_to_text("Flowchart", [x for x in flow_lines if x]),
        str(structured.get("tts_summary", "")),
    )


def _clean_node_label(text: str, fallback: str) -> str:
    cleaned = " ".join((text or "").replace("\n", " ").split())
    if not cleaned:
        cleaned = fallback
    cleaned = cleaned.replace('"', "'")
    return cleaned[:90]


def _build_mermaid_flowchart(structured: dict[str, Any], risk_flags: list[str]) -> str:
    scenario = structured.get("case_scenario", {}) or {}
    steps = structured.get("possible_steps", {}) or {}
    docs = structured.get("required_documentation", {}) or {}
    filing = structured.get("courts_and_filing_process", {}) or {}
    severity = structured.get("severity_assessment", {}) or {}
    helplines = structured.get("helplines_india", []) or []

    immediate = list(steps.get("immediate_actions", []))
    legal = list(steps.get("legal_actions", []))
    mandatory_docs = list(docs.get("mandatory", []))
    forum = list(filing.get("courts_forum", []))
    process = list(filing.get("application_process", []))

    issue_label = _clean_node_label(scenario.get("summary", ""), "Case intake")
    danger_yes = (
        "Call 112 and move to safe location"
        if "immediate_danger" in (risk_flags or [])
        else "Use helpline and local police support"
    )
    first_action = _clean_node_label(immediate[0] if immediate else "Record incident details", "Record incident details")
    evidence_step = _clean_node_label(mandatory_docs[0] if mandatory_docs else "Collect key evidence", "Collect key evidence")
    legal_step = _clean_node_label(legal[0] if legal else "Prepare legal complaint", "Prepare legal complaint")
    forum_step = _clean_node_label(forum[0] if forum else "Approach competent forum", "Approach competent forum")
    filing_step = _clean_node_label(process[0] if process else "File petition or complaint", "File petition or complaint")
    follow_up = _clean_node_label(process[1] if len(process) > 1 else "Track next hearing and orders", "Track next hearing and orders")
    severity_label = _clean_node_label(
        f"Severity: {severity.get('level', 'medium')} | {severity.get('time_sensitivity', 'Act within 24-48 hours')}",
        "Severity: medium",
    )

    helpline_text = " / ".join(
        [str(item.get("number", "")).strip() for item in helplines[:2] if str(item.get("number", "")).strip()]
    ) or "181 / 112"

    return "\n".join(
        [
            "flowchart TD",
            f'A["Issue Reported: {issue_label}"] --> B{{"Immediate danger?"}}',
            f'C["{danger_yes}\\nHelplines: {helpline_text}"] --> D["{first_action}"]',
            f'B -->|Yes| C',
            f'B -->|No| D["{first_action}"]',
            f'D --> E["{evidence_step}"]',
            f'E --> F["{legal_step}"]',
            f'F --> G["{forum_step}"]',
            f'G --> H["{filing_step}"]',
            f'H --> I["{follow_up}"]',
            f'I --> J["{severity_label}"]',
            "classDef emergency fill:#fff2f2,stroke:#d62828,stroke-width:2px,color:#8a1c1c;",
            "classDef process fill:#eef6ff,stroke:#1d5fd0,stroke-width:1.5px,color:#0f2f6b;",
            "classDef endpoint fill:#eefcf7,stroke:#168f6b,stroke-width:1.5px,color:#0a523c;",
            "class B,C emergency;",
            "class D,E,F,G,H,I process;",
            "class J endpoint;",
        ]
    )


def _render_mermaid_html(mermaid_source: str) -> str:
    graph_id = f"flowchart-{int(time.time() * 1000)}"
    escaped = escape(mermaid_source)
    return f"""
<div class='flowchart-wrap'>
  <div class='flowchart-title'>Guidance Flowchart</div>
  <div id='{graph_id}' class='mermaid'>{escaped}</div>
  <div class='flowchart-note'>If the graph does not render, copy the Mermaid source into mermaid.live.</div>
</div>
<script src='https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js'></script>
<script>
(() => {{
  if (!window.mermaid) return;
  window.mermaid.initialize({{ startOnLoad: false, securityLevel: 'loose', theme: 'default' }});
  const node = document.getElementById('{graph_id}');
  if (node) {{
    window.mermaid.run({{ nodes: [node] }});
  }}
}})();
</script>
"""


def _normalize_audio_mode(input_mode: str, query_text: str, audio_file: str | None) -> tuple[bool, str]:
    mode = (input_mode or "Auto").strip().lower()
    query = (query_text or "").strip()
    has_audio = bool(audio_file)

    if mode == "audio":
        return True, query
    if mode == "text":
        return False, query

    # Auto mode:
    use_audio = has_audio and not query
    if has_audio and query:
        use_audio = True
    return use_audio, query


def run_end_to_end(
    input_mode: str,
    query_text: str,
    audio_file: str | None,
    top_k: int,
    advisor_provider: str,
    stt_model_name: str,
    stt_model_dir: str,
    stt_local_files_only: bool,
    enable_tts: bool,
    tts_engine: str,
    tts_fallback_engine: str,
) -> tuple[Any, ...]:
    """Main UI callback for complete Veridiction flow."""
    start = time.perf_counter()
    try:
        use_audio, normalized_query = _normalize_audio_mode(input_mode, query_text, audio_file)

        transcript = ""
        audio_meta: dict[str, Any] = {}

        if use_audio:
            if not audio_file:
                raise ValueError("Audio mode selected but no audio input received.")
            transcriber = SERVICES.get_transcriber(
                model_name=stt_model_name,
                model_dir=stt_model_dir,
                local_only=stt_local_files_only,
            )
            transcription = transcriber.transcribe_file(
                audio_path=audio_file,
                language="en",
                beam_size=5,
                vad_filter=True,
            )
            transcript = str(transcription.get("text", "")).strip()
            audio_meta = {
                "audio_file": audio_file,
                "language": transcription.get("language", "en"),
                "language_probability": transcription.get("language_probability", 0.0),
                "duration": transcription.get("duration", 0.0),
                "segments": len(transcription.get("segments", [])),
            }
            if not transcript:
                raise ValueError("Transcription is empty. Try clearer speech and lower background noise.")
        else:
            transcript = normalized_query

        if not transcript:
            raise ValueError("Provide either text query or audio input.")

        flow = SERVICES.get_flow(top_k=top_k, advisor_provider=advisor_provider)
        output = flow.run(transcript)
        output["input_mode"] = "audio" if use_audio else "text"
        output["transcript"] = transcript
        if audio_meta:
            output["audio_metadata"] = audio_meta

        tts_path: str | None = None
        if enable_tts:
            tts = SERVICES.get_tts(tts_engine, tts_fallback_engine)
            tts_spoken_text = str(output.get("tts_summary", "")).strip() or str(output.get("final_text", ""))
            tts_result = tts.speak_to_file(
                text=tts_spoken_text,
                include_disclaimer=False,
            )
            tts_path = tts_result["audio_path"]
            output["tts"] = {
                "engine": tts_result["engine"],
                "audio_path": tts_result["audio_path"],
                "mime_type": tts_result["mime_type"],
                "size_bytes": tts_result["size_bytes"],
                "spoken_text": tts_spoken_text,
            }

        elapsed_ms = (time.perf_counter() - start) * 1000
        passages = output.get("retrieved_passages", []) or []
        passage_table = _format_passages_for_table(passages)
        structured = _build_structured_json(output, elapsed_ms)
        structured_response = output.get("structured_response", {}) or {}
        (
            case_scenario_text,
            possible_steps_text,
            required_docs_text,
            courts_process_text,
            severity_text,
            helplines_text,
            flowchart_text,
            tts_summary_text,
        ) = _extract_structured_sections(structured_response)
        safety = output.get("safety", {}) or {}
        mermaid_source = _build_mermaid_flowchart(
            structured=structured_response,
            risk_flags=list(safety.get("risk_flags", [])),
        )
        flowchart_html = _render_mermaid_html(mermaid_source)

        status = (
            "Run completed successfully. "
            f"Mode={output.get('input_mode', 'text')} | "
            f"Claim={output.get('claim_type', 'other')} | "
            f"Latency={elapsed_ms:.1f} ms"
        )

        return (
            transcript,
            str(output.get("claim_type", "other")),
            str(output.get("urgency", "low")),
            _ui_confidence(output),
            str(output.get("final_text", "")),
            case_scenario_text,
            possible_steps_text,
            required_docs_text,
            courts_process_text,
            severity_text,
            helplines_text,
            flowchart_text,
            mermaid_source,
            tts_summary_text,
            json.dumps(output.get("safety", {}), ensure_ascii=True, indent=2),
            passage_table,
            json.dumps(structured, ensure_ascii=True, indent=2),
            json.dumps(output, ensure_ascii=True, indent=2),
            tts_path,
            flowchart_html,
            status,
        )
    except Exception as exc:  # pragma: no cover
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        return (
            "",
            "error",
            "error",
            0.0,
            err,
            err,
            err,
            err,
            err,
            err,
            err,
            err,
            "",
            err,
            err,
            [],
            json.dumps({"error": str(exc)}, ensure_ascii=True, indent=2),
            err,
            None,
            "",
            "Run failed. See error details in output panes.",
        )


def clear_outputs() -> tuple[Any, ...]:
    return (
        "",
        "",
        "",
        0.0,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "{}",
        [],
        "{}",
        "{}",
        None,
        "",
        "Cleared.",
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Veridiction Law Assistant - End-to-End MVP") as demo:
        gr.HTML(
            """
            <div id='app-shell'>
              <div class='hero'>
                <h1>Veridiction Law Assistant</h1>
                <p>System Interface: Voice/Text Input -> Legal Pipeline -> Structured Response -> TTS Playback</p>
              </div>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML("<div class='section-title'>[I/O] Input Console</div>")
                    input_mode = gr.Radio(
                        choices=["Auto", "Text", "Audio"],
                        value="Auto",
                        label="Input Mode",
                    )
                    query_input = gr.Textbox(
                        label="Text Query",
                        lines=4,
                        placeholder="Example: My employer has not paid my salary for 3 months.",
                    )
                    audio_input = gr.Audio(
                        label="Voice Input (Microphone or Upload)",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )

                with gr.Accordion("[Settings] Advanced Controls", open=False):
                    with gr.Group(elem_classes=["card"]):
                        top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Retriever Top-K")
                        advisor_provider = gr.Dropdown(
                            choices=["auto", "grok", "fallback"],
                            value="grok",
                            label="Advisor Provider",
                        )
                        stt_model_name = gr.Textbox(value="distil-large-v3", label="STT Model Name")
                        stt_model_dir = gr.Textbox(value="data/models/faster-whisper", label="STT Model Cache Directory")
                        stt_local_files_only = gr.Checkbox(value=True, label="STT Local Files Only")
                        enable_tts = gr.Checkbox(value=True, label="Enable TTS")
                        tts_engine = gr.Dropdown(
                            choices=["edge_tts", "pyttsx3"],
                            value="edge_tts",
                            label="TTS Engine",
                        )
                        tts_fallback_engine = gr.Dropdown(
                            choices=["pyttsx3", "edge_tts"],
                            value="pyttsx3",
                            label="TTS Fallback Engine",
                        )

                with gr.Row():
                    run_btn = gr.Button("Run End-to-End", variant="primary", elem_classes=["btn-primary"])
                    clear_btn = gr.Button("Clear Outputs", variant="secondary", elem_classes=["btn-secondary"])

                gr.Examples(
                    examples=[
                        ["Auto", "My employer has not paid my salary for 3 months", None, 5, "grok", "distil-large-v3", "data/models/faster-whisper", True, True, "edge_tts", "pyttsx3"],
                        ["Auto", "My landlord is illegally evicting me without proper notice", None, 5, "grok", "distil-large-v3", "data/models/faster-whisper", True, True, "edge_tts", "pyttsx3"],
                        ["Auto", "Police arrested me without proper FIR or charges", None, 5, "grok", "distil-large-v3", "data/models/faster-whisper", True, True, "edge_tts", "pyttsx3"],
                    ],
                    inputs=[
                        input_mode,
                        query_input,
                        audio_input,
                        top_k,
                        advisor_provider,
                        stt_model_name,
                        stt_model_dir,
                        stt_local_files_only,
                        enable_tts,
                        tts_engine,
                        tts_fallback_engine,
                    ],
                    label="Quick Examples",
                )

            with gr.Column(scale=8):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML("<div class='section-title'>[Status] Run Summary</div>")
                    status_out = gr.Textbox(label="Execution Status", lines=2)
                    with gr.Row():
                        claim_type_out = gr.Textbox(label="Claim Type")
                        urgency_out = gr.Textbox(label="Urgency")
                        confidence_out = gr.Number(label="Confidence")

                with gr.Tab("Overview"):
                    with gr.Group(elem_classes=["card"]):
                        transcript_out = gr.Textbox(label="Transcript / Final Query", lines=4)
                        final_text_out = gr.Textbox(label="Final Advisor Text", lines=12)

                with gr.Tab("Legal Sections"):
                    with gr.Group(elem_classes=["card"]):
                        case_scenario_out = gr.Textbox(label="Case Scenario", lines=8)
                        possible_steps_out = gr.Textbox(label="Possible Steps", lines=10)
                        required_docs_out = gr.Textbox(label="Required Documentation", lines=10)
                        courts_process_out = gr.Textbox(label="Courts and Filing Process (Maharashtra)", lines=10)
                        severity_out = gr.Textbox(label="Severity Assessment", lines=6)
                        helplines_out = gr.Textbox(label="India Helplines", lines=8)
                        flowchart_out = gr.Textbox(label="Flowchart (Text Steps)", lines=8)
                        flowchart_mermaid_out = gr.Code(label="Flowchart Mermaid Source", language="markdown")

                with gr.Tab("Flowchart Graph"):
                    with gr.Group(elem_classes=["card"]):
                        flowchart_graph_out = gr.HTML(label="Guidance Flowchart")

                with gr.Tab("Safety"):
                    with gr.Group(elem_classes=["card"]):
                        safety_json_out = gr.Code(label="Safety JSON", language="json")

                with gr.Tab("Retrieval"):
                    with gr.Group(elem_classes=["card"]):
                        passages_table_out = gr.Dataframe(
                            headers=["Rank", "Score", "Source", "Passage Preview"],
                            datatype=["number", "number", "str", "str"],
                            wrap=True,
                            label="Retrieved Passages",
                        )

                with gr.Tab("Structured JSON"):
                    with gr.Group(elem_classes=["card"]):
                        structured_json_out = gr.Code(label="Structured Response JSON", language="json")
                        raw_json_out = gr.Code(label="Raw Pipeline JSON", language="json")

                with gr.Tab("Audio Output"):
                    with gr.Group(elem_classes=["card"]):
                        tts_summary_out = gr.Textbox(label="TTS Summary (Spoken)", lines=4)
                        tts_audio_out = gr.Audio(label="TTS Playback", type="filepath", interactive=False)
                        gr.HTML(
                            """
                            <div class='subtle'>
                              TTS is generated from a concise <b>tts_summary</b> for faster synthesis.
                            </div>
                            """
                        )

        run_btn.click(
            fn=run_end_to_end,
            inputs=[
                input_mode,
                query_input,
                audio_input,
                top_k,
                advisor_provider,
                stt_model_name,
                stt_model_dir,
                stt_local_files_only,
                enable_tts,
                tts_engine,
                tts_fallback_engine,
            ],
            outputs=[
                transcript_out,
                claim_type_out,
                urgency_out,
                confidence_out,
                final_text_out,
                case_scenario_out,
                possible_steps_out,
                required_docs_out,
                courts_process_out,
                severity_out,
                helplines_out,
                flowchart_out,
                flowchart_mermaid_out,
                tts_summary_out,
                safety_json_out,
                passages_table_out,
                structured_json_out,
                raw_json_out,
                tts_audio_out,
                flowchart_graph_out,
                status_out,
            ],
        )

        clear_btn.click(
            fn=clear_outputs,
            outputs=[
                transcript_out,
                claim_type_out,
                urgency_out,
                confidence_out,
                final_text_out,
                case_scenario_out,
                possible_steps_out,
                required_docs_out,
                courts_process_out,
                severity_out,
                helplines_out,
                flowchart_out,
                flowchart_mermaid_out,
                tts_summary_out,
                safety_json_out,
                passages_table_out,
                structured_json_out,
                raw_json_out,
                tts_audio_out,
                flowchart_graph_out,
                status_out,
            ],
        )

    return demo


def _find_open_port(start_port: int = 7860, end_port: int = 7890) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"Cannot find empty port in range: {start_port}-{end_port}")


def main() -> None:
    Path("data/tts").mkdir(parents=True, exist_ok=True)
    app = build_app()
    app.queue()
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    launch_port = _find_open_port(start_port=preferred_port, end_port=max(preferred_port + 30, 7890))
    print(f"Launching Gradio on http://127.0.0.1:{launch_port}")
    app.launch(server_name="127.0.0.1", server_port=launch_port, share=False, css=APP_CSS)


if __name__ == "__main__":
    main()
