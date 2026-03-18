# Indian-Law-Assistant-GenAI

End-to-end legal assistance prototype for Indian law scenarios with:
1. Hybrid legal retrieval (RAG)
2. Claim classification and urgency detection
3. LangGraph orchestration for advisor + safety output
4. Real-time speech-to-text from microphone (Step 4)
5. Text-to-speech response generation (Step 5)
6. Gradio system interface for full workflow testing (Step 6)

## Current Implementation Status

Completed:
1. Step 1: Advanced RAG retriever with TF-IDF, phrase boosts, synonym expansion, calibrated scoring
2. Step 2: Hybrid claim classifier with keyword + embedding scoring
3. Step 3: LangGraph flow (Retriever -> Advisor -> Safety) with deterministic fallback advisor
4. Step 4: Audio transcription using faster-whisper distil-large-v3 with live microphone mode
5. Step 5: TTS output utility with edge-tts primary and pyttsx3 offline fallback
6. Step 6: Gradio UI for end-to-end testing with text/voice input and TTS playback

## Methodology

### Step 1 Methodology: Retrieval

File: [rag/retriever.py](rag/retriever.py)

Approach:
1. Build local vector index from legal datasets using LlamaIndex
2. Retrieve dense candidates via sentence embeddings
3. Re-rank with lexical relevance:
	1. Phrase match boost
	2. TF-IDF weighted keyword boost
	3. Domain synonym boost
4. Calibrate final scores into consistent high-confidence range

Key techniques:
1. Aggressive candidate expansion (fetch 5x then rerank)
2. Corpus-level IDF calculation and caching
3. Legal domain synonym map
4. Score calibration to stable 0.80-0.95 style range for top results

### Step 2 Methodology: Classification

File: [nlp/classifier.py](nlp/classifier.py)

Approach:
1. Rule-based keyword scoring for deterministic signal
2. Embedding similarity to claim prototypes for semantic signal
3. Weighted fusion:
	1. Embedding weight: 0.65
	2. Keyword weight: 0.35
4. Urgency detection via lexical patterns + claim-type priors

Supported claim types:
1. unpaid_wages
2. domestic_violence
3. property_dispute
4. wrongful_termination
5. police_harassment
6. tenant_rights
7. consumer_fraud
8. other

### Step 3 Methodology: Orchestration and Safety

File: [agents/langgraph_flow.py](agents/langgraph_flow.py)

Pipeline graph:
1. Retriever node: classification + passage retrieval
2. Advisor node: LLM structured response or deterministic fallback
3. Safety node: risk flags + mandatory legal disclaimer

Advisor strategy:
1. Primary: local quantized Llama-based structured advisor
2. Fallback: deterministic template-based advisor with actionable steps

Safety strategy:
1. Detect high-risk contexts (violence/threat/urgent patterns)
2. Add escalation guidance
3. Always append mandatory disclaimer

### Step 4 Methodology: Speech-to-Text

File: [audio/transcriber.py](audio/transcriber.py)

Input modes:
1. Audio file transcription
2. Fixed-duration microphone recording
3. Live microphone capture with stop by:
	1. Enter key
	2. Silence threshold duration
	3. Max duration timeout

Reliability features:
1. Model cache directory support
2. Local-files-only mode for offline reuse after first download
3. Download warmup mode
4. Timeout and explicit first-run guidance logs

### Step 5 Methodology: Text-to-Speech

File: [tts/speak.py](tts/speak.py)

Synthesis strategy:
1. Primary engine: edge-tts (high quality free cloud voice)
2. Fallback engine: pyttsx3 (offline local TTS)
3. Safe text normalization before synthesis:
	1. Remove markdown/code fences
	2. Remove control characters
	3. Collapse whitespace
	4. Clamp max text length
4. Ensure disclaimer inclusion by default

Artifact output:
1. audio_path
2. audio_bytes
3. mime_type
4. size_bytes

### Step 6 Methodology: System Interface

File: [app_gradio.py](app_gradio.py)

UI flow:
1. Accept text or microphone/upload audio
2. Run STT if audio is selected
3. Run end-to-end legal graph
4. Optionally generate TTS
5. Display transcript, classification, passages, final response, raw JSON, and playable audio

## Models Used

### Embedding Models
1. sentence-transformers/all-MiniLM-L6-v2
	1. Used in retriever embeddings
	2. Used in classifier semantic similarity

### Advisor Model (Primary)
1. meta-llama/Llama-3.2-3B-Instruct
	1. Loaded in 4-bit quantized mode (bitsandbytes)
	2. Falls back to deterministic advisor if unavailable

### STT Model
1. faster-whisper distil-large-v3
	1. FasterWhisper ID: distil-large-v3
	2. HF repo alias used by library: Systran/faster-distil-whisper-large-v3

### TTS Voices/Engines
1. edge-tts voice default: en-IN-NeerjaNeural
2. pyttsx3 fallback for offline local generation

## Datasets Used

Retriever dataset sources (Hugging Face):
1. vihaannnn/Indian-Supreme-Court-Judgements-Chunked
2. Subimal10/indian-legal-data-cleaned

Retriever implementation details:
1. Deduplicates normalized passages
2. Extracts text from common fields (text/chunk/passage/content/clean_text/judgment_text/body)
3. Preserves legal metadata for traceability where available

## Tech Stack

Core language/runtime:
1. Python 3.11 (conda environment)

ML and NLP:
1. torch
2. sentence-transformers
3. transformers
4. bitsandbytes (for 4-bit advisor load)

Retrieval and data:
1. llama-index-core
2. llama-index-embeddings-huggingface
3. datasets (Hugging Face Datasets)

Audio:
1. faster-whisper
2. sounddevice
3. numpy

TTS:
1. edge-tts
2. pyttsx3

Workflow and API structure:
1. langgraph
2. pydantic

UI:
1. gradio

## Project Structure

Core modules:
1. [rag/retriever.py](rag/retriever.py)
2. [nlp/classifier.py](nlp/classifier.py)
3. [agents/langgraph_flow.py](agents/langgraph_flow.py)
4. [audio/transcriber.py](audio/transcriber.py)
5. [tts/speak.py](tts/speak.py)
6. [app_gradio.py](app_gradio.py)

Validation and reports:
1. [validate_retriever_advanced.py](validate_retriever_advanced.py)
2. [validate_step4_audio.py](validate_step4_audio.py)
3. [VALIDATION_QUERIES.py](VALIDATION_QUERIES.py)
4. [VALIDATION.md](VALIDATION.md)

## Setup and Run

Environment recommendation:
1. Use your conda env: veridiction
2. Use Python executable:
	C:/Users/rohan/miniconda3/envs/veridiction/python.exe

Install dependencies (if missing):
1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe -m pip install torch sentence-transformers transformers bitsandbytes
2. C:/Users/rohan/miniconda3/envs/veridiction/python.exe -m pip install llama-index-core llama-index-embeddings-huggingface datasets
3. C:/Users/rohan/miniconda3/envs/veridiction/python.exe -m pip install faster-whisper sounddevice numpy
4. C:/Users/rohan/miniconda3/envs/veridiction/python.exe -m pip install edge-tts pyttsx3 gradio langgraph pydantic

### Step 4 warmup and live transcription

Warmup model download once:
1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe audio/transcriber.py --download-only --model-name distil-large-v3 --model-dir data/models/faster-whisper

Live mic transcription using local cache:
1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe audio/transcriber.py --live-mic --local-files-only --model-dir data/models/faster-whisper --record-out data/audio_live.wav --max-seconds 30 --silence-seconds 2.0 --language en

### End-to-end pipeline (text/audio + optional TTS)

Text mode:
1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --query "My employer has not paid my salary for 3 months" --top-k 5

Live mic mode + local STT cache:
1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --live-mic --audio-model-dir data/models/faster-whisper --audio-local-files-only --record-out data/audio_live.wav --audio-max-seconds 30 --audio-silence-seconds 2.0 --audio-language en --top-k 5

Live mic + TTS response output:
1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe agents/langgraph_flow.py --live-mic --audio-model-dir data/models/faster-whisper --audio-local-files-only --record-out data/audio_live.wav --audio-max-seconds 30 --audio-silence-seconds 2.0 --audio-language en --top-k 5 --enable-tts --tts-output data/tts/final_response.mp3 --tts-engine edge_tts --tts-fallback-engine pyttsx3

### Step 5 standalone TTS

1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe tts/speak.py --text "Your complaint appears to be unpaid wages. Please gather salary slips and file a written complaint." --output data/tts/sample_response.mp3 --engine edge_tts --fallback-engine pyttsx3

### Step 6 UI launch

1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe app_gradio.py
2. Open: http://127.0.0.1:7860

## Validation

Retriever quality validation:
1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe validate_retriever_advanced.py --force-rebuild

Audio validation:
1. C:/Users/rohan/miniconda3/envs/veridiction/python.exe validate_step4_audio.py --audio-file data/sample.wav --report-out data/step4_validation_report.json

Manual test set:
1. Use [VALIDATION_QUERIES.py](VALIDATION_QUERIES.py)

## Deployment Note for Two-End Sync

For production sync between two user ends:
1. Keep a single backend service as source of truth
2. Clients send requests to backend API (not local scripts)
3. Persist transcript/response/audio artifacts in shared DB and object storage
4. Return stable session IDs and artifact URLs
5. Use websocket/polling for cross-client state sync

## Disclaimer

This project is an AI research prototype and not a substitute for professional legal advice. Always consult a qualified lawyer for legal decisions.