# SpeakLife

Self-hosted, real-time speech-to-text. Audio never leaves your own server, and
there's no per-minute billing once it's hosted.

Whisper has no native streaming mode, so the live feel comes from a
workaround: buffer incoming audio, re-transcribe the rolling window
repeatedly, and stitch results into stable text via a commit/revise
stabilizer (`backend/app/asr/stream_processor.py`) — locking in words once
two consecutive transcriptions agree on them, and only re-revising the
uncertain tail.

## Pipeline

1. Browser mic capture (Web Audio API + AudioWorklet) → 16kHz PCM16 over WebSocket
2. Silero VAD gates out silence
3. Rolling audio buffer holds the uncommitted window
4. faster-whisper transcribes the window on each step
5. Commit/revise stabilizer locks in agreed-upon words, revises the rest
6. Browser renders committed text solid, pending text dimmed

## Running locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000 and click **Start listening**.

First run downloads the Whisper model (`large-v3-turbo`, INT8) and the Silero
VAD model, so it'll be slow once — after that both are cached locally.

Config is environment-driven (see `backend/app/config.py`), prefixed
`SPEAKLIFE_`, e.g. `SPEAKLIFE_DEVICE=cuda` to use a GPU, or
`SPEAKLIFE_MODEL_SIZE=small` for a lighter model.

## Tests

```bash
cd backend
pytest
```

Tests inject fake ASR/VAD models, so they don't need a GPU, network access,
or real model weights.

## Docker

```bash
docker compose up --build
```

Serves the app on http://localhost:8000. CPU + INT8 by default; set
`SPEAKLIFE_DEVICE=cuda` in `docker-compose.yml` and use a CUDA-enabled base
image if running on a GPU host.

## Open questions

- Which GPU host to deploy to (RunPod / Lambda / Fly GPU / other)
- Tuning the buffer window/step sizes once real-time is running end to end
- Whether text-to-speech is in scope
- Whether the dyslexia-focused accessibility variant gets built later

See `SpeakLife-System-Design-Summary.md` for the full design rationale.
