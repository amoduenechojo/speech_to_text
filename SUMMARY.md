# SpeakLife — App Summary (2026-08-19)

Self-hosted, real-time speech-to-text. Audio never leaves the server, no
per-minute billing.

## Pipeline

1. Browser mic capture (Web Audio API + AudioWorklet) → downsample to 16kHz
   PCM16 → stream over WebSocket in 100ms chunks.
2. FastAPI backend (`backend/app/main.py`, `websocket_handler.py`) receives
   chunks at `/ws/transcribe`.
3. Silero VAD (`backend/app/asr/vad.py`) gates out silence.
4. Speech audio accumulates in a rolling buffer; faster-whisper
   (`backend/app/asr/engine.py`) re-transcribes the window on each step
   (~1s).
5. Commit/revise stabilizer (`backend/app/asr/stream_processor.py`) uses
   LocalAgreement-2: locks in words once two consecutive transcriptions
   agree, trims committed audio from the buffer, feeds committed text back
   in as Whisper's `initial_prompt`.
6. Frontend (`frontend/app.js`, `style.css`) renders committed text solid,
   pending text dimmed.

Config is env-driven via `SPEAKLIFE_*` vars — model size, device, VAD
threshold, buffer/step sizes (`backend/app/config.py`). Dev default model is
`base` (small/fast); production target is `large-v3-turbo`.

## Verification (2026-08-19)

- Ran the backend test suite: 11 tests pass (VAD, stream processor,
  websocket handling — all use fake injected models, no GPU/network
  needed).
- Found and removed a stray, unquoted `i am sorry` sitting after the
  function body in `websocket_handler.py` — an editing artifact that was a
  hard Python syntax error and broke test collection entirely. Fixed.
- `speech_to_text.py` at the repo root is an old, unrelated prototype using
  Google's cloud speech API (`speech_recognition` + `recognize_google`) —
  not part of the SpeakLife pipeline, and contradicts its self-hosted
  design. Left in place, flagged for possible cleanup.
- Did not do a live browser/mic test in this environment (no mic hardware
  available here); backend logic verified via test suite only.

## Open questions (from README)

- Which GPU host to deploy to (RunPod / Lambda / Fly GPU / other)
- Tuning the buffer window/step sizes once real-time is running end to end
- Whether text-to-speech is in scope
- Whether a dyslexia-focused accessibility variant gets built later

## Requested changes (2026-08-19), tracked for follow-up

1. Mic doesn't pick up voice when the user moves away from the laptop —
   **fixed 2026-08-20**: added `backend/app/audio.py::normalize_gain`
   (peak-normalizes quiet chunks toward a target level, capped at 6x gain
   so background noise isn't amplified into false VAD triggers), applied
   in `websocket_handler.py` before both VAD scoring and ASR buffering.
   Also made mic constraints explicit in `frontend/app.js`
   (`autoGainControl`/`noiseSuppression`/`echoCancellation: true`) instead
   of relying on per-browser defaults.
2. Transcription diction/clarity isn't great — likely the `base` model size
   trade-off (see config default above); the gain-normalization fix above
   should also help for quiet/far speech specifically. Bumping
   `SPEAKLIFE_MODEL_SIZE` is still the main lever for general accuracy.
3. Need a logo. Not started.
4. Add a refresh button to the frontend — **done 2026-08-19**: clears the
   caption and opens a fresh WebSocket (backend gives each connection its
   own clean `StreamProcessor`, so no backend change was needed).
5. Integrate an SLM (explicitly not an LLM) for **live punctuation/casing
   cleanup now, rant summarization later** (user's scoping call
   2026-08-20). In progress:
   - Tried `felflare/bert-restore-punctuation` — dead end, its published
     HF config only exposes generic `LABEL_0..LABEL_14`, no verified
     mapping to real punctuation marks. Not safe to use blind.
   - Switched to `oliverguhr/fullstop-punctuation-multilang-large` via the
     maintained `deepmultilingualpunctuation` PyPI package — verified
     `id2label` mapping (`0/'.', ',', '?', '-', ':'`), real usage code
     inspected directly from the installed package source. This one only
     restores punctuation, not casing — casing will be handled separately
     with a deterministic capitalize-after-sentence-end pass, not a
     second model.
   - Blocked on a ~2.2GB weight download failing repeatedly on SSL errors
     from HF's CDN in this sandbox; retrying in the background. Not yet
     wired into the app.
6. Pivot idea: a "rant space" — users speak freely, sessions get saved
   with start/stop timestamps, retrievable later. Would need persistent
   storage and a session/history UI; not yet started. SLM summarization
   (see 5) is meant to attach to this once it exists.
