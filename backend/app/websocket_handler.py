import asyncio

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .audio import normalize_gain
from .config import settings
from .asr.stream_processor import StreamProcessor


async def handle_connection(websocket: WebSocket, engine, vad) -> None:
    """One WebSocket connection's lifecycle: raw PCM16 audio in, transcript JSON out.

    Each connection gets its own StreamProcessor so buffering/commit state never
    leaks between concurrent speakers.
    """
    await websocket.accept()
    processor = StreamProcessor(
        engine=engine,
        sample_rate=settings.sample_rate,
        max_buffer_seconds=settings.max_buffer_seconds,
        min_step_seconds=settings.step_seconds,
    )

    chunk_count = 0
    speech_chunk_count = 0

    try:
        while True:
            message = await websocket.receive_bytes()
            chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            chunk = normalize_gain(chunk)
            chunk_count += 1

            is_speech = vad.contains_speech(chunk)
            if is_speech:
                speech_chunk_count += 1
                processor.add_audio(chunk)

            if settings.debug and chunk_count % 10 == 0:
                print(
                    f"[debug] chunks={chunk_count} speech_chunks={speech_chunk_count} "
                    f"peak_amplitude={float(np.abs(chunk).max()) if len(chunk) else 0.0:.4f} "
                    f"buffer_samples={len(processor.buffer)}",
                    flush=True,
                )

            if processor.ready():
                # processor.process() runs engine.transcribe() synchronously, which
                # is CPU-bound Whisper inference (device="cpu" in settings). Calling
                # it directly here would block the whole event loop for the duration
                # of inference — no audio gets read, no other connection gets
                # served — causing exactly the growing lag/backlog you're seeing.
                # asyncio.to_thread hands it to a worker thread instead.
                result = await asyncio.to_thread(processor.process)
                if settings.debug:
                    print(
                        f"[debug] transcript committed={result['committed']!r} pending={result['pending']!r}",
                        flush=True,
                    )
                try:
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "committed": result["committed"],
                            "pending": result["pending"],
                        }
                    )
                except RuntimeError as exc:
                    # Starlette raises RuntimeError (not WebSocketDisconnect) when a
                    # send is attempted after the client already sent a close frame —
                    # narrow the match on message so an unrelated RuntimeError from
                    # elsewhere in this loop (e.g. a real bug in processor.process())
                    # still propagates instead of being silently swallowed.
                    if "close message has been sent" not in str(exc):
                        raise
                    return
    except WebSocketDisconnect:
        return
