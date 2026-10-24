import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from app.asr.types import Word
from app.websocket_handler import handle_connection


class AlwaysSpeechVAD:
    def contains_speech(self, chunk):
        return True


class NeverSpeechVAD:
    def contains_speech(self, chunk):
        return False


class FakeEngine:
    def __init__(self, hypotheses):
        self._hypotheses = hypotheses
        self.calls = 0

    def transcribe(self, audio, initial_prompt=""):
        words = self._hypotheses[min(self.calls, len(self._hypotheses) - 1)]
        self.calls += 1
        return words


def make_words(spec):
    words = []
    t = 0.0
    for w in spec:
        words.append(Word(text=f" {w}", start=t, end=t + 0.3))
        t += 0.3
    return words


def build_app(engine, vad):
    app = FastAPI()

    @app.websocket("/ws/transcribe")
    async def ws_route(websocket: WebSocket):
        await handle_connection(websocket, engine, vad)

    return app


def pcm16_silence(num_samples):
    return np.zeros(num_samples, dtype=np.int16).tobytes()


def test_sends_transcript_json_once_a_full_step_of_speech_arrives():
    engine = FakeEngine([make_words(["hello", "wor"])])
    app = build_app(engine, AlwaysSpeechVAD())
    client = TestClient(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_bytes(pcm16_silence(16000))  # 1s at 16kHz == default step size
        message = ws.receive_json()

    assert message["type"] == "transcript"
    assert "committed" in message
    assert "pending" in message
    assert engine.calls == 1


def test_silence_never_reaches_the_engine():
    engine = FakeEngine([make_words(["hello"])])
    app = build_app(engine, NeverSpeechVAD())
    client = TestClient(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_bytes(pcm16_silence(16000))

    assert engine.calls == 0
