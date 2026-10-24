import copy
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.asr.engine import ASREngine
from app.asr.vad import SileroVAD
from app.config import settings
from app.websocket_handler import handle_connection

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = ASREngine(
        model_size=settings.model_size,
        device=settings.device,
        compute_type=settings.compute_type,
        language=settings.language,
    )
    # Load the Silero model once (avoids re-downloading/re-parsing torch.hub's
    # hubconf on every connection), but store the raw model rather than a
    # shared SileroVAD instance. The model carries hidden RNN state across
    # calls (see SileroVAD.reset()/model.reset_states()), so handing the same
    # instance to every connection would let concurrent speakers corrupt each
    # other's VAD state. Each connection instead gets its own SileroVAD
    # wrapping a deep copy of this template — see websocket_transcribe below.
    app.state.vad_model_template = SileroVAD._load_model()
    yield


app = FastAPI(title="SpeakLife", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
    vad = SileroVAD(
        threshold=settings.vad_threshold,
        sample_rate=settings.sample_rate,
        frame_samples=settings.vad_frame_samples,
        model=copy.deepcopy(websocket.app.state.vad_model_template),
    )
    await handle_connection(websocket, websocket.app.state.engine, vad)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")