"""EmotionVerse backend: FastAPI REST + Socket.IO, models loaded once at startup."""
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from fusion import fuse
from logging_setup import get_logger, setup_logging
from registry import ModelRegistry

setup_logging()
log = get_logger("main")

registry = ModelRegistry()


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load_all()  # blocking is fine: nothing is served until models are ready
    yield
    log.info("shutting down")


app = FastAPI(title="EmotionVerse", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=settings.cors.allowed_origins)
socket_app = socketio.ASGIApp(sio, app)


@app.get("/")
def root():
    return {"project": "EmotionVerse", "status": "running"}


@app.get("/health")
def health():
    return registry.health()


async def analyze_payload(data: dict) -> dict:
    """Run whichever modalities are present. One modality failing never affects the others."""
    timeout = settings.runtime.predict_timeout_s
    text, image, audio = data.get("text", ""), data.get("image", ""), data.get("audio", [])
    sample_rate = data.get("sample_rate", 16000)

    out = {"text": {}, "face": {}, "voice": {}}
    if text:
        out["text"] = await registry.predict("text", text, timeout=timeout)
    if image:
        out["face"] = await registry.predict("face", image, timeout=timeout)
    if audio:
        out["voice"] = await registry.predict("voice", audio, sample_rate, timeout=timeout)
    out["fused"] = fuse(out)
    return out


# ---------------- REST ---------------- #

@app.post("/predict/text")
async def predict_text(payload: dict):
    return await registry.predict("text", payload.get("text", ""), timeout=settings.runtime.predict_timeout_s)


@app.post("/predict/face")
async def predict_face(payload: dict):
    return await registry.predict("face", payload.get("image", ""), timeout=settings.runtime.predict_timeout_s)


@app.post("/predict/voice")
async def predict_voice(payload: dict):
    return await registry.predict("voice", payload.get("audio", []), payload.get("sample_rate", 16000),
                                  timeout=settings.runtime.predict_timeout_s)


@app.post("/predict/fusion")
async def predict_fusion(payload: dict):
    return await analyze_payload(payload)


# ---------------- Socket.IO ---------------- #

@sio.event
async def connect(sid, environ):
    log.info("client connected", extra={"sid": sid})
    await sio.emit("server_message", {"msg": "Connected to EmotionVerse"}, to=sid)


@sio.event
async def disconnect(sid):
    log.info("client disconnected", extra={"sid": sid})


@sio.event
async def analyze(sid, data):
    try:
        await sio.emit("emotion_result", await analyze_payload(data or {}), to=sid)
    except Exception:  # noqa: BLE001 - keep the connection alive no matter what
        log.exception("analyze failed", extra={"sid": sid})
        await sio.emit("emotion_result", {"error": "internal error"}, to=sid)
