"""EmotionVerse backend: FastAPI REST + Socket.IO, models loaded once at startup."""
import asyncio
import time
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from fusion import fuse
from logging_setup import get_logger, setup_logging
from registry import ModelRegistry
from validation import InputError, validate_audio, validate_image, validate_payload, validate_text

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

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.cors.allowed_origins,
    max_http_buffer_size=3_000_000,  # reject giant payloads at the transport
)
socket_app = socketio.ASGIApp(sio, app)


@app.get("/")
def root():
    return {"project": "EmotionVerse", "status": "running"}


@app.get("/health")
def health():
    return registry.health()


async def analyze_payload(clean: dict) -> dict:
    """Run whichever modalities are present, concurrently. One failing never affects the others."""
    timeout = settings.runtime.predict_timeout_s
    jobs = {}
    if "text" in clean:
        jobs["text"] = registry.predict("text", clean["text"], timeout=timeout)
    if "image" in clean:
        jobs["face"] = registry.predict("face", clean["image"], timeout=timeout)
    if "audio" in clean:
        jobs["voice"] = registry.predict("voice", clean["audio"], clean["sample_rate"], timeout=timeout)
    results = await asyncio.gather(*jobs.values())
    out = {"text": {}, "face": {}, "voice": {}, **dict(zip(jobs, results))}
    out["fused"] = fuse(out)
    return out


@app.exception_handler(InputError)
async def input_error_handler(_, exc: InputError):
    return JSONResponse({"error": exc.message, "code": "invalid_input"}, status_code=exc.status)


# ---------------- REST ---------------- #

@app.post("/predict/text")
async def predict_text(payload: dict):
    text = validate_text(payload.get("text"))
    return await registry.predict("text", text, timeout=settings.runtime.predict_timeout_s)


@app.post("/predict/face")
async def predict_face(payload: dict):
    image = validate_image(payload.get("image"))
    return await registry.predict("face", image, timeout=settings.runtime.predict_timeout_s)


@app.post("/predict/voice")
async def predict_voice(payload: dict):
    audio, sr = validate_audio(payload.get("audio"), payload.get("sample_rate", 16000))
    return await registry.predict("voice", audio, sr, timeout=settings.runtime.predict_timeout_s)


@app.post("/predict/fusion")
async def predict_fusion(payload: dict):
    return await analyze_payload(validate_payload(payload))


# ---------------- Socket.IO ---------------- #

_clients: dict = {}  # sid -> {"last": monotonic time of last accepted frame}
_busy = 0  # predictions in flight; frames arriving beyond the limit are dropped, not queued


@sio.event
async def connect(sid, environ):
    _clients[sid] = {"last": 0.0}
    log.info("client connected", extra={"sid": sid, "clients": len(_clients)})
    await sio.emit("server_message", {"msg": "Connected to EmotionVerse"}, to=sid)


@sio.event
async def disconnect(sid):
    _clients.pop(sid, None)
    log.info("client disconnected", extra={"sid": sid, "clients": len(_clients)})


@sio.event
async def analyze(sid, data):
    global _busy
    client = _clients.get(sid)
    if client is None:
        return
    now = time.monotonic()
    if (now - client["last"]) * 1000 < settings.runtime.min_frame_interval_ms:
        await sio.emit("emotion_result", {"dropped": "throttled"}, to=sid)
        return
    if _busy >= settings.runtime.max_concurrent_predictions:
        await sio.emit("emotion_result", {"dropped": "busy"}, to=sid)
        return
    try:
        clean = validate_payload(data)
    except InputError as exc:
        await sio.emit("emotion_result", {"error": exc.message, "code": "invalid_input"}, to=sid)
        return
    client["last"] = now
    _busy += 1
    try:
        await sio.emit("emotion_result", await analyze_payload(clean), to=sid)
    except Exception:  # noqa: BLE001 - keep the connection alive no matter what
        log.exception("analyze failed", extra={"sid": sid})
        await sio.emit("emotion_result", {"error": "internal error"}, to=sid)
    finally:
        _busy -= 1
