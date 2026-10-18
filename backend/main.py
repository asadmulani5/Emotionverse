"""EmotionVerse backend: FastAPI REST + Socket.IO, models loaded once at startup."""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from db import Database
from fusion import detect_mismatch, fuse
from logging_setup import get_logger, setup_logging
from registry import ModelRegistry
from summary import summarize
from validation import InputError, validate_audio, validate_image, validate_payload, validate_text

setup_logging()
log = get_logger("main")

registry = ModelRegistry()
database = Database()


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load_all()  # blocking is fine: nothing is served until models are ready
    await database.start()  # never raises; app runs without Mongo
    yield
    log.info("shutting down")
    await database.stop()


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
    return {**registry.health(), "db": database.status()}


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
    out["mismatch"] = detect_mismatch(out)
    return out


def _require_db():
    if not database.enabled or database.db is None or not database.ping():
        raise HTTPException(503, "database unavailable")


@app.get("/sessions")
def list_sessions(limit: int = Query(50, ge=1, le=200), skip: int = Query(0, ge=0)):
    _require_db()
    return {"sessions": database.list_sessions(limit, skip)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    _require_db()
    found = database.get_session(session_id)
    if not found:
        raise HTTPException(404, "session not found")
    found["summary"] = summarize(found["results"])
    return found


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
    session_id = uuid.uuid4().hex
    _clients[sid] = {"last": 0.0, "session_id": session_id}
    database.fire(database.create_session(session_id))
    log.info("client connected", extra={"sid": sid, "clients": len(_clients)})
    await sio.emit("server_message", {"msg": "Connected to EmotionVerse", "session_id": session_id}, to=sid)


@sio.event
async def disconnect(sid):
    client = _clients.pop(sid, None)
    if client:
        database.fire(database.end_session(client["session_id"]))
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
        t0 = time.monotonic()
        result = await analyze_payload(clean)
        log.debug("analyzed", extra={"sid": sid, "used": result["fused"]["used"],
                                   "ms": round((time.monotonic() - t0) * 1000)})
        database.save_result(client["session_id"], result)  # queued; never blocks the stream
        await sio.emit("emotion_result", result, to=sid)
    except Exception:  # noqa: BLE001 - keep the connection alive no matter what
        log.exception("analyze failed", extra={"sid": sid})
        await sio.emit("emotion_result", {"error": "internal error"}, to=sid)
    finally:
        _busy -= 1
