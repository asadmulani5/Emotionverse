# main.py
# Entry point for the EmotionVerse backend.
# Runs FastAPI (HTTP) and Socket.IO (real-time) together on one server.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
from models.text_model import load_text_model, predict_text_emotion

# FastAPI handles normal HTTP requests
app = FastAPI(title="EmotionVerse", version="1.0")

# Socket.IO handles real-time events — this is how
# the frontend streams face/audio/text data to us
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Merge both into one app — single server, single port
socket_app = socketio.ASGIApp(sio, app)
@app.on_event("startup")
async def startup():
    load_text_model()

# Without this, the browser blocks React (port 3000)
# from talking to Python (port 8000) — security rule
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HTTP ──────────────────────────────────────────────────


@app.post("/predict/text")
def predict_text(payload: dict):
    text = payload.get("text", "")
    if not text:
        return {"error": "no text provided"}
    return predict_text_emotion(text)

@app.get("/")
def health_check():
    # Visit localhost:8000 to confirm server is alive
    return {
        "project": "EmotionVerse",
        "status":  "running",
        "phase":   "1 — scaffold"
    }

# ── Socket.IO events ──────────────────────────────────────

@sio.event
async def connect(sid, environ):
    # Fires when a browser connects
    # sid = unique ID for that browser session
    print(f"[+] Connected: {sid}")
    await sio.emit("server_message", {
        "msg": "Connected to EmotionVerse"
    }, to=sid)

@sio.event
async def disconnect(sid):
    print(f"[-] Disconnected: {sid}")

@sio.event
async def analyze(sid, data):
    # This is the main event — frontend sends face/audio/text here
    # Real models plug in here Phase 2 onwards
    # For now returns placeholder so we can test the pipeline
    print(f"[~] analyze() from {sid} | keys: {list(data.keys())}")

    await sio.emit("emotion_result", {
        "emotions": {
            "happy":    0.10,
            "sad":      0.05,
            "angry":    0.05,
            "neutral":  0.60,
            "surprise": 0.10,
            "fear":     0.05,
            "disgust":  0.05
        },
        "dominant": "neutral",
        "note": "placeholder — real models coming Phase 2"
    }, to=sid)