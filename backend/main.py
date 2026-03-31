from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
from models.text_model import load_text_model, predict_text_emotion
from models.face_model import load_face_model, predict_face_emotion

app = FastAPI(title="EmotionVerse", version="1.0")

# Socket.IO handles real-time events 
# the frontend streams face/audio/text data to us
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Merge both into one app — single server, single port
socket_app = socketio.ASGIApp(sio, app)
@app.on_event("startup")
async def startup():
    load_text_model()
    load_face_model()


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

@app.post("/predict/face")
def predict_face(payload: dict):
    image_b64 = payload.get("image", "")
    if not image_b64:
        return {"error": "no image provided"}
    return predict_face_emotion(image_b64)

@app.get("/")
def health_check():
   
    return {
        "project": "EmotionVerse",
        "status":  "running",
        "phase":   "1 — scaffold"
    }

# ── Socket.IO events ──────────────────────────────────────

@sio.event
async def connect(sid, environ):
   
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