from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
from models.text_model import load_text_model, predict_text_emotion
from models.face_model import load_face_model, predict_face_emotion
from models.voice_model import load_voice_model, predict_voice_emotion
from fusion import weighted_fusion

app = FastAPI(title="EmotionVerse", version="1.0")

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, app)

@app.on_event("startup")
async def startup():
    load_text_model()
    load_face_model()
    load_voice_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {
        "project": "EmotionVerse",
        "status": "running",
        "phase": "5 — fusion engine active"
    }

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

@app.post("/predict/voice")
def predict_voice(payload: dict):
    audio_data = payload.get("audio", [])
    sample_rate = payload.get("sample_rate", 16000)
    if not audio_data:
        return {"error": "no audio provided"}
    return predict_voice_emotion(audio_data, sample_rate)

@app.post("/predict/fusion")
def predict_fusion(payload: dict):
    text = payload.get("text", "")
    image_b64 = payload.get("image", "")
    audio_data = payload.get("audio", [])
    sample_rate = payload.get("sample_rate", 16000)

    text_result  = predict_text_emotion(text) if text else {}
    face_result  = predict_face_emotion(image_b64) if image_b64 else {}
    voice_result = predict_voice_emotion(audio_data, sample_rate) if audio_data else {}

    fused = weighted_fusion(
        face_result.get("emotions", {}),
        voice_result.get("emotions", {}),
        text_result.get("emotions", {})
    )

    return {
        "text":  text_result,
        "face":  face_result,
        "voice": voice_result,
        "fused": fused
    }

@sio.event
async def connect(sid, environ):
    print(f"[+] Connected: {sid}")
    await sio.emit("server_message", {"msg": "Connected to EmotionVerse"}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"[-] Disconnected: {sid}")

@sio.event
async def analyze(sid, data):
    image_b64 = data.get("image", "")
    
    face_result = predict_face_emotion(image_b64) if image_b64 else {}
    emotions = face_result.get("emotions", {
        "happy": 0.1, "sad": 0.05, "angry": 0.05,
        "neutral": 0.6, "surprise": 0.1, "fear": 0.05, "disgust": 0.05
    })
    dominant = face_result.get("dominant", "neutral")

    await sio.emit("emotion_result", {
        "emotions": emotions,
        "dominant": dominant
    }, to=sid)