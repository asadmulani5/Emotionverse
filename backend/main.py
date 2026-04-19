from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from models.text_model import load_text_model, predict_text_emotion
from models.face_model import load_face_model, predict_face_emotion
from models.voice_model import load_voice_model, predict_voice_emotion
from fusion import weighted_fusion

app = FastAPI(title="EmotionVerse", version="1.0")

# Socket.IO setup
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, app)

# Load models on startup
@app.on_event("startup")
async def startup():
    load_text_model()
    load_face_model()
    load_voice_model()

# CORS (open for now, restrict later in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def health_check():
    return {
        "project": "EmotionVerse",
        "status": "running",
        "phase": "fusion engine active"
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- REST APIs ---------------- #

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
        "text": text_result,
        "face": face_result,
        "voice": voice_result,
        "fused": fused
    }


# ---------------- SOCKET EVENTS ---------------- #

@sio.event
async def connect(sid, environ):
    await sio.emit("server_message", {"msg": "Connected to EmotionVerse"}, to=sid)


@sio.event
async def disconnect(sid):
    pass


@sio.event
async def analyze(sid, data):
    try:
        text = data.get("text", "")
        image_b64 = data.get("image", "")
        audio_data = data.get("audio", [])
        sample_rate = data.get("sample_rate", 16000)

        # Run models
        text_result  = predict_text_emotion(text) if text else {}
        face_result  = predict_face_emotion(image_b64) if image_b64 else {}
        voice_result = predict_voice_emotion(audio_data, sample_rate) if audio_data else {}

        # Fusion
        fused = weighted_fusion(
            face_result.get("emotions", {}),
            voice_result.get("emotions", {}),
            text_result.get("emotions", {})
        )

        # Send result back
        await sio.emit("emotion_result", {
            "text": text_result,
            "face": face_result,
            "voice": voice_result,
            "fused": fused
        }, to=sid)

    except Exception as e:
        await sio.emit("emotion_result", {
            "error": str(e)
        }, to=sid)