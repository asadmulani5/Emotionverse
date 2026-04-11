import numpy as np
from transformers import pipeline

_classifier = None

def load_voice_model():
    global _classifier
    print("[voice_model] Loading wav2vec2 emotion model...")
    _classifier = pipeline(
        task="audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        top_k=None
    )
    print("[voice_model] Model loaded.")

def predict_voice_emotion(audio_data: list, sample_rate: int = 16000) -> dict:
    if _classifier is None:
        return {"error": "model not loaded"}

    try:
        audio_array = np.array(audio_data, dtype=np.float32)
        raw = _classifier({"array": audio_array, "sampling_rate": sample_rate})
        emotions = {item["label"]: round(item["score"], 4) for item in raw}
        dominant = max(emotions, key=emotions.get)
        return {"emotions": emotions, "dominant": dominant}

    except Exception as e:
        return {"error": f"voice emotion detection failed: {str(e)}"}