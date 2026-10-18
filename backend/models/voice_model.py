import numpy as np
from transformers import pipeline

from logging_setup import get_logger

log = get_logger("voice_model")
_classifier = None

def load_voice_model():
    global _classifier
    log.info("loading model", extra={"model": "voice"})
    _classifier = pipeline(
        task="audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        top_k=None,
        framework="pt"
    )
    log.info("model loaded", extra={"model": "voice"})

def predict_voice_emotion(audio_data: list, sample_rate: int = 16000) -> dict:
    if _classifier is None:
        return {"error": "model not loaded"}

    try:
        audio_array = np.asarray(audio_data, dtype=np.float32)
        if sample_rate != 16000:  # model expects 16 kHz; cheap linear resample
            n = int(len(audio_array) * 16000 / sample_rate)
            audio_array = np.interp(np.linspace(0, len(audio_array) - 1, n),
                                    np.arange(len(audio_array)), audio_array).astype(np.float32)
            sample_rate = 16000
        raw = _classifier({"array": audio_array, "sampling_rate": sample_rate})
        emotions = {item["label"]: round(item["score"], 4) for item in raw}
        dominant = max(emotions, key=emotions.get)
        return {"emotions": emotions, "dominant": dominant}

    except Exception as e:
        return {"error": f"voice emotion detection failed: {str(e)}"}