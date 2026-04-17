EMOTION_MAP = {
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    "fear": "fear",
    "surprise": "surprise",
    "love": "happy",
    "disgust": "disgust",
    "calm": "neutral",
    "fearful": "fear",
    "surprised": "surprise",
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "disgust": "disgust",
}

def normalize_emotions(raw: dict) -> dict:
    normalized = {}
    for label, score in raw.items():
        mapped = EMOTION_MAP.get(label.lower(), label)
        if mapped in normalized:
            normalized[mapped] += score
        else:
            normalized[mapped] = score
    return normalized

def weighted_fusion(face: dict, voice: dict, text: dict) -> dict:
    emotions = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]

    face  = normalize_emotions(face)
    voice = normalize_emotions(voice)
    text  = normalize_emotions(text)

    fused = {}
    for emotion in emotions:
        fused[emotion] = round(
            (0.4 * face.get(emotion, 0)) +
            (0.3 * voice.get(emotion, 0)) +
            (0.3 * text.get(emotion, 0)), 4
        )

    dominant = max(fused, key=fused.get)
    return {"emotions": fused, "dominant": dominant}