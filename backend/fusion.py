def normalize_text_emotions(text_emotions):
    mapping = {
        "sadness": "sad",
        "joy": "happy",
        "anger": "angry",
        "fear": "fear",
        "surprise": "surprise",
        "love": "happy"
    }

    normalized = {}

    for key, value in text_emotions.items():
        mapped_key = mapping.get(key, key)
        normalized[mapped_key] = normalized.get(mapped_key, 0) + value

    return normalized


def weighted_fusion(face, voice, text):
    # ✅ normalize text FIRST
    text = normalize_text_emotions(text)

    final = {
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "neutral": 0,
        "surprise": 0,
        "fear": 0,
        "disgust": 0
    }

    for k in final.keys():
        final[k] = (
            0.3 * face.get(k, 0) +
            0.7 * text.get(k, 0) +   # ✅ TEXT PRIORITY HIGH
            0.0 * voice.get(k, 0)
        )

    dominant = max(final, key=final.get)

    return {
        "emotions": final,
        "dominant": dominant
    }