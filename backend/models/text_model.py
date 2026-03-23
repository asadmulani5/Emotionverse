# text_model.py
# Detects emotion from text using DistilBERT.

from transformers import pipeline

# This variable holds the loaded model
# None until load_text_model() is called
_classifier = None

def load_text_model():
    """
    Downloads and loads the DistilBERT emotion model.
    Called once when the server starts — not on every request.
    Loading takes ~10 seconds, inference takes ~50ms.
    """
    global _classifier
    print("[text_model] Loading DistilBERT emotion model...")
    _classifier = pipeline(
        task="text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None  # return ALL emotion scores, not just the top one
    )
    print("[text_model] Model loaded.")

def predict_text_emotion(text: str) -> dict:
    """
    Takes a string and returns emotion probability scores.
    Example: "I am so happy today" -> { joy: 0.95, sadness: 0.02, ... }
    """
    if _classifier is None:
        return {"error": "model not loaded"}

    # Run the text through DistilBERT
    raw = _classifier(text)[0]

    # raw looks like: [{"label": "joy", "score": 0.95}, ...]
    # We convert it to a clean dict: { "joy": 0.95, ... }
    emotions = {item["label"]: round(item["score"], 4) for item in raw}

    # Find which emotion scored highest
    dominant = max(emotions, key=emotions.get)

    return {
        "emotions": emotions,
        "dominant": dominant
    }