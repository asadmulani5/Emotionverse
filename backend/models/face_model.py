import base64
import io
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline

_classifier = None

def load_face_model():
    global _classifier
    print("[face_model] Loading ViT face emotion model...")
    _classifier = pipeline(
        task="image-classification",
        model="trpakov/vit-face-expression",
        top_k=None
    )
    print("[face_model] Model loaded.")

_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade

def predict_face_emotion(image_b64: str) -> dict:
    if _classifier is None:
        return {"error": "model not loaded"}

    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return {"error": f"invalid image data: {str(e)}"}

    try:
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        cascade = _get_face_cascade()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return {"error": "no face detected in image"}

        x, y, w, h = faces[0]
        face_crop = image.crop((x, y, x + w, y + h))
    except Exception as e:
        return {"error": f"face detection failed: {str(e)}"}

    try:
        raw = _classifier(face_crop)
        emotions = {item["label"]: round(item["score"], 4) for item in raw}
        dominant = max(emotions, key=emotions.get)
        return {"emotions": emotions, "dominant": dominant}
    except Exception as e:
        return {"error": f"emotion classification failed: {str(e)}"}
