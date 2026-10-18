"""Input validation. Anything bad raises InputError (clean rejection, never a crash)."""
import base64
import binascii
import io
import math
from typing import Optional

import numpy as np
from PIL import Image

from config import settings


class InputError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.message = message
        self.status = status


def validate_text(text) -> str:
    if not isinstance(text, str):
        raise InputError("text must be a string")
    text = text.strip()
    if not text:
        raise InputError("text is empty")
    if len(text) > settings.limits.max_text_chars:
        raise InputError(f"text too long (max {settings.limits.max_text_chars} chars)", 413)
    return text


def validate_image(image) -> str:
    """Return clean base64 of a decodable image; reject empty/corrupt/oversized."""
    if not isinstance(image, str) or not image:
        raise InputError("image must be a non-empty base64 string")
    if image.startswith("data:") and "," in image:  # data URL from canvas.toDataURL
        image = image.split(",", 1)[1]
    if len(image) > settings.limits.max_image_bytes * 4 // 3 + 8:
        raise InputError("image too large", 413)
    try:
        raw = base64.b64decode(image, validate=True)
    except (binascii.Error, ValueError):
        raise InputError("image is not valid base64")
    if len(raw) > settings.limits.max_image_bytes:
        raise InputError("image too large", 413)
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001 - PIL raises many types for corrupt data
        raise InputError("image is corrupt or unsupported")
    if w < 32 or h < 32 or w > 4096 or h > 4096:
        raise InputError("image dimensions out of range (32..4096 px)")
    return image


def validate_audio(audio, sample_rate=16000) -> tuple:
    """Accept a list of floats or base64 float32-LE bytes. Returns (np.float32 array, sample_rate)."""
    lim = settings.limits
    if sample_rate not in lim.allowed_sample_rates:
        raise InputError(f"unsupported sample_rate (allowed: {lim.allowed_sample_rates})")
    if isinstance(audio, str):
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            raise InputError("audio is not valid base64")
        if len(raw) % 4:
            raise InputError("audio bytes are not float32 aligned")
        if len(raw) // 4 > lim.max_audio_samples:
            raise InputError("audio too long", 413)
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif isinstance(audio, list):
        if len(audio) > lim.max_audio_samples:
            raise InputError("audio too long", 413)
        try:
            arr = np.asarray(audio, dtype=np.float32)
        except (TypeError, ValueError):
            raise InputError("audio must be numbers")
        if arr.ndim != 1:
            raise InputError("audio must be a flat list")
    else:
        raise InputError("audio must be a list of floats or base64 float32")
    if arr.size < lim.min_audio_samples:
        raise InputError("audio too short")
    if not np.isfinite(arr).all():
        raise InputError("audio contains NaN/inf")
    return np.clip(arr, -1.0, 1.0), sample_rate


def validate_payload(data) -> dict:
    """Validate a combined payload; returns cleaned fields only for modalities present.

    Raises InputError if the payload is not a dict, nothing usable is present, or a
    *present* field is invalid. Empty/missing fields are simply skipped.
    """
    if not isinstance(data, dict):
        raise InputError("payload must be a JSON object")
    clean: dict = {}
    text: Optional[str] = data.get("text")
    if isinstance(text, str) and text.strip():
        clean["text"] = validate_text(text)
    elif text not in (None, ""):
        if not isinstance(text, str):
            raise InputError("text must be a string")
    if data.get("image"):
        clean["image"] = validate_image(data["image"])
    audio = data.get("audio")
    if audio is not None and not (isinstance(audio, (list, str)) and len(audio) == 0):
        clean["audio"], clean["sample_rate"] = validate_audio(audio, data.get("sample_rate", 16000))
    if not clean:
        raise InputError("no usable input (need text, image or audio)")
    return clean
