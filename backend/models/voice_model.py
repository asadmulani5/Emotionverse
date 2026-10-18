"""Voice (speech prosody) emotion model.

Uses a wav2vec2 checkpoint fine-tuned for speech emotion recognition (SUPERB ER,
trained on IEMOCAP: neutral / happy / angry / sad).

On load we verify that the important weights really came from the checkpoint.
If the classification head or the positional conv layer were randomly
initialised, predictions would be meaningless, so we raise and the registry
disables the voice modality instead of serving garbage.
"""
import os

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from logging_setup import get_logger

log = get_logger("voice_model")

MODEL_ID = os.getenv("VOICE_MODEL", "superb/wav2vec2-base-superb-er")
TARGET_SR = 16000
MIN_RMS = 0.005  # below this the clip is effectively silence

# SUPERB ER labels -> canonical labels used by fusion.py
LABELS = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}

# Weights that must come from the checkpoint, never be randomly initialised
CRITICAL = ("classifier", "projector", "pos_conv_embed")

_model = None
_extractor = None


def check_weights_loaded(missing_keys) -> None:
    """Raise if any critical weight was missing from the checkpoint (= random)."""
    bad = [k for k in missing_keys if any(part in k for part in CRITICAL)]
    if bad:
        raise RuntimeError(f"critical weights not loaded from checkpoint (random): {bad[:4]}")


def to_canonical(scores: dict) -> dict:
    out = {}
    for label, score in scores.items():
        key = LABELS.get(label.lower(), label.lower())
        out[key] = out.get(key, 0.0) + float(score)
    return out


def _load_with_legacy_weight_norm():
    """Older wav2vec2 checkpoints store the positional conv as weight_g/weight_v.
    Newer torch makes transformers build it as parametrizations.weight.original0/1,
    so those names don't match and the layer is left random. Hiding the new
    weight_norm while the model is built makes transformers use the legacy names."""
    params = torch.nn.utils.parametrizations
    new_wn = getattr(params, "weight_norm", None)
    if new_wn is not None:
        delattr(params, "weight_norm")
    try:
        return AutoModelForAudioClassification.from_pretrained(MODEL_ID, output_loading_info=True)
    finally:
        if new_wn is not None:
            params.weight_norm = new_wn


def load_voice_model():
    global _model, _extractor
    log.info("loading model", extra={"model": "voice", "id": MODEL_ID})
    _extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model, info = _load_with_legacy_weight_norm()
    check_weights_loaded(info.get("missing_keys", []))
    model.eval()
    _model = model
    log.info("model loaded", extra={"model": "voice", "labels": list(model.config.id2label.values())})


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio
    n = int(len(audio) * TARGET_SR / sr)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def predict_voice_emotion(audio_data, sample_rate: int = TARGET_SR) -> dict:
    if _model is None or _extractor is None:
        return {"error": "model not loaded"}
    try:
        audio = _resample(np.asarray(audio_data, dtype=np.float32), sample_rate)
        if audio.size == 0 or float(np.sqrt(np.mean(audio ** 2))) < MIN_RMS:
            return {"error": "no speech detected (too quiet)"}
        inputs = _extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(_model(**inputs).logits[0], dim=-1).tolist()
        raw = {_model.config.id2label[i]: p for i, p in enumerate(probs)}
        emotions = {k: round(v, 4) for k, v in to_canonical(raw).items()}
        return {"emotions": emotions, "dominant": max(emotions, key=emotions.get)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"voice emotion detection failed: {e}"}
