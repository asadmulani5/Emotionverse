"""Fusion engine: combine per-modality emotion scores into one distribution.

Works with 0..3 modalities. Missing/failed modalities are skipped and the
remaining weights are re-normalised, so the output is always a distribution.
"""
from typing import Dict, Optional

from config import settings

CANONICAL = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]

# Each model has its own label vocabulary; map them onto CANONICAL.
LABEL_MAP = {
    "sadness": "sad", "joy": "happy", "anger": "angry", "love": "happy",
    "fearful": "fear", "surprised": "surprise", "calm": "neutral",
}


def normalize_labels(emotions: Dict[str, float]) -> Dict[str, float]:
    """Map model-specific labels to the canonical set and renormalise to sum 1."""
    out: Dict[str, float] = {}
    for key, value in (emotions or {}).items():
        k = LABEL_MAP.get(key.lower(), key.lower())
        if k in CANONICAL:
            out[k] = out.get(k, 0.0) + float(value)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else {}


def _usable(result: Optional[dict]) -> Dict[str, float]:
    if not result or "error" in result:
        return {}
    return normalize_labels(result.get("emotions", {}))


def fuse(results: Dict[str, Optional[dict]], weights: Optional[Dict[str, float]] = None) -> dict:
    """results: {"face": {...}|None, "voice": ..., "text": ...} (model outputs)."""
    weights = weights or settings.fusion.weights
    scores = {m: _usable(r) for m, r in results.items()}
    scores = {m: s for m, s in scores.items() if s and weights.get(m, 0) > 0}
    if not scores:
        return {"emotions": {}, "dominant": None, "used": [], "weights": {}, "contributions": {}}

    total_w = sum(weights[m] for m in scores)
    eff = {m: weights[m] / total_w for m in scores}
    contributions = {
        m: {e: eff[m] * s.get(e, 0.0) for e in CANONICAL} for m, s in scores.items()
    }
    final = {e: sum(c[e] for c in contributions.values()) for e in CANONICAL}
    return {
        "emotions": final,
        "dominant": max(final, key=final.get),
        "used": sorted(scores),
        "weights": eff,
        "contributions": contributions,
    }
