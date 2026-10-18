"""Session summary: dominant expression and notable spikes, computed from saved results."""
from collections import Counter
from datetime import datetime
from typing import List

from config import settings


def _ts(r: dict) -> datetime:
    return datetime.fromisoformat(r["timestamp"])


def summarize(results: List[dict]) -> dict:
    fused = [(r, r.get("fused") or {}) for r in results]
    fused = [(r, f) for r, f in fused if f.get("emotions")]
    if not fused:
        return {"n_results": len(results), "dominant": None, "duration_s": 0, "distribution": {},
                "spikes": [], "mismatch_count": 0, "modalities_used": {}}

    t0 = _ts(fused[0][0])
    duration = (_ts(fused[-1][0]) - t0).total_seconds()

    totals: Counter = Counter()
    for _, f in fused:
        totals.update(f["emotions"])
    n = len(fused)
    mean = {e: v / n for e, v in totals.items()}
    dominant = max(mean, key=mean.get)

    cfg = settings.summary
    spikes, prev_spike = [], None
    for idx, (r, f) in enumerate(fused):
        for emo, score in f["emotions"].items():
            if emo == "neutral" or score < cfg.spike_min or score - mean.get(emo, 0) < cfg.spike_delta:
                continue
            # merge consecutive results of the same emotion into one spike (keep the peak)
            if prev_spike and prev_spike["emotion"] == emo and idx - prev_spike["_idx"] <= 1:
                prev_spike["_idx"] = idx
                if score > prev_spike["score"]:
                    prev_spike.update(score=round(score, 3), timestamp=r["timestamp"],
                                      offset_s=round((_ts(r) - t0).total_seconds(), 1))
                continue
            prev_spike = {"emotion": emo, "score": round(score, 3), "timestamp": r["timestamp"],
                          "offset_s": round((_ts(r) - t0).total_seconds(), 1), "_idx": idx}
            spikes.append(prev_spike)
    for s in spikes:
        s.pop("_idx", None)
    spikes = sorted(spikes, key=lambda s: -s["score"])[: cfg.max_spikes]
    spikes.sort(key=lambda s: s["offset_s"])

    used: Counter = Counter()
    for _, f in fused:
        used.update(f.get("used", []))
    return {
        "n_results": len(results),
        "duration_s": round(duration, 1),
        "dominant": dominant,
        "distribution": {e: round(v, 4) for e, v in sorted(mean.items(), key=lambda kv: -kv[1])},
        "spikes": spikes,
        "mismatch_count": sum(1 for r in results if (r.get("mismatch") or {}).get("detected")),
        "modalities_used": dict(used),
    }
