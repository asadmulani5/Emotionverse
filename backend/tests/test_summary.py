from datetime import datetime, timedelta, timezone

from summary import summarize

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def row(i, emo_scores, mismatch=False, used=("face",)):
    return {"timestamp": (T0 + timedelta(seconds=i)).isoformat(),
            "fused": {"emotions": emo_scores, "dominant": max(emo_scores, key=emo_scores.get), "used": list(used)},
            "mismatch": {"detected": mismatch}}


CALM = {"neutral": 0.9, "happy": 0.05, "sad": 0.05}
SAD = {"neutral": 0.1, "happy": 0.05, "sad": 0.85}


def test_empty():
    s = summarize([])
    assert s["dominant"] is None and s["spikes"] == []


def test_dominant_and_duration():
    s = summarize([row(i, CALM) for i in range(10)])
    assert s["dominant"] == "neutral" and s["duration_s"] == 9.0 and s["spikes"] == []


def test_spike_detected_with_timestamp_and_merged():
    rows = [row(i, CALM) for i in range(10)] + [row(10, SAD), row(11, SAD)] + [row(i, CALM) for i in range(12, 22)]
    s = summarize(rows)
    assert len(s["spikes"]) == 1
    sp = s["spikes"][0]
    assert sp["emotion"] == "sad" and sp["offset_s"] in (10.0, 11.0) and "T" in sp["timestamp"]


def test_two_separate_spikes():
    rows = [row(i, CALM) for i in range(5)] + [row(5, SAD)] + [row(i, CALM) for i in range(6, 12)] + [row(12, SAD)] + [row(i, CALM) for i in range(13, 20)]
    assert len(summarize(rows)["spikes"]) == 2


def test_mismatch_count_and_modalities():
    rows = [row(0, CALM, mismatch=True, used=("face", "text")), row(1, CALM)]
    s = summarize(rows)
    assert s["mismatch_count"] == 1 and s["modalities_used"] == {"face": 2, "text": 1}


def test_results_without_fused_are_ignored():
    s = summarize([{"timestamp": T0.isoformat(), "fused": {"emotions": {}}}])
    assert s["dominant"] is None
