import pytest

from fusion import CANONICAL, fuse, normalize_labels

W = {"face": 0.4, "voice": 0.3, "text": 0.3}


def res(**kw):
    return {"emotions": kw}


def test_normalize_maps_model_labels():
    out = normalize_labels({"joy": 0.5, "love": 0.25, "sadness": 0.25})
    assert out == {"happy": 0.75, "sad": 0.25}
    assert normalize_labels({"fearful": 1, "surprised": 1, "calm": 2}) == {
        "fear": 0.25, "surprise": 0.25, "neutral": 0.5}


def test_normalize_empty_or_unknown():
    assert normalize_labels({}) == {}
    assert normalize_labels({"banana": 1}) == {}


def test_three_modalities_sum_to_one():
    out = fuse({"face": res(happy=1), "voice": res(happy=1), "text": res(joy=1)}, W)
    assert out["dominant"] == "happy"
    assert sum(out["emotions"].values()) == pytest.approx(1)
    assert out["used"] == ["face", "text", "voice"]


def test_single_modality_weight_renormalised():
    out = fuse({"face": res(sad=1), "voice": None, "text": {"error": "x"}}, W)
    assert out["used"] == ["face"]
    assert out["weights"] == {"face": 1.0}
    assert out["dominant"] == "sad"


def test_two_modalities():
    out = fuse({"face": res(happy=1), "text": res(sadness=1)}, W)
    assert out["used"] == ["face", "text"]
    assert out["dominant"] == "happy"  # face 0.4/0.7 > text 0.3/0.7


def test_no_modalities():
    out = fuse({"face": None, "voice": {"error": "boom"}}, W)
    assert out["dominant"] is None and out["emotions"] == {} and out["used"] == []


def test_zero_weight_modality_ignored():
    out = fuse({"face": res(happy=1), "text": res(sad=1)}, {"face": 0, "text": 1})
    assert out["used"] == ["text"]


def test_contributions_add_up_to_final():
    out = fuse({"face": res(happy=1), "text": res(sadness=1)}, W)
    for e in CANONICAL:
        assert out["emotions"][e] == pytest.approx(
            sum(c[e] for c in out["contributions"].values()))


from fusion import detect_mismatch


def test_mismatch_detected_when_confident_and_different():
    out = detect_mismatch({"face": res(happy=0.9, sad=0.1), "text": res(sadness=0.8, joy=0.2)}, 0.6)
    assert out["detected"] is True
    p = out["pairs"][0]
    assert (p["a"], p["b"], p["a_emotion"], p["b_emotion"]) == ("face", "text", "happy", "sad")


def test_no_mismatch_when_agree():
    assert not detect_mismatch({"face": res(happy=1), "text": res(joy=1)}, 0.6)["detected"]


def test_no_mismatch_below_threshold():
    r = {"face": res(happy=0.5, sad=0.5 - 1e-9, fear=1e-9), "text": res(sadness=0.9, joy=0.1)}
    assert not detect_mismatch(r, 0.6)["detected"]


def test_threshold_is_configurable():
    r = {"face": res(happy=0.7, sad=0.3), "voice": res(angry=0.7, sad=0.3)}
    assert not detect_mismatch(r, 0.8)["detected"]
    assert detect_mismatch(r, 0.6)["detected"]


def test_neutral_ignored_by_default_but_can_count():
    r = {"face": res(neutral=0.9, happy=0.1), "text": res(sadness=0.9, joy=0.1)}
    assert not detect_mismatch(r, 0.6)["detected"]
    assert detect_mismatch(r, 0.6, ignore_neutral=False)["detected"]


def test_mismatch_ignores_failed_or_single_modalities():
    assert not detect_mismatch({"face": res(happy=1), "voice": {"error": "x"}}, 0.6)["detected"]
    assert not detect_mismatch({}, 0.6)["detected"]


def test_three_way_reports_each_disagreeing_pair():
    r = {"face": res(happy=1), "voice": res(angry=1), "text": res(sadness=1)}
    assert len(detect_mismatch(r, 0.6)["pairs"]) == 3
