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
