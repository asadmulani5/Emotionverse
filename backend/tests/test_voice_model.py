import numpy as np
import pytest

from models import voice_model as vm


def test_check_passes_when_only_unimportant_keys_missing():
    vm.check_weights_loaded(["wav2vec2.masked_spec_embed"])


@pytest.mark.parametrize("key", [
    "classifier.weight",
    "projector.bias",
    "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
])
def test_check_fails_when_critical_weights_are_random(key):
    with pytest.raises(RuntimeError):
        vm.check_weights_loaded([key])


def test_labels_mapped_to_canonical():
    out = vm.to_canonical({"neu": 0.1, "hap": 0.6, "ang": 0.2, "sad": 0.1})
    assert set(out) == {"neutral", "happy", "angry", "sad"}
    assert out["happy"] == pytest.approx(0.6)


def test_silence_is_rejected(monkeypatch):
    monkeypatch.setattr(vm, "_model", object())
    monkeypatch.setattr(vm, "_extractor", object())
    result = vm.predict_voice_emotion(np.zeros(16000).tolist(), 16000)
    assert "error" in result


def test_not_loaded_returns_error(monkeypatch):
    monkeypatch.setattr(vm, "_model", None)
    assert vm.predict_voice_emotion([0.1] * 16000)["error"] == "model not loaded"
