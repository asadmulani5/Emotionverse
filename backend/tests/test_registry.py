import sys
import types

import pytest

import registry as reg


def fake_module(name, loader_raises=False, predict_raises=False):
    mod = types.ModuleType(name)

    def load():
        if loader_raises:
            raise RuntimeError("weights missing")

    def predict(*a):
        if predict_raises:
            raise ValueError("boom")
        return {"emotions": {"happy": 1.0}, "dominant": "happy"}

    mod.load, mod.predict = load, predict
    sys.modules[name] = mod


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.delenv("DISABLED_MODELS", raising=False)
    specs = {
        "face": ("fake_face", "load", "predict"),
        "text": ("fake_text", "load", "predict"),
        "voice": ("fake_voice", "load", "predict"),
    }
    monkeypatch.setattr(reg, "SPECS", specs)
    return specs


def test_failed_load_disables_only_that_model(registry):
    fake_module("fake_face"); fake_module("fake_text"); fake_module("fake_voice", loader_raises=True)
    r = reg.ModelRegistry()
    r.load_all()
    assert r.available() == ["face", "text"]
    assert r.status["voice"]["status"] == "failed"
    assert r.health()["status"] == "degraded"
    assert r.predict_blocking("voice", [0.1])["error"].endswith("unavailable")


def test_import_failure_is_isolated(registry):
    fake_module("fake_face"); fake_module("fake_text")
    sys.modules.pop("fake_voice", None)  # import will fail
    r = reg.ModelRegistry()
    r.load_all()
    assert r.status["voice"]["status"] == "failed"
    assert r.is_available("face")


def test_all_failed_is_down_but_no_exception(registry):
    for n in ("fake_face", "fake_text", "fake_voice"):
        fake_module(n, loader_raises=True)
    r = reg.ModelRegistry()
    r.load_all()
    assert r.health()["status"] == "down"


def test_disabled_models_env(registry, monkeypatch):
    for n in ("fake_face", "fake_text", "fake_voice"):
        fake_module(n)
    monkeypatch.setenv("DISABLED_MODELS", "voice")
    r = reg.ModelRegistry()
    r.load_all()
    assert r.status["voice"]["status"] == "disabled"
    assert r.available() == ["face", "text"]


async def test_prediction_crash_is_contained(registry):
    fake_module("fake_face", predict_raises=True); fake_module("fake_text"); fake_module("fake_voice")
    r = reg.ModelRegistry()
    r.load_all()
    bad = await r.predict("face", "x")
    good = await r.predict("text", "x")
    assert "error" in bad and good["dominant"] == "happy"


async def test_timeout(registry):
    import time
    fake_module("fake_face"); fake_module("fake_text"); fake_module("fake_voice")
    sys.modules["fake_face"].predict = lambda *a: time.sleep(0.5)
    r = reg.ModelRegistry()
    r.load_all()
    out = await r.predict("face", "x", timeout=0.05)
    assert out["error"].endswith("timed out")
