import pytest
from httpx import ASGITransport, AsyncClient

import main


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=main.app), base_url="http://t") as c:
        yield c


async def test_health_reports_models(client):
    r = (await client.get("/health")).json()
    assert r["status"] == "down"  # all disabled in tests
    assert set(r["models"]) == {"face", "text", "voice"}


@pytest.mark.parametrize("path,body,code", [
    ("/predict/text", {"text": ""}, 422),
    ("/predict/text", {}, 422),
    ("/predict/face", {"image": "junk"}, 422),
    ("/predict/voice", {"audio": [0.1]}, 422),
    ("/predict/fusion", {}, 422),
    ("/predict/text", {"text": "x" * 5000}, 413),
])
async def test_bad_inputs_rejected_cleanly(client, path, body, code):
    r = await client.post(path, json=body)
    assert r.status_code == code
    assert r.json()["code"] == "invalid_input"


async def test_valid_text_with_model_unavailable_returns_error_not_crash(client):
    r = await client.post("/predict/text", json={"text": "hello"})
    assert r.status_code == 200 and "error" in r.json()


async def test_fusion_survives_all_models_down(client):
    r = await client.post("/predict/fusion", json={"text": "hello"})
    assert r.status_code == 200
    assert r.json()["fused"]["dominant"] is None
