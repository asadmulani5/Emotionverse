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


async def test_sessions_routes_503_when_db_disabled(client):
    assert (await client.get("/sessions")).status_code == 503
    assert (await client.get("/sessions/abc")).status_code == 503
    assert (await client.get("/health")).json()["db"] == {"enabled": False, "connected": False}


CID = "client-aaaaaaaaaaaaaaaa"
OTHER = "client-bbbbbbbbbbbbbbbb"


@pytest.fixture
def mongo_db(monkeypatch):
    import mongomock
    import db as dbmod
    monkeypatch.setattr(dbmod.settings.db, "enabled", True)
    d = dbmod.Database(client=mongomock.MongoClient())
    monkeypatch.setattr(main, "database", d)
    return d, dbmod


async def test_session_detail_includes_summary(client, mongo_db):
    d, dbmod = mongo_db
    await d.create_session("abc", {"client_id": CID})
    d.db.results.insert_one({"session_id": "abc", "timestamp": dbmod.utcnow(),
                             "fused": {"emotions": {"happy": 0.9, "neutral": 0.1}, "used": ["text"]},
                             "mismatch": {"detected": True}})
    h = {"X-Client-Id": CID}
    body = (await client.get("/sessions/abc", headers=h)).json()
    assert body["summary"]["dominant"] == "happy"
    assert body["summary"]["mismatch_count"] == 1
    assert "client_id" not in body["session"]
    assert (await client.get("/sessions/nope", headers=h)).status_code == 404
    sessions = (await client.get("/sessions", headers=h)).json()["sessions"]
    assert len(sessions) == 1 and "client_id" not in sessions[0]


async def test_sessions_are_private_per_client(client, mongo_db):
    d, _ = mongo_db
    await d.create_session("mine", {"client_id": CID})
    await d.create_session("theirs", {"client_id": OTHER})
    h = {"X-Client-Id": CID}
    ids = [s["session_id"] for s in (await client.get("/sessions", headers=h)).json()["sessions"]]
    assert ids == ["mine"]
    assert (await client.get("/sessions/theirs", headers=h)).status_code == 404


@pytest.mark.parametrize("headers", [{}, {"X-Client-Id": "short"}, {"X-Client-Id": "bad id with spaces!!"}])
async def test_sessions_require_valid_client_id(client, mongo_db, headers):
    assert (await client.get("/sessions", headers=headers)).status_code == 400
    assert (await client.get("/sessions/mine", headers=headers)).status_code == 400


def test_valid_client_id():
    assert main.valid_client_id(CID) == CID
    assert main.valid_client_id(None) is None
    assert main.valid_client_id("x" * 100) is None
    assert main.valid_client_id("has spaces in it!!") is None
