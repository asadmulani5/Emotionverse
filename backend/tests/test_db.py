import asyncio

import mongomock
import pytest

import db as dbmod
from db import Database


@pytest.fixture
def database(monkeypatch):
    monkeypatch.setattr(dbmod.settings.db, "enabled", True)
    monkeypatch.setattr(dbmod.settings.db, "flush_interval_s", 0.05)
    d = Database(client=mongomock.MongoClient())
    d.ensure_indexes()
    return d


RESULT = {"face": {"emotions": {"happy": 1}, "dominant": "happy", "extra": 1},
          "voice": {"error": "x"}, "text": {}, "fused": {"emotions": {"happy": 1}, "dominant": "happy"}}


def test_indexes_created(database):
    names = database.db.results.index_information()
    assert any(list(v["key"]) == [("session_id", 1), ("timestamp", 1)] for v in names.values())


async def test_session_and_batched_results_roundtrip(database):
    await database.start()
    await database.create_session("s1")
    for _ in range(3):
        assert database.save_result("s1", RESULT)
    await asyncio.sleep(0.3)
    await database.stop()
    got = database.get_session("s1")
    assert got["session"]["n_results"] == 3
    assert len(got["results"]) == 3
    r = got["results"][0]
    assert r["face"] == {"emotions": {"happy": 1}, "dominant": "happy"}  # slimmed
    assert r["voice"] == {"error": "x"}
    assert "T" in r["timestamp"]
    assert database.list_sessions()[0]["session_id"] == "s1"


async def test_save_result_never_blocks_and_drops_when_full(database):
    database.queue = asyncio.Queue(maxsize=1)
    assert database.save_result("s", RESULT) is True
    assert database.save_result("s", RESULT) is False


def test_disabled_is_a_noop(monkeypatch):
    monkeypatch.setattr(dbmod.settings.db, "enabled", False)
    d = Database()
    assert d.save_result("s", RESULT) is False
    assert d.status() == {"enabled": False, "connected": False}


async def test_flush_failure_is_swallowed(database):
    class Boom:
        def insert_many(self, *_): raise RuntimeError("mongo down")
    database.db = type("D", (), {"results": Boom(), "sessions": database.db.sessions})()
    database.save_result("s", RESULT)
    await database._flush(database._drain())  # must not raise
    assert database.connected is False


def test_unreachable_mongo_does_not_crash(monkeypatch):
    monkeypatch.setattr(dbmod.settings.db, "enabled", True)
    monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:1")
    monkeypatch.setattr(dbmod.settings.db, "server_selection_timeout_ms", 100)
    d = Database()
    assert d.ping() is False
    d.ensure_indexes()  # logs, no raise
