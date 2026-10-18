"""MongoDB persistence. Optional: the app must run fine if Mongo is down or unset.

Writes are queued and flushed in batches by a background task so the live
stream never waits on the database. Only emotion scores are stored - never raw
images, audio or typed text.
"""
import asyncio
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

from config import settings
from logging_setup import get_logger

log = get_logger("db")

DEFAULT_LOCAL_URI = "mongodb://localhost:27017"  # not a secret; docker-compose mongo


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    def __init__(self, client=None):
        self.enabled = settings.db.enabled
        self.client = client
        self.db = None
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._task: Optional[asyncio.Task] = None
        self._bg: set = set()
        self._last_warn = 0.0
        self.connected = False
        if self.enabled and self.client is None:
            self._connect()
        if self.client is not None:
            self.db = self.client[settings.db.database]

    # ---- connection ----
    def _connect(self) -> None:
        uri = os.getenv("MONGO_URI") or DEFAULT_LOCAL_URI
        try:
            from pymongo import MongoClient
            # lazy: MongoClient() does not connect until first use, so this never blocks startup
            self.client = MongoClient(uri, serverSelectionTimeoutMS=settings.db.server_selection_timeout_ms)
        except Exception:  # noqa: BLE001 - bad URI etc. must not stop the app
            log.exception("could not create mongo client; persistence disabled")
            self.enabled = False

    def ping(self) -> bool:
        if not self.enabled or self.client is None:
            self.connected = False
            return False
        try:
            self.client.admin.command("ping")
            self.connected = True
        except Exception:  # noqa: BLE001
            self.connected = False
        return self.connected

    def ensure_indexes(self) -> None:
        if self.db is None:
            return
        try:
            self.db.results.create_index([("session_id", 1), ("timestamp", 1)])
            self.db.sessions.create_index("session_id", unique=True)
            self.db.sessions.create_index([("started_at", -1)])
            self.connected = True
        except Exception as exc:  # noqa: BLE001
            self._warn("index creation failed", exc)

    def _warn(self, msg: str, exc: Exception) -> None:
        """Rate-limited warning so a dead DB doesn't flood the logs."""
        self.connected = False
        if time.monotonic() - self._last_warn > 30:
            self._last_warn = time.monotonic()
            log.warning(msg, extra={"error": f"{type(exc).__name__}: {exc}"[:200]})

    def status(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "connected": False}
        return {"enabled": True, "connected": self.ping(), "queued": self.queue.qsize()}

    # ---- lifecycle (call from lifespan) ----
    async def start(self) -> None:
        if not self.enabled:
            log.info("database disabled")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._startup_blocking)
        self._task = asyncio.create_task(self._flusher())

    def _startup_blocking(self) -> None:
        if self.ping():
            self.ensure_indexes()
            log.info("database connected")
        else:
            log.warning("database unreachable at startup; will keep retrying in background")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._flush(self._drain())  # last chance to persist what's queued

    def fire(self, coro) -> None:
        """Run a DB coroutine in the background so socket handlers never wait on Mongo."""
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ---- sessions ----
    async def create_session(self, session_id: str, meta: Optional[dict] = None) -> None:
        if not self.enabled or self.db is None:
            return
        doc = {"session_id": session_id, "started_at": utcnow(), "ended_at": None,
               "n_results": 0, **(meta or {})}
        await self._run(lambda: self.db.sessions.insert_one(doc), "create_session failed")

    async def end_session(self, session_id: str) -> None:
        if not self.enabled or self.db is None:
            return
        await self._run(lambda: self.db.sessions.update_one(
            {"session_id": session_id}, {"$set": {"ended_at": utcnow()}}), "end_session failed")

    # ---- results (non-blocking) ----
    def save_result(self, session_id: str, result: dict) -> bool:
        """Queue a result for batched insert. Never blocks; drops (returns False) if full/disabled."""
        if not self.enabled or self.db is None:
            return False
        doc = {
            "session_id": session_id,
            "timestamp": utcnow(),
            "face": _slim(result.get("face")),
            "voice": _slim(result.get("voice")),
            "text": _slim(result.get("text")),
            "fused": result.get("fused"),
            "mismatch": result.get("mismatch"),
        }
        try:
            self.queue.put_nowait(doc)
            return True
        except asyncio.QueueFull:
            log.warning("result queue full; dropping result")
            return False

    def _drain(self) -> List[dict]:
        docs = []
        while not self.queue.empty():
            docs.append(self.queue.get_nowait())
        return docs

    async def _flusher(self) -> None:
        cfg = settings.db
        while True:
            batch: List[dict] = []
            deadline = time.monotonic() + cfg.flush_interval_s
            while len(batch) < cfg.batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            if batch:
                await self._flush(batch)

    async def _flush(self, batch: List[dict]) -> None:
        if not batch or self.db is None:
            return

        def write():
            self.db.results.insert_many(batch)
            counts: dict = {}
            for d in batch:
                counts[d["session_id"]] = counts.get(d["session_id"], 0) + 1
            for sid, n in counts.items():
                self.db.sessions.update_one({"session_id": sid}, {"$inc": {"n_results": n}})
            self.connected = True

        await self._run(write, "batch write failed; results dropped")

    async def _run(self, fn, err_msg: str):
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, fn)
        except Exception as exc:  # noqa: BLE001 - DB trouble never reaches the live stream
            self._warn(err_msg, exc)
            return None

    # ---- queries (blocking; call from sync routes / executor) ----
    def list_sessions(self, limit: int = 50, skip: int = 0) -> List[dict]:
        cur = self.db.sessions.find({}, {"_id": 0}).sort("started_at", -1).skip(skip).limit(limit)
        return [_iso(d) for d in cur]

    def get_session(self, session_id: str, max_results: int = 5000) -> Optional[dict]:
        sess = self.db.sessions.find_one({"session_id": session_id}, {"_id": 0})
        if not sess:
            return None
        cur = self.db.results.find({"session_id": session_id}, {"_id": 0}).sort("timestamp", 1).limit(max_results)
        return {"session": _iso(sess), "results": [_iso(d) for d in cur]}


def _slim(r: Optional[dict]) -> Optional[dict]:
    """Keep only scores + dominant (or the error) from a model result."""
    if not r:
        return None
    if "error" in r:
        return {"error": r["error"]}
    return {"emotions": r.get("emotions"), "dominant": r.get("dominant")}


def _iso(doc: dict) -> dict:
    def conv(v):
        if isinstance(v, datetime):
            return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).isoformat()
        return v
    return {k: conv(v) for k, v in doc.items()}
