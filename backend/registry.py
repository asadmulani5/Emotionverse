"""Model registry: load each model once at startup, isolate failures.

A model that fails to import/load is marked unavailable; the server still
starts and the other modalities keep working.
"""
import asyncio
import importlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import psutil

from logging_setup import get_logger

log = get_logger("registry")

# modality -> (module, loader fn, predictor fn)
SPECS = {
    "face": ("models.face_model", "load_face_model", "predict_face_emotion"),
    "text": ("models.text_model", "load_text_model", "predict_text_emotion"),
    "voice": ("models.voice_model", "load_voice_model", "predict_voice_emotion"),
}


def rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1_048_576, 1)


class ModelRegistry:
    def __init__(self, disabled: Optional[set] = None):
        env = {m.strip() for m in os.getenv("DISABLED_MODELS", "").split(",") if m.strip()}
        self.disabled = set(disabled or ()) | env
        self.status: Dict[str, dict] = {m: {"status": "pending"} for m in SPECS}
        self._predict_fns: Dict[str, Callable] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="infer")
        self.started_at = time.time()

    # ---- loading ----
    def load_all(self) -> None:
        log.info("loading models", extra={"ram_mb": rss_mb()})
        for name, (module, loader, predictor) in SPECS.items():
            if name in self.disabled:
                self.status[name] = {"status": "disabled"}
                log.warning("model disabled by config", extra={"model": name})
                continue
            before = rss_mb()
            t0 = time.time()
            try:
                mod = importlib.import_module(module)
                getattr(mod, loader)()
                self._predict_fns[name] = getattr(mod, predictor)
                self.status[name] = {"status": "loaded", "load_seconds": round(time.time() - t0, 1)}
                log.info("model ready", extra={"model": name, "ram_mb": rss_mb(),
                                                "ram_delta_mb": round(rss_mb() - before, 1)})
            except Exception as exc:  # noqa: BLE001 - a bad model must not stop the server
                self.status[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
                log.exception("model failed to load; modality disabled", extra={"model": name})
        log.info("startup complete", extra={"ram_mb": rss_mb(), "available": self.available()})

    def available(self) -> list:
        return [m for m, s in self.status.items() if s["status"] == "loaded"]

    def is_available(self, modality: str) -> bool:
        return modality in self._predict_fns and self.status[modality]["status"] == "loaded"

    # ---- prediction ----
    def predict_blocking(self, modality: str, *args) -> dict:
        """Run a model; never raises. Always returns a dict (with 'error' on failure)."""
        if not self.is_available(modality):
            return {"error": f"{modality} model unavailable"}
        try:
            result = self._predict_fns[modality](*args)
            return result if isinstance(result, dict) else {"error": "bad model output"}
        except Exception as exc:  # noqa: BLE001
            log.exception("prediction crashed", extra={"model": modality})
            return {"error": f"{modality} prediction failed: {type(exc).__name__}"}

    async def predict(self, modality: str, *args, timeout: float = 5.0) -> dict:
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self._executor, self.predict_blocking, modality, *args)
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            log.warning("prediction timed out", extra={"model": modality, "timeout_s": timeout})
            return {"error": f"{modality} prediction timed out"}

    # ---- health ----
    def health(self) -> dict:
        n_ok = len(self.available())
        overall = "ok" if n_ok == len(SPECS) else ("degraded" if n_ok else "down")
        return {
            "status": overall,
            "models": self.status,
            "ram_mb": rss_mb(),
            "uptime_s": round(time.time() - self.started_at, 1),
        }
