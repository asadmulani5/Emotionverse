# DEVLOG

## Audit — fix startup, add project docs
**What changed:** Rebuilt venv on Python 3.12 (the old one was empty, on 3.14, which the pinned torch 2.2.2 can't run). Removed unused `deepface`/`tensorflow`/`keras`/`librosa`/`sounddevice` from requirements. Forced `framework="pt"` in all three Hugging Face pipelines. Split requirements into runtime and dev. Deleted Render-only `runtime.txt`. Added CLAUDE.md and this file.
**Why:** With deepface installed, TensorFlow/Keras 3 got into the environment and `transformers` tried to load TF classes, crashing startup. Nothing in our code uses TF.
**Key concept:** A `transformers` pipeline picks a backend (PyTorch or TensorFlow) by what's installed. Pinning `framework="pt"` and not installing TF makes startup deterministic. Also: pinned versions only work on Python versions that have wheels for them.
**Verified:** server starts, all 3 models load, /predict/text and /predict/voice return results, frontend builds.

## Task 8 — Structured logging
**What changed:** New `backend/logging_setup.py` (JSON-lines formatter, `LOG_LEVEL` env). All `print` calls replaced by `logger.info/exception`.
**Why:** Machine-parseable logs with fields (model name, session id, timings) are searchable in prod; prints are not.
**Key concept:** Structured logging = log a fixed message plus key/value fields (`extra={...}`), emitted as JSON, instead of formatting data into a sentence.

## Task 16 (+ fusion rewrite) — Config file
**What changed:** `backend/config.yaml` + typed loader `config.py` (pydantic). Fusion weights, mismatch threshold, input limits, timeouts, DB batching, CORS origins all live there. `fusion.py` rewritten: normalises each model's labels (joy→happy, fearful→fear, calm→neutral...), re-normalises weights over only the modalities that returned a result, and returns per-modality contributions. First unit tests added.
**Why:** Voice weight was hard-coded to 0 and voice labels didn't match the canonical set, so voice never counted. Hard-coded numbers also made tuning require code edits.
**Key concept:** Late fusion = each model scores independently, then we take a weighted average of probability distributions. If a modality is missing we drop it and re-scale the remaining weights so the result is still a valid distribution.

## Tasks 2, 4, 5 — Model registry, isolation, /health
**What changed:** New backend/registry.py loads face/text/voice once at startup, each in its own try/except; failures mark that modality failed/disabled and the server still starts. Predictions run in a thread pool through predict(), which never raises and has a timeout. GET /health reports per-model status, RAM (MB) and uptime. main.py rewritten to use a lifespan handler and the registry; DISABLED_MODELS env var turns models off. Text input is now truncated to 512 tokens.
**Why:** One broken model (missing weights, OOM, bad input) used to crash the whole request or startup. Small hosts also need the option to run with fewer models.
**Key concept:** Graceful degradation: treat each model as an optional plugin. Wrap every call so errors become data ({error: ...}) instead of exceptions, and let fusion work with whatever is left.

## Blocked
(none yet)
