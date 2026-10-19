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

## Tasks 3, 6 — Input validation, throttling, drop-when-busy
**What changed:** New backend/validation.py rejects empty/oversized/corrupt text, images (decode + dimension check), audio (length, NaN, sample rate, list or base64 float32). REST returns 422/413 JSON; Socket.IO emits {error, code}. Socket analyze now drops frames ({dropped: 'throttled'|'busy'}) instead of queueing: per-connection min interval and a global in-flight limit from config. Modalities run concurrently with a per-model timeout. Voice audio is resampled to 16 kHz. Socket.IO max payload capped at 3 MB.
**Why:** A bad frame or giant payload shouldn't crash the server or pile up work; real-time video should favour fresh frames over complete ones.
**Key concept:** Backpressure: if the consumer is slower than the producer, drop the oldest/newest work rather than queue it, or latency grows without bound. Validate at the boundary, then trust internal code.

## Tasks 9–12 — MongoDB persistence
**What changed:** New backend/db.py. Connects with MONGO_URI from .env (falls back to local docker mongo URL); lazy client, so startup never blocks and the app runs if Mongo is down/unset. Each socket connection gets a uuid session_id and a doc in 'sessions' (created/ended in background tasks). Every result is queued with save_result() and a background flusher inserts batches (batch_size / flush_interval in config) into 'results' — the live stream never waits on the DB, a full queue drops rather than blocks. Only scores are stored, never images/audio/typed text. Indexes: results(session_id, timestamp), sessions.session_id unique, sessions.started_at. Routes: GET /sessions, GET /sessions/{id} (503 if DB down, 404 if missing). /health includes db status. Added .env.example (placeholders only) and python-dotenv.
**Why:** Replay and summaries need history; privacy and reliability need it to be optional and non-blocking.
**Key concept:** Write-behind queue: the request path only appends to an in-memory queue (O(1)); a separate worker batches inserts. Trade-off: if the process dies, up to ~2 s of results are lost — acceptable for this use. Compound index (session_id, timestamp) serves 'all results for a session in time order' without a sort.

## Tasks 18, 19 (backend), 20 (backend) — Mismatch, explainability data, session summary
**What changed:** fusion.detect_mismatch(): for each pair of modalities, flags when dominant emotions differ and both confidences >= config threshold (neutral ignored by default). analyze results now carry fused.contributions (each modality's weighted share per emotion — the explainability data) and a 'mismatch' block. New backend/summary.py computes dominant expression (mean fused distribution), duration, spikes (non-neutral emotion >= spike_min and >= spike_delta above its session mean, merged when consecutive, with timestamps), mismatch count and modalities used; returned as 'summary' by GET /sessions/{id}.
**Why:** These are the product's differentiators: instead of one opaque label, show *why* and when signals disagree.
**Key concept:** Explainability by construction: late fusion is a weighted sum, so each modality's contribution is exactly weight x its score - no post-hoc approximation needed. Spike detection compares a point to its own session baseline rather than an absolute threshold.

## Tasks 7 (frontend), 15, 17, part of 26 — Frontend foundation
**What changed:** Removed CRA boilerplate (logo, reportWebVitals, App.css, stale test, README, manifest icons, web-vitals dep). Rewrote the frontend: config.js (backend URL from REACT_APP_BACKEND_URL), hooks/useEmotionSocket (auto-reconnect with status connecting/connected/reconnecting/unreachable, stop-and-wait sending), WebcamCapture (releases camera when off, friendly denied/not-found messages), ExpressionPanel, Timeline, StatusBar, ErrorBanner, light/dark CSS variables. Messages for server down, reconnecting, camera denied, no face, server-rejected input. Wording is 'Detected expression', never 'you feel'. Render-only runtime.txt was already removed in the audit.
**Why:** The old UI mixed inline styles, hard-coded localhost, and silently swallowed failures; the user must always see what's wrong.
**Key concept:** Stop-and-wait flow control: send the next frame only after the last result (or a timeout) so a slow server never builds a backlog. Reconnect state machine: 'unreachable' (never connected) vs 'reconnecting' (lost connection) - same retry loop, different message.

## Task 22 — Consent / privacy screen
**What changed:** New ConsentScreen modal shown on first load (state saved in localStorage via a safe wrapper). LivePage (camera, socket, analysis) is only mounted after the user accepts, so no camera access and no socket connection happen before consent. 'No thanks' shows a screen confirming nothing was captured. A 'Privacy & consent' button revokes consent. App.js now holds consent + theme toggle (system/light/dark); live logic moved to pages/LivePage.jsx. Text covers: estimates not feelings, self-use only (EU AI Act prohibits it in workplaces/schools), what is sent where, what is saved (scores only).
**Why:** Emotion sensing is sensitive; consent must be informed and come before any capture.
**Key concept:** Consent gating by mounting: instead of an if-check inside each feature, don't render the feature (and its effects) at all until consent - effects can't run if the component never mounts.

## Task 21 — Face / voice / text toggles (+ microphone capture)
**What changed:** ModalityToggles lets the user switch each input on/off. Off means the hardware is released (camera/mic tracks stopped) and nothing is sent. New MicCapture records a rolling 3 s window with Web Audio, skips silence, resamples to 16 kHz in the browser (utils/audio.js) and sends base64 float32 — voice was never wired into the frontend before. Friendly mic-denied/not-found messages. Backend logs per-request used modalities and latency at DEBUG.
**Why:** Users should control exactly what is sensed, and voice fusion needs real audio.
**Key concept:** Privacy by construction: a toggle that merely hides the UI isn't enough; the effect that owns the MediaStream is keyed on  so turning it off runs cleanup and stops the tracks. Resampling client-side cuts upload size ~3x.

## Blocked
(none yet)

## Mismatch alerts + explainability panel (UI)
- What: MismatchAlert shows when two inputs confidently disagree (e.g. smiling face, sad text) plus recent history. ContributionPanel shows how much each input contributed to the top expression.
- Why: Makes the fusion explainable instead of a black-box number, and treats disagreement between signals as a useful insight.
- Fixes: throttled/busy frames no longer wipe the panels; modality names capitalized.
- Key concept: fused score per emotion = sum over inputs of (re-balanced weight x that input's score). Each input's share of the top emotion is its term divided by the total.

## Sessions page: summary + replay
- What: Live/Sessions tabs. Sessions lists saved sessions (empty ones hidden), and opening one shows a summary (dominant expression, average mix, spikes, mismatch count) and a replay with play/pause, slider and 1x/2x/4x speed.
- Why: Turns stored data into something useful: you can look back at how your detected expressions changed over a session.
- Local dev DB: docker run -d --name emotionverse-mongo -p 27017:27017 -v emotionverse-mongo-data:/data/db --restart unless-stopped mongo:7
- Key concept: replay waits the real time gap between saved readings divided by speed, capped at 2s so long pauses don't stall playback.
- Found: voice model's classifier head isn't loaded from the checkpoint (random weights). Tracked as a separate fix.

## Fix voice model (random weights)
- Problem: the old checkpoint loaded with a randomly initialised classifier head AND a random positional conv layer, so voice predictions were meaningless.
- Cause 1: that checkpoint's head layer names didn't match the model class.
- Cause 2: newer torch makes transformers name the positional conv weights parametrizations.weight.original0/1, while older checkpoints store weight_g/weight_v, so those weights silently failed to load.
- Fix: switched to superb/wav2vec2-base-superb-er; load with the legacy weight_norm so names match; check_weights_loaded() raises if any critical weight is missing, so the registry disables voice instead of serving garbage; near-silent clips are rejected.
- Key concept: "Some weights were newly initialized" means those layers are random. Always read model loading warnings; a model can run without errors and still be producing noise.

## Frontend tests
- What: 18 Jest tests in 8 files: format, smoothing, audio and emotion-label helpers; mismatch sentence; SessionSummary rendering; ConsentScreen buttons; App flow (consent -> live -> sessions, remembered consent, revoking consent).
- Why: there were no frontend tests, so CI would fail with "No tests found". Tests also guard the consent flow, which is the privacy promise of the app.
- How: React Testing Library renders components in jsdom (a fake browser) and clicks buttons like a user. The Live and Sessions pages are stubbed with jest.mock in the App test because they need a webcam and a backend.
- Run: cd frontend && CI=true npm test -- --watchAll=false
