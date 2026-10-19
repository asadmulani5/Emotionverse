# EmotionVerse — project guide for Claude sessions

Private, local-first, explainable "emotion mirror" for personal self-reflection.
NOT for monitoring employees or students. UI wording: "detected expression", never "you feel".

## Hard rules
- NEVER write secrets (passwords, tokens, API keys, connection strings) into any tracked file, log, commit, PR, issue, DEVLOG or this file. Secrets live in untracked `.env` (user fills it in VS Code) or platform settings (HF Space secrets, Vercel env, GitHub Actions secrets).
- Only `.env.example` (placeholders) is tracked. gitleaks runs as a pre-commit hook. Run `git diff --cached` before every push.
- Never commit to main. One issue + one branch (feat/ fix/ docs/ test/ chore/) + one PR per task, "Closes #N", squash-merge, delete branch. Conventional commits, imperative.
- After every task: DEVLOG.md entry (what / why / key concept in simple words) and update Progress below.

## Architecture
React (CRA) frontend <-Socket.IO / REST-> FastAPI + python-socketio backend (ASGI `main:socket_app`).
Backend runs 3 Hugging Face models (face: trpakov/vit-face-expression + OpenCV Haar crop; text: distilbert emotion; voice: wav2vec2 SER) and a weighted fusion engine. MongoDB (optional) stores sessions/results.

## Run (macOS, Python 3.12 venv at ./venv — 3.14 is NOT supported by pinned torch)
```
python3.12 -m venv venv && venv/bin/pip install -r backend/requirements-dev.txt
cd backend && ../venv/bin/python -m uvicorn main:socket_app --port 8000
cd frontend && npm install && npm start     # http://localhost:3000
cd backend && ../venv/bin/python -m pytest  # backend tests
cd frontend && CI=true npm test -- --watchAll=false
```

## Folder map
- backend/main.py — app, REST + Socket.IO events; backend/registry.py — model loading/isolation/health
- backend/fusion.py — N-modality weighted fusion; backend/config.yaml + config.py — all tunables
- backend/tests/ — pytest (run from backend/)
- backend/models/{face,text,voice}_model.py — model loaders + predict
- backend/requirements*.txt
- frontend/src/App.js, components/{WebcamCapture,EmotionDashboard}.jsx
- .github/ — PR template (CI added later)

## Progress
Legend: [x] done, [ ] todo, [!] blocked (see DEVLOG "Blocked")
- [x] repo hygiene (.gitignore, PR template, gitleaks hook)
- [x] 1 Audit: py3.12 env, removed deepface/tensorflow, CLAUDE.md, DEVLOG.md
- [x] 8 structured logging
- [x] 2,4,5 registry, isolation, /health, RAM logging
- [x] 3,6 input validation, timeouts, drop-when-busy
- [ ] 7 Socket.IO disconnect handling + frontend reconnect UI
- [x] 9-12 MongoDB (db.py, sessions/results, batching, indexes, REST)
- [x] 13 Frontend past-sessions page + replay
- [x] 16 config file (backend/config.yaml, config.py); fusion.py rewritten + tests
- [x] 7 (frontend reconnect), 15 UI errors, 17 cleanup
- [x] 14 Tests (frontend: 18 tests across utils, components and app flow; backend: 77)
- [x] 18 mismatch (backend), 19 contributions (backend), 20 summary (backend)
- [x] 22 consent screen
- [x] 21 toggles + mic capture
- [x] 18-19 Signature UI: mismatch alerts + explainability panel
- [x] 20 Summary view UI
- [x] 24 Docker + compose, 25 CI (required on main)
- [ ] 26 UI polish
- [ ] 27 HF Spaces deploy (ASK user about HF PRO first)
- [ ] 28 Vercel deploy
- [ ] 29 README
- [ ] 30 Release v0.1.0

## Known issues
- Voice model: superb/wav2vec2-base-superb-er (neutral/happy/angry/sad, roughly 60-65% accurate, acted speech). Excited speech is often read as angry (both high arousal).
