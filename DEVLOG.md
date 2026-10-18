# DEVLOG

## Audit — fix startup, add project docs
**What changed:** Rebuilt venv on Python 3.12 (the old one was empty, on 3.14, which the pinned torch 2.2.2 can't run). Removed unused `deepface`/`tensorflow`/`keras`/`librosa`/`sounddevice` from requirements. Forced `framework="pt"` in all three Hugging Face pipelines. Split requirements into runtime and dev. Deleted Render-only `runtime.txt`. Added CLAUDE.md and this file.
**Why:** With deepface installed, TensorFlow/Keras 3 got into the environment and `transformers` tried to load TF classes, crashing startup. Nothing in our code uses TF.
**Key concept:** A `transformers` pipeline picks a backend (PyTorch or TensorFlow) by what's installed. Pinning `framework="pt"` and not installing TF makes startup deterministic. Also: pinned versions only work on Python versions that have wheels for them.
**Verified:** server starts, all 3 models load, /predict/text and /predict/voice return results, frontend builds.

## Blocked
(none yet)
