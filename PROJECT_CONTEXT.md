# PROJECT CONTEXT — Interceptor Drone Detection System
_Paste this into a Claude Project's knowledge/instructions so any new conversation has full context immediately._

## What this is
A full-stack drone detection web app. Visitor's browser camera (or an uploaded
file) → Flask backend → Roboflow-hosted computer vision model → results drawn
as boxes, logged automatically, shown on a dashboard. Built as **Phase 1** of
a larger goal: Phase 2 will connect a confirmed detection to Mission Planner
to arm/disarm a target drone (not yet built — stubbed hook only).

## Current status (as of this document)
- ✅ Phase 1 complete and deployed publicly
- ✅ 16 pytest tests passing, CI runs them on every push
- ✅ Clean git history on GitHub, no secrets committed
- ⏳ Phase 2 (Mission Planner) — not started

## Links
- **Repo:** https://github.com/ayodhya11/interceptor-drone-project
- **Live demo:** https://interceptor-drone-project.onrender.com/
- **Hosting:** Render (free tier — sleeps after 15 min idle, first load can take 30-60s to wake)

## Tech stack
Python 3.12 · Flask 3.0 · OpenCV (headless) · Roboflow `inference-sdk` ·
gunicorn (prod server) · pytest + GitHub Actions (CI) · Chart.js (dashboard
chart, via CDN) · vanilla HTML/CSS/JS (no frontend framework, deliberately)

## Architecture (in one paragraph)
The browser captures the visitor's own camera via `getUserMedia()` — never
the server's — because the server has no physical webcam once deployed to
the cloud. Two client-side loops run: a fast one (~60fps) keeps video smooth
by redrawing the last-known detection boxes every frame, and a slower one
(~1.4/sec) POSTs a frame to `/infer_frame`, which calls Roboflow and returns
predictions as JSON. Every detection is logged to a flat JSONL file
(`detection_log.py`), which the dashboard reads and aggregates into stats +
a chart, with zero manual data entry. Detection-confirmation state (has a
drone been seen 2 frames in a row?) lives in the Flask session, scoped
per-visitor — this was a deliberate fix after an earlier version used one
shared global object, which would have caused concurrent visitors to
interfere with each other.

## Key files
- `app.py` — all Flask routes (login, dashboard, upload, webcam, `/infer_frame`, `/save_snapshot`, `/download_report`)
- `detector.py` — Roboflow client wrapper; confidence threshold + class filter; demo-mode fallback predictions
- `detection_log.py` — append-only JSONL log + aggregation for the dashboard
- `roboflow_stats.py` — best-effort fetch of Roboflow project metadata (currently unused by the UI, left in for future use)
- `templates/*.html` — Jinja2 templates, one per page
- `static/css/style.css` — hand-written "tactical GCS" themed stylesheet (amber/cyan/gunmetal palette)
- `tests/` — pytest suite (16 tests), all run in demo mode (no network calls, deterministic)
- `.github/workflows/tests.yml` — CI: runs the test suite on every push to `main`
- `Procfile` — `web: gunicorn app:app ...` — how Render starts the app in production
- `docs/PROJECT_REPORT.pdf` — 5-page project writeup (overview, architecture, decisions, roadmap) with real screenshots

## Key engineering decisions (useful context if extending this)
1. **Browser-side camera capture, not server-side** — makes live detection work for every visitor once deployed publicly, not just on a local machine with an attached webcam. This replaced an earlier server-camera (`cv2.VideoCapture`) approach.
2. **Per-session state via Flask `session`, not a global object** — fixes a real concurrency bug found during development (one shared webcam-state object meant simultaneous visitors would interfere with each other).
3. **Confidence threshold + class filter + 2-consecutive-frame confirmation** — reduces false positives (the hosted public model sometimes flagged people as drones). Configurable via `.env` (`CONFIDENCE_THRESHOLD`, `CLASS_FILTER`). Honest limitation: this can't fully fix a dataset-level problem — that needs retraining with negative examples.
4. **Demo mode** — if `ROBOFLOW_API_KEY` isn't set, the app never looks broken: uploads and webcam both show a clearly labeled sample detection (`drone (demo)`) instead of erroring. This is also what CI tests run against (no real network calls in tests).
5. **Flat JSONL log instead of a database** — deliberate simplicity for current scale; flagged as the first thing to replace if this needed to survive traffic/restarts (Render's free tier filesystem isn't persistent).

## Known limitations (worth stating proactively, not hiding)
- Render free tier: filesystem resets on redeploy/restart — detection log and uploads aren't permanent.
- Email-only login — no password, no rate limiting; fine for a demo, not for anything handling real hardware.
- The underlying Roboflow model (`drone-uskpc/1`) is a public/community model, not one trained by the project owner — false-positive behavior is a dataset issue, not purely fixable in app code.
- Root Directory on Render must be **blank** (app files sit at repo root, not in a subfolder) — this tripped up the first deploy attempt.
- Render requires `PYTHON_VERSION=3.12.7` env var set explicitly — newer default Python versions aren't yet supported by the `inference-sdk` dependency.

## Local setup (quick reference)
```bash
pip install -r requirements.txt
cp .env.example .env   # add ROBOFLOW_API_KEY, or leave blank for demo mode
python app.py           # http://127.0.0.1:5000
```

## What's next (Phase 2)
Wire the Mission Planner hook (currently just a `print()` stub in `app.py`'s
`/infer_frame` route) to `pymavlink` — on a confirmed detection (2 consecutive
frames), connect to Mission Planner and arm/disarm the target drone.

## How this project was built (for context on working style)
Iterative, conversational build over multiple sessions: started from a rough
existing local prototype (webcam + ChatGPT-generated detection code), rebuilt
from scratch with a proper Flask structure, swapped to Roboflow hosted
inference, added auto-logging/dashboard/demo-mode/tests/CI, then deployed
publicly and pushed to GitHub with a clean commit history. Preference:
practical, working software over perfect architecture — but with honest
documentation of shortcuts and limitations rather than hiding them.
