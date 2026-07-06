# Drone GCS — Real-Time Drone Detection Console

A Flask web app that detects drones from a live webcam or an uploaded image/video,
using a Roboflow-hosted computer vision model. Built as Phase 1 of a larger
project: **Phase 2 will connect a confirmed detection to Mission Planner to
arm/disarm the target drone automatically.**

**Live demo:** _add your deployed URL here once hosted_
**Demo mode:** if no API key is configured, the app automatically falls back
to a labeled sample detection instead of failing — so this repo is fully
explorable without any setup. (See `demo/sample_detection.jpg`.)

## What this demonstrates
- End-to-end ML product thinking: dataset → hosted inference → product UI, not just a notebook
- Browser-based camera capture (not server-side), so live detection works for every visitor once deployed publicly, and multiple people can use it at once without interfering with each other (state is scoped per-session, not a shared global)
- Automatic, zero-manual-entry telemetry: every detection is logged (class, confidence, box size) and surfaced on a dashboard with a live chart
- Defensive engineering: confidence thresholding + class filtering + temporal confirmation to reduce false positives; graceful degradation (demo mode) when a dependency (the API key) isn't present
- A clean data export (one-click JSON report) for downstream use

## Screenshots
_Add 2-3 dashboard/webcam screenshots here before sharing — this is what recruiters actually look at first._

## Architecture
```
Browser (login → dashboard → webcam / upload)
        │
        ▼
Flask app (app.py)
        │
        ├── detector.py ──────► Roboflow serverless inference API
        ├── webcam_stream.py ──► threaded capture + inference loop
        ├── detection_log.py ──► auto-logs every detection (JSONL)
        └── roboflow_stats.py ─► best-effort fetch of official model metrics
```

## Tech stack
Flask · OpenCV · Roboflow Inference SDK · Chart.js · vanilla JS/CSS (no frontend framework — deliberately lightweight)

## Run it locally
```bash
pip install -r requirements.txt
cp .env.example .env      # add your ROBOFLOW_API_KEY, or leave blank for demo mode
python app.py
```
Open http://127.0.0.1:5000

## Deploy it
See `DEPLOYMENT.md` for step-by-step Render deployment instructions.

## Live webcam, anywhere
Webcam detection captures from **the visitor's own browser camera**
(`getUserMedia`), not the server's — so it works identically whether you're
running this locally or it's deployed publicly. Each visitor gets their own
independent session (Flask session, cookie-based), so multiple people can use
it at once without interfering with each other. Camera permission is
requested by the browser the first time (standard browser security — it
remembers the choice per-site after that, same as any other website).

Note: `getUserMedia` requires a secure context — `https://` in production, or
`http://localhost`/`127.0.0.1` for local dev. Render (and most hosts) provide
HTTPS automatically, so this just works once deployed.

## Roadmap
- [x] Phase 1 — webcam + upload detection, auto-logged results, dashboard
- [x] Browser-based webcam capture (works when deployed publicly, not just locally)
- [ ] Phase 2 — Mission Planner integration (arm/disarm on first confirmed detection)
