# Deploying to Render (free tier)

Render is a solid free option for a Flask app like this: Git-based deploys,
free web service tier, and environment variables set safely in their
dashboard (never committed to your repo). The free tier sleeps after
inactivity — the first request after a while takes ~30-60s to wake up. That's
fine for a portfolio/demo link.

## 1. Push this repo to GitHub first
See the main README / your terminal history — you need this on GitHub before
Render can deploy it.

## 2. Create a Render account
Go to https://render.com and sign up (GitHub login is fastest — it also
makes connecting your repo a one-click step).

## 3. Create a new Web Service
- Dashboard → **New** → **Web Service**
- Connect your GitHub account if prompted, then select your `drone-detection` repo

## 4. Configure the service
Render usually auto-detects Python. Confirm/set:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** leave blank if a `Procfile` is present (it already is —
  `web: gunicorn app:app --bind 0.0.0.0:$PORT ...`), otherwise use that exact line.
- **Instance Type:** Free

## 5. Add your environment variables
Under **Environment**, add:
```
ROBOFLOW_API_KEY = your_actual_key
ROBOFLOW_MODEL_ID = drone-uskpc/1
CONFIDENCE_THRESHOLD = 0.65
CLASS_FILTER = drone
FLASK_SECRET = some-random-string-you-make-up
```
This is the whole point of using environment variables instead of `.env` in
the repo — your key never touches GitHub.

If you'd rather not expose your key publicly at all, just **skip** setting
`ROBOFLOW_API_KEY` — the app will run in demo mode automatically, which is a
perfectly reasonable choice for a public portfolio link.

## 6. Deploy
Click **Create Web Service**. Render will build and deploy — watch the logs.
Once it says "Your service is live," you'll get a URL like:
```
https://drone-gcs.onrender.com
```

## 7. Update your README
Paste that URL into the "Live demo" line at the top of `README.md`, commit, and push.

## Notes
- The free tier's filesystem is **not persistent** — uploaded files and the
  detection log (`static/uploads/`, `static/results/`, `detections_log.jsonl`)
  will reset on redeploy or restart. Fine for a demo; for anything long-lived
  you'd want a real database or object storage.
- Live webcam detection will not work on Render (see the "Known limitation"
  note in the main README) — the app falls back to demo mode there
  automatically, so this isn't a broken-looking failure.
