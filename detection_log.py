"""
Detection log — automatic record of every detection the app makes.

Every time the webcam or an uploaded file detects something, we append one line
of JSON here. The dashboard reads this file to build the Results section
(counts, average confidence, box sizes, a simple chart) with zero manual entry.

Storage: a plain JSONL file (one JSON object per line) — no database needed.
"""
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "static", "detections_log.jsonl")

MAX_LOG_ENTRIES = 5000   # trim oldest entries past this to keep the file small


def log_detections(source, predictions):
    """
    Append one entry per prediction. `source` is a short label like
    "webcam" or "upload:filename.jpg" so results are traceable.
    Safe to call with an empty predictions list (does nothing).
    """
    if not predictions:
        return
    ts = time.time()
    lines = []
    for p in predictions:
        entry = {
            "ts": ts,
            "source": source,
            "class": p.get("class", "object"),
            "confidence": round(float(p.get("confidence", 0)), 4),
            "width": round(float(p.get("width", 0)), 1),
            "height": round(float(p.get("height", 0)), 1),
            "x": round(float(p.get("x", 0)), 1),
            "y": round(float(p.get("y", 0)), 1),
        }
        lines.append(json.dumps(entry))

    with open(LOG_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")

    _trim_if_needed()


def _trim_if_needed():
    if not os.path.exists(LOG_PATH):
        return
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()
    if len(lines) > MAX_LOG_ENTRIES:
        with open(LOG_PATH, "w") as f:
            f.writelines(lines[-MAX_LOG_ENTRIES:])


def read_all():
    """Return every logged entry, oldest first."""
    if not os.path.exists(LOG_PATH):
        return []
    entries = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summary():
    """
    Aggregate stats for the dashboard: total count, per-class breakdown,
    average confidence, average box size, and a time-bucketed series for
    the chart (detections per hour, last 24 buckets).
    """
    entries = read_all()
    if not entries:
        return {
            "total": 0, "by_class": {}, "avg_confidence": 0,
            "avg_width": 0, "avg_height": 0, "recent": [], "chart": [],
        }

    by_class = {}
    total_conf = total_w = total_h = 0
    for e in entries:
        by_class[e["class"]] = by_class.get(e["class"], 0) + 1
        total_conf += e["confidence"]
        total_w += e["width"]
        total_h += e["height"]

    n = len(entries)

    # bucket into hourly counts for the last 24 hours (simple, dependency-free chart data)
    now = time.time()
    buckets = [0] * 24
    for e in entries:
        age_hours = (now - e["ts"]) / 3600
        if 0 <= age_hours < 24:
            buckets[23 - int(age_hours)] += 1

    return {
        "total": n,
        "by_class": by_class,
        "avg_confidence": round(total_conf / n, 3),
        "avg_width": round(total_w / n, 1),
        "avg_height": round(total_h / n, 1),
        "recent": list(reversed(entries[-25:])),   # most recent 25, newest first
        "chart": buckets,
    }
