"""
Best-effort fetch from Roboflow's PROJECT API (roboflow.com/api) — a different
endpoint from the serverless inference API used elsewhere in this app.

This is where dataset counts (total images, classes, train/val/test split) and
training metrics (mAP, precision, recall) actually live. It only works if:
  - your API key has access to the project's metadata, and
  - (for training metrics) the model version was trained on Roboflow itself

If it's not available, the dashboard shows a clear fallback message — we never
fake these numbers.
"""
import os
import requests

import detector

_cache = None  # cached for the life of the process; one dashboard load = one call


def _fetch_raw(timeout=4):
    """Fetch the raw project JSON once. Returns (data, error_reason)."""
    global _cache
    if _cache is not None:
        return _cache

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        _cache = (None, "No API key configured.")
        return _cache

    url = f"https://api.roboflow.com/{detector.MODEL_ID}"
    try:
        resp = requests.get(url, params={"api_key": key}, timeout=timeout)
        if resp.status_code != 200:
            _cache = (None, f"Roboflow project API returned {resp.status_code}.")
            return _cache
        _cache = (resp.json(), None)
        return _cache
    except requests.RequestException as e:
        _cache = (None, f"Could not reach Roboflow: {e}")
        return _cache
    except ValueError:
        _cache = (None, "Unexpected response from Roboflow.")
        return _cache


def fetch_model_stats():
    """mAP / precision / recall for this trained version, if available."""
    data, reason = _fetch_raw()
    if data is None:
        return {"available": False, "reason": reason}

    model = data.get("model") or {}
    map_score, precision, recall = model.get("map"), model.get("precision"), model.get("recall")
    if map_score is None and precision is None and recall is None:
        return {"available": False, "reason": "This model version doesn't expose training metrics via the API."}
    return {"available": True, "map": map_score, "precision": precision, "recall": recall}


def fetch_dataset_stats():
    """Total images / classes / train-val-test split, if available."""
    data, reason = _fetch_raw()
    if data is None:
        return {"available": False, "reason": reason}

    project = data.get("project") or {}
    total_images = project.get("images")
    classes = project.get("classes")   # usually {"drone": <count>, ...}
    splits = project.get("splits")     # usually {"train": n, "valid": n, "test": n}

    if total_images is None and not classes and not splits:
        return {"available": False, "reason": "Dataset counts aren't exposed for this project/key."}

    return {
        "available": True,
        "total_images": total_images,
        "classes": list(classes.keys()) if classes else [],
        "splits": splits or {},
    }
