"""
Roboflow hosted inference.
Talks to the serverless API for model `drone-uskpc/1` (configurable via env).
The API key is read from the ROBOFLOW_API_KEY environment variable — never hardcoded.
"""
import os
import cv2

API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "drone-uskpc/1")

# Confidence threshold: raise this if the model flags humans/background as drones.
# 0.0-1.0. Try 0.6-0.75 if you're seeing false positives.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.65"))

# Only accept these class names (comma-separated in .env). Empty = accept any class
# the model returns. Setting this to "drone" ignores any other class the model
# might emit, which helps if it was trained with extra labels.
_class_filter_env = os.environ.get("CLASS_FILTER", "drone").strip()
CLASS_FILTER = [c.strip() for c in _class_filter_env.split(",") if c.strip()] or None

_client = None


def api_configured():
    """True if an API key is available."""
    return bool(os.environ.get("ROBOFLOW_API_KEY"))


def get_client():
    """Create the Roboflow client once, on first use, configured with our thresholds."""
    global _client
    if _client is None:
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from inference_sdk import InferenceHTTPClient
        from inference_sdk.http.entities import InferenceConfiguration

        _client = InferenceHTTPClient(api_url=API_URL, api_key=key)
        _client.configure(InferenceConfiguration(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            class_filter=CLASS_FILTER,
        ))
    return _client


def infer(image):
    """
    Run inference. `image` may be a file path, URL, or a numpy BGR frame.
    Returns a list of prediction dicts:
      {x, y, width, height (center-based, pixels), confidence, class, class_id}
    Already filtered server-side by CONFIDENCE_THRESHOLD and CLASS_FILTER.
    """
    result = get_client().infer(image, model_id=MODEL_ID)
    if isinstance(result, dict):
        return result.get("predictions", [])
    return []


def draw(frame, predictions, color=(0, 176, 255)):
    """Draw boxes + labels onto a BGR frame (in place) and return it."""
    for p in predictions:
        cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        label = f'{p.get("class", "object")} {p.get("confidence", 0):.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def infer_image_file(path):
    """Run inference on an image file. Returns (annotated_bgr, count, raw_predictions)."""
    preds = infer(path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    annotated = draw(img, preds)
    return annotated, len(preds), preds
