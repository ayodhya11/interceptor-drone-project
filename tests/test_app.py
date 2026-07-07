"""
Test suite for the Drone GCS Flask app.

Runs entirely in demo mode (no API key) so it's fast and makes no real network
calls — it verifies the app's own behavior, not Roboflow's model.
"""
import base64
import io
import json

from PIL import Image

import detection_log


def _sample_data_url():
    """A tiny valid JPEG, base64-encoded as a data URL, like the browser sends."""
    img = Image.new("RGB", (320, 240), color=(50, 60, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + b64


# ---------------------------------------------------------------- auth ----

def test_dashboard_redirects_when_logged_out(client):
    r = client.get("/dashboard")
    assert r.status_code in (302, 308)


def test_login_with_valid_email_reaches_dashboard(client):
    r = client.post("/", data={"email": "test@example.com"}, follow_redirects=True)
    assert r.status_code == 200
    assert b"Drone Detection" in r.data


def test_login_rejects_invalid_email(client):
    r = client.post("/", data={"email": "not-an-email"})
    assert r.status_code == 200
    assert b"valid email address" in r.data


def test_logout_clears_session(logged_in_client):
    r = logged_in_client.get("/logout", follow_redirects=True)
    assert r.status_code == 200
    # should be bounced back to the login page, not the dashboard
    r2 = logged_in_client.get("/dashboard")
    assert r2.status_code in (302, 308)


# ------------------------------------------------------------ demo mode ----

def test_dashboard_shows_demo_mode_banner(logged_in_client):
    r = logged_in_client.get("/dashboard")
    assert r.status_code == 200
    assert b"Demo mode" in r.data


def test_upload_page_loads(logged_in_client):
    r = logged_in_client.get("/detect/upload")
    assert r.status_code == 200


def test_upload_returns_demo_result(logged_in_client):
    data = {"file": (io.BytesIO(b"not a real image, demo mode ignores content"), "test.jpg")}
    r = logged_in_client.post("/detect/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"demo/sample_detection.jpg" in r.data
    assert b"Detections found" in r.data


def test_webcam_page_shows_demo_banner_and_live_controls(logged_in_client):
    r = logged_in_client.get("/detect/webcam")
    assert r.status_code == 200
    # camera capture itself doesn't depend on the API key, so the real controls
    # are always present — only the banner + HUD label change in demo mode
    assert b"Demo mode" in r.data
    assert b"Start camera" in r.data
    assert b"getUserMedia" in r.data


# ---------------------------------------------------------- infer_frame ----

def test_infer_frame_returns_demo_prediction(logged_in_client):
    r = logged_in_client.post("/infer_frame", json={"image": _sample_data_url()})
    assert r.status_code == 200
    data = r.get_json()
    assert data["demo"] is True
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["class"] == "drone (demo)"


def test_infer_frame_confirms_after_two_consecutive_hits(logged_in_client):
    logged_in_client.get("/detect/webcam")  # resets consecutive-hit state for this session
    r1 = logged_in_client.post("/infer_frame", json={"image": _sample_data_url()})
    assert r1.get_json()["drone_confirmed"] is False
    r2 = logged_in_client.post("/infer_frame", json={"image": _sample_data_url()})
    assert r2.get_json()["drone_confirmed"] is True


def test_infer_frame_rejects_missing_image(logged_in_client):
    r = logged_in_client.post("/infer_frame", json={})
    assert r.status_code == 400


def test_infer_frame_populates_detection_log(logged_in_client):
    logged_in_client.post("/infer_frame", json={"image": _sample_data_url()})
    s = detection_log.summary()
    assert s["total"] >= 1


# --------------------------------------------------------- snapshot/export ----

def test_save_snapshot(logged_in_client):
    r = logged_in_client.post("/save_snapshot", json={"image": _sample_data_url()})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_download_report_shape(logged_in_client):
    r = logged_in_client.get("/download_report")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "summary" in data
    assert "model" in data
    assert "detection_log" in data


# ------------------------------------------------------------ detection_log ----

def test_summary_is_empty_with_no_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(detection_log, "LOG_PATH", str(tmp_path / "empty.jsonl"))
    s = detection_log.summary()
    assert s["total"] == 0
    assert s["by_class"] == {}


def test_summary_aggregates_logged_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(detection_log, "LOG_PATH", str(tmp_path / "log.jsonl"))
    detection_log.log_detections("test", [
        {"class": "drone", "confidence": 0.9, "width": 100, "height": 50, "x": 10, "y": 20},
        {"class": "drone", "confidence": 0.7, "width": 80, "height": 40, "x": 5, "y": 5},
    ])
    s = detection_log.summary()
    assert s["total"] == 2
    assert s["by_class"]["drone"] == 2
    assert s["avg_confidence"] == 0.8
