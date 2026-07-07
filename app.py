"""
Drone Detection & Tracking — Ground Control Station
====================================================
Local Flask console that runs the Roboflow hosted model `drone-uskpc/1` for:
  - Email-only login
  - Dashboard (project aim, dataset, training, annotations, results)
  - Detection from an uploaded image/video
  - Real-time webcam detection

Detection uses Roboflow's serverless API — no local model file needed.
The API key is read from the ROBOFLOW_API_KEY environment variable (see .env.example).

PHASE 1 (this build): webcam + upload detection.
PHASE 2 (later):      Mission Planner arm/disarm on first detection
                      (hook: WebcamInference._on_first_detection in webcam_stream.py).

Run:
    pip install -r requirements.txt
    cp .env.example .env        # then put your key in .env
    python app.py
Open http://127.0.0.1:5000
"""

import os
import time
import json
import base64
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, Response, flash, jsonify, send_file
)
from werkzeug.utils import secure_filename

import detector
import detection_log
import mission_planner

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

ALLOWED_IMG = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VID = {"mp4", "avi", "mov", "mkv"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-this-secret-key-later")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB uploads

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "training"), exist_ok=True)

CONFIRM_FRAMES = 2   # consecutive detections needed before the alert/Mission Planner hook fires


@app.template_filter("datetimeformat")
def datetimeformat(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# ----------------------------------------------------------------------------
# Auth (email-only)
# ----------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "email" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        if "@" in email and "." in email:
            session["email"] = email
            return redirect(url_for("dashboard"))
        flash("Please enter a valid email address.")
    if "email" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    training_imgs = sorted(
        f for f in os.listdir(os.path.join(BASE_DIR, "static", "training"))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    results = detection_log.summary()
    return render_template(
        "dashboard.html",
        email=session["email"],
        model_ready=detector.api_configured(),
        demo_mode=not detector.api_configured(),
        model_id=detector.MODEL_ID,
        conf_threshold=detector.CONFIDENCE_THRESHOLD,
        training_imgs=training_imgs,
        results=results,
    )


# ----------------------------------------------------------------------------
# Upload detection (image or video)
# ----------------------------------------------------------------------------
def _ext(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@app.route("/detect/upload", methods=["GET", "POST"])
@login_required
def detect_upload():
    result_file = result_type = count = error = None
    demo_mode = not detector.api_configured()

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "No file selected."
        elif demo_mode:
            # No live API key — show the bundled sample so the page never looks broken.
            result_file, result_type, count = "demo/sample_detection.jpg", "image", 1
        else:
            ext = _ext(file.filename)
            fname = secure_filename(file.filename)
            in_path = os.path.join(UPLOAD_DIR, fname)
            file.save(in_path)
            try:
                if ext in ALLOWED_IMG:
                    result_file, count = _detect_image(in_path, fname)
                    result_type = "image"
                elif ext in ALLOWED_VID:
                    result_file, count = _detect_video(in_path, fname)
                    result_type = "video"
                else:
                    error = "Unsupported file type."
            except Exception as e:  # noqa: BLE001
                error = f"Detection failed: {e}"

    return render_template(
        "detect_upload.html",
        email=session["email"],
        result_file=result_file, result_type=result_type,
        count=count, error=error, demo_mode=demo_mode,
    )


def _detect_image(in_path, fname):
    annotated, count, preds = detector.infer_image_file(in_path)
    detection_log.log_detections(f"upload:{fname}", preds)
    out_name = f"det_{int(time.time())}_{fname}"
    cv2.imwrite(os.path.join(RESULT_DIR, out_name), annotated)
    return f"results/{out_name}", count


def _detect_video(in_path, fname):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = f"det_{int(time.time())}_{os.path.splitext(fname)[0]}.mp4"
    writer = cv2.VideoWriter(
        os.path.join(RESULT_DIR, out_name),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preds = detector.infer(frame)     # note: one API call per frame (slow on long clips)
        detection_log.log_detections(f"upload:{fname}", preds)
        total += len(preds)
        writer.write(detector.draw(frame, preds))
    cap.release()
    writer.release()
    return f"results/{out_name}", total


# ----------------------------------------------------------------------------
# Webcam detection (real-time MJPEG stream)
# ----------------------------------------------------------------------------
@app.route("/detect/webcam")
@login_required
def detect_webcam():
    # fresh state each time the page is opened
    session["consecutive_hits"] = 0
    session["drone_alerted"] = False
    return render_template(
        "detect_webcam.html",
        email=session["email"],
        model_ready=detector.api_configured(),
    )


def _decode_data_url(data_url):
    """Turn a 'data:image/jpeg;base64,...' string into a BGR numpy frame."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.route("/infer_frame", methods=["POST"])
@login_required
def infer_frame():
    """
    Runs on a single frame captured in the VISITOR'S OWN browser (via getUserMedia).
    This is what makes live detection work once the app is deployed publicly —
    there's no server-side camera involved at all.
    """
    data = request.get_json(silent=True) or {}
    data_url = data.get("image", "")
    if not data_url:
        return jsonify(error="no image provided"), 400

    try:
        frame = _decode_data_url(data_url)
        if frame is None:
            return jsonify(error="could not decode image"), 400
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 400

    demo = not detector.api_configured()
    if demo:
        h, w = frame.shape[:2]
        preds = detector.demo_predictions(w, h)
    else:
        try:
            preds = detector.infer(frame)
        except Exception as e:  # noqa: BLE001
            return jsonify(error=str(e), predictions=[], demo=False), 200

    if preds:
        detection_log.log_detections("webcam-browser" + (":demo" if demo else ""), preds)

    # per-visitor state via the Flask session (cookie-based) — correct even
    # with many people using the deployed site at the same time
    hits = session.get("consecutive_hits", 0)
    hits = hits + 1 if preds else 0
    session["consecutive_hits"] = hits
    drone_confirmed = hits >= CONFIRM_FRAMES

    mp_result = None
    if drone_confirmed and not session.get("drone_alerted"):
        session["drone_alerted"] = True
        # PHASE 2: connect to Mission Planner and arm the vehicle.
        # Stays in simulation (log-only) mode unless both
        # MISSION_PLANNER_CONNECTION and ARM_ON_DETECTION=true are set —
        # see mission_planner.py for the safety rationale.
        mp_result = mission_planner.trigger(
            arm=True, reason=f"webcam confirmed ({hits} consecutive frames)"
        )

    return jsonify(
        predictions=preds, demo=demo, drone_confirmed=drone_confirmed,
        mission_planner=mp_result,
    )


@app.route("/save_snapshot", methods=["POST"])
@login_required
def save_snapshot():
    """Save a frame the browser already drew boxes onto (sent as a data URL)."""
    data = request.get_json(silent=True) or {}
    data_url = data.get("image", "")
    if not data_url:
        return jsonify(ok=False, error="No image data received."), 400
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 400

    out_name = f"snapshot_{int(time.time())}.jpg"
    with open(os.path.join(RESULT_DIR, out_name), "wb") as f:
        f.write(img_bytes)
    return jsonify(ok=True, url=url_for("static", filename=f"results/{out_name}"))


@app.route("/mission_planner/status")
@login_required
def mission_planner_status():
    """JSON snapshot for the dashboard: sim mode?, last action, last error."""
    return jsonify(mission_planner.status())


@app.route("/mission_planner/disarm", methods=["POST"])
@login_required
def mission_planner_disarm():
    """
    Manual, human-initiated disarm. Deliberately NOT automatic — a human
    should always be the one to stand a vehicle down, even though arming
    can be triggered by a confirmed detection.
    """
    result = mission_planner.trigger(arm=False, reason=f"manual disarm by {session['email']}")
    session["drone_alerted"] = False
    session["consecutive_hits"] = 0
    return jsonify(result)


@app.route("/upload_training", methods=["POST"])
@login_required
def upload_training():
    files = request.files.getlist("training_files")
    saved = 0
    for file in files:
        if not file or file.filename == "":
            continue
        ext = _ext(file.filename)
        if ext not in {"png", "jpg", "jpeg"}:
            continue
        fname = secure_filename(file.filename)
        file.save(os.path.join(BASE_DIR, "static", "training", fname))
        saved += 1
    if saved:
        flash(f"Uploaded {saved} training image(s).")
    else:
        flash("No valid PNG/JPG files were uploaded.")
    return redirect(url_for("dashboard"))


@app.route("/download_report")
@login_required
def download_report():
    """Bundle everything about the model + detection history into one downloadable JSON file."""
    results = detection_log.summary()
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_by": session["email"],
        "model": {
            "provider": "Roboflow (hosted serverless inference)",
            "model_id": detector.MODEL_ID,
            "confidence_threshold": detector.CONFIDENCE_THRESHOLD,
            "class_filter": detector.CLASS_FILTER,
        },
        "summary": {
            "total_detections": results["total"],
            "detections_by_class": results["by_class"],
            "average_confidence": results["avg_confidence"],
            "average_box_width_px": results["avg_width"],
            "average_box_height_px": results["avg_height"],
        },
        "detection_log": detection_log.read_all(),
    }
    out_path = os.path.join(BASE_DIR, "static", "results", "drone_gcs_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return send_file(out_path, as_attachment=True,
                      download_name=f"drone_gcs_report_{int(time.time())}.json")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
