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
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

import cv2
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, Response, flash, jsonify, send_file
)
from werkzeug.utils import secure_filename

import detector
import detection_log
from webcam_stream import WebcamInference

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

ALLOWED_IMG = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VID = {"mp4", "avi", "mov", "mkv"}
WEBCAM_INDEX = int(os.environ.get("WEBCAM_INDEX", "0"))

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-this-secret-key-later")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB uploads

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "training"), exist_ok=True)

# single shared webcam session
cam = WebcamInference(index=WEBCAM_INDEX)


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
    return render_template(
        "detect_webcam.html",
        email=session["email"],
        model_ready=detector.api_configured(),
    )


@app.route("/video_feed")
@login_required
def video_feed():
    if not detector.api_configured():
        return "ROBOFLOW_API_KEY is not set.", 503
    cam.start()
    if not cam.running:
        return "Could not open webcam.", 503
    return Response(cam.frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_feed", methods=["POST"])
@login_required
def stop_feed():
    cam.stop()
    return ("", 204)


@app.route("/snapshot", methods=["POST"])
@login_required
def snapshot():
    """Save the current webcam frame (with boxes drawn) to static/results/."""
    if not cam.running or cam.frame is None:
        return jsonify(ok=False, error="Webcam is not running."), 400
    with cam.lock:
        frame = cam.frame.copy()
        preds = list(cam.predictions)
    annotated = detector.draw(frame, preds)
    out_name = f"snapshot_{int(time.time())}.jpg"
    cv2.imwrite(os.path.join(RESULT_DIR, out_name), annotated)
    return jsonify(ok=True, url=url_for("static", filename=f"results/{out_name}"), count=len(preds))


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


@app.route("/status")
@login_required
def status():
    return jsonify(drone_seen=cam.drone_seen, running=cam.running)


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
