"""
Threaded webcam inference.

Because the Roboflow call takes network time (~200ms-1s), we decouple it from
display: one thread grabs frames continuously, another runs inference on the
latest frame as fast as the API allows, and the stream draws the most recent
predictions on every frame. Result: smooth video, boxes update as fast as the API can.
"""
import threading
import time

import cv2

import detector
import detection_log


class WebcamInference:
    def __init__(self, index=0, infer_interval=0.3, confirm_frames=2):
        self.index = index
        self.infer_interval = infer_interval
        self.confirm_frames = confirm_frames   # consecutive hits needed before alerting
        self.cap = None
        self.frame = None
        self.predictions = []          # what's drawn on screen (every hit, unfiltered)
        self._consecutive_hits = 0
        self.running = False
        self.drone_seen = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            self.running = False
            return
        self.running = True
        self.drone_seen = False
        self._consecutive_hits = 0
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame

    def _infer_loop(self):
        while self.running:
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
            if frame is not None:
                try:
                    preds = detector.infer(frame)
                    with self.lock:
                        self.predictions = preds

                    if preds:
                        detection_log.log_detections("webcam", preds)
                        self._consecutive_hits += 1
                    else:
                        self._consecutive_hits = 0

                    # require a few consecutive hits before firing the alert /
                    # Mission Planner hook, so one flickering frame doesn't trigger it
                    if self._consecutive_hits >= self.confirm_frames and not self.drone_seen:
                        self.drone_seen = True
                        self._on_first_detection()
                except Exception as e:  # noqa: BLE001
                    print("Webcam inference error:", e)
            time.sleep(self.infer_interval)

    def _on_first_detection(self):
        """PHASE 2 hook: connect to Mission Planner and arm/disarm here."""
        print(">>> DRONE DETECTED — Mission Planner trigger point (not armed yet)")

    def frames(self):
        """Generator yielding JPEG frames for MJPEG streaming."""
        while self.running:
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
                preds = list(self.predictions)
            if frame is None:
                time.sleep(0.03)
                continue
            annotated = detector.draw(frame, preds)
            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.03)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
