"""
Shared pytest fixtures.

All tests run in DEMO MODE on purpose: no ROBOFLOW_API_KEY is set, so nothing
here ever makes a real network call to Roboflow. That means these tests are
fast, free, and deterministic in CI — they check the app's own logic (routes,
session state, logging), not the third-party model.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Make sure no real key leaks in from the environment running these tests
os.environ.pop("ROBOFLOW_API_KEY", None)

import pytest
import app as flask_app_module
import detection_log


@pytest.fixture
def client(tmp_path, monkeypatch):
    # isolate each test's detection log so tests never interfere with each other
    # or with a real local detections_log.jsonl
    monkeypatch.setattr(detection_log, "LOG_PATH", str(tmp_path / "detections_log.jsonl"))
    flask_app_module.app.config.update(TESTING=True)
    with flask_app_module.app.test_client() as c:
        yield c


@pytest.fixture
def logged_in_client(client):
    client.post("/", data={"email": "test@example.com"}, follow_redirects=True)
    return client
