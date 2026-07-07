"""
Tests for mission_planner.py.

These run entirely in simulation mode — no MAVLink connection, no pymavlink
network calls — matching how the rest of the suite runs in demo mode with
no real Roboflow calls. Real hardware/SITL testing is a manual step, not CI.
"""
import importlib

import mission_planner


def reload_mp(monkeypatch, connection="", arm_on_detection="false"):
    """Reload the module with specific env vars so module-level config is reread."""
    monkeypatch.setenv("MISSION_PLANNER_CONNECTION", connection)
    monkeypatch.setenv("ARM_ON_DETECTION", arm_on_detection)
    importlib.reload(mission_planner)
    return mission_planner


def test_defaults_to_simulation_mode(monkeypatch):
    mp = reload_mp(monkeypatch, connection="", arm_on_detection="false")
    assert mp.simulation_mode() is True


def test_connection_alone_is_not_enough_to_go_live(monkeypatch):
    # Two-key interlock: a connection string without ARM_ON_DETECTION=true
    # must still stay in simulation.
    mp = reload_mp(monkeypatch, connection="udp:127.0.0.1:14550", arm_on_detection="false")
    assert mp.simulation_mode() is True


def test_both_keys_set_leaves_simulation_mode(monkeypatch):
    mp = reload_mp(monkeypatch, connection="udp:127.0.0.1:14550", arm_on_detection="true")
    assert mp.simulation_mode() is False


def test_trigger_arm_in_simulation_mode(monkeypatch):
    mp = reload_mp(monkeypatch)
    result = mp.trigger(arm=True, reason="test")
    assert result["ok"] is True
    assert result["simulated"] is True
    assert result["action"] == "ARM"

    st = mp.status()
    assert st["simulation_mode"] is True
    assert st["armed"] is True
    assert st["last_action"] == "ARM (sim)"


def test_trigger_disarm_in_simulation_mode(monkeypatch):
    mp = reload_mp(monkeypatch)
    result = mp.trigger(arm=False, reason="manual disarm test")
    assert result["ok"] is True
    assert result["simulated"] is True
    assert result["action"] == "DISARM"
    assert mp.status()["armed"] is False


def test_trigger_never_raises_even_if_something_is_wrong(monkeypatch):
    # Force "live" config but with a bogus connection string that will fail
    # fast rather than actually reaching hardware — trigger() must still
    # return a result dict, never propagate an exception.
    mp = reload_mp(monkeypatch, connection="udp:127.0.0.1:1", arm_on_detection="true")
    result = mp.trigger(arm=True, reason="test-live-failure-path")
    assert result["ok"] is False
    assert result["simulated"] is False
    assert "error" in result
