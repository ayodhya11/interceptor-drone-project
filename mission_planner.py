"""
Mission Planner / ArduPilot integration (Phase 2).
==================================================
Connects to a MAVLink endpoint (Mission Planner's UDP output, SITL, or a real
flight controller) via pymavlink and issues arm/disarm commands when the
detection pipeline confirms a target.

SAFETY-FIRST BY DESIGN — read this before enabling live mode:

  - Defaults to SIMULATION mode. Nothing is ever sent to a real vehicle
    unless BOTH env vars below are set. This is a deliberate two-key
    interlock, not just a single flag:
        MISSION_PLANNER_CONNECTION   e.g. "udp:127.0.0.1:14550"
        ARM_ON_DETECTION             must literally be "true"
    Leaving either unset keeps you in log-only simulation, which is what
    CI and local dev should stay in.

  - This module only ever sends the standard MAV_CMD_COMPONENT_ARM_DISARM
    command — the same one any GCS (Mission Planner, QGroundControl) sends
    when a human clicks "Arm" in its UI. It carries no special privileges
    beyond what MAVLink already exposes to any authenticated client on
    that link, and it does nothing a flight controller's own arming
    checks (GPS lock, EKF health, safety switch, etc.) wouldn't still
    gate on the vehicle side.

  - Every action — simulated or real — is logged and recorded in
    `status()`, so there's always an audit trail of what fired and when.

  - Auto-arm on an *unattended* CV trigger is a real physical-safety
    decision, not just a coding one — especially given this project's own
    documented false-positive behavior (PROJECT_CONTEXT.md, decision #3).
    Recommended pattern: keep ARM_ON_DETECTION off in production and require
    a human to confirm via the manual endpoint before anything arms. Only
    flip ARM_ON_DETECTION on for a controlled test (e.g. SITL) where an
    unexpected arm has zero real-world consequence.

Config (.env):
    MISSION_PLANNER_CONNECTION   MAVLink connection string. Unset = simulation.
    ARM_ON_DETECTION             "true" to allow the automatic hook to send
                                  real commands. Default: "false" (log-only).
    MAVLINK_ARM_TIMEOUT          Seconds to wait for heartbeat/ACK. Default: 5.
"""
import os
import time
import threading

CONNECTION_STRING = os.environ.get("MISSION_PLANNER_CONNECTION", "").strip()
ARM_ON_DETECTION = os.environ.get("ARM_ON_DETECTION", "false").strip().lower() == "true"
ARM_TIMEOUT = float(os.environ.get("MAVLINK_ARM_TIMEOUT", "5"))

_lock = threading.Lock()
_connection = None
_last_state = {
    "armed": False,
    "last_action": None,
    "last_action_ts": None,
    "last_error": None,
}


def simulation_mode():
    """True if we won't talk to a real vehicle at all (the safe default)."""
    return not (CONNECTION_STRING and ARM_ON_DETECTION)


def _get_connection():
    """Lazily open (and cache) the MAVLink connection. Raises on failure."""
    global _connection
    if _connection is None:
        from pymavlink import mavutil  # lazy import: pymavlink stays optional in demo/sim mode
        _connection = mavutil.mavlink_connection(CONNECTION_STRING)
        _connection.wait_heartbeat(timeout=ARM_TIMEOUT)
    return _connection


def _send_arm_command(arm: bool):
    """Send MAV_CMD_COMPONENT_ARM_DISARM (param1=1 arm, 0 disarm) and wait for an ACK."""
    from pymavlink import mavutil
    conn = _get_connection()
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1 if arm else 0,
        0, 0, 0, 0, 0, 0,
    )
    ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=ARM_TIMEOUT)
    if ack is None:
        raise TimeoutError("No COMMAND_ACK received from the flight controller.")
    if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        raise RuntimeError(f"Flight controller rejected the command (result={ack.result}).")


def trigger(arm: bool, reason: str = "confirmed detection"):
    """
    Called from the detection hook (or a manual route). Thread-safe.
    Never raises — logs and records the outcome instead, so a MAVLink
    hiccup can't crash a request handler mid-response to a browser.
    """
    with _lock:
        action = "ARM" if arm else "DISARM"

        if simulation_mode():
            msg = (f"[SIM] Would {action} vehicle — reason: {reason}. "
                   f"(Set MISSION_PLANNER_CONNECTION + ARM_ON_DETECTION=true to go live.)")
            print(msg)
            _last_state.update(armed=arm, last_action=f"{action} (sim)",
                                last_action_ts=time.time(), last_error=None)
            return {"ok": True, "simulated": True, "action": action}

        try:
            _send_arm_command(arm)
            print(f">>> {action} command sent and ACKed — reason: {reason}")
            _last_state.update(armed=arm, last_action=action,
                                last_action_ts=time.time(), last_error=None)
            return {"ok": True, "simulated": False, "action": action}
        except Exception as e:  # noqa: BLE001
            print(f"!!! {action} FAILED: {e}")
            _last_state.update(last_action=f"{action} (FAILED)",
                                last_action_ts=time.time(), last_error=str(e))
            return {"ok": False, "simulated": False, "action": action, "error": str(e)}


def status():
    """Snapshot for the dashboard/status route: sim mode?, last action, last error."""
    return {
        "simulation_mode": simulation_mode(),
        "connection_string": CONNECTION_STRING or None,
        **_last_state,
    }
