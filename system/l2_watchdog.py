#!/usr/bin/env python3
"""Watchdog for the tam-l2-collector service (Coinbase Level 2 + trades tape).

Prints exactly one line:
  OK ...        collector healthy (service active, heartbeat fresh)
  RECOVERED ... collector was down/stale; watchdog restored it and verified
  FAILED ...    recovery attempted but the collector is still not healthy

Every run appends one line to data/l2_watchdog.log. Stdlib only.
Recovery never touches the collector's code or database: it reinstalls the
systemd unit from the in-project copy (system/tam-l2-collector.service) when
missing/different, enables it, and starts/restarts the service.
"""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/home/hatch/workspace/trading-agent-muse")
STATUS = PROJECT / "data" / "l2_status.json"
UNIT_SRC = PROJECT / "system" / "tam-l2-collector.service"
UNIT_DST = Path("/etc/systemd/system/tam-l2-collector.service")
LOG = PROJECT / "data" / "l2_watchdog.log"
SERVICE = "tam-l2-collector"
STALE_SEC = 180  # heartbeat rewrites every ~10s; 180s is generous
VERIFY_SEC = 75


def log(line: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with LOG.open("a") as f:
            f.write(f"{ts} {line}\n")
    except OSError:
        pass


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def service_active() -> bool:
    try:
        r = run(["systemctl", "is-active", SERVICE])
        return r.stdout.strip() == "active"
    except Exception:
        return False


def status_info():
    """Return (age_seconds, status_dict) or (None, None) if unreadable."""
    try:
        st = json.loads(STATUS.read_text())
        return time.time() - float(st["updated_ts"]), st
    except Exception:
        return None, None


def healthy():
    age, st = status_info()
    if not service_active():
        return False, "service not active"
    if age is None:
        return False, "status file missing/unreadable"
    if age > STALE_SEC:
        return False, f"heartbeat stale ({age:.0f}s old)"
    if not st.get("connected"):
        return False, "status says not connected"
    return True, f"heartbeat {age:.0f}s old, last_minute={st.get('last_minute_written')}"


def main() -> int:
    ok, why = healthy()
    if ok:
        line = f"OK {why}"
        print(line)
        log(line)
        return 0

    reasons = why
    actions = []
    _, pre_st = status_info()
    pre_ts = float(pre_st["updated_ts"]) if pre_st else 0.0
    try:
        if not UNIT_DST.exists() or UNIT_DST.read_bytes() != UNIT_SRC.read_bytes():
            shutil.copyfile(UNIT_SRC, UNIT_DST)
            actions.append("unit reinstalled from project copy")
            run(["systemctl", "daemon-reload"])
        run(["systemctl", "enable", SERVICE])
        if service_active():
            run(["systemctl", "restart", SERVICE])
            actions.append("service restarted")
        else:
            run(["systemctl", "start", SERVICE])
            actions.append("service started")
    except Exception as e:  # noqa: BLE001 - report, don't crash
        line = f"FAILED was: {reasons}; recovery error: {e}"
        print(line)
        log(line)
        return 1

    deadline = time.time() + VERIFY_SEC
    while time.time() < deadline:
        ok, why = healthy()
        _, now_st = status_info()
        advanced = bool(now_st) and float(now_st["updated_ts"]) > pre_ts
        if ok and advanced:
            line = f"RECOVERED was: {reasons}; did: {'; '.join(actions)}; now: {why}"
            print(line)
            log(line)
            return 0
        time.sleep(5)

    line = (f"FAILED was: {reasons}; did: {'; '.join(actions)}; "
            f"still: {why}; heartbeat_advanced={advanced}")
    print(line)
    log(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
