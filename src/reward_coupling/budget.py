"""Persistent wall-clock hard cap shared across commands; no GPU throughput guesses."""

import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def run_budgeted(command, ledger, cap_seconds, deadline_epoch=None):
    if cap_seconds <= 0 or not command:
        raise ValueError("Need command and positive cap")
    path = Path(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        events = [json.loads(s) for s in handle if s.strip()]
        if events and events[-1]["event"] == "start":
            raise RuntimeError(
                "Unclosed budget reservation; reconcile interrupted run before continuing"
            )
        if any(e.get("cap_seconds", cap_seconds) != cap_seconds for e in events):
            raise ValueError("Cannot change a ledger's budget cap")
        used = sum(e.get("elapsed_seconds", 0) for e in events)
        remaining = cap_seconds - used
        if deadline_epoch is not None:
            remaining = min(remaining, deadline_epoch - time.time())
        if remaining <= 0:
            raise TimeoutError("Total budget exhausted")

        def append(event):
            handle.write(json.dumps(event, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        append(
            {
                "event": "start",
                "command": command,
                "cap_seconds": cap_seconds,
                "remaining_seconds": remaining,
                "started_at_epoch": time.time(),
                "deadline_epoch": deadline_epoch,
            }
        )
        start = time.monotonic()
        process = subprocess.Popen(command, start_new_session=True)
        status = "complete"
        try:
            returncode = process.wait(timeout=remaining)
            if returncode:
                status = "failed"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            returncode, status = 124, "budget_exhausted"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            append(
                {
                    "event": "end",
                    "status": status,
                    "elapsed_seconds": time.monotonic() - start,
                    "ended_at_epoch": time.time(),
                }
            )
        return {
            "status": status,
            "returncode": returncode,
            "charged_unit": "whole_process_wall_seconds",
        }
