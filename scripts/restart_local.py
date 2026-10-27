"""Restart this repository's managed local API and worker after editing .env."""

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"


def running(pid):
    return (
        subprocess.run(
            ["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0
    )


def main():
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Install uv before starting the local services.")
    if not (ROOT / ".env").is_file():
        raise SystemExit("Create .env before starting the local services.")
    RUNTIME.mkdir(exist_ok=True)
    database = RUNTIME / "shar.sqlite"
    if database.exists():
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
            active = db.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','running','retry_wait')"
            ).fetchone()[0]
        if active:
            raise SystemExit("Finish or cancel active jobs before restarting the local services.")
    state_path = RUNTIME / "services.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    processes = []
    for role, info in previous.items():
        pid = info["pid"]
        if not running(pid):
            continue
        command = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True)
        cwd = subprocess.check_output(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], text=True)
        if (
            role not in {"worker", "serve"}
            or not command.strip().endswith(f" shar {role}")
            or f"n{ROOT}" not in cwd.splitlines()
            or os.getpgid(pid) != pid
        ):
            raise SystemExit(f"PID {pid} is no longer a managed service; leaving it untouched.")
        processes.append(pid)
    for pid in processes:
        os.killpg(pid, signal.SIGTERM)
    for _ in range(50):
        if not any(running(pid) for pid in processes):
            break
        time.sleep(0.1)
    else:
        raise SystemExit("A service has not exited. Check runtime/logs before retrying.")
    # Read keys only. Let uv load the values, overriding stale values inherited from a shell.
    env = os.environ.copy()
    for line in (ROOT / ".env").read_text().splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if match:
            env.pop(match[1], None)
    env["UV_PROJECT_ENVIRONMENT"] = str((ROOT / ".venv").resolve())
    env["SHAR_STORAGE_ROOT"] = str(RUNTIME)
    logs = RUNTIME / "logs"
    logs.mkdir(exist_ok=True, mode=0o700)
    services = {}
    started = time.time()
    for role in ("worker", "serve"):
        log = logs / f"{role}.log"
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab", buffering=0) as output:
            process = subprocess.Popen(
                [uv, "run", "--no-sync", "--env-file", str(ROOT / ".env"), "shar", role],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
        services[role] = {"pid": process.pid, "log": str(log)}
    state_path.write_text(json.dumps(services, indent=2))
    for attempt in range(600):
        if not all(running(s["pid"]) for s in services.values()):
            break
        if attempt and attempt % 50 == 0:
            print("Waiting for API and worker startup…", flush=True)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/ready", timeout=0.5) as response:
                fresh = float((RUNTIME / "worker-heartbeat").read_text()) >= started
                if (
                    response.status == 200
                    and fresh
                    and all(running(s["pid"]) for s in services.values())
                ):
                    print(
                        "API and worker restarted. Open http://127.0.0.1:8000 and refresh the page."
                    )
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise SystemExit(
        "Services did not become ready; inspect runtime/logs/serve.log and worker.log."
    )


if __name__ == "__main__":
    main()
