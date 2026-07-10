"""Locked-down Docker runner for frozen candidate submissions."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class CandidateRunError(RuntimeError):
    pass


def _drain_capped(stream: Any, buffer: bytearray, limit: int) -> None:
    while True:
        chunk = stream.read(8_192)
        if not chunk:
            return
        remaining = limit - len(buffer)
        if remaining > 0:
            buffer.extend(chunk[:remaining])


def run_candidate_docker(
    submission: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    *,
    image: str = "cam1k-runner:latest",
    timeout_seconds: int = 240,
    log_limit_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    submission = Path(submission).resolve()
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o777)
    name = f"cam1k-candidate-{uuid.uuid4().hex[:12]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=256",
        "--memory=4g",
        "--cpus=4",
        "--ulimit=nofile=256:256",
        "--user=65532:65532",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=512m",
        "--env=HOME=/tmp",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=OMP_NUM_THREADS=4",
        "--env=OPENBLAS_NUM_THREADS=4",
        f"--mount=type=bind,src={submission},dst=/submission,readonly",
        f"--mount=type=bind,src={input_root},dst=/input,readonly",
        f"--mount=type=bind,src={output_root},dst=/output",
        image,
        "python",
        "/submission/solution.py",
        "--capture",
        "/input/capture",
        "--output",
        "/output/maps.npy",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    log = bytearray()
    reader = threading.Thread(
        target=_drain_capped, args=(process.stdout, log, log_limit_bytes), daemon=True
    )
    reader.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(
            ["docker", "kill", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        process.kill()
        returncode = process.wait(timeout=10)
    finally:
        reader.join(timeout=5)
        process.stdout.close()
    metadata = {
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "log": log.decode("utf-8", errors="replace"),
        "log_truncated": len(log) >= log_limit_bytes,
        "container_name": name,
    }
    if timed_out:
        raise CandidateRunError(f"candidate exceeded {timeout_seconds}s: {metadata}")
    if returncode:
        raise CandidateRunError(f"candidate exited {returncode}: {metadata}")
    return metadata
