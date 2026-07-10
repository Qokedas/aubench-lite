#!/usr/bin/env python3
"""Run a frontier model as a coding agent in the isolated CAM1K development container.

Supports the OpenAI Responses API (gpt-*) and the Anthropic Messages API (claude-*).
Each episode is fully isolated: its own workspace, its own networkless container, its
own transcript, and its own frozen submission + grade report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from cam1k_eval.formats import write_json
from cam1k_eval.grader import grade_submission
from cam1k_eval.submission import freeze_submission

ROOT = Path(__file__).resolve().parents[1]
SHELL_OUTPUT_LIMIT = 64 * 1024

_SHELL_DESCRIPTION = (
    "Run one shell command in the isolated development container. "
    "The working directory is /workspace. There is no network access."
)
_SHELL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute."},
        "timeout_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 600,
            "description": "Command wall timeout.",
        },
    },
    "required": ["command", "timeout_seconds"],
    "additionalProperties": False,
}

OPENAI_SHELL_TOOL = {
    "type": "function",
    "name": "shell",
    "description": _SHELL_DESCRIPTION,
    "parameters": _SHELL_SCHEMA,
    "strict": True,
}

# Anthropic strict schemas reject numeric min/max constraints; enforce in code instead.
ANTHROPIC_SHELL_TOOL = {
    "name": "shell",
    "description": _SHELL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute."},
            "timeout_seconds": {
                "type": "integer",
                "description": "Command wall timeout in seconds, between 1 and 600.",
            },
        },
        "required": ["command", "timeout_seconds"],
        "additionalProperties": False,
    },
    "strict": True,
}

USER_PROMPT = (
    "Work autonomously in /workspace. Read /workspace/BRIEF.md and complete the "
    "request in it. Use the shell tool for everything. Finish only after "
    "qokedas-submit reports the submission is ready."
)
INSTRUCTIONS = (
    "You are the evaluated coding agent, not a consultant. Continue making concrete progress "
    "with shell calls until the submission is complete. Do not request user input. Hidden data "
    "is unavailable and must not be guessed or exfiltrated."
)


def _run(
    command: list[str], *, timeout: int = 300, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _copy_public_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "agent" / "BRIEF.md", workspace / "BRIEF.md")
    (workspace / "submission").mkdir()
    (workspace / "deliverables").mkdir()
    (workspace / "experiments").mkdir()
    # Docker Desktop bind mounts preserve enough host ownership information that a
    # dedicated container UID needs explicit write permission on this episode only.
    for directory in (workspace, *workspace.rglob("*")):
        directory.chmod(0o777 if directory.is_dir() else 0o666)


def _start_agent(workspace: Path, name: str) -> None:
    public_data = (ROOT / "data" / "benchmark" / "public").resolve()
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=512",
        "--memory=16g",
        "--cpus=8",
        "--user=10001:10001",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=2g",
        "--env=HOME=/home/agent",
        f"--mount=type=bind,src={workspace.resolve()},dst=/workspace",
        f"--mount=type=bind,src={public_data},dst=/data,readonly",
        "cam1k-agent:latest",
        "sleep",
        "infinity",
    ]
    last_error = ""
    # Docker Desktop's bind-file bridge can briefly lag an atomic dataset-directory
    # replacement. Retry only the container creation; the episode has not reached the
    # model yet, so this is deterministic and cannot duplicate agent actions.
    for attempt in range(5):
        result = _run(command, check=False)
        if result.returncode == 0:
            return
        last_error = result.stderr.strip() or result.stdout.strip()
        _run(["docker", "rm", "--force", name], check=False)
        time.sleep(1 + attempt)
    raise RuntimeError(f"agent container failed to start after retries: {last_error}")


def _drain(stream: Any, output: bytearray) -> None:
    while True:
        chunk = stream.read(8_192)
        if not chunk:
            return
        if len(output) < SHELL_OUTPUT_LIMIT:
            output.extend(chunk[: SHELL_OUTPUT_LIMIT - len(output)])


def _shell(container: str, command: str, timeout_seconds: int) -> str:
    wrapped = [
        "docker",
        "exec",
        "--user=10001:10001",
        "--workdir=/workspace",
        container,
        "timeout",
        "--signal=KILL",
        str(timeout_seconds),
        "bash",
        "-lc",
        command,
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        wrapped,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    output = bytearray()
    reader = threading.Thread(target=_drain, args=(process.stdout, output), daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout_seconds + 15)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = 124
    reader.join(timeout=5)
    process.stdout.close()
    result = {
        "exit_code": returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output": output.decode("utf-8", errors="replace"),
        "truncated": len(output) >= SHELL_OUTPUT_LIMIT,
    }
    return json.dumps(result)


def _api_key(env_file: Path, name: str) -> str:
    values = dotenv_values(env_file)
    value = values.get(name) or os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} was not found in {env_file} or the environment")
    return value


def _with_retry(call: Any, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return call(**kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _append_transcript(transcript: Path, payload: dict[str, Any]) -> None:
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_openai_episode(
    arguments: argparse.Namespace, container: str, transcript: Path
) -> tuple[int, str]:
    from openai import OpenAI

    client = OpenAI(api_key=_api_key(arguments.env_file, "OPENAI_API_KEY"))
    prompt = USER_PROMPT
    input_items: list[Any] = [{"role": "user", "content": prompt}]
    actions = 0
    final_text = ""
    while actions < arguments.max_actions:
        response = _with_retry(
            client.responses.create,
            model=arguments.model,
            instructions=INSTRUCTIONS,
            input=input_items,
            tools=[OPENAI_SHELL_TOOL],
            tool_choice="auto",
            reasoning={"effort": arguments.reasoning_effort},
            max_output_tokens=arguments.max_output_tokens,
            # The Responses API may return reasoning item references. Persisting the
            # response keeps those IDs valid when the next tool output is supplied.
            store=True,
        )
        _append_transcript(transcript, response.model_dump(mode="json"))
        input_items.extend(response.output)
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            final_text = response.output_text
            break
        for call in calls:
            if actions >= arguments.max_actions:
                break
            actions += 1
            try:
                payload = json.loads(call.arguments)
                output = _shell(
                    container, str(payload["command"]), int(payload["timeout_seconds"])
                )
            except Exception as exc:
                output = json.dumps({"error": str(exc)})
            input_items.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": output}
            )
    return actions, final_text


def _run_anthropic_episode(
    arguments: argparse.Namespace, container: str, transcript: Path
) -> tuple[int, str]:
    import anthropic

    client = anthropic.Anthropic(
        api_key=_api_key(arguments.env_file, "ANTHROPIC_API_KEY"),
        timeout=900.0,
        max_retries=3,
    )
    request: dict[str, Any] = {
        "model": arguments.model,
        "max_tokens": arguments.max_output_tokens,
        "system": INSTRUCTIONS,
        "tools": [ANTHROPIC_SHELL_TOOL],
        "output_config": {"effort": arguments.reasoning_effort},
        # Auto-cache the growing conversation prefix across loop iterations.
        "cache_control": {"type": "ephemeral"},
    }
    if "fable" not in arguments.model and "mythos" not in arguments.model:
        # Fable 5 rejects any explicit thinking config (always on); Opus needs opt-in.
        request["thinking"] = {"type": "adaptive"}
    prompt = USER_PROMPT
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    def _turn() -> Any:
        with client.messages.stream(**request, messages=messages) as stream:
            return stream.get_final_message()

    actions = 0
    final_text = ""
    while actions < arguments.max_actions:
        response = _with_retry(_turn)
        _append_transcript(transcript, response.model_dump(mode="json"))
        if response.stop_reason == "refusal":
            final_text = "[refusal stop_reason — no submission produced by this turn]"
            break
        # Echo content blocks (including thinking) back unchanged for the next turn.
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            break
        results: list[dict[str, Any]] = []
        for block in tool_uses:
            if actions >= arguments.max_actions:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": "action budget exhausted"}),
                        "is_error": True,
                    }
                )
                continue
            actions += 1
            try:
                output = _shell(
                    container,
                    str(block.input["command"]),
                    max(1, min(600, int(block.input["timeout_seconds"]))),
                )
            except Exception as exc:
                output = json.dumps({"error": str(exc)})
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )
        # All tool results for one assistant turn go back in a single user message.
        messages.append({"role": "user", "content": results})
    return actions, final_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="dotenv file holding OPENAI_API_KEY / ANTHROPIC_API_KEY (kept on the host; "
        "never mounted into any container)",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--max-output-tokens", type=int, default=32_768)
    parser.add_argument("--runner-image", default="cam1k-runner:latest")
    parser.add_argument("--grade-timeout", type=int, default=240)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()

    if not (ROOT / "data" / "benchmark" / "private" / "manifest.json").is_file():
        raise RuntimeError("benchmark data is missing; run cam1k-eval data build")
    model_slug = re.sub(r"[^a-z0-9]+", "-", arguments.model.lower()).strip("-")
    episode_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{model_slug}-{uuid.uuid4().hex[:8]}"
    episode_root = ROOT / "episodes" / episode_id
    workspace = episode_root / "workspace"
    transcript = episode_root / "responses.jsonl"
    _copy_public_workspace(workspace)
    episode_root.mkdir(parents=True, exist_ok=True)

    container = f"cam1k-agent-{episode_id.lower()}"
    _start_agent(workspace, container)

    runner = (
        _run_anthropic_episode if arguments.model.startswith("claude") else _run_openai_episode
    )
    actions = 0
    final_text = ""
    api_error: str | None = None
    try:
        actions, final_text = runner(arguments, container, transcript)
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
    finally:
        _run(["docker", "rm", "--force", container], check=False)

    report: dict[str, Any] = {
        "episode_id": episode_id,
        "model": arguments.model,
        "reasoning_effort": arguments.reasoning_effort,
        "actions": actions,
        "model_final_text": final_text,
        "api_error": api_error,
    }
    submission = workspace / "submission"
    try:
        frozen = episode_root / "frozen-submission"
        report["freeze"] = freeze_submission(submission, frozen)
        report["grade"] = grade_submission(
            frozen,
            ROOT / "data" / "benchmark" / "private",
            image=arguments.runner_image,
            seed=arguments.seed,
            timeout_seconds=arguments.grade_timeout,
        )
    except Exception as exc:
        report["freeze_or_grade_error"] = f"{type(exc).__name__}: {exc}"
    report_path = ROOT / "reports" / f"{episode_id}.json"
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
