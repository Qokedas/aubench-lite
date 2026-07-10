"""Host-side CLI for CAM1K Beamforming Eval."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .dataset import build_benchmark, download_sources, inspect_rosbag
from .grader import grade_submission


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="inspect, download, or build data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    inspect_parser = data_commands.add_parser("inspect-bag")
    inspect_parser.add_argument("bag")
    download_parser = data_commands.add_parser("download")
    download_parser.add_argument("--root", default="data/source")
    download_parser.add_argument("--capture", action="append", dest="captures")
    build_parser = data_commands.add_parser("build")
    build_parser.add_argument("--source", default="data/source")
    build_parser.add_argument("--output", default="data/benchmark")

    grade = commands.add_parser("grade", help="run a frozen submission on private cases")
    grade.add_argument("submission")
    grade.add_argument("--private", default="data/benchmark/private")
    grade.add_argument("--public", default="data/benchmark/public")
    grade.add_argument("--image", default="cam1k-runner:latest")
    grade.add_argument("--seed", type=int)
    grade.add_argument("--timeout", type=int, default=240)
    grade.add_argument("--output")

    doctor = commands.add_parser("doctor", help="check local runtime prerequisites")
    doctor.add_argument("--bag", help="optional ROS1 bag to inspect for CAM1K pressure topics")

    arguments = parser.parse_args()
    if arguments.command == "data" and arguments.data_command == "inspect-bag":
        report = inspect_rosbag(arguments.bag)
        _print(report)
        if not report["contains_cam1k_pressure"]:
            raise SystemExit(
                "This ROS bag has no CAM1K pressure topic. "
                "Use the dataset set3 sound/container files."
            )
    elif arguments.command == "data" and arguments.data_command == "download":
        _print(download_sources(arguments.root, captures=arguments.captures))
    elif arguments.command == "data" and arguments.data_command == "build":
        _print(build_benchmark(arguments.source, arguments.output))
    elif arguments.command == "grade":
        report = grade_submission(
            arguments.submission,
            arguments.private,
            public_root=arguments.public,
            image=arguments.image,
            seed=arguments.seed,
            timeout_seconds=arguments.timeout,
        )
        if arguments.output:
            Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
            Path(arguments.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _print(report)
    elif arguments.command == "doctor":
        docker = subprocess.run(
            ["docker", "version", "--format", "{{.Client.Version}} {{.Server.Version}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        report = {
            "docker": {
                "ok": docker.returncode == 0,
                "version": docker.stdout.strip(),
                "error": docker.stderr.strip(),
            },
        }
        if arguments.bag:
            bag_path = Path(arguments.bag)
            report["bag"] = (
                inspect_rosbag(bag_path) if bag_path.is_file() else {"missing": str(bag_path)}
            )
        _print(report)


if __name__ == "__main__":
    main()
