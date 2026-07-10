"""Private grader that never executes candidate code in its own process.

Each private case is materialized as a raw capture whose sound rows are shuffled by a
fresh secret permutation, with ``channelOrdering`` in the accompanying metadata
composed to match. A program that honestly honors the device metadata is unaffected;
one that hard-codes the visible capture's row order fails. Grading is skill-based:
see ``scoring`` for the uninformed-floor construction.
"""

from __future__ import annotations

import hashlib
import secrets
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .constants import BAND_CENTERS
from .formats import (
    expected_map_shape,
    load_json,
    load_npy,
    open_capture,
    sha256_file,
    validate_maps,
    write_capture,
)
from .runner import CandidateRunError, run_candidate_docker
from .scoring import (
    aggregate_case_scores,
    build_floor_family,
    canonicalize_batch,
    case_floors,
    score_case_skill,
)


def _zero_metrics(error: str) -> dict[str, Any]:
    return {
        "map_skill": 0.0,
        "hotspot_skill": 0.0,
        "raw_map_score": 0.0,
        "raw_hotspot_score": 0.0,
        "ssim": 0.0,
        "correlation": 0.0,
        "l1_similarity": 0.0,
        "blank_fraction": 1.0,
        "mean_floor": 0.0,
        "frame_band_count": 0,
        "error": error[:4_096],
    }


def _materialize_secret_capture(
    case_root: Path,
    destination: Path,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Write a permuted copy of a private raw case for one candidate run."""
    capture = open_capture(case_root)
    permutation = rng.permutation(capture["sound"].shape[0])
    permuted_sound = np.ascontiguousarray(capture["sound"][permutation]).astype("<i4")
    permuted_ordering = capture["ordering"][permutation]
    write_capture(destination / "capture", permuted_sound, capture["container"], permuted_ordering)
    for value in (destination / "capture").iterdir():
        value.chmod(0o444)
    (destination / "capture").chmod(0o555)
    destination.chmod(0o555)
    return {
        "channel_permutation_sha256": hashlib.sha256(
            permutation.astype("<i4").tobytes()
        ).hexdigest(),
        "num_samples": int(permuted_sound.shape[1]),
    }


def _climatologies(references: dict[str, np.ndarray], kinds: dict[str, str]) -> list[np.ndarray]:
    """Per-band mean canonical maps over hidden reference groups.

    Strong empirical floors for audio-independent strategies, including ones the
    explicit family does not enumerate; not a formal optimum of the composite metric.
    """
    groups: dict[str, list[str]] = {"all": list(references)}
    for name, kind in kinds.items():
        groups.setdefault(kind, []).append(name)
        prefix = name.rsplit("_", 1)[0]
        if kind == "real":
            groups.setdefault(prefix, []).append(name)
    climatologies = []
    for members in groups.values():
        if not members:
            continue
        stacked = [
            canonicalize_batch(
                references[name].reshape(-1, *references[name].shape[2:])
            ).reshape(references[name].shape).mean(axis=0)
            for name in members
        ]
        climatologies.append(np.mean(stacked, axis=0))
    return climatologies


def grade_submission(
    submission: str | Path,
    private_root: str | Path,
    *,
    public_root: str | Path | None = None,
    image: str = "cam1k-runner:latest",
    seed: int | None = None,
    timeout_seconds: int = 240,
) -> dict[str, Any]:
    from .submission import validate_submission

    submission = Path(submission).resolve()
    private_root = Path(private_root).resolve()
    public_root = (
        Path(public_root).resolve() if public_root else private_root.parent / "public"
    )
    submission_manifest = validate_submission(submission)
    image_result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if image_result.returncode:
        raise RuntimeError(f"runner image is unavailable: {image}: {image_result.stderr.strip()}")
    image_id = image_result.stdout.strip()
    private_manifest = load_json(private_root / "manifest.json")
    effective_seed = secrets.randbits(63) if seed is None else seed
    rng = np.random.default_rng(effective_seed)

    anchor = load_npy(
        public_root / "vendor_export" / "maps.npy", max_bytes=64 * 1024 * 1024, mmap_mode=None
    )
    references = {
        case["name"]: np.asarray(
            load_npy(
                private_root / "cases" / case["name"] / "reference_maps.npy",
                max_bytes=64 * 1024 * 1024,
                mmap_mode=None,
            ),
            dtype=np.float32,
        )
        for case in private_manifest["cases"]
    }
    kinds = {case["name"]: case["kind"] for case in private_manifest["cases"]}
    family = build_floor_family(anchor, _climatologies(references, kinds))
    centers = np.asarray(BAND_CENTERS, dtype=np.float64)

    records: list[dict[str, Any]] = []
    for case in private_manifest["cases"]:
        case_root = private_root / "cases" / case["name"]
        reference = references[case["name"]]
        floors = case_floors(family, reference, centers)
        with (
            tempfile.TemporaryDirectory(prefix="cam1k-private-input-") as input_tmp,
            tempfile.TemporaryDirectory(prefix="cam1k-private-output-") as output_tmp,
        ):
            input_root = Path(input_tmp)
            output_root = Path(output_tmp)
            transform = _materialize_secret_capture(case_root, input_root, rng)
            execution: dict[str, Any] | None = None
            try:
                execution = run_candidate_docker(
                    submission,
                    input_root,
                    output_root,
                    image=image,
                    timeout_seconds=timeout_seconds,
                )
                expected = expected_map_shape(transform["num_samples"])
                output = output_root / "maps.npy"
                maps = load_npy(
                    output, max_bytes=int(np.prod(expected)) * 4 + 1_024, mmap_mode=None
                )
                validate_maps(maps, expected)
                metrics = score_case_skill(np.asarray(maps), reference, floors)
                output_hash = sha256_file(output)
            except (CandidateRunError, FileNotFoundError, ValueError) as exc:
                metrics = _zero_metrics(str(exc))
                output_hash = None
            input_root.chmod(0o755)
            (input_root / "capture").chmod(0o755)
            records.append(
                {
                    "name": case["name"],
                    "kind": case["kind"],
                    "metrics": metrics,
                    "execution": execution,
                    "output_sha256": output_hash,
                    "transform": transform,
                }
            )
    aggregate = aggregate_case_scores(records)
    aggregate.update(
        {
            "schema_version": "cam1k-grade-v2-skill",
            "seed": effective_seed,
            "runner_image": image,
            "runner_image_id": image_id,
            "submission": submission_manifest,
            "private_manifest_sha256": sha256_file(private_root / "manifest.json"),
        }
    )
    return aggregate
