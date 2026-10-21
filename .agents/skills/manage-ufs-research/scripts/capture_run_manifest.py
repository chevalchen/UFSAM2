#!/usr/bin/env python3
"""Create an immutable run manifest and refuse uncommitted experiment code."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path | None = None, required: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if required and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(f"Command failed: {' '.join(command)}\n{detail}")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def tracked_file(path_text: str, label: str, repo: Path) -> dict[str, object]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside the Git repository: {path}") from exc
    run(["git", "ls-files", "--error-unmatch", "--", relative], cwd=repo)
    return {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def external_file(path_text: str, label: str, repo: Path) -> dict[str, object]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    return {
        "path": display_path(path, repo),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def environment_fingerprint() -> dict[str, object]:
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpus"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # Environment capture must not hide the Git evidence.
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a clean Git snapshot and write a reproducible run manifest."
    )
    parser.add_argument("--experiment", required=True, help="Stable Experiment ID.")
    parser.add_argument("--config", required=True, help="Tracked resolved experiment config.")
    parser.add_argument("--episodes", required=True, help="Tracked episode/split manifest.")
    parser.add_argument("--checkpoint", help="Input checkpoint to hash, if used.")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional immutable input file to hash; repeat as needed.",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--command", required=True, help="Exact command that will be run.")
    parser.add_argument("--output", required=True, help="Manifest path outside the repository.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(run(["git", "rev-parse", "--show-toplevel"])).resolve()
    dirty = run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo)
    if dirty:
        raise SystemExit(f"Refusing formal run from a dirty worktree:\n{dirty}")

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing run manifest: {output}")
    try:
        output.relative_to(repo)
    except ValueError:
        pass
    else:
        raise SystemExit("Run manifests and outputs must be written outside the Git repository.")

    inputs: dict[str, object] = {
        "config": tracked_file(args.config, "config", repo),
        "episodes": tracked_file(args.episodes, "episodes", repo),
    }
    if args.checkpoint:
        inputs["checkpoint"] = external_file(args.checkpoint, "checkpoint", repo)
    for item in args.artifact:
        if "=" not in item:
            raise SystemExit(f"Invalid --artifact value, expected NAME=PATH: {item}")
        name, path = item.split("=", 1)
        if not name or name in inputs:
            raise SystemExit(f"Invalid or duplicate artifact name: {name!r}")
        inputs[name] = external_file(path, f"artifact {name}", repo)

    branch = run(["git", "branch", "--show-current"], cwd=repo, required=False) or None
    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": args.experiment,
        "git": {
            "commit": run(["git", "rev-parse", "HEAD"], cwd=repo),
            "branch": branch,
            "clean": True,
        },
        "run": {
            "seed": args.seed,
            "fold": str(args.fold),
            "command": args.command,
        },
        "inputs": inputs,
        "environment": environment_fingerprint(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
