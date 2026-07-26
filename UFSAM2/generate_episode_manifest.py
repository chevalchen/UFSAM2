"""Generate disjoint, replayable EXP-001 episode partitions on the data server."""

import argparse
import subprocess

import opts
from datasets import build_dataset
from util.episode_manifest import (
    file_sha256,
    generate_episode_manifest,
    write_episode_manifest,
)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(args: argparse.Namespace) -> None:
    if args.dataset_file == "multi":
        raise ValueError("Episode manifests must name one concrete dataset.")
    dataset = build_dataset(args.dataset_file, image_set=args.manifest_source_split, args=args)
    counts = {
        "train": args.manifest_train_episodes,
        "calibration": args.manifest_calibration_episodes,
        "validation": args.manifest_validation_episodes,
    }
    manifest = generate_episode_manifest(
        dataset,
        dataset_name=args.dataset_file,
        source_split=args.manifest_source_split,
        fold=args.fold,
        shots=args.shots,
        counts=counts,
        seed=args.seed,
        data_root=args.data_root,
        source_git_sha=_git_sha(),
        max_attempts_per_episode=args.manifest_max_attempts_per_episode,
    )
    write_episode_manifest(manifest, args.manifest_output)
    print(f"Wrote {args.manifest_output}")
    print(f"SHA256: {file_sha256(args.manifest_output)}")
    print(f"Partition counts: {manifest['partition_counts']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate deterministic EXP-001 episode partitions",
        parents=[opts.get_args_parser()],
    )
    parser.add_argument("--manifest_output", required=True)
    parser.add_argument("--manifest_source_split", default="train")
    parser.add_argument("--manifest_train_episodes", type=int, required=True)
    parser.add_argument("--manifest_calibration_episodes", type=int, required=True)
    parser.add_argument("--manifest_validation_episodes", type=int, required=True)
    parser.add_argument("--manifest_max_attempts_per_episode", type=int, default=500)
    main(parser.parse_args())
