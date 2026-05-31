"""Select individual ALFWorld rollout txt files for memory/SFT generation.

This is trajectory-level selection: each selected txt becomes one downstream
record with one trajectory. It intentionally keeps successful and failed
rollouts from mixed envs instead of requiring every trajectory in an env to
share the same outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
SFT_DIR = REPO_DIR / "examples" / "sft_data_generation"
sys.path.insert(0, str(SFT_DIR / "preprocess"))

from parse_alfworld import parse_trajectory_file  # noqa: E402


def dump_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def infer_work_dir(output_file: Path) -> Path:
    parent = output_file.resolve().parent
    if parent.name == "01_processed":
        return parent.parent
    return parent


def sample_name(relative_path: str, index: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in relative_path)
    return f"{index:03d}_{safe}.txt"


def build_record(
    *,
    parsed: dict,
    source_path: Path,
    input_dir: Path,
    outcome: str,
    index: int,
) -> tuple[dict, dict]:
    source_env_id = source_path.parent.name
    source_file = source_path.name
    pseudo_env_id = f"traj_{outcome.lower()}_{index:03d}"
    relative_path = os.path.relpath(source_path, input_dir)

    record = {
        "env_id": pseudo_env_id,
        "task": parsed.get("task_desc", ""),
        "type": "all_success" if outcome == "Success" else "all_fail",
        "trajectories": [parsed.get("steps", [])],
        "source": {
            "path": str(source_path),
            "relative_path": relative_path,
            "origin_env_id": source_env_id,
            "origin_file": source_file,
            "outcome": outcome,
        },
    }
    selected_file = {
        "index": index,
        "pseudo_env_id": pseudo_env_id,
        "outcome": outcome,
        "path": str(source_path),
        "relative_path": relative_path,
        "origin_env_id": source_env_id,
        "origin_file": source_file,
        "size_bytes": source_path.stat().st_size,
        "trajectory_length": parsed.get("length", 0),
        "task": parsed.get("task_desc", ""),
    }
    return record, selected_file


def write_raw_artifacts(
    *,
    input_dir: Path,
    raw_stage_dir: Path,
    selected_files: list[dict],
    sample_limit: int,
) -> None:
    samples_dir = raw_stage_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dump_json({"rollout_dir": str(input_dir), "files": selected_files}, raw_stage_dir / "selected_files.json")
    for i, row in enumerate(selected_files[:sample_limit], 1):
        source = Path(row["path"])
        target = samples_dir / sample_name(row["relative_path"], i)
        shutil.copyfile(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Root directory containing env*/test*.txt rollouts.")
    parser.add_argument("--output_file", required=True, help="Downstream processed_trajectories.json path.")
    parser.add_argument("--target_success", type=int, default=113)
    parser.add_argument("--target_fail", type=int, default=110)
    parser.add_argument("--sample_limit", type=int, default=6)
    parser.add_argument(
        "--raw_stage_dir",
        default=None,
        help="Optional 00_raw_rollouts artifact directory. Defaults to WORK_DIR/00_raw_rollouts when output is under 01_processed.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_file = Path(args.output_file).resolve()
    work_dir = infer_work_dir(output_file)
    processed_dir = output_file.parent
    raw_stage_dir = Path(args.raw_stage_dir).resolve() if args.raw_stage_dir else work_dir / "00_raw_rollouts"

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    txt_files = sorted(input_dir.glob("env*/*.txt"))
    success: list[tuple[Path, dict]] = []
    fail: list[tuple[Path, dict]] = []
    skipped: list[dict] = []

    for path in txt_files:
        try:
            parsed = parse_trajectory_file(str(path))
        except Exception as exc:
            skipped.append({"path": str(path), "reason": f"parse_error: {exc}"})
            continue
        if not parsed.get("steps"):
            skipped.append({"path": str(path), "reason": "no parsed steps"})
            continue
        target = success if parsed.get("is_success") else fail
        target.append((path.resolve(), parsed))

    if len(success) < args.target_success or len(fail) < args.target_fail:
        raise SystemExit(
            "Not enough rollout trajectories: "
            f"success={len(success)}/{args.target_success}, fail={len(fail)}/{args.target_fail}. "
            "Generate more rollouts before running SFT generation."
        )

    selected_records: list[dict] = []
    selected_files: list[dict] = []

    for idx, (path, parsed) in enumerate(success[: args.target_success], 1):
        record, row = build_record(
            parsed=parsed,
            source_path=path,
            input_dir=input_dir,
            outcome="Success",
            index=idx,
        )
        selected_records.append(record)
        selected_files.append(row)

    for idx, (path, parsed) in enumerate(fail[: args.target_fail], 1):
        record, row = build_record(
            parsed=parsed,
            source_path=path,
            input_dir=input_dir,
            outcome="Failure",
            index=idx,
        )
        selected_records.append(record)
        selected_files.append(row)

    processed_dir.mkdir(parents=True, exist_ok=True)
    dump_json(selected_records, output_file)
    dump_json({"rollout_dir": str(input_dir), "files": selected_files}, processed_dir / "selected_rollout_files.json")
    if skipped:
        dump_json(skipped, processed_dir / "skipped_rollout_files.json")

    write_raw_artifacts(
        input_dir=input_dir,
        raw_stage_dir=raw_stage_dir,
        selected_files=selected_files,
        sample_limit=args.sample_limit,
    )

    counts = Counter(item["type"] for item in selected_records)
    report = "\n".join(
        [
            "# ALFWorld trajectory-level rollout selection",
            "",
            f"- Input rollout dir: `{input_dir}`",
            f"- Output file: `{output_file}`",
            f"- Available successful trajectories: {len(success)}",
            f"- Available failed trajectories: {len(fail)}",
            f"- Selected successful trajectories: {counts.get('all_success', 0)}",
            f"- Selected failed trajectories: {counts.get('all_fail', 0)}",
            f"- Skipped files: {len(skipped)}",
            "",
            "Each selected txt is represented as one pseudo-env with one trajectory.",
        ]
    )
    (processed_dir / "selection_report.md").write_text(report + "\n", encoding="utf-8")

    print(
        "selected_success={success} selected_fail={fail} total={total}".format(
            success=counts.get("all_success", 0),
            fail=counts.get("all_fail", 0),
            total=len(selected_records),
        )
    )
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
