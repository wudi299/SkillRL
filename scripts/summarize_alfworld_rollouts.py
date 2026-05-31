#!/usr/bin/env python3
"""Summarize and select ALFWorld rollout txt files.

This uses the same parser as the SFT pipeline, so the all-success/all-fail
counts match the data selection logic used before memory and SFT generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.sft_data_generation.preprocess.parse_alfworld import parse_trajectory_file


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _summarize_env(env_dir: Path, min_trajs_per_env: int) -> dict[str, Any] | None:
    traj_files = sorted(env_dir.glob("*.txt"))
    if len(traj_files) < min_trajs_per_env:
        return None

    parsed = [parse_trajectory_file(str(path)) for path in traj_files]
    success = [item for item in parsed if item["is_success"]]
    fail = [item for item in parsed if not item["is_success"]]
    status = "mixed"
    if len(success) >= min_trajs_per_env and not fail:
        status = "all_success"
    elif len(fail) >= min_trajs_per_env and not success:
        status = "all_fail"

    lengths = [item["length"] for item in parsed]
    return {
        "env_id": env_dir.name,
        "status": status,
        "trajectory_count": len(parsed),
        "success_count": len(success),
        "fail_count": len(fail),
        "avg_length": mean(lengths) if lengths else 0,
        "task": next((item.get("task_desc", "") for item in parsed if item.get("task_desc")), ""),
        "files": [str(path) for path in traj_files],
        "success_files": [item["file_path"] for item in success],
        "fail_files": [item["file_path"] for item in fail],
    }


def _write_report(
    output_path: Path,
    input_dir: Path,
    summaries: list[dict[str, Any]],
    target_success_envs: int,
    target_fail_envs: int,
) -> None:
    all_success = [item for item in summaries if item["status"] == "all_success"]
    all_fail = [item for item in summaries if item["status"] == "all_fail"]
    mixed = [item for item in summaries if item["status"] == "mixed"]

    total_trajectories = sum(item["trajectory_count"] for item in summaries)
    total_success = sum(item["success_count"] for item in summaries)
    total_fail = sum(item["fail_count"] for item in summaries)
    avg_len = mean([item["avg_length"] for item in summaries]) if summaries else 0

    lines = [
        "# ALFWorld Rollout Summary",
        "",
        "## Summary",
        f"- Input dir: {input_dir}",
        f"- Env dirs scanned: {len(summaries)}",
        f"- Total trajectories: {total_trajectories}",
        f"- Successful trajectories: {total_success}",
        f"- Failed trajectories: {total_fail}",
        f"- Average trajectory length: {avg_len:.2f}",
        "",
        "## Env Classification",
        f"- All-success envs: {len(all_success)} / target {target_success_envs}",
        f"- All-fail envs: {len(all_fail)} / target {target_fail_envs}",
        f"- Mixed envs: {len(mixed)}",
        "",
        "## Readiness",
    ]
    ready = len(all_success) >= target_success_envs and len(all_fail) >= target_fail_envs
    if ready:
        lines.append("- Ready for 113 success / 110 fail selection: yes")
    else:
        lines.append("- Ready for 113 success / 110 fail selection: no")
        lines.append(f"- More all-success envs needed: {max(0, target_success_envs - len(all_success))}")
        lines.append(f"- More all-fail envs needed: {max(0, target_fail_envs - len(all_fail))}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target_success_envs", type=int, default=113)
    parser.add_argument("--target_fail_envs", type=int, default=110)
    parser.add_argument("--min_trajs_per_env", type=int, default=3)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env_dirs = sorted(path for path in input_dir.glob("env*") if path.is_dir())
    summaries = []
    for env_dir in env_dirs:
        summary = _summarize_env(env_dir, args.min_trajs_per_env)
        if summary is not None:
            summaries.append(summary)

    all_success = [item for item in summaries if item["status"] == "all_success"]
    all_fail = [item for item in summaries if item["status"] == "all_fail"]
    selected = {
        "input_dir": str(input_dir),
        "target_success_envs": args.target_success_envs,
        "target_fail_envs": args.target_fail_envs,
        "selected_success_envs": all_success[: args.target_success_envs],
        "selected_fail_envs": all_fail[: args.target_fail_envs],
    }

    _write_json(output_dir / "rollout_summary.json", {"envs": summaries})
    _write_json(output_dir / "selected_rollouts_113_success_110_fail.json", selected)
    _write_report(
        output_dir / "rollout_pilot_report.md",
        input_dir,
        summaries,
        args.target_success_envs,
        args.target_fail_envs,
    )

    print(
        "all_success={success} all_fail={fail} mixed={mixed}".format(
            success=len(all_success),
            fail=len(all_fail),
            mixed=sum(1 for item in summaries if item["status"] == "mixed"),
        )
    )
    print(f"wrote {output_dir / 'rollout_pilot_report.md'}")


if __name__ == "__main__":
    main()
