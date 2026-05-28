"""Create samples, reports, and archives for auditable SkillRL runs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_utils import dump_json, ensure_dir, load_json, utc_now


KNOWN_PRICES_PER_1K = {
    "gpt-4o": {"input": 0.00125, "output": 0.0100},
}


def _sample_name(prefix: str, index: int, suffix: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix)
    return f"{safe}_sample_{index:03d}{suffix}"


def _truncate_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated]..."


def command_raw(args: argparse.Namespace) -> None:
    stage_dir = Path(args.stage_dir)
    samples_dir = stage_dir / "samples"
    ensure_dir(str(samples_dir))

    files = sorted(glob.glob(os.path.join(args.rollout_dir, "**", "*.txt"), recursive=True))
    selected = files[: args.limit] if args.limit and args.limit > 0 else files

    rows = [
        {
            "index": i,
            "path": path,
            "relative_path": os.path.relpath(path, args.rollout_dir),
            "size_bytes": os.path.getsize(path),
        }
        for i, path in enumerate(selected)
    ]
    dump_json({"rollout_dir": args.rollout_dir, "files": rows}, str(stage_dir / "selected_files.json"))

    for i, path in enumerate(selected[: args.sample_limit], 1):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = _truncate_text(f.read())
        rel = os.path.relpath(path, args.rollout_dir)
        out = samples_dir / _sample_name(rel, i, ".txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)


def _json_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def summarize_json(data: Any) -> dict[str, Any]:
    items = _json_items(data)
    summary: dict[str, Any] = {"top_level_type": type(data).__name__, "item_count": len(items)}
    if items and isinstance(items[0], dict):
        key_counts = Counter()
        type_counts = Counter()
        for item in items:
            key_counts.update(item.keys())
            value = item.get("type") or item.get("outcome")
            if value:
                type_counts[str(value)] += 1
        summary["keys"] = dict(key_counts)
        summary["type_or_outcome_counts"] = dict(type_counts)
    return summary


def command_sample_json(args: argparse.Namespace) -> None:
    stage_dir = Path(args.stage_dir)
    samples_dir = stage_dir / "samples"
    ensure_dir(str(samples_dir))
    data = load_json(args.input)
    dump_json(summarize_json(data), str(stage_dir / "summary.json"))
    for i, item in enumerate(_json_items(data)[: args.limit], 1):
        dump_json(item, str(samples_dir / _sample_name(args.name, i, ".json")))


def command_skill_bank(args: argparse.Namespace) -> None:
    stage_dir = Path(args.stage_dir)
    ensure_dir(str(stage_dir))
    skill_bank = load_json(args.input)
    dump_json(skill_bank.get("general_skills", []), str(stage_dir / "general_skills.json"))
    category_key = "task_specific_skills"
    if "query_type_skills" in skill_bank:
        category_key = "query_type_skills"
    dump_json(skill_bank.get(category_key, {}), str(stage_dir / "task_specific_skills.json"))
    dump_json(skill_bank.get("common_mistakes", []), str(stage_dir / "common_mistakes.json"))


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return load_json(str(path))


def _count_llm_calls(work_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    files = sorted(work_dir.glob("**/llm_calls.jsonl"))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                model = record.get("model") or "unknown"
                usage = record.get("usage") or {}
                totals[model]["calls"] += 1
                totals[model]["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                totals[model]["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                totals[model]["total_tokens"] += int(usage.get("total_tokens") or 0)
    cost = {"models": {}, "known_model_estimated_usd": 0.0, "note": "USD estimates are only computed for models listed in KNOWN_PRICES_PER_1K."}
    for model, row in totals.items():
        price = KNOWN_PRICES_PER_1K.get(model)
        estimated = None
        if price:
            estimated = row["prompt_tokens"] / 1000 * price["input"] + row["completion_tokens"] / 1000 * price["output"]
            cost["known_model_estimated_usd"] += estimated
        cost["models"][model] = {**row, "estimated_usd": estimated}
    return cost, totals


def _first(items: Any) -> Any:
    return items[0] if isinstance(items, list) and items else None


def _json_preview(value: Any, max_chars: int = 2500) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return _truncate_text(text, max_chars)


def command_report(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    report_dir = Path(args.output_dir)
    ensure_dir(str(report_dir))

    processed = _load_json_if_exists(work_dir / "01_processed" / "processed_trajectories.json")
    cleaned = _load_json_if_exists(work_dir / "01_processed" / "processed_trajectories_cleaned.json")
    memories = _load_json_if_exists(work_dir / "02_memory" / "generated_memories.json")
    skill_bank = _load_json_if_exists(work_dir / "03_skill_bank" / "skill_bank.json") or {}
    distilled = _load_json_if_exists(work_dir / "04_distillation" / "distilled_trajectories.json")
    sft = _load_json_if_exists(work_dir / "05_sft_data" / "alfworld_sft_data.json")
    raw_files = _load_json_if_exists(work_dir / "00_raw_rollouts" / "selected_files.json") or {"files": []}

    counts = {
        "raw_rollout_files": len(raw_files.get("files", [])),
        "processed_envs": len(processed or []),
        "cleaned_envs": len(cleaned or []),
        "generated_memories": len(memories or []),
        "distilled_trajectories": len(distilled or []),
        "sft_examples": len(sft or []),
        "general_skills": len(skill_bank.get("general_skills", [])),
        "task_specific_skills": sum(len(v) for v in skill_bank.get("task_specific_skills", {}).values()),
        "common_mistakes": len(skill_bank.get("common_mistakes", [])),
    }
    dump_json(counts, str(report_dir / "counts.json"))

    cost, _ = _count_llm_calls(work_dir)
    dump_json(cost, str(report_dir / "cost_estimate.json"))

    sample_rollout_text = ""
    sample_txts = sorted((work_dir / "00_raw_rollouts" / "samples").glob("*.txt"))
    if sample_txts:
        sample_rollout_text = sample_txts[0].read_text(encoding="utf-8", errors="replace")

    task_skills = skill_bank.get("task_specific_skills", {})
    first_task_skill = None
    first_task_cat = None
    for cat, skills in task_skills.items():
        if skills:
            first_task_cat = cat
            first_task_skill = skills[0]
            break

    summary = f"""# SkillRL ALFWorld 可追踪运行报告

## 基本信息
- Run ID: `{args.run_id}`
- 工作目录: `{work_dir}`
- 轨迹输入目录: `{args.rollout_dir}`
- Memory model: `{args.memory_model}`
- Aggregate model: `{args.aggregate_model}`
- Distill model: `{args.distill_model}`

## 数量统计
- 原始 rollout txt 文件: {counts['raw_rollout_files']}
- 解析后 env 条目: {counts['processed_envs']}
- 去重后 env 条目: {counts['cleaned_envs']}
- 老师模型生成 memory: {counts['generated_memories']}
- 蒸馏 ShareGPT 轨迹: {counts['distilled_trajectories']}
- 冷启动 SFT 样本: {counts['sft_examples']}
- 通用 skill: {counts['general_skills']}
- 专精 skill: {counts['task_specific_skills']}
- 常见错误: {counts['common_mistakes']}

## 原始轨迹样例
```text
{_truncate_text(sample_rollout_text, 2500)}
```

## 解析后的轨迹样例
```json
{_json_preview(_first(cleaned))}
```

## 老师模型总结后的 Memory 样例
```json
{_json_preview(_first(memories))}
```

## 冷启动 SFT 数据样例
```json
{_json_preview(_first(sft))}
```

## 通用 Skill 样例
```json
{_json_preview(_first(skill_bank.get('general_skills', [])))}
```

## 专精 Skill 样例
- Category: `{first_task_cat}`

```json
{_json_preview(first_task_skill)}
```

## 关键文件
- `00_raw_rollouts/selected_files.json`
- `01_processed/processed_trajectories_cleaned.json`
- `02_memory/generated_memories.json`
- `03_skill_bank/skill_bank.json`
- `04_distillation/distilled_trajectories.json`
- `05_sft_data/alfworld_sft_data.json`
- `06_report/counts.json`
- `06_report/cost_estimate.json`

## LLM Trace
- Memory trace: `02_memory/llm_calls.jsonl`
- Skill aggregation trace: `03_skill_bank/llm_calls.jsonl`
- Distillation trace: `04_distillation/llm_calls.jsonl`
"""
    (report_dir / "summary.md").write_text(summary, encoding="utf-8")


def command_manifest(args: argparse.Namespace) -> None:
    data = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "repo_dir": args.repo_dir,
        "work_dir": args.work_dir,
        "rollout_dir": args.rollout_dir,
        "models": {
            "memory": args.memory_model,
            "aggregate": args.aggregate_model,
            "distill": args.distill_model,
        },
        "limit": args.limit,
        "trace_llm": args.trace_llm,
        "stages": [
            "00_raw_rollouts",
            "01_processed",
            "02_memory",
            "03_skill_bank",
            "04_distillation",
            "05_sft_data",
            "06_report",
        ],
    }
    dump_json(data, os.path.join(args.work_dir, "manifest.json"))


def command_package(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    output_path = work_dir / f"skillrl_run_{args.run_id}.tar.gz"
    with tarfile.open(output_path, "w:gz") as tar:
        for path in work_dir.rglob("*"):
            if path == output_path:
                continue
            tar.add(path, arcname=str(path.relative_to(work_dir.parent)))
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    raw = sub.add_parser("raw")
    raw.add_argument("--rollout-dir", required=True)
    raw.add_argument("--stage-dir", required=True)
    raw.add_argument("--limit", type=int, default=0)
    raw.add_argument("--sample-limit", type=int, default=3)
    raw.set_defaults(func=command_raw)

    sample = sub.add_parser("sample-json")
    sample.add_argument("--input", required=True)
    sample.add_argument("--stage-dir", required=True)
    sample.add_argument("--name", required=True)
    sample.add_argument("--limit", type=int, default=3)
    sample.set_defaults(func=command_sample_json)

    skill = sub.add_parser("skill-bank")
    skill.add_argument("--input", required=True)
    skill.add_argument("--stage-dir", required=True)
    skill.set_defaults(func=command_skill_bank)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--repo-dir", required=True)
    manifest.add_argument("--work-dir", required=True)
    manifest.add_argument("--rollout-dir", required=True)
    manifest.add_argument("--memory-model", required=True)
    manifest.add_argument("--aggregate-model", required=True)
    manifest.add_argument("--distill-model", required=True)
    manifest.add_argument("--limit", required=True)
    manifest.add_argument("--trace-llm", required=True)
    manifest.set_defaults(func=command_manifest)

    report = sub.add_parser("report")
    report.add_argument("--work-dir", required=True)
    report.add_argument("--run-id", required=True)
    report.add_argument("--rollout-dir", required=True)
    report.add_argument("--output-dir", required=True)
    report.add_argument("--memory-model", required=True)
    report.add_argument("--aggregate-model", required=True)
    report.add_argument("--distill-model", required=True)
    report.set_defaults(func=command_report)

    package = sub.add_parser("package")
    package.add_argument("--work-dir", required=True)
    package.add_argument("--run-id", required=True)
    package.set_defaults(func=command_package)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
