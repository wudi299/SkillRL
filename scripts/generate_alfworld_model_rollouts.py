#!/usr/bin/env python3
"""Generate ALFWorld rollout txt files with a local instruction model.

This script is intentionally separate from generate_alfworld_rollouts.py, which
uses ALFWorld's hand-coded expert. Here the model must choose each action from
the current observation and admissible actions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
TASK_RE = re.compile(r"Your task is to:\s*(.*?)(?:\n|$)", re.IGNORECASE)


SYSTEM_PROMPT = """You are an ALFWorld text-game agent.
Choose exactly one admissible action for the current situation.
Respond in this exact format:
<think>brief reasoning in English</think>
<action>one admissible action copied exactly</action>

Useful ALFWorld strategy:
- Keep the task goal in mind at every step.
- Search likely receptacles and containers systematically.
- Use open/examine/take actions when they are admissible.
- For hot objects use microwave/stove when available; for cool objects use fridge; for clean objects use sinkbasin.
- After preparing or finding the target object, put it in the required receptacle.

Do not invent actions. Do not wrap the action in quotes. Do not use Chinese. Do not add any other text."""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _format_admissible(actions: Iterable[str]) -> str:
    kept = [a for a in actions if a != "help"]
    return "\n ".join(f"'{a}'," for a in kept)


def _format_obs(obs: str, admissible_actions: Iterable[str]) -> str:
    return (
        f"{obs.strip()}\n\n"
        "Your admissible actions of the current situation are: "
        f"[{_format_admissible(admissible_actions)}].\n\n"
        "Now it's your turn to take an action. "
        "Once you've finished your reasoning, you should choose an admissible action "
        "for current step and present it within <action> </action> tags."
    )


def _extract_task(obs: str) -> str:
    match = TASK_RE.search(obs or "")
    return match.group(1).strip() if match else ""


def _strip_wrapping_quotes(text: str) -> str:
    text = text.strip().strip("`").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def _build_user_prompt(
    obs: str,
    admissible_actions: Iterable[str],
    task_desc: str,
    history: list[str],
) -> str:
    actions = [a for a in admissible_actions if a != "help"]
    action_block = "\n".join(f"- {action}" for action in actions)
    history_block = "\n".join(history[-8:]) if history else "(none yet)"
    return (
        f"Task: {task_desc or 'unknown'}\n\n"
        f"Recent history:\n{history_block}\n\n"
        f"Current observation:\n{obs.strip()}\n\n"
        "Admissible actions:\n"
        f"{action_block}\n\n"
        "Choose the next action. Copy exactly one admissible action."
    )


def _normalise_action(action: str) -> str:
    action = _strip_wrapping_quotes(action)
    action = re.sub(r"^\s*(action|next action)\s*:\s*", "", action, flags=re.IGNORECASE)
    action = action.strip().rstrip(".。")
    return " ".join(action.strip().lower().split())


def _candidate_actions(raw_output: str) -> list[str]:
    raw_output = raw_output or ""
    candidates: list[str] = []
    match = ACTION_RE.search(raw_output or "")
    if match:
        candidates.append(match.group(1))

    without_think = THINK_RE.sub("", raw_output).strip()
    for line in without_think.splitlines():
        line = line.strip().lstrip("-*0123456789. ")
        if line:
            candidates.append(line)
    if without_think:
        candidates.append(without_think[-160:])
    return candidates


def _parse_model_action(raw_output: str, admissible_actions: Iterable[str]) -> tuple[str, bool, str]:
    admissible = {
        _normalise_action(action): action
        for action in admissible_actions
        if action and action != "help"
    }
    candidates = _candidate_actions(raw_output)
    for candidate in candidates:
        normalised = _normalise_action(candidate)
        if normalised in admissible:
            return admissible[normalised], True, ""

    cleaned = THINK_RE.sub("", raw_output or "").strip()
    lowered = _normalise_action(cleaned)
    for normalised, original in sorted(admissible.items(), key=lambda item: len(item[0]), reverse=True):
        if normalised and normalised in lowered:
            return original, True, ""

    fallback = _normalise_action(candidates[-1] if candidates else raw_output or "")
    if not fallback:
        return fallback, False, "empty_action"
    if re.search(r"[\u4e00-\u9fff]", raw_output or ""):
        return fallback, False, "contains_chinese"
    if not ACTION_RE.search(raw_output or ""):
        return fallback, False, "missing_action_tags"
    if not THINK_RE.search(raw_output or ""):
        return fallback, False, "missing_think_tags"
    return fallback, False, "not_in_admissible_actions"


def _classify_task(gamefile: str | None) -> str:
    gamefile = gamefile or ""
    tasks = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]
    for task in tasks:
        if task in gamefile:
            return task
    return "unknown"


class VllmActionModel:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        gpu_memory_utilization: float,
        dtype: str,
        trust_remote_code: bool,
    ) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )

    def _format_chat(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    def generate_actions(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        formatted = [self._format_chat(prompt) for prompt in prompts]
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(data), ensure_ascii=False) + "\n")


def _indent_block(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _write_trajectory(path: Path, env_idx: int, traj_idx: int, steps: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"=== Trajectory for Test {traj_idx}, Env {env_idx} ===\n")
        for step in steps:
            f.write(
                f"{step['step_id']} | Action: {step['action']} | "
                f"Reward: {step['reward']:.3f} | Done: {step['done']}\n"
            )
            if step.get("raw_model_output"):
                f.write("Raw Model Output:\n")
                f.write(f"{_indent_block(step['raw_model_output'].strip())}\n")
                f.write(f"Parsed Action: {step.get('parsed_action', '')}\n")
                f.write(f"Valid Action: {step.get('valid_action', False)}\n")
                if step.get("invalid_reason"):
                    f.write(f"Invalid Reason: {step['invalid_reason']}\n")
            f.write(f"Obs: {step['obs']}\n\n")


def _empty_stats(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "args": vars(args),
        "totals": {
            "env_num": args.env_num,
            "trajs_per_env": args.trajs_per_env,
            "target_trajectories": args.env_num * args.trajs_per_env,
            "completed_trajectories": 0,
            "successful_trajectories": 0,
            "failed_trajectories": 0,
            "model_calls": 0,
            "invalid_actions": 0,
            "steps": 0,
        },
        "by_env": {},
        "by_task": {},
    }


def _update_env_summary(stats: dict[str, Any], env_key: str, env_records: list[dict[str, Any]]) -> None:
    success_count = sum(1 for record in env_records if record["success"])
    fail_count = len(env_records) - success_count
    status = "mixed"
    if len(env_records) > 0 and success_count == len(env_records):
        status = "all_success"
    elif len(env_records) > 0 and fail_count == len(env_records):
        status = "all_fail"

    stats["by_env"][env_key] = {
        "status": status,
        "success_count": success_count,
        "fail_count": fail_count,
        "trajectory_count": len(env_records),
        "steps": sum(record["steps"] for record in env_records),
        "invalid_actions": sum(record["invalid_actions"] for record in env_records),
        "task_type": env_records[0].get("task_type", "unknown") if env_records else "unknown",
    }


def _write_report(output_dir: Path, stats: dict[str, Any]) -> None:
    status_counts = Counter(record["status"] for record in stats["by_env"].values())
    totals = stats["totals"]
    completed = max(1, totals["completed_trajectories"])
    invalid_rate = totals["invalid_actions"] / max(1, totals["model_calls"])
    success_rate = totals["successful_trajectories"] / completed
    avg_steps = totals["steps"] / completed

    lines = [
        "# ALFWorld Qwen7B Rollout Pilot Report",
        "",
        "## Summary",
        f"- Completed trajectories: {totals['completed_trajectories']}",
        f"- Successful trajectories: {totals['successful_trajectories']}",
        f"- Failed trajectories: {totals['failed_trajectories']}",
        f"- Trajectory success rate: {success_rate:.4f}",
        f"- Model calls: {totals['model_calls']}",
        f"- Invalid actions: {totals['invalid_actions']} ({invalid_rate:.4f})",
        f"- Average steps per trajectory: {avg_steps:.2f}",
        "",
        "## Env Classification",
        f"- All-success envs: {status_counts.get('all_success', 0)}",
        f"- All-fail envs: {status_counts.get('all_fail', 0)}",
        f"- Mixed envs: {status_counts.get('mixed', 0)}",
        "",
        "## Task Counts",
    ]
    task_counts = defaultdict(int)
    for record in stats["by_env"].values():
        task_counts[record.get("task_type", "unknown")] += 1
    for task_type, count in sorted(task_counts.items()):
        lines.append(f"- {task_type}: {count}")
    lines.extend(
        [
            "",
            "## Files",
            f"- Stats: {output_dir / 'rollout_stats.json'}",
            f"- Invalid actions: {output_dir / 'invalid_actions.jsonl'}",
        ]
    )
    (output_dir / "rollout_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="/root/autodl-tmp/skillrl-data/rollouts/alfworld_qwen7b_base")
    parser.add_argument("--env_num", type=int, default=100)
    parser.add_argument(
        "--env_offset",
        type=int,
        default=0,
        help="Offset added to env directory IDs so repeated batches do not overwrite earlier rollouts.",
    )
    parser.add_argument("--trajs_per_env", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument(
        "--history_turns",
        type=int,
        default=8,
        help="Number of recent action/observation snippets to include in each model prompt.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_dataset", default="eval_in_distribution")
    parser.add_argument("--num_cpus_per_worker", type=float, default=0.05)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    from agent_system.environments.env_package.alfworld import build_alfworld_envs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_log_path = output_dir / "invalid_actions.jsonl"
    stats_path = output_dir / "rollout_stats.json"
    stats = _empty_stats(args)

    alf_config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
        )
    )
    resources_per_worker = {"num_cpus": args.num_cpus_per_worker, "num_gpus": 0.0}

    model = VllmActionModel(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    envs = build_alfworld_envs(
        alf_config_path,
        seed=args.seed,
        env_num=args.env_num,
        group_n=1,
        is_train=True,
        env_kwargs={"eval_dataset": args.eval_dataset},
        resources_per_worker=resources_per_worker,
    )

    env_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for traj_idx in range(1, args.trajs_per_env + 1):
            text_obs, _image_obs, infos = envs.reset()
            done = [False] * args.env_num
            trajectories: list[list[dict[str, Any]]] = [[] for _ in range(args.env_num)]
            histories: list[list[str]] = [[] for _ in range(args.env_num)]
            trajectory_invalid_counts = [0] * args.env_num
            trajectory_step_counts = [0] * args.env_num
            trajectory_task_types = [
                _classify_task(info.get("extra.gamefile") or info.get("gamefile")) for info in infos
            ]
            trajectory_tasks = [_extract_task(obs) for obs in text_obs]

            for env_idx in range(args.env_num):
                trajectories[env_idx].append(
                    {
                        "step_id": "Step -1",
                        "action": "None",
                        "parsed_action": "None",
                        "raw_model_output": "",
                        "valid_action": True,
                        "invalid_reason": "",
                        "reward": 0.0,
                        "done": False,
                        "obs": _format_obs(text_obs[env_idx], envs.get_admissible_commands[env_idx]),
                    }
                )
                histories[env_idx].append(f"Initial observation: {text_obs[env_idx].strip()}")

            for step_idx in range(args.max_steps):
                active_env_indices = [idx for idx in range(args.env_num) if not done[idx]]
                if not active_env_indices:
                    break

                prompts = [
                    _build_user_prompt(
                        text_obs[idx],
                        envs.get_admissible_commands[idx],
                        trajectory_tasks[idx],
                        histories[idx][-args.history_turns :],
                    )
                    for idx in active_env_indices
                ]
                raw_outputs = model.generate_actions(prompts)
                stats["totals"]["model_calls"] += len(raw_outputs)

                actions = ["look"] * args.env_num
                step_meta: dict[int, dict[str, Any]] = {}
                for env_idx, raw_output in zip(active_env_indices, raw_outputs):
                    admissible_commands = [a for a in envs.get_admissible_commands[env_idx] if a != "help"]
                    admissible = [_normalise_action(a) for a in admissible_commands]
                    parsed_action, valid_action, invalid_reason = _parse_model_action(
                        raw_output,
                        admissible_commands,
                    )
                    if not valid_action:
                        trajectory_invalid_counts[env_idx] += 1
                        stats["totals"]["invalid_actions"] += 1
                        if not invalid_reason:
                            invalid_reason = "not_in_admissible_actions"
                        _append_jsonl(
                            invalid_log_path,
                            {
                                "traj_idx": traj_idx,
                                "env_idx": env_idx,
                                "step_idx": step_idx,
                                "reason": invalid_reason,
                                "parsed_action": parsed_action,
                                "raw_model_output": raw_output,
                                "admissible_actions": envs.get_admissible_commands[env_idx],
                            },
                        )
                    fallback_action = "look" if "look" in admissible else (admissible_commands[0] if admissible_commands else "look")
                    actions[env_idx] = parsed_action if valid_action else fallback_action
                    step_meta[env_idx] = {
                        "raw_model_output": raw_output,
                        "parsed_action": parsed_action,
                        "valid_action": valid_action,
                        "invalid_reason": "" if valid_action else invalid_reason,
                    }

                next_obs, _image_obs, rewards, dones, infos = envs.step(actions)

                for env_idx in active_env_indices:
                    done[env_idx] = bool(dones[env_idx])
                    trajectory_step_counts[env_idx] += 1
                    meta = step_meta[env_idx]
                    trajectories[env_idx].append(
                        {
                            "step_id": f"Step {step_idx:02d}",
                            "action": actions[env_idx],
                            "parsed_action": meta["parsed_action"],
                            "raw_model_output": meta["raw_model_output"],
                            "valid_action": meta["valid_action"],
                            "invalid_reason": meta["invalid_reason"],
                            "reward": float(rewards[env_idx]),
                            "done": done[env_idx],
                            "obs": _format_obs(
                                next_obs[env_idx],
                                envs.get_admissible_commands[env_idx],
                            ),
                        }
                    )
                    histories[env_idx].append(
                        "Step {step}: action={action}; reward={reward:.3f}; done={done}; observation={obs}".format(
                            step=step_idx,
                            action=actions[env_idx],
                            reward=float(rewards[env_idx]),
                            done=done[env_idx],
                            obs=str(next_obs[env_idx]).strip(),
                        )
                    )

                text_obs = next_obs

            for env_idx, steps in enumerate(trajectories):
                global_env_idx = args.env_offset + env_idx
                env_key = f"env{global_env_idx:06d}"
                out_path = output_dir / env_key / f"test{traj_idx}.txt"
                _write_trajectory(out_path, global_env_idx, traj_idx, steps)
                final = steps[-1]
                success = float(final["reward"]) >= 10.0
                stats["totals"]["completed_trajectories"] += 1
                stats["totals"]["successful_trajectories"] += int(success)
                stats["totals"]["failed_trajectories"] += int(not success)
                stats["totals"]["steps"] += trajectory_step_counts[env_idx]
                env_records[env_key].append(
                    {
                        "traj_idx": traj_idx,
                        "path": str(out_path),
                        "success": success,
                        "reward": float(final["reward"]),
                        "done": bool(final["done"]),
                        "steps": trajectory_step_counts[env_idx],
                        "invalid_actions": trajectory_invalid_counts[env_idx],
                        "task_type": trajectory_task_types[env_idx],
                    }
                )
                _update_env_summary(stats, env_key, env_records[env_key])
                _write_json(stats_path, stats)
                print(
                    f"wrote {out_path} steps={len(steps)} reward={final['reward']:.3f} "
                    f"done={final['done']} success={success} invalid={trajectory_invalid_counts[env_idx]}"
                )
    finally:
        stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(stats_path, stats)
        _write_report(output_dir, stats)
        envs.close()
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
