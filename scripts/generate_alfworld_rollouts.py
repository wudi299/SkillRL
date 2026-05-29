#!/usr/bin/env python3
"""Generate ALFWorld rollout txt files for the auditable SFT pipeline."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import ray

from agent_system.environments.env_package.alfworld import build_alfworld_envs


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


def _expert_action(info: Dict[str, Any], admissible_actions: List[str]) -> str:
    plan = info.get("extra.expert_plan") or info.get("expert_plan") or []
    if isinstance(plan, str):
        plan = [plan]
    for action in plan:
        if action and action in admissible_actions:
            return action
    if "look" in admissible_actions:
        return "look"
    return admissible_actions[0] if admissible_actions else "look"


def _write_trajectory(path: Path, env_idx: int, traj_idx: int, steps: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"=== Trajectory for Test {traj_idx}, Env {env_idx} ===\n")
        for step in steps:
            f.write(
                f"{step['step_id']} | Action: {step['action']} | "
                f"Reward: {step['reward']:.3f} | Done: {step['done']}\n"
            )
            f.write(f"Obs: {step['obs']}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/root/autodl-tmp/skillrl-data/rollouts/alfworld")
    parser.add_argument("--env_num", type=int, default=1)
    parser.add_argument("--trajs_per_env", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--policy",
        choices=["expert"],
        default="expert",
        help="expert uses ALFWorld's handcoded expert plan and does not call an LLM.",
    )
    args = parser.parse_args()

    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        "../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    alf_config_path = os.path.abspath(alf_config_path)
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}

    envs = build_alfworld_envs(
        alf_config_path,
        seed=args.seed,
        env_num=args.env_num,
        group_n=1,
        is_train=True,
        env_kwargs={"eval_dataset": "eval_in_distribution"},
        resources_per_worker=resources_per_worker,
    )

    output_dir = Path(args.output_dir)
    try:
        for traj_idx in range(1, args.trajs_per_env + 1):
            text_obs, _image_obs, infos = envs.reset()
            done = [False] * args.env_num
            trajectories: List[List[Dict[str, Any]]] = [[] for _ in range(args.env_num)]

            for env_idx in range(args.env_num):
                trajectories[env_idx].append(
                    {
                        "step_id": "Step -1",
                        "action": "None",
                        "reward": 0.0,
                        "done": False,
                        "obs": _format_obs(text_obs[env_idx], envs.get_admissible_commands[env_idx]),
                    }
                )

            for step_idx in range(args.max_steps):
                actions = []
                for env_idx in range(args.env_num):
                    if done[env_idx]:
                        actions.append("look")
                        continue
                    actions.append(_expert_action(infos[env_idx], envs.get_admissible_commands[env_idx]))

                next_obs, _image_obs, rewards, dones, infos = envs.step(actions)

                for env_idx in range(args.env_num):
                    if done[env_idx]:
                        continue
                    done[env_idx] = bool(dones[env_idx])
                    trajectories[env_idx].append(
                        {
                            "step_id": f"Step {step_idx:02d}",
                            "action": actions[env_idx],
                            "reward": float(rewards[env_idx]),
                            "done": done[env_idx],
                            "obs": _format_obs(
                                next_obs[env_idx],
                                envs.get_admissible_commands[env_idx],
                            ),
                        }
                    )

                if all(done):
                    break

            for env_idx, steps in enumerate(trajectories):
                out_path = output_dir / f"env{env_idx:03d}" / f"test{traj_idx}.txt"
                _write_trajectory(out_path, env_idx, traj_idx, steps)
                final = steps[-1]
                print(
                    f"wrote {out_path} steps={len(steps)} "
                    f"reward={final['reward']:.3f} done={final['done']}"
                )
    finally:
        envs.close()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
