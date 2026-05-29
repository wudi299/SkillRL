# SFT data generation

Turn agent rollout trajectories into the SFT data format used by
[Jianwen/SkillRL-SFT-Data](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data),
for ALFWorld, WebShop, and Search-agent.

The rollout step is **not** included — run your own model on the target
environment, dump trajectories to disk, then run the pipeline here.

## Pipeline

```
your trajectory dump (txt)
  │
  ▼
[1] preprocess/    parse + dedupe              → processed_trajectories.json
[2] skill_memory/  per-traj memories + aggregate → skill_bank.json
[3] distillation/  o3 reasoning + (ALFWorld) synthetic done step
[4] postprocess/   sharegpt → alpaca pairs + (WebShop) validate
  │
  ▼
{instruction, output} JSON  (alpaca, matches HF schema)
```

## Quick start

```bash
pip install openai
export OPENAI_API_KEY=...
export ROLLOUT_DIR=/path/to/your/rollout/txts
export WORK_DIR=/path/to/output

bash run_alfworld.sh   # or run_webshop.sh / run_search.sh
```

## Auditable ALFWorld smoke run

For the local-GitHub-AutoDL workflow, use the auditable runner first. It keeps
each intermediate artifact under one run directory and writes a Chinese report
with samples from every stage.

On the H800 instance, first sync code and prepare the full environment:

```bash
cd /root/autodl-tmp/SkillRL
bash scripts/autodl_h800_workflow.sh all
```

This command installs the full GPU environment and verifies imports. It does
not call teacher models by default.

```bash
cd /root/autodl-tmp/SkillRL
source /root/miniconda3/etc/profile.d/conda.sh
conda activate skillrl
source /root/autodl-tmp/skillrl-data/env.sh

export OPENAI_API_KEY=...
export ROLLOUT_DIR=/root/autodl-tmp/skillrl-data/rollouts/alfworld
export WORK_DIR=/root/autodl-tmp/skillrl-runs/alfworld_smoke_001

LIMIT=3 TRACE_LLM=1 bash scripts/run_alfworld_audit_pipeline.sh
```

The same smoke run can also be launched through the AutoDL wrapper:

```bash
LIMIT=1 TRACE_LLM=1 bash scripts/autodl_h800_workflow.sh smoke
```

Key outputs:

```
${WORK_DIR}/manifest.json
${WORK_DIR}/00_raw_rollouts/selected_files.json
${WORK_DIR}/01_processed/processed_trajectories_cleaned.json
${WORK_DIR}/02_memory/generated_memories.json
${WORK_DIR}/02_memory/llm_calls.jsonl
${WORK_DIR}/03_skill_bank/skill_bank.json
${WORK_DIR}/03_skill_bank/general_skills.json
${WORK_DIR}/03_skill_bank/task_specific_skills.json
${WORK_DIR}/04_distillation/distilled_trajectories.json
${WORK_DIR}/04_distillation/llm_calls.jsonl
${WORK_DIR}/05_sft_data/alfworld_sft_data.json
${WORK_DIR}/06_report/summary.md
${WORK_DIR}/skillrl_run_alfworld_smoke_001.tar.gz
```

Set `TRACE_LLM=0` to skip full prompt/response traces. Trace files never store
API keys, but they do store prompts and model outputs, so keep them out of Git.

Parameter defaults for the first AutoDL pass:

| Parameter | Default | When to change |
|---|---:|---|
| `LIMIT` | `1` through the wrapper, `3` in the raw runner | Increase to `3` and `10` after the first successful smoke test |
| `TRACE_LLM` | `1` | Set to `0` only when prompt/response traces are not needed |
| `WORK_DIR` | `/root/autodl-tmp/skillrl-runs/alfworld_smoke_001` | Use a new directory name for every experiment |
| `MAX_JOBS` | `8` | Use `4` if `flash-attn` compilation is memory-heavy |
| `FLASH_ATTN_REQUIRED` | `1` | Set to `0` to continue when GitHub release downloads for flash-attn fail |
| `MEMORY_MODEL` | `gpt-4o` | Override only for cost or latency testing |
| `AGGREGATE_MODEL` | `gpt-4o` | Override only for cost or latency testing |
| `DISTILL_MODEL` | `o3` | Override only after the baseline smoke run works |

## Per-environment differences

| | ALFWorld | WebShop | Search |
|---|---|---|---|
| Success filter | `Reward: 10.000` | `Reward: 10.000` (`click[buy now]` matches target) | `Reward: 1.0` |
| Synthetic `done` step | **Yes** | No (`click[buy now]` terminates env) | No (`<answer>` terminates) |
| Skill categories | 6 task types (pick / clean / heat / cool / examine / two-obj) | 3 (general / apparel / electronics) | 2 (direct_retrieval / multi_hop) |
| Postprocess validation | not needed | normalize action vocabulary | not needed |

## Input rollout txt format (ALFWorld / WebShop)

The parsers expect one .txt per rollout, organized as:

```
<ROLLOUT_DIR>/
  env000/
    test1.txt
    test2.txt
    ...
  env001/
    ...
```

Each .txt has one block per step:

```
=== Trajectory for Test 0, Env 5 ===
Step -1 | Action: None | Reward: 0.000 | Done: False
Obs: -= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a cabinet 1, ...

Your task is to: put a clean plate in box

Your admissible actions of the current situation are: [open cabinet 1, go to countertop 1, ...].

Now it's your turn to take an action. ...

Step 00 | Action: go to countertop 1 | Reward: 0.000 | Done: False
Obs: You arrive at countertop 1. ...
Your admissible actions of the current situation are: [...].

...

Step 12 | Action: put plate 1 in box 1 | Reward: 10.000 | Done: True
Obs: You put the object down successfully.
Your admissible actions of the current situation are: [...].
```

Conventions:

- `Step -1` is the initial state (`Action: None`).
- `Step N` (N ≥ 0) is the Nth action; its `Obs` is the env response
  **after** taking the action plus the admissible actions for the new
  state.
- The winning step has `Reward: 10.000` and `Done: True`.

A reference rollout writer is in `verl-agent/examples/prompt_agent/qwen_alfworld.py`.

## Input rollout txt format (Search)

```
Your question: when was i can only imagine movie released?

History:
Step 0:<search>I Can Only Imagine film release date</search> <information>...</information>
Step 1:<search>"I Can Only Imagine" theatrical release</search> <information>...</information>

Now it's your turn to take an action.
Action 2: <answer>March 16, 2018</answer>

Reward: 1.0
```

File numbering is used to infer the QA dataset (`nq`, `popqa`,
`hotpotqa`, etc.) — adjust `infer_data_source` in `parse_search.py` if
your numbering differs.

## Bring your own JSON

If your rollout produces a different on-disk format, skip
`preprocess/parse_*.py` and feed your own JSON to stage 2 directly. The
schema downstream stages expect:

```jsonc
[
  {
    "env_id": "env001",
    "task": "put a clean plate in box",
    "type": "all_success",   // or "all_fail"
    "trajectories": [
      [
        { "step_id": "Step -1", "action": null,
          "observation": "<initial state>",
          "admissible_actions": [...], "reward": 0.0, "done": false },
        { "step_id": "Step 00", "action": "go to countertop 1",
          "observation": "<state after action>",
          "admissible_actions": [...], "reward": 0.0, "done": false },
        ...
      ],
      ...
    ]
  }
]
```

## Output schema

```jsonc
[
  {
    "instruction": "You are an expert agent operating in the ALFRED Embodied Environment.\nYour task is to: ...\n\n## Retrieved Relevant Experience\n\n### General Principles\n- ...\n\n### <Category> Skills\n- ...\n\n### Mistakes to Avoid\n- ...\n\n## Current Progress\nYour current observation is: ...\nYour admissible actions of the current situation are: [...]\n\nNow it's your turn to take an action.\n...",
    "output": "<think>...</think>\n<action>...</action>"
  }
]
```

Same column layout as the HF dataset (`instruction`, `output`); can be
loaded directly by alpaca-style SFT trainers.

## Layout

```
sft_data_generation/
├── README.md
├── run_alfworld.sh / run_webshop.sh / run_search.sh   # entry points
│
├── preprocess/        # [1] txt → JSON
│   ├── parse_alfworld.py
│   ├── parse_webshop.py
│   ├── parse_search.py
│   └── dedupe_repetitions.py
│
├── skill_memory/      # [2] per-traj memories + aggregate (gpt-4o)
│   ├── generate_memory_alfworld.py
│   ├── generate_memory_webshop.py
│   ├── generate_memory_search.py
│   └── aggregate_skills.py
│
├── distillation/      # [3] o3 reasoning + (ALFWorld) synthetic done
│   ├── skill_retrieval.py
│   ├── distill_alfworld.py
│   ├── distill_webshop.py
│   └── distill_search.py
│
└── postprocess/       # [4] sharegpt → alpaca + (WebShop) validate
    ├── sharegpt_to_pairs.py
    └── validate_and_fix_webshop.py
```
