# SkillRL Project Instructions

Use these defaults whenever working in this repository.

## Repository And Remotes

- Local Windows repo: `D:\top\SkillRL`
- GitHub origin: `https://github.com/wudi299/SkillRL.git`
- Original upstream: `https://github.com/aiming-lab/SkillRL.git`
- Normal pushes go to `origin main`, not `upstream`.

## AutoDL Layout

- AutoDL code path: `/root/autodl-tmp/SkillRL`
- AutoDL data path: `/root/autodl-tmp/skillrl-data`
- AutoDL Conda env path: `/root/autodl-tmp/envs/skillrl`
- AutoDL env file: `/root/autodl-tmp/skillrl-data/env.sh`
- AutoDL run output path: `/root/autodl-tmp/skillrl-runs`
- ALFWorld rollout path: `/root/autodl-tmp/skillrl-data/rollouts/alfworld`
- First smoke work dir: `/root/autodl-tmp/skillrl-runs/alfworld_smoke_001`

## Important Scripts

- `scripts/autodl_h800_workflow.sh`: AutoDL H800 setup wrapper.
- `scripts/prepare_autodl_env.sh`: Conda, PyTorch, vLLM, flash-attn setup and verification.
- `scripts/run_alfworld_audit_pipeline.sh`: auditable ALFWorld pipeline that saves raw, processed, memory, skill-bank, distillation, and SFT artifacts.

## Preferred AutoDL Commands

Prepare or refresh the H800 environment:

```bash
cd /root/autodl-tmp/SkillRL
git pull origin main
bash scripts/autodl_h800_workflow.sh all
```

If `flash-attn` compilation is too heavy:

```bash
MAX_JOBS=4 bash scripts/autodl_h800_workflow.sh all
```

Run the first auditable ALFWorld smoke test after rollout files are ready:

```bash
export OPENAI_API_KEY="..."
export ROLLOUT_DIR=/root/autodl-tmp/skillrl-data/rollouts/alfworld
export WORK_DIR=/root/autodl-tmp/skillrl-runs/alfworld_smoke_001
LIMIT=1 TRACE_LLM=1 bash scripts/autodl_h800_workflow.sh smoke
```

## Operating Rules

- Keep datasets, model caches, checkpoints, and experiment results out of GitHub.
- Results should stay under `/root/autodl-tmp/skillrl-runs`; download only the packaged `skillrl_run_*.tar.gz` when collecting outputs.
- Distinguish temporary shell variables from persisted variables in `/root/autodl-tmp/skillrl-data/env.sh`.
- Explain AutoDL and Git operations plainly in Chinese when discussing them with the user.
