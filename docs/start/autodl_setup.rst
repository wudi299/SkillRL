AutoDL Setup
============

AutoDL instances already run inside a managed container, so do not use the
Docker preparation script there. Use the data disk under ``/root/autodl-tmp``
for code, Conda environments, datasets, and model caches.

Recommended Layout
------------------

.. code-block:: bash

   /root/autodl-tmp/SkillRL
   /root/autodl-tmp/envs/skillrl
   /root/autodl-tmp/skillrl-data

Clone the repository:

.. code-block:: bash

   cd /root/autodl-tmp
   git clone https://github.com/wudi299/SkillRL.git
   cd SkillRL

H800 Setup Workflow
-------------------

On the H800 instance, use the workflow wrapper first. It pulls the latest
GitHub code, checks the shell scripts, installs the full GPU environment, and
runs the import/CUDA verification:

.. code-block:: bash

   cd /root/autodl-tmp/SkillRL
   bash scripts/autodl_h800_workflow.sh all

The wrapper does not run the ALFWorld teacher-model smoke test by default,
because that step consumes API credits and requires rollout files. To run it
after adding rollouts and setting an API key:

.. code-block:: bash

   cd /root/autodl-tmp/SkillRL
   export OPENAI_API_KEY=...
   export ROLLOUT_DIR=/root/autodl-tmp/skillrl-data/rollouts/alfworld
   export WORK_DIR=/root/autodl-tmp/skillrl-runs/alfworld_smoke_001
   LIMIT=1 TRACE_LLM=1 bash scripts/autodl_h800_workflow.sh smoke

Useful tuning knobs:

- ``LIMIT=1`` for the first smoke test, then ``LIMIT=3`` and ``LIMIT=10``.
- ``TRACE_LLM=1`` while debugging so prompt/response traces are preserved.
- ``WORK_DIR`` should be changed for each run, for example
  ``/root/autodl-tmp/skillrl-runs/alfworld_smoke_002``.
- ``MAX_JOBS=4`` if ``flash-attn`` compilation is too slow or memory-heavy;
  otherwise keep the default ``MAX_JOBS=8``.
- ``FLASH_ATTN_REQUIRED=0`` if GitHub release downloads are unstable and you
  need to finish the rest of the environment first.

No-GPU Instance
---------------

On a no-GPU or small-memory instance, install the base environment first:

.. code-block:: bash

   cd /root/autodl-tmp/SkillRL
   INSTALL_GPU_DEPS=0 bash scripts/prepare_autodl_env.sh base

This installs the Conda environment, PyTorch CUDA wheel, project
requirements except ``flash-attn``, and the editable SkillRL package.
It also writes ``/root/autodl-tmp/skillrl-data/env.sh``.

H800/GPU Instance
-----------------

On an H800 instance, install all possible runtime dependencies:

.. code-block:: bash

   cd /root/autodl-tmp/SkillRL
   INSTALL_GPU_DEPS=1 bash scripts/prepare_autodl_env.sh full

The script installs ``vllm`` and ``flash-attn`` only when explicitly enabled
or when GPU detection succeeds in auto mode.

Manual Activation
-----------------

For later sessions:

.. code-block:: bash

   source /root/miniconda3/etc/profile.d/conda.sh
   conda activate skillrl
   source /root/autodl-tmp/skillrl-data/env.sh
   cd /root/autodl-tmp/SkillRL

If name activation does not work in an old shell, use the full prefix once:

.. code-block:: bash

   conda activate /root/autodl-tmp/envs/skillrl

Verification
------------

Run:

.. code-block:: bash

   bash scripts/prepare_autodl_env.sh verify

The verification prints import status for ``verl``, ``agent_system``,
``ray``, ``vllm``, ``flash_attn``, and CUDA availability.

Notes
-----

- Prefer a Python 3.10 Conda environment even if the AutoDL image shows
  Python 3.12.
- Prefer at least 100 GB data disk for full WebShop, Search index, and model
  caches.
- If ``vllm==0.11.0`` conflicts with PyTorch or CUDA on the selected image,
  keep the error output and use the repository-tested default
  ``vllm==0.8.5.post1`` with ``torch==2.6.0`` and CUDA 12.4.
- Keep ``/root/autodl-tmp/skillrl-runs`` out of Git. Download only the
  generated ``skillrl_run_*.tar.gz`` package when collecting results.
