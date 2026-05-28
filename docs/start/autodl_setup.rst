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
   conda activate /root/autodl-tmp/envs/skillrl
   source /root/autodl-tmp/skillrl-data/env.sh
   cd /root/autodl-tmp/SkillRL

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
  keep the error output and fall back to a version matching the image.
