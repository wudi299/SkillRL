No-GPU Server Preparation for SkillRL
=====================================

This guide prepares SkillRL code, Docker environment, and task data on a
no-GPU Linux server. It does not start GPU training or run ``vllm``.

Directory Layout
----------------

Use the same paths now and later on the GPU server:

.. code-block:: bash

   /workspace/SkillRL          # repository
   /data/skillrl               # task data and generated env.sh
   /data/skillrl/hf-cache      # Hugging Face cache

Host Usage
----------

Clone or sync this repository to ``/workspace/SkillRL`` first, then run:

.. code-block:: bash

   cd /workspace/SkillRL
   bash scripts/prepare_nogpu_server.sh host

The host phase pulls the Docker image, mounts the repository and data
directories, then runs the container preparation phase.

Useful overrides:

.. code-block:: bash

   REPO_DIR=/workspace/SkillRL \
   DATA_DIR=/data/skillrl \
   WEBSHOP_DATA_SIZE=all \
   bash scripts/prepare_nogpu_server.sh host

If WebShop full data download is unstable, retry with:

.. code-block:: bash

   WEBSHOP_DATA_SIZE=small CONTAINER_NAME=skillrl-prep-small \
   bash scripts/prepare_nogpu_server.sh host

If you only want to prepare package dependencies and Search parquet first:

.. code-block:: bash

   SKIP_WEBSHOP=1 SKIP_SEARCH_INDEX=1 \
   bash scripts/prepare_nogpu_server.sh host

Container Usage
---------------

If you already entered the Docker container manually, run:

.. code-block:: bash

   cd /workspace/SkillRL
   bash scripts/prepare_nogpu_server.sh container

To re-run only verification:

.. code-block:: bash

   source /data/skillrl/env.sh
   cd /workspace/SkillRL
   bash scripts/prepare_nogpu_server.sh verify

What Gets Prepared
------------------

- SkillRL editable install with ``pip install --no-deps -e .``.
- ALFWorld dependencies and downloaded game/PDDL/cache files.
- Java JRE if it is missing, then WebShop environment data via
  ``setup.sh -d all`` by default.
- Search third-party environment, processed SearchR1 parquet files, and
  SearchR1 index/corpus files.
- SkillBank file existence checks for ALFWorld, WebShop, and Search.
- ``/data/skillrl/env.sh`` with reusable paths for later GPU training.

Expected Verification
---------------------

The script checks:

.. code-block:: bash

   python -c "import verl; import agent_system; print('repo imports ok')"
   python -c "import alfworld; print('alfworld import ok')"
   python -c "import gym; print('gym import ok')"
   test -f /data/skillrl/searchR1_processed_direct/train.parquet
   test -f /data/skillrl/searchR1/e5_Flat.index
   test -f /data/skillrl/searchR1/wiki-18.jsonl

No CUDA check is expected on the no-GPU server.

Move To GPU Server
------------------

On the GPU server, mount or copy the same two directories:

.. code-block:: bash

   /workspace/SkillRL
   /data/skillrl

Then start the same Docker image with ``--gpus all`` and source:

.. code-block:: bash

   source /data/skillrl/env.sh

Run GPU smoke tests and training only after confirming ``nvidia-smi`` and
``torch.cuda.is_available()`` inside the GPU container.
