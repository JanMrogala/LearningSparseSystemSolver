#!/bin/bash
# Experiment entrypoint — called by CER inside the Singularity container.
#
# Environment variables set by CER:
#   CER_COMMIT      - full git commit hash
#   WANDB_PROJECT   - W&B project name (from cer.yaml)
#   WANDB_RUN_NAME  - cer-<commit_short>
#   WANDB_TAGS      - commit hash (for querying via `cer results`)
#   WANDB_API_KEY   - W&B API key

set -euo pipefail

# Data lives at the repo root; config resolves via LSSS_DATA_DIR/dataset_name
export LSSS_DATA_DIR=$(pwd)

# Make lsss importable without requiring hatchling in the container
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Install any deps the container might be missing
pip install -q pexpect pytorch-lightning hydra-core omegaconf wandb

python scripts/train.py
