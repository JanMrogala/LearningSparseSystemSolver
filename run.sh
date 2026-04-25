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

pip install -q -e . 2>/dev/null || true

python scripts/train.py
