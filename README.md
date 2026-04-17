# LearningSparseSystemSolver

Reimplementation of *Learning to Compute Gröbner Bases* (Kera et al., arXiv:2311.12904). MVP targets shape-position ideals over F_7 with n=2 variables.

## Setup

Requires Singular on `PATH` and Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart (MVP)

```bash
# 1. Generate dataset (10K train / 1K test, all verified via Singular)
python scripts/gen_dataset.py --config-name mvp

# 2. Train a small Transformer
python scripts/train.py --config-name mvp wandb.mode=offline

# 3. Evaluate with polynomial-acc, support-acc, and ideal-equality metrics
python scripts/evaluate.py --config-name eval \
  eval.ckpt_path=outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/last.ckpt \
  wandb.mode=offline
```

## Structure

- `src/lsss/algebra/`: polynomial ring, Singular bridge, shape-position generator, verification.
- `src/lsss/data/`: JSONL dataset format, prefix-notation tokenizer, PyTorch dataset.
- `src/lsss/model/`: from-scratch encoder-decoder Transformer.
- `src/lsss/train/`: Lightning module and data module.
- `src/lsss/eval/`: greedy decode, metrics, evaluation entry point.
- `configs/`: Hydra config groups (MVP and paper-scale presets).
- `docs/superpowers/`: design specs and implementation plans.

## Testing

```bash
pytest                          # unit + config tests (fast)
pytest -m integration           # requires Singular
```

## References

- Design spec: `docs/superpowers/specs/2026-04-18-learning-to-compute-grobner-bases-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-18-grobner-mvp-implementation.md`
