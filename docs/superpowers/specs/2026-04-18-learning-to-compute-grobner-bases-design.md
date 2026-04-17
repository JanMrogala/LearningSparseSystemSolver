# Learning to Compute Gröbner Bases: Reimplementation Design

**Date:** 2026-04-18
**Source paper:** Kera, Ishihara, Kambe, Vaccon, Yokoyama. *Learning to Compute Gröbner Bases* (arXiv:2311.12904v3, Nov 2024).
**Scope:** End-to-end reimplementation of the paper's two contributions (dataset generation and Transformer learnability), starting with a minimal viable slice and keeping the architecture open to a later sparse-systems extension.

## 1. Goals

1. Reproduce the paper's backward dataset-generation algorithm (Alg. 1) and verify it against Singular's forward generators on small instances.
2. Reproduce the paper's Transformer learnability results (Tables 1-2) with both discrete and hybrid coefficient embeddings.
3. Keep the architecture modular so that a later sparse-systems generator can be plugged in without touching the model, trainer, or evaluator.

## 2. Non-goals

1. Reproducing the full experimental matrix in the first pass. MVP targets one slice (shape position, 𝔽₇, n=2, discrete embedding).
2. Implementing the Cauchy-module generator. Shape position only for the initial spec.
3. Implementing FGLM change-of-order. Shape-position generation already produces the lex basis directly, and lex is the target order for MVP.
4. Implementing a forward-generation baseline for Table 1 beyond a thin Singular wrapper. The paper's forward baseline uses SageMath with three Singular algorithms; a SageMath dependency is out of scope for MVP and can be added later as a sibling module.
5. Out-of-distribution generalization experiments (paper Sec. F.4). MVP tracks in-distribution metrics only.

## 3. Architectural overview

Approach: **two-stage pipeline with on-disk dataset**. Dataset generation and training are separate executables sharing a versioned JSONL dataset format. Singular is called only from the generator and from the evaluator; the training loop has zero algebra dependency.

```
+--------------------+     +--------------------+     +--------------------+
| gen_dataset.py     |     | train.py           |     | evaluate.py        |
| (Singular-heavy)   |     | (PyTorch + Light.) |     | (PyTorch + Sing.)  |
+---------+----------+     +---------+----------+     +---------+----------+
          |                          |                          |
          v                          v                          v
     JSONL shards <----- W&B run -----+---------- W&B run ------+
     + meta.json             + checkpoints
```

Top-level stack: **PyTorch (from scratch, no HuggingFace), PyTorch Lightning, Weights & Biases, Hydra (OmegaConf), Singular via subprocess.**

## 4. Repository layout

```
LearningSparseSystemSolver/
├── pyproject.toml                # torch, pytorch-lightning, wandb, hydra-core, omegaconf,
│                                 # pexpect (singular bridge), numpy, tqdm, pytest, pytest-xdist
├── README.md
├── configs/                      # Hydra config groups (see §11)
├── src/lsss/
│   ├── algebra/                  # only module that touches Singular
│   │   ├── singular.py           # long-lived subprocess wrapper
│   │   ├── polynomial.py         # Poly dataclass, arithmetic, term orders
│   │   ├── sampling.py           # unimodular upper-triangular matrices, permutations, random polys
│   │   ├── shape_position.py     # Alg. 1 implementation (sans FGLM)
│   │   ├── fglm.py               # change-of-order wrapper (stub for MVP, filled in post-MVP)
│   │   └── verify.py             # reduced-GB check and ideal-equality check
│   ├── data/
│   │   ├── generate.py           # entry point: (F, G) pairs to JSONL shards
│   │   ├── dataset_format.py     # JSONL schema + schema-version integer
│   │   ├── tokenizer.py          # prefix-notation tokenizer, vocab, hybrid/discrete paths
│   │   └── torch_dataset.py      # map-style Dataset, collation
│   ├── model/
│   │   ├── embedding.py          # DiscreteEmbedding, HybridEmbedding
│   │   ├── transformer.py        # encoder-decoder from scratch (uses F.sdpa kernel)
│   │   └── heads.py              # ClassificationHead (tied to embedding), RegressionHead
│   ├── train/
│   │   ├── lit_module.py         # pl.LightningModule
│   │   ├── lit_datamodule.py     # pl.LightningDataModule
│   │   └── train.py              # Hydra entry point
│   ├── eval/
│   │   ├── evaluate.py           # Hydra entry point
│   │   ├── metrics.py            # polynomial-acc, support-acc, ideal-equality
│   │   └── decode.py             # greedy decode (beam width 1), handles hybrid case
│   └── utils/
│       ├── config.py             # dataclass-backed structured configs
│       ├── seeding.py
│       └── logging.py
├── scripts/
│   ├── gen_dataset.py            # thin CLI wrapper
│   ├── train.py
│   └── evaluate.py
└── tests/                        # see §12
```

**Boundary discipline:** `model/`, `train/`, `eval/decode.py`, `eval/metrics.py` (except `ideal_equality`) have zero imports from `algebra/`. The only bridge from training/eval to algebra is through the JSONL format in `data/`.

## 5. Algebra subsystem

### 5.1 Internal polynomial representation

```python
@dataclass(frozen=True)
class Poly:
    terms: tuple[tuple[Coeff, tuple[int, ...]], ...]   # sorted by term order
    n_vars: int
    field: Field                                       # Fp(p), QQ, or RR
```

`Coeff` is a sum type: `int` for 𝔽ₚ, `Fraction` for ℚ (exact rationals, matches the paper's tokenization of `a/b` with `a,b ∈ {-5,…,5}`), `float` for ℝ.

### 5.2 Singular bridge (`algebra/singular.py`)

One long-lived Singular subprocess per worker via `pexpect`. Polynomials are serialized to Singular input syntax; output is parsed back into `Poly`. Exposes:

1. `reduced_groebner(F, order) -> G`
2. `fglm(G, from_order, to_order) -> G'` (stub in MVP, always called with `from_order=to_order=lex`)
3. `ideal_equal(F1, F2) -> bool` (computes reduced GBs of both, compares)

Configured via `generation.singular.binary` and `generation.singular.timeout_seconds`. Subprocess calls that exceed the timeout raise and the sample is resampled (if `generation.verify.resample_on_failure=true`).

### 5.3 Shape-position generation (`algebra/shape_position.py`)

Direct implementation of the paper's Alg. 1:

1. **Sample G.** Univariate monic `h ∈ k[xₙ]` of degree `d_h`, with `d_h ∈ [1, d]`. Sample `g₁, …, g_{n-1}`, each univariate in `xₙ` of degree `< d_h`. Compose G as `{h, x₁ − g₁, …, x_{n−1} − g_{n-1}}`. This is already a reduced lex-Gröbner basis of a 0-dimensional ideal in shape position.
2. **Sample transforms.** `s ~ Uniform[n, s_max]`. Sample `U₁ ∈ ST(s, k[x]_{≤d'})` and `U₂' ∈ ST(n, k[x]_{≤d'})` (unimodular upper-triangular, all-ones diagonal). Pad `U₂'` to `U₂ ∈ k[x]^{s×n}` with zero block below. Sample permutation matrix `P ∈ {0,1}^{s×s}`.
3. **Compute F.** `F = U₁ · P · U₂ · G` via polynomial matrix-vector products. O(n² + s²) polynomial multiplications.
4. **Change order (post-MVP).** If target order ≠ lex, call FGLM on G. Not exercised in MVP.

**Coefficient-range defaults** (configurable):
- G coefficients: `a/b` with `a, b ∈ {-5, …, 5}` (for ℚ).
- F-multiplier coefficients: `a/b` with `a, b ∈ {-100, …, 100}` (for ℚ).
- Integer ranges for 𝔽ₚ are the full field.

**Density control (`density_sigma`):** When sampling entries of `U₂'` above the diagonal, each monomial in `k[x]_{≤d'}` is independently included with probability σ. Paper uses σ = 1.0, 0.6, 0.3, 0.2 for n = 2, 3, 4, 5.

### 5.4 Verification (`algebra/verify.py`)

Two checks, both available at two distinct stages.

**`check_reduced_gb(G, order) -> bool`:** Calls `reduced_groebner(G, order)` and compares to G. Catches bugs in h-degree vs g-degree and monic normalization.

**`ideal_equal(F, G) -> bool`:** Computes reduced GBs of both, compares. Catches bugs in U₁·P·U₂ construction.

**Stage 1: during generation.** After each sample is produced in `shape_position.py`. Gated by `generation.verify.enabled`. Default: on for MVP, off for paper-scale. Pair is discarded and resampled on failure up to `generation.verify.max_resample_attempts` times.

**Stage 2: during evaluation.** After the model predicts G_pred, `ideal_equal(F, G_pred)` is computed as the "ideal-equality" metric. Gated by `eval.check_ideal_equality`. Runs on full 1K test set (~1K Singular calls per eval run, manageable).

**Not verified at training time.** Training has zero algebra dependency.

### 5.5 Parallelism

Dataset generation is embarrassingly parallel. `gen_dataset.py` spawns `generation.num_workers` processes, each with its own Singular subprocess and RNG seeded from `(global_seed, worker_id)`. Each worker writes its own JSONL shard; a final pass writes `index.json`. Deterministic given the Hydra config.

## 6. Data subsystem

### 6.1 On-disk format (`data/dataset_format.py`)

**Sharded JSONL**, one `(F, G)` pair per line. JSONL chosen over Parquet for human inspectability and streaming simplicity. Dataset size is GB-scale max, not TB-scale.

```
data/<dataset_name>/
├── meta.json                 # Hydra config snapshot, field, n, sizes, schema_version
├── train/
│   ├── shard-00000.jsonl
│   └── ...
├── test/shard-00000.jsonl
└── index.json                # per-shard size, byte offsets for random access
```

Per-line schema:
```json
{
  "id": 42,
  "F": [[[coeff, [e1, ..., en]], ...], ...],
  "G": [[[coeff, [e1, ..., en]], ...], ...],
  "field": "QQ",
  "n": 3,
  "seed": 12345,
  "verified": true
}
```

Coefficients: `["a", "b"]` (string num/denom) for ℚ; `int` for 𝔽ₚ; `float` for ℝ. Schema version lives in `meta.json`; bumping it invalidates old caches.

**Raw polynomials only; no pre-tokenized tensors on disk.** Tokenization happens in `LightningDataModule.setup()`. Keeps the JSONL tokenizer-agnostic so vocab changes do not invalidate the generated dataset.

### 6.2 Tokenizer (`data/tokenizer.py`)

**Prefix notation**, matching the paper's example `{x² − 1/2y, 7/2xy² − 5/3x − 2}` tokenized as `[C1, E2, E0, +, C-1, /, C2, E0, E1, <sep>, C1, E0, E1]`.

**Vocabulary:**
- Variables: `E0, E1, …, E_{n-1}`.
- Exponents: `C0, C1, …, C_{d_max}`.
- Operators: `+, *, /, -`.
- Structural: `<bos>, <eos>, <sep>, <pad>`.
- **Discrete path coefficients:** one token per 𝔽ₚ element (`C0, …, C_{p-1}`). Discrete ℚ is intractable (paper notes >1M integer tokens), so discrete+ℚ is not supported.
- **Hybrid path coefficients:** a single `<coeff>` token; real value rides in a sidecar tensor.

**Sample encoding (dict of tensors):**
```python
{
  "tokens": LongTensor[L],
  "coeff_values": FloatTensor[L],   # 0.0 outside <coeff> positions
  "coeff_mask": BoolTensor[L],
  "target_tokens": LongTensor[M],
  "target_coeff_values": FloatTensor[M],
  "target_coeff_mask": BoolTensor[M],
  "coeff_scale": Float,             # per-sample scale factor, un-applied at decode time
}
```

Discrete mode: `coeff_mask` is all False; value tensors unused. The uniform schema keeps the model interface clean.

**Coefficient scaling** (hybrid path, ℚ/ℝ): per-sample scaling to match training range (paper Sec. 5, "we can scale the coefficients of given polynomials globally so that they match our training coefficient range"). `coeff_scale` stored so predictions can be un-scaled at decode time.

**No truncation.** F and G are tightly coupled; truncation would break the learnability signal. `max_src_len` and `max_tgt_len` are configured high enough to cover the dataset (measured during generation, written to `meta.json`).

### 6.3 PyTorch dataset (`data/torch_dataset.py`)

Map-style `torch.utils.data.Dataset`. Uses `index.json` for O(1) random access. Collation pads to batch max length. Tokenization runs once in `LightningDataModule.setup()` and is cached in memory; at paper-scale (1M samples) this is expected to fit in <10 GB RAM for 𝔽ₚ datasets. For ℚ/ℝ with hybrid encoding the per-sample tensor is larger; if RAM becomes tight we will switch to on-the-fly tokenization in `__getitem__`.

## 7. Model subsystem

### 7.1 Embeddings (`model/embedding.py`)

Two classes sharing `forward(tokens, coeff_values, coeff_mask) -> (B, L, D)`.

**`DiscreteEmbedding`:** `nn.Embedding(vocab_size, D)`. Ignores the value/mask tensors. Used for 𝔽ₚ runs.

**`HybridEmbedding`:**
- `nn.Embedding(vocab_size, D)` for non-coefficient tokens.
- `f_E: ℝ → ℝ^D`, one hidden-layer ReLU MLP (paper Eq. D.1), width `d_model`.
- Output: `where(coeff_mask[..., None], f_E(coeff_values[..., None]), token_embedding(tokens))`.
- The `<coeff>` discrete embedding still exists and is trained (its table entry is unused at coefficient positions but shape uniformity matters).

**Positional embedding:** learned absolute (paper: "absolute positional embedding is learned from scratch"). Added to the token embedding before the encoder/decoder.

### 7.2 Transformer (`model/transformer.py`)

Encoder-decoder, written from scratch in PyTorch. Attention uses `torch.nn.functional.scaled_dot_product_attention` (Flash-Attention on A6000) wrapped in our own multi-head module for transparency.

**Paper defaults** (`model/paper.yaml`): 6/6 layers, 8 heads, d=512, d_ff=2048, dropout 0.1. Pre-LN blocks.

**MVP defaults** (`model/mvp.yaml`): 3/3 layers, 4 heads, d=256, d_ff=1024.

### 7.3 Heads (`model/heads.py`)

**`ClassificationHead`:** `Linear(D, vocab_size)` with weight-tying to the input embedding matrix. Cross-entropy over non-pad positions.

**`RegressionHead`:** `Linear(D, 1)`. MSE only at positions where `target_coeff_mask` is True. Disabled (loss weight 0) in discrete mode; weights still exist for interface uniformity.

**Combined loss:** `L = L_CE + λ · L_MSE`, with `λ = 0.01` (paper). Configurable via `embedding.regression_loss_weight`.

### 7.4 Decoding (`eval/decode.py`)

**Greedy (beam width 1)**, matching the paper.

**Discrete mode:** standard argmax over classification logits.

**Hybrid mode:** at each step, argmax the classification head; if the predicted token is `<coeff>`, read the regression head's scalar as the emitted value; feed both back as the next input. Stop at `<eos>` or `model.max_tgt_len`. Un-scale coefficients by `coeff_scale` before emitting the polynomial set.

## 8. Training subsystem

### 8.1 `GroebnerDataModule` (`train/lit_datamodule.py`)

- `prepare_data`: no-op (dataset generation is a separate script).
- `setup(stage)`: loads `meta.json`, constructs tokenizer, tokenizes all shards into in-memory `Dataset` objects.
- `train_dataloader`, `val_dataloader`, `test_dataloader`: paper batch size 16, `num_workers` from config, pinned memory, persistent workers.

Val split: 1K held-out slice carved from the train shards during generation. Test set stays untouched until the evaluation script.

### 8.2 `GroebnerTransformer` (`train/lit_module.py`)

Standard Lightning module. Highlights:

- `training_step`: forward, combined loss, log `train/loss`, `train/ce`, `train/mse`.
- `validation_step`: teacher-forced forward for loss; greedy decode on first `train.val_decode_subset_size` (default 64) samples per val check for running polynomial-acc and support-acc.
- `on_validation_epoch_end`: log a W&B Table with 8 decoded (F, G_true, G_pred) triples for eyeball checks.
- `configure_optimizers`: AdamW(β=(0.9, 0.999), weight_decay=0), linear decay from `train.lr` (1e-4 paper default) over `total_steps`.
- **Full ideal-equality metric is not computed during training.** Too slow; runs only in the evaluation script.

### 8.3 `scripts/train.py`

```python
@hydra.main(config_path="../configs", config_name="mvp")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    dm = GroebnerDataModule(cfg)
    model = GroebnerTransformer(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu",
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        logger=WandbLogger(**cfg.wandb),
        callbacks=[
            ModelCheckpoint(monitor="val/support_acc", mode="max", save_top_k=2, save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_val=cfg.train.grad_clip,
        val_check_interval=cfg.train.val_check_interval,
    )
    trainer.fit(model, dm)
```

Hydra provides the output directory (`outputs/YYYY-MM-DD/HH-MM-SS/`); checkpoints and the W&B run dir live inside it.

**Resumption:** `trainer.fit(..., ckpt_path=...)`. Config snapshotted by Hydra and W&B.

**Precision:** `bf16-mixed` default on A6000 (native bf16 support).

## 9. Evaluation subsystem

Separate script, not a Lightning callback, because it pulls in Singular and runs on the full 1K test set.

### 9.1 Metrics (`eval/metrics.py`)

1. **Polynomial accuracy:** strict equality of decoded G with ground truth (sets of polynomials, each polynomial term-by-term after sorting).
2. **Support accuracy:** same set of monomial supports, coefficients ignored.
3. **Ideal equality:** `verify.ideal_equal(F, G_pred)`. Singular call per sample. Gated by `eval.check_ideal_equality`.

Per-metric the eval script also reports per-polynomial and per-coefficient accuracy as diagnostic breakdowns.

### 9.2 `scripts/evaluate.py`

```python
@hydra.main(config_path="../configs", config_name="eval")
def main(cfg):
    model = GroebnerTransformer.load_from_checkpoint(cfg.eval.ckpt_path)
    dm = GroebnerDataModule(cfg)
    dm.setup("test")
    results = run_eval(model, dm.test_dataloader(), cfg)
    # W&B: append to original training run if cfg.wandb.run_id is set, otherwise new run.
    # Disk: write JSON summary to cfg.eval.output_json.
```

## 10. Reproducibility

1. Hydra config snapshot per run (`config.yaml` in output dir).
2. Full Hydra config logged to W&B as the run config.
3. `pl.seed_everything(cfg.seed, workers=True)` seeds PyTorch, NumPy, Python.
4. Dataset seed and worker-seed derivation baked into `meta.json`, so regenerating from the same Hydra config produces the same dataset.
5. Git commit and diff captured by W&B automatically.

## 11. Hydra configuration

Each entry point consumes a subset of the top-level config:
- `scripts/gen_dataset.py` uses `data`, `generation`, `seed`, `output_dir`. It ignores `model`, `embedding`, `train`, `wandb`.
- `scripts/train.py` uses `data` (to locate the dataset), `model`, `embedding`, `train`, `wandb`, `seed`. It ignores `generation`.
- `scripts/evaluate.py` uses `data`, `model`, `embedding`, `eval`, `wandb`, `seed`.

Unused groups are still composed (so `mvp.yaml` works for all three scripts), but their values have no effect. This keeps a single top-level preset usable across the pipeline.

```
configs/
├── mvp.yaml                    # top-level MVP preset
├── paper.yaml                  # top-level paper-scale preset
├── eval.yaml                   # top-level evaluation preset
├── data/<kind>_<field>_n<n>.yaml
├── model/{mvp,paper}.yaml
├── embedding/{discrete,hybrid}.yaml
├── train/{mvp,paper}.yaml
├── generation/{mvp,paper}.yaml
└── wandb/default.yaml
```

### 11.1 `configs/mvp.yaml`

```yaml
defaults:
  - data: shape_position_fp7_n2
  - model: mvp
  - embedding: discrete
  - train: mvp
  - generation: mvp
  - wandb: default
  - _self_

seed: 0
output_dir: ${hydra:runtime.output_dir}
```

### 11.2 `configs/data/shape_position_fp7_n2.yaml`

```yaml
kind: shape_position
field: Fp7
n_vars: 2
d_max: 5
d_prime: 3
s_max: 5
term_order: lex
coeff_range_G: [-5, 5]
coeff_range_F: [-100, 100]
density_sigma: 1.0
dataset_name: shape_position_fp7_n2
dataset_path: ${oc.env:LSSS_DATA_DIR,data}/${.dataset_name}
```

### 11.3 `configs/generation/mvp.yaml`

```yaml
n_train: 10000
n_test: 1000
num_workers: 8
verify:
  enabled: true
  check_reduced_gb: true
  check_ideal_equality: true
  resample_on_failure: true
  max_resample_attempts: 10
singular:
  binary: Singular
  timeout_seconds: 30
```

### 11.4 `configs/generation/paper.yaml`

```yaml
n_train: 1000000
n_test: 1000
num_workers: 48
verify:
  enabled: false
  check_reduced_gb: false
  check_ideal_equality: false
  resample_on_failure: false
  max_resample_attempts: 0
singular:
  binary: Singular
  timeout_seconds: 30
```

### 11.5 `configs/model/paper.yaml`

```yaml
n_layers_encoder: 6
n_layers_decoder: 6
n_heads: 8
d_model: 512
d_ff: 2048
dropout: 0.1
max_src_len: 5000
max_tgt_len: 2000
```

### 11.5a `configs/model/mvp.yaml`

```yaml
n_layers_encoder: 3
n_layers_decoder: 3
n_heads: 4
d_model: 256
d_ff: 1024
dropout: 0.1
max_src_len: 2000
max_tgt_len: 1000
```

### 11.5b `configs/wandb/default.yaml`

```yaml
project: lsss-groebner
entity: null                     # fill in with your W&B entity, or leave null for default
name: null                       # Hydra run dir used if null
tags: []
mode: online                     # offline | disabled for CI / local debugging
run_id: null                     # set in eval to append to an existing training run
```

### 11.6 `configs/embedding/hybrid.yaml`

```yaml
kind: hybrid
mlp_hidden_layers: 1
mlp_hidden_width: ${model.d_model}
coeff_scale_mode: per_sample
regression_loss_weight: 0.01
regression_correct_threshold: 0.1
```

### 11.7 `configs/train/paper.yaml`

```yaml
epochs: 8
batch_size: 16
lr: 1.0e-4
schedule: linear_decay
grad_clip: 1.0
precision: bf16-mixed
devices: 1
val_check_interval: 0.25
val_decode_subset_size: 64
```

### 11.8 `configs/eval.yaml`

```yaml
defaults:
  - data: shape_position_fp7_n2
  - model: paper
  - embedding: discrete
  - wandb: default
  - _self_

eval:
  ckpt_path: ???
  check_ideal_equality: true
  beam_width: 1
  max_decode_len: ${model.max_tgt_len}
  output_json: ${hydra:runtime.output_dir}/results.json
```

### 11.9 Command-line examples

```bash
# MVP end-to-end
python scripts/gen_dataset.py --config-name mvp
python scripts/train.py --config-name mvp

# Paper-scale 𝔽₇, n=3
python scripts/gen_dataset.py --config-name paper data=shape_position_fp7_n3
python scripts/train.py --config-name paper data=shape_position_fp7_n3

# Evaluate a checkpoint
python scripts/evaluate.py --config-name eval \
  eval.ckpt_path=outputs/2026-04-18/12-00-00/checkpoints/last.ckpt
```

## 12. Testing strategy

Three tiers; pytest with markers `@pytest.mark.integration` (needs Singular) and `@pytest.mark.gpu` (needs CUDA).

**Unit (fast, pure-Python):**
- `test_polynomial.py`: arithmetic, term order, equality, normalization (Fp/QQ/RR).
- `test_sampling.py`: unimodular matrices have det=1; permutations valid; coefficient ranges respected.
- `test_shape_position.py`: sampled G has shape-position form; `deg(h) > deg(gⱼ)`.
- `test_tokenizer.py`: tokenize(detokenize(x)) = x; prefix notation matches paper example.
- `test_embedding.py`: hybrid embedding shape; gradient flows through both paths.
- `test_model.py`: forward pass shapes; generation loop terminates.

**Integration (needs Singular):**
- `test_singular_bridge.py`: reduced-GB roundtrip on small known examples; `ideal_equal` correct on crafted equal/unequal pairs.
- `test_ideal_equality.py`: full `verify.py` on 20 hand-picked cases.
- `test_generate_pipeline.py`: generate 10 samples with verification on, all pass.

**Smoke (needs GPU, nightly CI):**
- `test_train_smoke.py`: 1 epoch on 100 samples with MVP config; loss decreases, no NaN, W&B disabled.
- `test_eval_smoke.py`: eval a random-init model end-to-end; produces JSON summary.

## 13. Milestones

**M1: Algebra foundations.** `Poly`, Singular bridge, unit tests. Deliverable: round-trip a polynomial through Singular. ~400 LOC + 200 tests.

**M2: Dataset generator.** Alg. 1 (sans FGLM), `verify.py`, `gen_dataset.py`, MVP Hydra config. Deliverable: `gen_dataset.py --config-name mvp` produces 10K verified pairs. ~500 LOC + 300 tests.

**M3: Tokenizer and dataset.** Prefix tokenizer with discrete + hybrid paths, `GroebnerDataModule`. Deliverable: DataLoader yields batched tensors from the M2 dataset. ~400 LOC + 200 tests.

**M4: Model and training.** From-scratch Transformer, embeddings, heads, Lightning module, W&B logging. Deliverable: `train.py --config-name mvp` trains end-to-end, loss decreases. ~800 LOC + 200 tests.

**M5: Evaluation.** Greedy decode, three metrics, evaluation script. Deliverable: `evaluate.py` produces JSON + W&B table. ~300 LOC + 150 tests.

**M6: MVP reproduction.** End-to-end run on shape position, 𝔽₇, n=2, discrete embedding. Expected: non-trivial polynomial- and support-accuracy; ballpark of paper's 93.7% / 95.4% at n=2, with some degradation from smaller MVP model.

**M7 (post-MVP, separate plan):** Paper-scale reproduction across all fields and n∈{2,3,4,5}, both embeddings; reproduce Table 2.

**M8 (future, separate plan):** Sparse-systems extension. Adds sibling generator under `algebra/sparse_*.py` and Hydra group; model/train/eval untouched.

**Total MVP:** ~2400 LOC source + ~1050 LOC tests.

## 14. Risks and open questions

1. **Singular subprocess throughput.** At paper scale with verification off, generation is fast enough. At MVP with verification on, each sample costs 2 Singular calls; needs benchmarking to decide whether to keep verification fully on for 1M-sample paper runs or off with periodic spot-checks.
2. **Hybrid regression accuracy.** Paper reports that hybrid embedding underperforms discrete on polynomial accuracy (regression errors accumulate in autoregressive generation). Our implementation inherits this; we should not expect hybrid to match discrete on 𝔽ₚ metrics.
3. **𝔽₃₁ learnability gap.** Paper reports sharp degradation at 𝔽₃₁ (46-50% accuracy) vs 𝔽₇ (70-90%). This is an open problem in the paper and we expect to reproduce the gap, not close it.
4. **Coefficient scaling for ℚ.** Paper mentions "scale the coefficients globally to match our training coefficient range" but does not specify the scaling rule precisely. We use a per-sample max-abs normalization; may need adjustment during implementation.
5. **FGLM dependence.** MVP avoids FGLM entirely. Adding a non-lex target order post-MVP will require implementing the FGLM wrapper; not expected to be hard (Singular has it built in) but untested.
6. **Schema version migrations.** JSONL schema-version bumps invalidate cached datasets. Mitigation: schema is simple and stable; we document the version in the spec and in `meta.json`.
