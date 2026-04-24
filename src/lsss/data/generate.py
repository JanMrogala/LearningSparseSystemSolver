from __future__ import annotations

import json
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lsss.algebra.field import Fp, QQ, RR
from lsss.algebra.shape_position import generate_pair
from lsss.algebra.singular import SingularSession
from lsss.algebra.verify import check_generated_pair
from lsss.data.dataset_format import SCHEMA_VERSION, write_shard


def _resolve_field(name: str):
    if name.startswith("Fp"):
        return Fp(int(name[2:]))
    if name == "QQ":
        return QQ()
    if name == "RR":
        return RR()
    raise ValueError(f"Unknown field {name!r}")


def _worker(args):
    (
        worker_id,
        base_seed,
        n_samples,
        cfg_dict,
        split,
        out_path,
    ) = args
    cfg = OmegaConf.create(cfg_dict)
    field = _resolve_field(cfg.data.field)
    rng = random.Random(hash((base_seed, worker_id, split)))

    samples = []
    session_ctx = None
    session = None
    if cfg.generation.verify.enabled:
        session_ctx = SingularSession(
            binary=cfg.generation.singular.binary,
            timeout_seconds=cfg.generation.singular.timeout_seconds,
        )
        session = session_ctx.__enter__()

    try:
        sample_id = worker_id * n_samples
        produced = 0
        attempt_budget = n_samples * (
            cfg.generation.verify.max_resample_attempts + 1
        )
        attempts = 0
        while produced < n_samples and attempts < attempt_budget:
            attempts += 1
            sample_seed = rng.randint(0, 2**31 - 1)
            sample_rng = random.Random(sample_seed)
            F, G = generate_pair(
                n_vars=cfg.data.n_vars,
                d_max=cfg.data.d_max,
                d_prime=cfg.data.d_prime,
                s_max=cfg.data.s_max,
                field=field,
                density_sigma=cfg.data.density_sigma,
                rng=sample_rng,
            )
            verified = False
            if cfg.generation.verify.enabled:
                verified = check_generated_pair(
                    session, F, G,
                    n_vars=cfg.data.n_vars,
                    field=field,
                    order=cfg.data.term_order,
                )
                if not verified:
                    if cfg.generation.verify.resample_on_failure:
                        continue
            samples.append(
                {
                    "id": sample_id,
                    "F": F,
                    "G": G,
                    "field": field,
                    "n": cfg.data.n_vars,
                    "seed": sample_seed,
                    "verified": verified or not cfg.generation.verify.enabled,
                }
            )
            sample_id += 1
            produced += 1
            if produced % 100 == 0:
                print(f"[worker {worker_id} {split}] {produced}/{n_samples}", flush=True)
    finally:
        if session_ctx is not None:
            session_ctx.__exit__(None, None, None)

    write_shard(out_path, samples)
    return len(samples)


def generate_dataset(cfg: DictConfig) -> None:
    dataset_path = Path(cfg.data.dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)

    for split, n_samples in [
        ("train", cfg.generation.n_train),
        ("test", cfg.generation.n_test),
    ]:
        split_dir = dataset_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        num_workers = max(1, cfg.generation.num_workers)
        per_worker = _split_counts(n_samples, num_workers)
        jobs = []
        for w, count in enumerate(per_worker):
            if count == 0:
                continue
            jobs.append(
                (
                    w,
                    cfg.seed,
                    count,
                    cfg_dict,
                    split,
                    split_dir / f"shard-{w:05d}.jsonl",
                )
            )
        if num_workers == 1:
            for job in tqdm(jobs, desc=f"{split}"):
                _worker(job)
        else:
            with Pool(num_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_worker, jobs),
                    total=len(jobs),
                    desc=split,
                ):
                    pass

    meta = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg_dict,
    }
    (dataset_path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _split_counts(total: int, workers: int) -> list[int]:
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]
