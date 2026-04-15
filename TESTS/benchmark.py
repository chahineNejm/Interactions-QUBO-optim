"""Run the SB solver on every instance of a dataset and save results.

Usage:
    python -m benchmarks.benchmark g05_60
    python -m benchmarks.benchmark g05_60 --steps 5000 --xi0 0.04 --repeats 3
    python -m benchmarks.benchmark g05_60 --device cuda --parallel 2048

Results are written to: <project>/results/<dataset>__<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver.solver import SBConfig, SBSolver
from benchmarks.datasets import (DATASETS, cuts_from_energies, data_root,
                                 load_instance)


# --------------------------------------------------------------------------- #
def _build_config(name: str, args: argparse.Namespace) -> SBConfig:
    """Per-dataset defaults, overridable from the CLI."""
    d = DATASETS[name]["defaults"].copy()
    if args.xi0      is not None: d["xi0"]        = args.xi0
    if args.p_max    is not None: d["p_max"]      = args.p_max
    if args.dt       is not None: d["dt"]         = args.dt
    if args.steps    is not None: d["n_steps"]    = args.steps
    if args.parallel is not None: d["n_parallel"] = args.parallel
    return SBConfig(device=args.device, seed=args.seed, **d)


def _summarize(cuts: np.ndarray, opt: int) -> dict:
    best = float(cuts.max())
    return {
        "best_cut":     best,
        "mean_cut":     float(cuts.mean()),
        "std_cut":      float(cuts.std()),
        "gap_abs":      opt - best,
        "gap_rel":      (opt - best) / opt,
        "success_rate": float((cuts >= opt - 1e-6).mean()),
    }


# --------------------------------------------------------------------------- #
def run_instance(name: str, idx: int, cfg: SBConfig,
                 n_repeats: int, seed0: int) -> dict:
    J, W, opt = load_instance(name, idx)
    cuts = []
    t0 = time.perf_counter()
    for r in range(n_repeats):
        cfg_r = SBConfig(**{**cfg.__dict__, "seed": seed0 + r})
        res = SBSolver(J, config=cfg_r).run()
        cuts.append(cuts_from_energies(res.all_energies.numpy(), W))
    cuts = np.concatenate(cuts)
    out = {
        "instance": f"{name}_{idx}",
        "N":        J.shape[0],
        "opt":      opt,
        "W_total":  W,
        "seconds":  time.perf_counter() - t0,
        "cuts":     cuts.tolist(),
    }
    out.update(_summarize(cuts, opt))
    return out


def benchmark(name: str, args: argparse.Namespace) -> dict:
    cfg = _build_config(name, args)
    print(f"=== {name}  N={DATASETS[name]['N']}  device={cfg.device} ===")
    print(f"hyperparams: K={cfg.K} delta={cfg.delta} xi0={cfg.xi0} "
          f"p_max={cfg.p_max} dt={cfg.dt} steps={cfg.n_steps} "
          f"B={cfg.n_parallel} repeats={args.repeats}")
    print(f"{'instance':<14} {'opt':>5} {'best':>5} {'mean':>7} "
          f"{'gap':>4} {'rel%':>6} {'succ%':>7} {'time(s)':>8}")
    print("-" * 64)

    reports = []
    for i in range(DATASETS[name]["n_inst"]):
        r = run_instance(name, i, cfg, args.repeats, args.seed)
        reports.append(r)
        print(f"{r['instance']:<14} {r['opt']:>5} {r['best_cut']:>5.0f} "
              f"{r['mean_cut']:>7.1f} {r['gap_abs']:>4.0f} "
              f"{100*r['gap_rel']:>5.2f}% {100*r['success_rate']:>6.2f}% "
              f"{r['seconds']:>8.2f}")

    n_solved = sum(r["gap_abs"] <= 0 for r in reports)
    mean_gap = float(np.mean([r["gap_rel"] for r in reports]))
    mean_succ = float(np.mean([r["success_rate"] for r in reports]))
    print("-" * 64)
    print(f"Solved exactly: {n_solved}/{len(reports)}    "
          f"Mean rel. gap: {100*mean_gap:.3f}%    "
          f"Mean success rate: {100*mean_succ:.3f}%")

    return {
        "dataset":      name,
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "config":       {k: (str(v) if isinstance(v, torch.dtype) else v)
                         for k, v in cfg.__dict__.items()},
        "n_repeats":    args.repeats,
        "n_solved":     n_solved,
        "n_total":      len(reports),
        "mean_gap_rel": mean_gap,
        "mean_success": mean_succ,
        "instances":    reports,
    }


# --------------------------------------------------------------------------- #
def _save(report: dict) -> Path:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = report["timestamp"].replace(":", "-")
    path = out_dir / f"{report['dataset']}__{stamp}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {path.relative_to(out_dir.parent)}")
    return path


# --------------------------------------------------------------------------- #
def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(DATASETS),
                   help="dataset name (use `download` to fetch first)")
    p.add_argument("--xi0",      type=float, default=None)
    p.add_argument("--p_max",    type=float, default=None)
    p.add_argument("--dt",       type=float, default=None)
    p.add_argument("--steps",    type=int,   default=None)
    p.add_argument("--parallel", type=int,   default=None)
    p.add_argument("--repeats",  type=int,   default=1)
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> Path:
    args = _parse(argv)
    report = benchmark(args.dataset, args)
    return _save(report)


if __name__ == "__main__":
    main(sys.argv)
