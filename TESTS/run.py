"""Plug-and-play entry point: download → benchmark → visualize.

Usage:
    python -m benchmarks.run                      # default: g05_60
    python -m benchmarks.run g05_80
    python -m benchmarks.run g05_60 --steps 5000 --xi0 0.04 --repeats 3
    python -m benchmarks.run g05_60 g05_80        # several datasets in a row
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import benchmark as bench_mod
from benchmarks import download as dl_mod
from benchmarks import visualize as viz_mod
from benchmarks.datasets import DATASETS


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*", default=["g05_60"],
                   help=f"one or more of {list(DATASETS)} (default: g05_60)")
    p.add_argument("--xi0",      type=float, default=None)
    p.add_argument("--p_max",    type=float, default=None)
    p.add_argument("--dt",       type=float, default=None)
    p.add_argument("--steps",    type=int,   default=None)
    p.add_argument("--parallel", type=int,   default=None)
    p.add_argument("--repeats",  type=int,   default=1)
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-show",  action="store_true",
                   help="save plots without opening windows")
    args = p.parse_args(argv[1:])

    for name in args.datasets:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset {name!r}; "
                             f"available: {list(DATASETS)}")
        print(f"\n========== {name} ==========")
        dl_mod.download(name)

        # benchmark.benchmark expects the same argparse Namespace shape
        bench_args = argparse.Namespace(
            dataset=name, xi0=args.xi0, p_max=args.p_max, dt=args.dt,
            steps=args.steps, parallel=args.parallel, repeats=args.repeats,
            seed=args.seed, device=args.device,
        )
        report = bench_mod.benchmark(name, bench_args)
        json_path = bench_mod._save(report)
        viz_mod.visualize(json_path, show=not args.no_show)


if __name__ == "__main__":
    main(sys.argv)
