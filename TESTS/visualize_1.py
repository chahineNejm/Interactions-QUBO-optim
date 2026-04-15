"""Generate plots from a benchmark results JSON.

Usage:
    python -m benchmarks.visualize results/g05_60__2026-04-15T14-37-12.json
    python -m benchmarks.visualize --latest                # newest JSON
    python -m benchmarks.visualize --latest --no-show      # save only

Produces three figures next to the JSON:
  *_summary.png    bar charts: optimality gap (%) and success rate (%) per instance
  *_histograms.png cut-value histograms across the ensemble, with optimum overlaid
  *_quality.png    ranked cut-value distribution per instance (cdf-style)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
def _short(name: str) -> str:
    """`g05_60_3` -> `#3`."""
    return "#" + name.split("_")[-1]


# --------------------------------------------------------------------------- #
def plot_summary(report: dict, out_path: Path, show: bool) -> None:
    insts = report["instances"]
    names = [_short(r["instance"]) for r in insts]
    gaps  = [100 * r["gap_rel"]      for r in insts]
    succ  = [100 * r["success_rate"] for r in insts]
    colors = ["tab:green" if g <= 0 else "tab:red" for g in gaps]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(names, gaps, color=colors)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("instance"); axes[0].set_ylabel("relative gap (%)")
    axes[0].set_title(f"{report['dataset']}: optimality gap "
                      f"({report['n_solved']}/{report['n_total']} exact)")

    axes[1].bar(names, succ, color="tab:blue")
    axes[1].set_xlabel("instance"); axes[1].set_ylabel("success rate (%)")
    axes[1].set_title(f"{report['dataset']}: ensemble success rate")
    axes[1].set_yscale("symlog", linthresh=0.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  wrote {out_path.name}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
def plot_histograms(report: dict, out_path: Path, show: bool) -> None:
    insts = report["instances"]
    n = len(insts)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, insts):
        cuts = np.array(r["cuts"])
        ax.hist(cuts, bins=30, color="steelblue", alpha=0.85)
        ax.axvline(r["opt"], color="red", lw=1.5)
        ax.set_title(f"{_short(r['instance'])}  opt={r['opt']}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(insts):]:
        ax.axis("off")
    fig.supxlabel("cut value"); fig.supylabel("count")
    fig.suptitle(f"{report['dataset']}: ensemble cut distribution "
                 f"(red line = known optimum)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_path.name}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
def plot_quality(report: dict, out_path: Path, show: bool) -> None:
    """For each instance, plot the gap-to-opt of every replica, sorted."""
    insts = report["instances"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.cm.viridis
    for k, r in enumerate(insts):
        cuts = np.sort(np.array(r["cuts"]))[::-1]            # best first
        gaps = (r["opt"] - cuts) / r["opt"] * 100
        rank = np.arange(1, len(gaps) + 1) / len(gaps) * 100
        ax.plot(rank, gaps, color=cmap(k / max(1, len(insts) - 1)),
                lw=1.0, label=_short(r["instance"]))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("ensemble percentile (best → worst)")
    ax.set_ylabel("gap to optimum (%)")
    ax.set_title(f"{report['dataset']}: quality of every replica")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_yscale("symlog", linthresh=0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  wrote {out_path.name}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
def visualize(json_path: Path, show: bool = True) -> List[Path]:
    with open(json_path) as f:
        report = json.load(f)
    stem = json_path.stem
    out_dir = json_path.parent
    paths = [
        out_dir / f"{stem}__summary.png",
        out_dir / f"{stem}__histograms.png",
        out_dir / f"{stem}__quality.png",
    ]
    print(f"Visualizing {json_path.name}:")
    plot_summary   (report, paths[0], show)
    plot_histograms(report, paths[1], show)
    plot_quality   (report, paths[2], show)
    return paths


# --------------------------------------------------------------------------- #
def _latest() -> Path:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    files = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("no results JSON found in results/")
    return files[-1]


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", help="path to a results JSON")
    p.add_argument("--latest", action="store_true",
                   help="visualize the most recent results JSON")
    p.add_argument("--no-show", action="store_true",
                   help="save plots but do not open windows")
    args = p.parse_args(argv[1:])
    if args.latest:
        path = _latest()
    elif args.path:
        path = Path(args.path)
    else:
        p.error("provide a JSON path or use --latest")
    visualize(path, show=not args.no_show)


if __name__ == "__main__":
    main(sys.argv)
