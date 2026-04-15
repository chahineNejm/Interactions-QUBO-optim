"""Dataset registry for the SB benchmark.

Each dataset entry has:
  - N        : number of nodes
  - n_inst   : number of instances
  - url(i)   : URL of the i-th instance file (.csv with header `n,m,opt`)
  - defaults : recommended SB hyperparameters for this dataset

The CSV format (rudy-style with the optimum prepended) is:
    n,m,opt_cut
    i,j,w        # 1-indexed nodes, repeated m times
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import torch

# --------------------------------------------------------------------------- #
# Mirror of the BiqMac g05 files (n=60, 80, 100)
_BASE = ("https://raw.githubusercontent.com/Libsmj/"
         "CSCI-4961-GoemansWilliamson/main/Maxcut%20Test%20Data")


def _g05(n: int) -> dict:
    """Build the descriptor for the g05_n family (10 instances)."""
    return {
        "N": n,
        "n_inst": 10,
        "url": lambda i, n=n: f"{_BASE}/g05_{n}_{i}.csv",
        "filename": lambda i, n=n: f"g05_{n}_{i}.csv",
        "defaults": {
            60:  dict(xi0=0.05,  p_max=1.0, dt=0.15, n_steps=3000, n_parallel=1024),
            80:  dict(xi0=0.04,  p_max=1.0, dt=0.15, n_steps=5000, n_parallel=1024),
            100: dict(xi0=0.035, p_max=1.0, dt=0.15, n_steps=8000, n_parallel=1024),
        }[n],
    }


DATASETS: Dict[str, dict] = {
    "g05_60":  _g05(60),
    "g05_80":  _g05(80),
    "g05_100": _g05(100),
}


# --------------------------------------------------------------------------- #
def data_root() -> Path:
    """Where downloaded files live: <project>/data/."""
    return Path(__file__).resolve().parent.parent / "data"


def instance_path(name: str, i: int) -> Path:
    return data_root() / name / DATASETS[name]["filename"](i)


def load_instance(name: str, i: int) -> Tuple[torch.Tensor, float, int]:
    """Return (J, W_total, opt_cut) for the i-th instance of dataset `name`.

    Convention: J is the symmetric coupling matrix with J_ij = -w_ij,
    matching the report's Ising form H = -1/2 s^T J s - h^T s.
    For Max-Cut: cut(s) = (W_total - H(s)) / 2.
    """
    path = instance_path(name, i)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m benchmarks.download {name}`."
        )
    rows = list(csv.reader(open(path)))
    n, _, opt = int(rows[0][0]), int(rows[0][1]), int(rows[0][2])
    J = torch.zeros(n, n)
    W = 0.0
    for ii, jj, ww in rows[1:]:
        ii, jj, ww = int(ii) - 1, int(jj) - 1, float(ww)
        J[ii, jj] -= ww
        J[jj, ii] -= ww
        W += ww
    return J, W, opt


def cuts_from_energies(E, W) -> "np.ndarray | torch.Tensor":
    """cut = (W_total - H_Ising) / 2"""
    return 0.5 * (W - E)
