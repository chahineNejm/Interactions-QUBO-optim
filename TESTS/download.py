"""Download dataset files into <project>/data/<dataset>/.

Usage:
    python -m benchmarks.download                  # all datasets
    python -m benchmarks.download g05_60           # one dataset
    python -m benchmarks.download g05_60 g05_80    # several
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# allow `python benchmarks/download.py` as well as `python -m benchmarks.download`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets import DATASETS, data_root


def download(name: str) -> None:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {list(DATASETS)}")
    spec = DATASETS[name]
    out_dir = data_root() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] target: {out_dir}")
    for i in range(spec["n_inst"]):
        fname = spec["filename"](i)
        out = out_dir / fname
        if out.exists():
            print(f"  [{i+1:>2}/{spec['n_inst']}] {fname}  (cached)")
            continue
        url = spec["url"](i)
        urllib.request.urlretrieve(url, out)
        print(f"  [{i+1:>2}/{spec['n_inst']}] {fname}")


def main(argv: list[str]) -> None:
    names = argv[1:] if len(argv) > 1 else list(DATASETS)
    for name in names:
        download(name)


if __name__ == "__main__":
    main(sys.argv)
