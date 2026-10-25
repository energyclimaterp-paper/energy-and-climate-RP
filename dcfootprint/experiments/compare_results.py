"""Compare two results folders file by file (CSV numerically, JSON/MD textually).

Usage: python dcfootprint/experiments/compare_results.py <dir_a> <dir_b> [rtol]
Prints, per file: identical | numerically equal within rtol | DIFFERENT (with the largest
relative difference and where it is). Exit code 1 if anything is DIFFERENT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _num_diff(a: pd.DataFrame, b: pd.DataFrame):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return f"shape/columns differ {a.shape} vs {b.shape}"
    worst, where = 0.0, None
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.abs(xv - yv) / np.maximum(np.maximum(np.abs(xv), np.abs(yv)), 1e-12)
            d[both_nan] = 0.0
            if np.nanmax(d, initial=0) > worst:
                worst, where = float(np.nanmax(d)), c
        elif not x.fillna("<NA>").astype(str).equals(y.fillna("<NA>").astype(str)):
            return f"text column {c!r} differs"
    return worst, where


def main(a: Path, b: Path, rtol: float) -> int:
    bad = 0
    names = sorted({p.name for p in a.iterdir() if p.is_file()} | {p.name for p in b.iterdir() if p.is_file()})
    for n in names:
        pa, pb = a / n, b / n
        if not (pa.exists() and pb.exists()):
            print(f"{n:<36} only in {'A' if pa.exists() else 'B'}"); bad += 1; continue
        if pa.read_bytes() == pb.read_bytes():
            print(f"{n:<36} identical"); continue
        if n.endswith(".csv"):
            r = _num_diff(pd.read_csv(pa), pd.read_csv(pb))
            if isinstance(r, str) or r[0] > rtol:
                print(f"{n:<36} DIFFERENT  {r}"); bad += 1
            else:
                print(f"{n:<36} equal within rtol (max rel diff {r[0]:.2e} in {r[1]})")
        elif n.endswith(".json"):
            ja, jb = json.loads(pa.read_text()), json.loads(pb.read_text())
            print(f"{n:<36} {'equal' if ja == jb else 'DIFFERENT'}"); bad += ja != jb
        else:
            print(f"{n:<36} DIFFERENT (text)"); bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9))
