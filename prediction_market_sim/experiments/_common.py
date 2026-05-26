"""Shared helpers for experiment scripts: paths, RNG, plot styling, CSV writing."""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: save figures to files
import matplotlib.pyplot as plt  # noqa: E402

# Make `import srpm` work when running scripts directly from the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def out_path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


def save_fig(fig, filename: str) -> str:
    path = out_path(filename)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def write_csv(filename: str, header: list[str], rows: list[list]) -> str:
    path = out_path(filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  saved {path}")
    return path
