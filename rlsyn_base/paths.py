from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def repo_root() -> Path:
    """
    Return repository root directory (the parent of this package).
    """
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """
    Root-level data directory: <repo>/data
    """
    return repo_root() / "data"


def data_path(*parts: str) -> str:
    """
    Convenience helper returning a string path under <repo>/data.
    (String is convenient for downstream libs that don't accept Path.)
    """
    return str(data_dir().joinpath(*parts))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_smiles_lines(path: str | Path, *, limit: Optional[int] = None) -> list[str]:
    """
    Read one-SMILES-per-line file. Empty lines are skipped.
    """
    out: list[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(s)
            if limit is not None and len(out) >= limit:
                break
    return out


def read_first_column_csv(path: str | Path, *, limit: Optional[int] = None) -> list[str]:
    """
    Read first column from a CSV-like file. This is intentionally lightweight:
    it does not require pandas.
    """
    out: list[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            first = s.split(",")[0].strip()
            # very small heuristic to skip header rows
            if first.lower() in {"smiles", "smi"}:
                continue
            out.append(first)
            if limit is not None and len(out) >= limit:
                break
    return out


