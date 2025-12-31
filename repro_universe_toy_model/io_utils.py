from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

import pandas as pd


def package_root() -> Path:
    """Return the package directory (repo-relative results live under this)."""
    return Path(__file__).resolve().parent


def results_root() -> Path:
    return package_root() / "results"


def ensure_results_layout() -> None:
    root = results_root()
    (root / "sweep_r").mkdir(parents=True, exist_ok=True)
    (root / "sweep_s").mkdir(parents=True, exist_ok=True)
    (root / "sweep_pfail").mkdir(parents=True, exist_ok=True)
    (root / "scatter_sample").mkdir(parents=True, exist_ok=True)


def get_logger(*, mode: str = "full") -> logging.Logger:
    """
    Configure and return the diagnostics logger.

    Logs to `results/diagnostics.log` (or `_quick`) and also to stderr.
    """
    ensure_results_layout()
    logger = logging.getLogger("repro_universe_toy_model")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times in-process.
    if getattr(logger, "_configured", False):
        return logger

    log_name = "diagnostics.log" if mode == "full" else f"diagnostics_{mode}.log"
    log_path = results_root() / log_name

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger._configured = True  # type: ignore[attr-defined]
    return logger


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Atomic CSV write (temp file -> rename).

    Ensures an interrupted run never leaves a partial/corrupt CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid4().hex}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_csv_or_empty(path: Path, *, expected_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if not path.exists():
        if expected_columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=list(expected_columns))
    df = pd.read_csv(path)
    if expected_columns is not None:
        for c in expected_columns:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {path}")
    return df


def upsert_row(
    df: pd.DataFrame,
    row: dict,
    *,
    key_cols: list[str],
) -> pd.DataFrame:
    """
    Insert/replace a row in a DataFrame using a composite key.
    Returns a new DataFrame (does not mutate in-place).
    """
    if df.empty:
        return pd.DataFrame([row])

    # Build a boolean mask for matching keys.
    mask = pd.Series(True, index=df.index)
    for k in key_cols:
        if k in df.columns and df[k].dtype.kind == "f" and isinstance(row.get(k), (float, int)):
            # Stable comparisons across CSV round-trips.
            mask &= df[k].round(12) == round(float(row[k]), 12)
        else:
            mask &= df[k] == row[k]

    df2 = df.copy()
    if mask.any():
        idx = df2.index[mask][0]
        for k, v in row.items():
            df2.at[idx, k] = v
    else:
        df2 = pd.concat([df2, pd.DataFrame([row])], ignore_index=True)
    return df2


def seed_indices_present(
    seed_df: pd.DataFrame,
    *,
    key: dict,
    seed_index_col: str = "seed_index",
) -> set[int]:
    if seed_df.empty:
        return set()
    mask = pd.Series(True, index=seed_df.index)
    for k, v in key.items():
        if k in seed_df.columns and seed_df[k].dtype.kind == "f" and isinstance(v, (float, int)):
            mask &= seed_df[k].round(12) == round(float(v), 12)
        else:
            mask &= seed_df[k] == v
    if not mask.any():
        return set()
    return set(int(x) for x in seed_df.loc[mask, seed_index_col].tolist())


