from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .io_utils import (
    atomic_write_csv,
    read_csv_or_empty,
    results_root,
    seed_indices_present,
    upsert_row,
)
from .model import simulate_seed


def _mode_suffix(mode: str) -> str:
    return "" if mode == "full" else f"_{mode}"


def _n(x: float) -> float:
    """
    Normalise parameter floats for stable CSV resume keys.

    This avoids mismatches like 0.3 vs 0.30000000000000004 after CSV round-trips.
    """
    return float(round(float(x), 12))


def _seed_for(*, base_seed: int, sweep_offset: int, seed_index: int) -> int:
    # Deterministic seeding rule (paired across parameter values within each sweep type).
    return int(base_seed + sweep_offset * 1000 + seed_index)


def _std_across_seeds(x: pd.Series) -> float:
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1))


def _nanmean(x: pd.Series) -> float:
    arr = x.to_numpy(dtype=float, copy=False)
    if np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _aggregate_seed_rows(seed_rows: pd.DataFrame) -> dict:
    n_seeds = int(len(seed_rows))
    out = {
        "mean_rho_V": float(seed_rows["rho_V"].mean()) if n_seeds else float("nan"),
        "std_rho_V": _std_across_seeds(seed_rows["rho_V"]) if n_seeds else float("nan"),
        "mean_T": float(seed_rows["mean_T"].mean()) if n_seeds else float("nan"),
        "mean_B": float(seed_rows["mean_B"].mean()) if n_seeds else float("nan"),
        "frac_intelligent": float(seed_rows["frac_intelligent"].mean()) if n_seeds else float("nan"),
        "frac_failed": float(seed_rows["frac_failed"].mean()) if n_seeds else float("nan"),
        "frac_success": float(seed_rows["frac_success"].mean()) if n_seeds else float("nan"),
        "mean_fC_success": _nanmean(seed_rows["mean_fC_success"]) if n_seeds else float("nan"),
        "tail_top1_fraction": _nanmean(seed_rows["tail_top1_fraction"]) if n_seeds else float("nan"),
        "n_seeds": n_seeds,
    }
    return out


def _param_mask(df: pd.DataFrame, *, r: float, s: float, p_fail: float) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    r = _n(r)
    s = _n(s)
    p_fail = _n(p_fail)
    return (df["r"].round(12) == r) & (df["s"].round(12) == s) & (df["p_fail"].round(12) == p_fail)


def _ensure_csv_with_headers(path: Path, columns: list[str]) -> None:
    if path.exists():
        return
    atomic_write_csv(pd.DataFrame(columns=columns), path)


def _append_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([row])
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def run_sweep_r(
    *,
    mode: str,
    n_ensemble: int,
    n_seeds: int,
    logger_warn: Callable[[str], None],
    logger_info: Callable[[str], None],
) -> None:
    """Sweep 1 — ρ vs r (Sign-Flip Sweep)."""
    sweep_type = "sweep_r"
    sweep_offset = 0
    run_id = f"vacuum_toy_{mode}"

    out_dir = results_root() / "sweep_r"
    suffix = _mode_suffix(mode)
    summary_path = out_dir / f"sweep_r_summary{suffix}.csv"
    seed_path = out_dir / f"sweep_r_seed_diagnostics{suffix}.csv"

    summary_cols = [
        "run_id",
        "sweep_type",
        "r",
        "s",
        "p_fail",
        "mean_rho_V",
        "std_rho_V",
        "mean_T",
        "mean_B",
        "frac_intelligent",
        "frac_failed",
        "frac_success",
        "mean_fC_success",
        "n_seeds",
    ]
    seed_cols = [
        "run_id",
        "sweep_type",
        "r",
        "s",
        "p_fail",
        "seed_index",
        "seed",
        "rho_V",
        "mean_T",
        "mean_B",
        "frac_intelligent",
        "frac_failed",
        "frac_success",
        "mean_fC_success",
        "tail_top1_fraction",
    ]

    s_fixed = 0.5
    p_fail_fixed = 0.5

    logger_info(f"START sweep_r: r in {config.R_VALUES}, s={s_fixed}, p_fail={p_fail_fixed}")

    _ensure_csv_with_headers(seed_path, seed_cols)
    _ensure_csv_with_headers(summary_path, summary_cols)

    seed_df = read_csv_or_empty(seed_path, expected_columns=seed_cols)
    summary_df = read_csv_or_empty(summary_path, expected_columns=summary_cols)

    s_fixed_n = _n(s_fixed)
    p_fail_fixed_n = _n(p_fail_fixed)

    for r in tqdm(config.R_VALUES, desc=f"{sweep_type}{suffix}", leave=True):
        r_n = _n(r)
        key = {"r": r_n, "s": s_fixed_n, "p_fail": p_fail_fixed_n}
        present = seed_indices_present(seed_df, key=key, seed_index_col="seed_index")
        if len(present) >= n_seeds:
            continue

        missing = [i for i in range(n_seeds) if i not in present]
        logger_info(f"{sweep_type}: running r={r} missing_seeds={missing}")

        for seed_index in missing:
            seed = _seed_for(base_seed=config.BASE_SEED, sweep_offset=sweep_offset, seed_index=seed_index)
            rng = np.random.default_rng(seed)

            stats, _ = simulate_seed(
                n_ensemble=n_ensemble,
                r=r_n,
                s=s_fixed_n,
                p_fail=p_fail_fixed_n,
                rng=rng,
                mu_T=config.MU_T,
                alpha_fc=config.ALPHA_FC,
                beta_fc=config.BETA_FC,
                p_int=config.P_INT,
                gamma_fail=config.GAMMA_FAIL,
                gamma_succ=config.GAMMA_SUCC,
                need_tail_metric=False,
                return_arrays=False,
                warn_hook=logger_warn,
                context=f"{sweep_type} r={r} seed_index={seed_index}",
            )

            row = {
                "run_id": run_id,
                "sweep_type": sweep_type,
                "r": r_n,
                "s": s_fixed_n,
                "p_fail": p_fail_fixed_n,
                "seed_index": int(seed_index),
                "seed": int(seed),
                "rho_V": float(stats.rho_V),
                "mean_T": float(stats.mean_T),
                "mean_B": float(stats.mean_B),
                "frac_intelligent": float(stats.frac_intelligent),
                "frac_failed": float(stats.frac_failed),
                "frac_success": float(stats.frac_success),
                "mean_fC_success": float(stats.mean_fC_success),
                "tail_top1_fraction": float("nan"),
            }
            seed_df = _append_row(seed_df, row)
            atomic_write_csv(seed_df, seed_path)

        # Recompute aggregated row for this r and upsert into summary
        mask = _param_mask(seed_df, r=r_n, s=s_fixed_n, p_fail=p_fail_fixed_n)
        seed_rows = seed_df.loc[mask].copy()
        agg = _aggregate_seed_rows(seed_rows)
        summary_row = {
            "run_id": run_id,
            "sweep_type": sweep_type,
            "r": r_n,
            "s": s_fixed_n,
            "p_fail": p_fail_fixed_n,
            "mean_rho_V": agg["mean_rho_V"],
            "std_rho_V": agg["std_rho_V"],
            "mean_T": agg["mean_T"],
            "mean_B": agg["mean_B"],
            "frac_intelligent": agg["frac_intelligent"],
            "frac_failed": agg["frac_failed"],
            "frac_success": agg["frac_success"],
            "mean_fC_success": agg["mean_fC_success"],
            "n_seeds": agg["n_seeds"],
        }
        summary_df = upsert_row(summary_df, summary_row, key_cols=["sweep_type", "r", "s", "p_fail"])
        summary_df = summary_df[summary_cols]
        atomic_write_csv(summary_df, summary_path)

    # Interpolate r* only if sign flip exists in the completed range.
    done = summary_df.copy()
    done = done[
        (done["sweep_type"] == sweep_type)
        & (done["s"] == s_fixed_n)
        & (done["p_fail"] == p_fail_fixed_n)
    ]
    done = done.sort_values("r")
    done = done[done["n_seeds"] >= n_seeds]
    if len(done) >= 2:
        rhos = done["mean_rho_V"].to_numpy(dtype=float, copy=False)
        rs = done["r"].to_numpy(dtype=float, copy=False)
        has_neg = bool(np.any(rhos < 0))
        has_pos = bool(np.any(rhos > 0))
        if not (has_neg and has_pos):
            logger_warn("sweep_r: no sign-flip in range (no r* estimated)")
        else:
            r_star: Optional[float] = None
            for i in range(len(rs) - 1):
                y1, y2 = rhos[i], rhos[i + 1]
                if (y1 <= 0.0 <= y2) or (y2 <= 0.0 <= y1):
                    if y1 == y2:
                        r_star = float(rs[i])
                    else:
                        r_star = float(rs[i] + (0.0 - y1) * (rs[i + 1] - rs[i]) / (y2 - y1))
                    break
            if r_star is None or not math.isfinite(r_star):
                logger_warn("sweep_r: sign flip detected but interpolation failed")
            else:
                logger_info(f"sweep_r: estimated r* ≈ {r_star:.6g}")

    logger_info("END sweep_r")


def run_sweep_s(
    *,
    mode: str,
    n_ensemble: int,
    n_seeds: int,
    logger_warn: Callable[[str], None],
    logger_info: Callable[[str], None],
) -> None:
    """Sweep 2 — ρ vs s (Optimisation Strength)."""
    sweep_type = "sweep_s"
    sweep_offset = 1
    run_id = f"vacuum_toy_{mode}"

    out_dir = results_root() / "sweep_s"
    suffix = _mode_suffix(mode)
    summary_path = out_dir / f"sweep_s_summary{suffix}.csv"
    seed_path = out_dir / f"sweep_s_seed_diagnostics{suffix}.csv"

    summary_cols = [
        "run_id",
        "sweep_type",
        "r",
        "s",
        "p_fail",
        "mean_rho_V",
        "std_rho_V",
        "mean_T",
        "mean_B",
        "mean_fC_success",
        "frac_success",
        "n_seeds",
    ]
    seed_cols = [
        "run_id",
        "sweep_type",
        "r",
        "s",
        "p_fail",
        "seed_index",
        "seed",
        "rho_V",
        "mean_T",
        "mean_B",
        "frac_intelligent",
        "frac_failed",
        "frac_success",
        "mean_fC_success",
        "tail_top1_fraction",
    ]

    r_fixed = 2.0
    p_fail_fixed = 0.5

    logger_info(f"START sweep_s: s in {config.S_VALUES}, r={r_fixed}, p_fail={p_fail_fixed}")

    _ensure_csv_with_headers(seed_path, seed_cols)
    _ensure_csv_with_headers(summary_path, summary_cols)

    seed_df = read_csv_or_empty(seed_path, expected_columns=seed_cols)
    summary_df = read_csv_or_empty(summary_path, expected_columns=summary_cols)

    r_fixed_n = _n(r_fixed)
    p_fail_fixed_n = _n(p_fail_fixed)

    for s in tqdm(config.S_VALUES, desc=f"{sweep_type}{suffix}", leave=True):
        s_n = _n(s)
        key = {"r": r_fixed_n, "s": s_n, "p_fail": p_fail_fixed_n}
        present = seed_indices_present(seed_df, key=key, seed_index_col="seed_index")
        if len(present) >= n_seeds:
            continue

        missing = [i for i in range(n_seeds) if i not in present]
        logger_info(f"{sweep_type}: running s={s} missing_seeds={missing}")

        for seed_index in missing:
            seed = _seed_for(base_seed=config.BASE_SEED, sweep_offset=sweep_offset, seed_index=seed_index)
            rng = np.random.default_rng(seed)

            stats, _ = simulate_seed(
                n_ensemble=n_ensemble,
                r=r_fixed_n,
                s=s_n,
                p_fail=p_fail_fixed_n,
                rng=rng,
                mu_T=config.MU_T,
                alpha_fc=config.ALPHA_FC,
                beta_fc=config.BETA_FC,
                p_int=config.P_INT,
                gamma_fail=config.GAMMA_FAIL,
                gamma_succ=config.GAMMA_SUCC,
                need_tail_metric=False,
                return_arrays=False,
                warn_hook=logger_warn,
                context=f"{sweep_type} s={s} seed_index={seed_index}",
            )

            row = {
                "run_id": run_id,
                "sweep_type": sweep_type,
                "r": r_fixed_n,
                "s": s_n,
                "p_fail": p_fail_fixed_n,
                "seed_index": int(seed_index),
                "seed": int(seed),
                "rho_V": float(stats.rho_V),
                "mean_T": float(stats.mean_T),
                "mean_B": float(stats.mean_B),
                "frac_intelligent": float(stats.frac_intelligent),
                "frac_failed": float(stats.frac_failed),
                "frac_success": float(stats.frac_success),
                "mean_fC_success": float(stats.mean_fC_success),
                "tail_top1_fraction": float("nan"),
            }
            seed_df = _append_row(seed_df, row)
            atomic_write_csv(seed_df, seed_path)

        mask = _param_mask(seed_df, r=r_fixed_n, s=s_n, p_fail=p_fail_fixed_n)
        seed_rows = seed_df.loc[mask].copy()
        agg = _aggregate_seed_rows(seed_rows)
        summary_row = {
            "run_id": run_id,
            "sweep_type": sweep_type,
            "r": r_fixed_n,
            "s": s_n,
            "p_fail": p_fail_fixed_n,
            "mean_rho_V": agg["mean_rho_V"],
            "std_rho_V": agg["std_rho_V"],
            "mean_T": agg["mean_T"],
            "mean_B": agg["mean_B"],
            "mean_fC_success": agg["mean_fC_success"],
            "frac_success": agg["frac_success"],
            "n_seeds": agg["n_seeds"],
        }
        summary_df = upsert_row(summary_df, summary_row, key_cols=["sweep_type", "r", "s", "p_fail"])
        summary_df = summary_df[summary_cols]
        atomic_write_csv(summary_df, summary_path)

    logger_info("END sweep_s")


def run_sweep_pfail(
    *,
    mode: str,
    n_ensemble: int,
    n_seeds: int,
    logger_warn: Callable[[str], None],
    logger_info: Callable[[str], None],
) -> None:
    """Sweep 3 — ρ vs p_fail (Robustness & Tail Dominance)."""
    sweep_type = "sweep_pfail"
    sweep_offset = 2
    run_id = f"vacuum_toy_{mode}"

    out_dir = results_root() / "sweep_pfail"
    suffix = _mode_suffix(mode)
    summary_path = out_dir / f"sweep_pfail_summary{suffix}.csv"
    seed_path = out_dir / f"sweep_pfail_seed_diagnostics{suffix}.csv"

    summary_cols = [
        "run_id",
        "sweep_type",
        "r",
        "s",
        "p_fail",
        "mean_rho_V",
        "std_rho_V",
        "mean_T",
        "mean_B",
        "frac_success",
        "tail_top1_fraction",
        "n_seeds",
    ]
    seed_cols = [
        "run_id",
        "sweep_type",
        "r",
        "s",
        "p_fail",
        "seed_index",
        "seed",
        "rho_V",
        "mean_T",
        "mean_B",
        "frac_intelligent",
        "frac_failed",
        "frac_success",
        "mean_fC_success",
        "tail_top1_fraction",
    ]

    r_fixed = 2.0
    s_fixed = 0.5

    logger_info(f"START sweep_pfail: p_fail in {config.P_FAIL_VALUES}, r={r_fixed}, s={s_fixed}")

    _ensure_csv_with_headers(seed_path, seed_cols)
    _ensure_csv_with_headers(summary_path, summary_cols)

    seed_df = read_csv_or_empty(seed_path, expected_columns=seed_cols)
    summary_df = read_csv_or_empty(summary_path, expected_columns=summary_cols)

    r_fixed_n = _n(r_fixed)
    s_fixed_n = _n(s_fixed)

    for p_fail in tqdm(config.P_FAIL_VALUES, desc=f"{sweep_type}{suffix}", leave=True):
        p_fail_n = _n(p_fail)
        key = {"r": r_fixed_n, "s": s_fixed_n, "p_fail": p_fail_n}
        present = seed_indices_present(seed_df, key=key, seed_index_col="seed_index")
        if len(present) >= n_seeds:
            continue

        missing = [i for i in range(n_seeds) if i not in present]
        logger_info(f"{sweep_type}: running p_fail={p_fail} missing_seeds={missing}")

        for seed_index in missing:
            seed = _seed_for(base_seed=config.BASE_SEED, sweep_offset=sweep_offset, seed_index=seed_index)
            rng = np.random.default_rng(seed)

            stats, _ = simulate_seed(
                n_ensemble=n_ensemble,
                r=r_fixed_n,
                s=s_fixed_n,
                p_fail=p_fail_n,
                rng=rng,
                mu_T=config.MU_T,
                alpha_fc=config.ALPHA_FC,
                beta_fc=config.BETA_FC,
                p_int=config.P_INT,
                gamma_fail=config.GAMMA_FAIL,
                gamma_succ=config.GAMMA_SUCC,
                need_tail_metric=True,
                return_arrays=False,
                warn_hook=logger_warn,
                context=f"{sweep_type} p_fail={p_fail} seed_index={seed_index}",
            )

            row = {
                "run_id": run_id,
                "sweep_type": sweep_type,
                "r": r_fixed_n,
                "s": s_fixed_n,
                "p_fail": p_fail_n,
                "seed_index": int(seed_index),
                "seed": int(seed),
                "rho_V": float(stats.rho_V),
                "mean_T": float(stats.mean_T),
                "mean_B": float(stats.mean_B),
                "frac_intelligent": float(stats.frac_intelligent),
                "frac_failed": float(stats.frac_failed),
                "frac_success": float(stats.frac_success),
                "mean_fC_success": float(stats.mean_fC_success),
                "tail_top1_fraction": float(stats.tail_top1_fraction),
            }
            seed_df = _append_row(seed_df, row)
            atomic_write_csv(seed_df, seed_path)

        mask = _param_mask(seed_df, r=r_fixed_n, s=s_fixed_n, p_fail=p_fail_n)
        seed_rows = seed_df.loc[mask].copy()
        agg = _aggregate_seed_rows(seed_rows)
        summary_row = {
            "run_id": run_id,
            "sweep_type": sweep_type,
            "r": r_fixed_n,
            "s": s_fixed_n,
            "p_fail": p_fail_n,
            "mean_rho_V": agg["mean_rho_V"],
            "std_rho_V": agg["std_rho_V"],
            "mean_T": agg["mean_T"],
            "mean_B": agg["mean_B"],
            "frac_success": agg["frac_success"],
            "tail_top1_fraction": agg["tail_top1_fraction"],
            "n_seeds": agg["n_seeds"],
        }
        summary_df = upsert_row(summary_df, summary_row, key_cols=["sweep_type", "r", "s", "p_fail"])
        summary_df = summary_df[summary_cols]
        atomic_write_csv(summary_df, summary_path)

    logger_info("END sweep_pfail")


def export_scatter_sample(
    *,
    mode: str,
    n_scatter: int,
    logger_warn: Callable[[str], None],
    logger_info: Callable[[str], None],
) -> Path:
    """Export one patch-level scatter sample dataset (no plotting)."""
    r = 2.0
    s = 0.5
    p_fail = 0.5
    seed = config.BASE_SEED

    out_dir = results_root() / "scatter_sample"
    suffix = _mode_suffix(mode)
    out_path = out_dir / f"scatter_r2.0_s0.5_pfail0.5{suffix}.csv"

    logger_info(f"scatter_sample: generating N={n_scatter} r={r} s={s} p_fail={p_fail} seed={seed}")

    rng = np.random.default_rng(seed)
    stats, arrays = simulate_seed(
        n_ensemble=n_scatter,
        r=r,
        s=s,
        p_fail=p_fail,
        rng=rng,
        mu_T=config.MU_T,
        alpha_fc=config.ALPHA_FC,
        beta_fc=config.BETA_FC,
        p_int=config.P_INT,
        gamma_fail=config.GAMMA_FAIL,
        gamma_succ=config.GAMMA_SUCC,
        need_tail_metric=False,
        return_arrays=True,
        warn_hook=logger_warn,
        context="scatter_sample",
    )
    assert arrays is not None

    I = arrays["I"].astype(np.int8, copy=False)
    F = arrays["F"].astype(np.int8, copy=False)
    Sflag = arrays["S"].astype(np.int8, copy=False)

    category = np.full(n_scatter, "non_int", dtype=object)
    category[I.astype(bool)] = "fail"
    category[Sflag.astype(bool)] = "success"

    df = pd.DataFrame(
        {
            "patch_id": np.arange(n_scatter, dtype=np.int64),
            "T": arrays["T"].astype(np.float64, copy=False),
            "B": arrays["B"].astype(np.int64, copy=False),
            "fC_0": arrays["fC0"].astype(np.float64, copy=False),
            "fC": arrays["fC"].astype(np.float64, copy=False),
            "I": I,
            "F": F,
            "S": Sflag,
            "category": category,
        }
    )

    atomic_write_csv(df, out_path)
    logger_info(
        f"scatter_sample: wrote {out_path.name} (mean_T={stats.mean_T:.6g}, mean_B={stats.mean_B:.6g})"
    )
    return out_path


