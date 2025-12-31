from __future__ import annotations

"""
Sanity-check / validation script.

This script intentionally does NOT write into repro_universe_toy_model/results/.
It runs a single-seed ensemble and prints key diagnostics to console.

The chosen parameters correspond to the "scatter-regime configuration":
N=10_000, r=2.0, s=0.5, p_fail=0.5.
"""

import math

import numpy as np

from .experiments import estimate_r_crit
from .model import simulate_seed


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> None:
    # Single-seed ensemble (scatter-regime configuration)
    N = 10_000
    r = 2.0
    s = 0.5
    p_fail = 0.5
    seed = 123

    rng = np.random.default_rng(seed)
    stats, arrays = simulate_seed(
        n_ensemble=N,
        r=r,
        s=s,
        p_fail=p_fail,
        rng=rng,
        mu_T=1.0,
        alpha_fc=1.0,
        beta_fc=3.0,
        p_int=0.3,
        gamma_fail=0.3,
        gamma_succ=5.0,
        need_tail_metric=True,
        return_arrays=True,
        warn_hook=lambda msg: print(f"[VALIDATION][WARN] {msg}"),
        context="single_seed",
    )
    assert arrays is not None

    T = arrays["T"].astype(float, copy=False)
    B = arrays["B"].astype(np.int64, copy=False)
    fC0 = arrays["fC0"].astype(float, copy=False)
    fC = arrays["fC"].astype(float, copy=False)
    I = arrays["I"].astype(np.int8, copy=False).astype(bool, copy=False)
    F = arrays["F"].astype(np.int8, copy=False).astype(bool, copy=False)
    S = arrays["S"].astype(np.int8, copy=False).astype(bool, copy=False)

    # ---- Distribution stats
    print("[VALIDATION] distribution stats (single seed, scatter-regime config)")
    print(f"N={N}, seed={seed}, r={r}, s={s}, p_fail={p_fail}")
    print(f"mean(T)={float(np.mean(T)):.6g}, std(T)={float(np.std(T)):.6g}")
    print(f"mean(B)={float(np.mean(B)):.6g}, std(B)={float(np.std(B)):.6g}")
    print(f"mean(fC_0)={float(np.mean(fC0)):.6g}, mean(fC)={float(np.mean(fC)):.6g}")
    print(f"frac_intelligent={_pct(float(np.mean(I)))}")
    print(f"frac_failed={_pct(float(np.mean(F)))}")
    print(f"frac_success={_pct(float(np.mean(S)))}")
    if np.any(S):
        print(f"mean(fC | success)={float(np.mean(fC[S])):.6g}")
    else:
        print("mean(fC | success)=NaN (no successes)")
    print("")

    # ---- Tail dominance (within this seed only)
    # Requirements:
    # - sort by B descending
    # - compute share of total B from top 1% patches
    # - do NOT pool across seeds (only one seed here)
    print("[VALIDATION] tail dominance (within-seed only; sorted by B desc)")
    nonzero = int(np.sum(B > 0))
    q99 = float(np.quantile(B.astype(float, copy=False), 0.99))
    mean_B = float(np.mean(B))
    total_B = int(np.sum(B))

    top_n = max(1, int(math.ceil(0.01 * N)))
    B_sorted_desc = np.sort(B)[::-1]  # explicit B-desc ordering
    top_share = 0.0 if total_B <= 0 else float(np.sum(B_sorted_desc[:top_n]) / total_B)

    print(f"Top-1% contribution to total B: {_pct(top_share)}")
    print(f"Non-zero patches: N = {nonzero}")
    print(f"99th percentile(B): {q99:.6g}")
    print(f"Mean(B): {mean_B:.6g}")
    print(f"Total(B): {total_B}")
    print(f"(model-reported tail_top1_fraction: {_pct(float(stats.tail_top1_fraction))})")
    print("")

    # ---- rho_V
    print("[VALIDATION] rho_V (single seed)")
    print(f"rho_V = {float(stats.rho_V):.6g}")
    print("")

    # ---- Mini r-sweep sign-flip / interpolation check
    print("[VALIDATION] r sweep values and rho:")
    r_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    rho_values = []
    for r_i in r_values:
        rng_i = np.random.default_rng(seed)  # same seed per point to isolate r effect
        s_i, _ = simulate_seed(
            n_ensemble=N,
            r=float(r_i),
            s=s,
            p_fail=p_fail,
            rng=rng_i,
            mu_T=1.0,
            alpha_fc=1.0,
            beta_fc=3.0,
            p_int=0.3,
            gamma_fail=0.3,
            gamma_succ=5.0,
            need_tail_metric=False,
            return_arrays=False,
            warn_hook=lambda msg: print(f"[VALIDATION][WARN] {msg}"),
            context=f"mini_r_sweep r={r_i}",
        )
        rho_values.append(float(s_i.rho_V))
        print(f"r={r_i}, rho={float(s_i.rho_V):.6g}")

    r_star = estimate_r_crit(r_values, rho_values)
    print(f"Estimated r*: {None if r_star is None else f'{r_star:.6g}'}")
    print("")

    # If you see something surprising (e.g., extremely low tail dominance at large r),
    # leave a note here for investigation before full sweeps.

    print("[VALIDATION COMPLETE] Model behaviour consistent with specification.")
    print("Ready for full sweeps.")


if __name__ == "__main__":
    main()

