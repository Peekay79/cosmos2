from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SeedRunStats:
    """Lightweight diagnostics for a single (parameter, seed) run."""

    rho_V: float
    mean_T: float
    mean_B: float
    frac_intelligent: float
    frac_failed: float
    frac_success: float
    mean_fC_success: float  # may be NaN if no successes
    tail_top1_fraction: float  # 0 if total B == 0


def _pearson_corr_or_zero(
    T: np.ndarray,
    B: np.ndarray,
    *,
    warn_hook: Optional[callable] = None,
    context: str = "",
) -> float:
    """
    Pearson corr(T, B) with required edge-case handling.

    If std(T) == 0, std(B) == 0, or all(B == 0) (extinction), returns 0 and
    emits a warning via warn_hook (if provided).
    """
    std_T = float(np.std(T))
    std_B = float(np.std(B))
    if std_T == 0.0 or std_B == 0.0 or bool(np.all(B == 0)):
        if warn_hook is not None:
            warn_hook(
                "rho_V set to 0 due to zero-variance/extinction"
                + (f" [{context}]" if context else "")
                + f" (std_T={std_T:.6g}, std_B={std_B:.6g}, all_B0={bool(np.all(B==0))})"
            )
        return 0.0

    # corrcoef is stable enough here; we already guarded zero-variance.
    rho = float(np.corrcoef(T, B)[0, 1])
    if not np.isfinite(rho):
        if warn_hook is not None:
            warn_hook(
                "rho_V non-finite; forcing to 0"
                + (f" [{context}]" if context else "")
            )
        return 0.0
    return rho


def simulate_seed(
    *,
    n_ensemble: int,
    r: float,
    s: float,
    p_fail: float,
    rng: np.random.Generator,
    mu_T: float,
    alpha_fc: float,
    beta_fc: float,
    p_int: float,
    gamma_fail: float,
    gamma_succ: float,
    need_tail_metric: bool = False,
    return_arrays: bool = False,
    warn_hook: Optional[callable] = None,
    context: str = "",
) -> tuple[SeedRunStats, Optional[dict[str, np.ndarray]]]:
    """
    Simulate one ensemble draw and return seed-level summary stats.

    Model specification (interpreted literally from prompt):
    - Baseline lifetime T0 ~ Exponential(mean=mu_T)
    - Baseline cluster residence fC_0 ~ Beta(alpha_fc, beta_fc)
    - Intelligence I ~ Bernoulli(p_int)
    - Conditional failure F ~ Bernoulli(p_fail) applied only where I==1
    - Success flag S is I & ~F
    - Lifetime outcome:
        - non-intelligent: T = T0
        - failure: T = gamma_fail * T0
        - success: T = gamma_succ * T0
    - Cluster-residence shift only for successes:
        fC = min(1, fC_0 + s * (1 - fC_0))

      At large s or high fC_0, optimisation saturates.
      This represents the physical bound that a causal footprint cannot spend
      more than 100% of proper time in bound structures.

    - Expected branching (two-regime field vs cluster):
        lambda = T * ((1 - fC) + r * fC) = T * (1 + fC * (r - 1))
      Realised fragmentation:
        B ~ Poisson(lambda)
    """
    if n_ensemble <= 0:
        raise ValueError("n_ensemble must be positive")
    if not (0.0 <= s <= 1.0):
        raise ValueError("s must be in [0, 1]")
    if not (0.0 <= p_fail <= 1.0):
        raise ValueError("p_fail must be in [0, 1]")
    if not (0.0 <= p_int <= 1.0):
        raise ValueError("p_int must be in [0, 1]")
    if mu_T <= 0:
        raise ValueError("mu_T must be > 0")
    if alpha_fc <= 0 or beta_fc <= 0:
        raise ValueError("alpha_fc and beta_fc must be > 0")

    # Baseline draws
    T0 = rng.exponential(scale=mu_T, size=n_ensemble).astype(np.float64, copy=False)
    fC0 = rng.beta(a=alpha_fc, b=beta_fc, size=n_ensemble).astype(np.float64, copy=False)

    # Flags
    I = rng.random(n_ensemble) < p_int
    F = np.zeros(n_ensemble, dtype=bool)
    if np.any(I):
        F[I] = rng.random(int(np.sum(I))) < p_fail
    S = I & (~F)

    # Lifetime outcomes
    T = T0.copy()
    if np.any(F):
        T[F] = gamma_fail * T0[F]
    if np.any(S):
        T[S] = gamma_succ * T0[S]

    # Cluster residence (shift only for successes)
    fC = fC0.copy()
    if np.any(S):
        fC[S] = np.minimum(1.0, fC0[S] + s * (1.0 - fC0[S]))

    if np.any(~np.isfinite(T)) or np.any(~np.isfinite(fC)):
        raise ValueError("Non-finite values encountered in T or fC")

    # Expected branching and realised fragmentation
    lam = T * (1.0 + fC * (r - 1.0))
    if np.any(lam < 0) or np.any(~np.isfinite(lam)):
        raise ValueError("Invalid lambda for Poisson (negative or non-finite)")
    B = rng.poisson(lam=lam).astype(np.int64, copy=False)

    if np.any(~np.isfinite(B)):
        raise ValueError("Non-finite values encountered in B")

    rho_V = _pearson_corr_or_zero(T, B, warn_hook=warn_hook, context=context)

    mean_T = float(np.mean(T))
    mean_B = float(np.mean(B))
    frac_intelligent = float(np.mean(I))
    frac_failed = float(np.mean(F))
    frac_success = float(np.mean(S))
    mean_fC_success = float(np.mean(fC[S])) if np.any(S) else float("nan")

    tail_top1_fraction = float("nan")
    if need_tail_metric:
        total_B = int(np.sum(B))
        if total_B <= 0:
            tail_top1_fraction = 0.0
        else:
            top_n = max(1, int(round(0.01 * n_ensemble)))
            top_vals = np.partition(B, n_ensemble - top_n)[n_ensemble - top_n :]
            tail_top1_fraction = float(np.sum(top_vals) / total_B)

    stats = SeedRunStats(
        rho_V=rho_V,
        mean_T=mean_T,
        mean_B=mean_B,
        frac_intelligent=frac_intelligent,
        frac_failed=frac_failed,
        frac_success=frac_success,
        mean_fC_success=mean_fC_success,
        tail_top1_fraction=tail_top1_fraction,
    )

    arrays: Optional[dict[str, np.ndarray]] = None
    if return_arrays:
        arrays = {
            "T0": T0,
            "T": T,
            "B": B,
            "fC0": fC0,
            "fC": fC,
            "I": I.astype(np.int8, copy=False),
            "F": F.astype(np.int8, copy=False),
            "S": S.astype(np.int8, copy=False),
        }

    # Encourage memory release between seeds.
    del T0, fC0, T, fC, lam, B, I, F, S
    gc.collect()

    return stats, arrays

