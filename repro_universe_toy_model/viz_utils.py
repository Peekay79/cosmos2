"""
Visualisation utilities for repro_universe_toy_model.

This module is intentionally *read-only* with respect to numerical results:
it only reads existing CSVs under results/ and writes figure files under
results/figures/.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


COLOURS = {
    "non_int": "#0072B2",   # blue
    "fail":    "#D55E00",   # orange
    "success": "#009E73",   # green
}


def _try_import_seaborn() -> tuple[bool, object | None]:
    try:
        import seaborn as sns  # type: ignore

        return True, sns
    except ImportError:
        return False, None


def _base_dirs() -> tuple[str, str]:
    """Return (package_dir, results_dir) using absolute paths."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(pkg_dir, "results")
    return pkg_dir, results_dir


def _read_csv_prefer_full(
    full_path: str, *, fallback_path: str | None = None, label: str
) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(full_path)
    except FileNotFoundError:
        if fallback_path is not None:
            try:
                print(f"[WARN] Missing {label} at {full_path}; falling back to {fallback_path}")
                return pd.read_csv(fallback_path)
            except FileNotFoundError:
                pass
        print(f"[WARN] Missing {label} CSV: {full_path}")
        return None


def _pick_main_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has a run_id column and multiple runs, pick the "largest" run by
    (ensemble_n, seeds_n) if available, otherwise keep the first run_id.
    """
    if "run_id" not in df.columns:
        return df
    run_ids = df["run_id"].dropna().unique().tolist()
    if len(run_ids) <= 1:
        return df

    if {"ensemble_n", "seeds_n"}.issubset(df.columns):
        sizes = (
            df.groupby("run_id", dropna=False)[["ensemble_n", "seeds_n"]]
            .max()
            .reset_index()
            .sort_values(["ensemble_n", "seeds_n"], ascending=[False, False])
        )
        chosen = sizes.iloc[0]["run_id"]
        return df[df["run_id"] == chosen].copy()

    # Fallback: stable choice (first run_id by appearance)
    chosen = run_ids[0]
    return df[df["run_id"] == chosen].copy()


def _col(df: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Missing required column (tried: {names}); have: {list(df.columns)}")


def _compute_ci95(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 95% CI for mean_rho_V using std_rho_V and n_seeds:
      mean +/- 1.96 * std / sqrt(n_seeds)
    Returns (lower, upper) arrays (may contain NaNs if inputs missing/invalid).
    """
    mean_col = _col(df, "mean_rho_V")
    std_col = _col(df, "std_rho_V")
    n_col = _col(df, "n_seeds", "seeds_n")

    mean = pd.to_numeric(df[mean_col], errors="coerce").to_numpy()
    std = pd.to_numeric(df[std_col], errors="coerce").to_numpy()
    n = pd.to_numeric(df[n_col], errors="coerce").to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        se = std / np.sqrt(n)
        half = 1.96 * se
        lower = mean - half
        upper = mean + half
    return lower, upper


def _maybe_read_r_crit(results_dir: str) -> Optional[float]:
    """
    Optionally read r_crit from a text file if present.
    We check a couple plausible locations.
    """
    candidates = [
        os.path.join(results_dir, "sweep_r", "r_crit_estimate.txt"),
        os.path.join(results_dir, "r_crit_estimate.txt"),
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            val = float(txt)
            if np.isfinite(val):
                return val
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return None


def _estimate_r_crit_from_sign_flip(df_r: pd.DataFrame) -> Tuple[Optional[float], bool]:
    """
    Estimate r_crit by looking for a sign-flip in mean_rho_V across adjacent r.
    Returns (r_crit_or_none, saw_sign_flip_bool).
    """
    r_col = _col(df_r, "r")
    y_col = _col(df_r, "mean_rho_V")

    d = df_r[[r_col, y_col]].copy()
    d[r_col] = pd.to_numeric(d[r_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna().sort_values(r_col)
    if len(d) < 2:
        return None, False

    r = d[r_col].to_numpy()
    y = d[y_col].to_numpy()

    # Treat exact zeros as tiny (so we can still detect flips across 0).
    y_adj = y.copy()
    y_adj[y_adj == 0.0] = 0.0
    sign = np.sign(y_adj)

    # Find first adjacent pair with opposite non-zero sign
    for i in range(len(sign) - 1):
        if sign[i] == 0 or sign[i + 1] == 0:
            continue
        if sign[i] * sign[i + 1] < 0:
            y1, y2 = y[i], y[i + 1]
            r1, r2 = r[i], r[i + 1]
            if y2 == y1:
                return float((r1 + r2) / 2.0), True
            # Linear interpolation to y=0
            return float(r1 + (0.0 - y1) * (r2 - r1) / (y2 - y1)), True
    return None, False


def generate_main_figure():
    """
    Load existing CSV outputs (if present) and generate the 4-panel Figure 1.

    The function must not modify any CSVs. It only reads from results/ and
    writes Figure_1_Main.png / Figure_1_Main.pdf under results/figures/.
    """

    # Imports and style
    has_sns, sns = _try_import_seaborn()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if has_sns:
        sns.set_theme(style="whitegrid")  # type: ignore[union-attr]

    # rcParams should override seaborn theme if seaborn is present
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
            "grid.linewidth": 0.6,
        }
    )

    _, results_dir = _base_dirs()

    # Data sources (prefer full; optionally fall back to _quick if present)
    sweep_r_path = os.path.join(results_dir, "sweep_r", "sweep_r_summary.csv")
    sweep_s_path = os.path.join(results_dir, "sweep_s", "sweep_s_summary.csv")
    sweep_pfail_path = os.path.join(results_dir, "sweep_pfail", "sweep_pfail_summary.csv")
    scatter_path = os.path.join(
        results_dir, "scatter_sample", "scatter_r2.0_s0.5_pfail0.5.csv"
    )

    df_r = _read_csv_prefer_full(
        sweep_r_path,
        fallback_path=os.path.join(results_dir, "sweep_r", "sweep_r_summary_quick.csv"),
        label="sweep_r_summary",
    )
    df_s = _read_csv_prefer_full(
        sweep_s_path,
        fallback_path=os.path.join(results_dir, "sweep_s", "sweep_s_summary_quick.csv"),
        label="sweep_s_summary",
    )
    df_p = _read_csv_prefer_full(
        sweep_pfail_path,
        fallback_path=os.path.join(results_dir, "sweep_pfail", "sweep_pfail_summary_quick.csv"),
        label="sweep_pfail_summary",
    )
    df_sc = _read_csv_prefer_full(
        scatter_path,
        fallback_path=os.path.join(
            results_dir, "scatter_sample", "scatter_r2.0_s0.5_pfail0.5_quick.csv"
        ),
        label="scatter_sample",
    )

    if df_r is None and df_s is None and df_p is None and df_sc is None:
        print("[WARN] No result CSVs found; skipping figure generation.")
        return

    # Figure layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_r, ax_s, ax_pfail, ax_scatter = axes.flat
    any_panel = False

    # Panel (A): rho_V vs r
    try:
        if df_r is None:
            raise FileNotFoundError(sweep_r_path)
        d = _pick_main_run(df_r)
        r_col = _col(d, "r")
        d[r_col] = pd.to_numeric(d[r_col], errors="coerce")
        d = d.dropna(subset=[r_col]).sort_values(r_col)

        ci_lo, ci_hi = _compute_ci95(d)
        y = pd.to_numeric(d[_col(d, "mean_rho_V")], errors="coerce").to_numpy()
        x = pd.to_numeric(d[r_col], errors="coerce").to_numpy()
        rho_lower, rho_upper = ci_lo, ci_hi
        yerr = np.vstack([y - rho_lower, rho_upper - y])

        ax_r.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="-o",
            capsize=3,
            color=COLOURS["success"],
            ecolor=COLOURS["success"],
            markersize=4.5,
        )
        ax_r.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax_r.set_xlabel(r"$r$ (cluster advantage)")
        ax_r.set_ylabel(r"$\rho_V$")
        ax_r.set_title(r"(A) Correlation vs cluster advantage $r$")
        # Light horizontal gridlines only (journal-like)
        ax_r.grid(True, axis="y", alpha=0.20)
        ax_r.grid(False, axis="x")

        # Explicitly annotate the tested r-range (reviewer-proof cue)
        if len(x) > 0 and np.isfinite(x).any():
            r_min = float(np.nanmin(x))
            r_max = float(np.nanmax(x))
            ax_r.text(
                0.02,
                0.04,
                rf"Tested range: {r_min:.1f} ≤ r ≤ {r_max:.1f}",
                transform=ax_r.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.6),
            )

        r_crit = _maybe_read_r_crit(results_dir)
        if r_crit is None:
            r_crit, saw_flip = _estimate_r_crit_from_sign_flip(d)
        else:
            saw_flip = True

        if r_crit is not None:
            ax_r.axvline(
                r_crit,
                linestyle="--",
                color=COLOURS["fail"],
                linewidth=1.0,
            )
            ax_r.text(
                0.98,
                0.95,
                fr"$r_{{crit}} \approx {r_crit:.3g}$",
                transform=ax_r.transAxes,
                fontsize=9,
                va="top",
                ha="right",
            )
        else:
            # No sign-flip observed
            # Place inside axes, top-left, in axes coordinates so it never overlaps the curve
            ax_r.text(
                0.02, 0.95,
                "No sign-flip in sampled r-range",
                transform=ax_r.transAxes,
                ha="left", va="top",
                fontsize=9,
            )

        any_panel = True
    except FileNotFoundError:
        print(f"[WARN] sweep_r panel skipped (missing CSV): {sweep_r_path}")
        ax_r.set_axis_off()
    except Exception as e:
        print(f"[WARN] sweep_r panel skipped (error): {e}")
        ax_r.set_axis_off()

    # Panel (B): rho_V vs s
    try:
        if df_s is None:
            raise FileNotFoundError(sweep_s_path)
        d = _pick_main_run(df_s)
        s_col = _col(d, "s")
        d[s_col] = pd.to_numeric(d[s_col], errors="coerce")
        d = d.dropna(subset=[s_col]).sort_values(s_col)

        ci_lo, ci_hi = _compute_ci95(d)
        y = pd.to_numeric(d[_col(d, "mean_rho_V")], errors="coerce").to_numpy()
        x = pd.to_numeric(d[s_col], errors="coerce").to_numpy()
        rho_lower, rho_upper = ci_lo, ci_hi
        yerr = np.vstack([y - rho_lower, rho_upper - y])

        ax_s.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="-o",
            capsize=3,
            color=COLOURS["success"],
            ecolor=COLOURS["success"],
            markersize=4.5,
        )
        ax_s.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax_s.set_xlabel(r"$s$ (optimisation strength)")
        ax_s.set_ylabel(r"$\rho_V$")
        ax_s.set_title(r"(B) Correlation vs optimisation strength $s$")
        # Light horizontal gridlines only (journal-like)
        ax_s.grid(True, axis="y", alpha=0.20)
        ax_s.grid(False, axis="x")
        any_panel = True
    except FileNotFoundError:
        print(f"[WARN] sweep_s panel skipped (missing CSV): {sweep_s_path}")
        ax_s.set_axis_off()
    except Exception as e:
        print(f"[WARN] sweep_s panel skipped (error): {e}")
        ax_s.set_axis_off()

    # Panel (C): rho_V vs p_fail (bar)
    try:
        if df_p is None:
            raise FileNotFoundError(sweep_pfail_path)
        d = _pick_main_run(df_p)
        pf_col = _col(d, "p_fail")
        d[pf_col] = pd.to_numeric(d[pf_col], errors="coerce")
        d = d.dropna(subset=[pf_col]).sort_values(pf_col)

        ci_lo, ci_hi = _compute_ci95(d)
        y = pd.to_numeric(d[_col(d, "mean_rho_V")], errors="coerce").to_numpy()
        x = pd.to_numeric(d[pf_col], errors="coerce").to_numpy()
        rho_lower, rho_upper = ci_lo, ci_hi
        yerr = np.vstack([y - rho_lower, rho_upper - y])

        # Use actual p_fail values for positioning so tick spacing matches the sweep
        ax_pfail.bar(x, y, width=0.12, color=COLOURS["fail"], alpha=0.9)
        ax_pfail.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="#222222",
            elinewidth=1.3,
            capsize=3,
            capthick=1.3,
            alpha=0.9,
        )
        ax_pfail.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax_pfail.set_xticks(x)
        ax_pfail.set_xticklabels([f"{v:.1f}" for v in x])
        ax_pfail.set_xlabel(r"$p_{\mathrm{fail}}$")
        ax_pfail.set_ylabel(r"$\rho_V$")
        ax_pfail.set_title(r"(C) Robustness to failure probability")
        ax_pfail.grid(True, axis="y", alpha=0.20)
        ax_pfail.grid(False, axis="x")

        # Numeric labels above bars (make small differences obvious)
        for xv, yv in zip(x, y):
            if not np.isfinite(xv) or not np.isfinite(yv):
                continue
            ax_pfail.text(
                float(xv),
                float(yv) + 0.01,
                f"{float(yv):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Ensure labels are not clipped by the y-limits
        if len(rho_upper) > 0 and np.isfinite(rho_upper).any():
            top = float(np.nanmax(rho_upper) + 0.05)
            bottom, _ = ax_pfail.get_ylim()
            ax_pfail.set_ylim(bottom=bottom, top=top)

        any_panel = True
    except FileNotFoundError:
        print(f"[WARN] sweep_pfail panel skipped (missing CSV): {sweep_pfail_path}")
        ax_pfail.set_axis_off()
    except Exception as e:
        print(f"[WARN] sweep_pfail panel skipped (error): {e}")
        ax_pfail.set_axis_off()

    # Panel (D): scatter (T,B) by category
    try:
        if df_sc is None:
            raise FileNotFoundError(scatter_path)
        df_scatter = df_sc.copy()
        # Validate expected columns exist (keep plotting-only behavior)
        _col(df_scatter, "T")
        _col(df_scatter, "B")
        _col(df_scatter, "category")
        df_scatter["T"] = pd.to_numeric(df_scatter["T"], errors="coerce")
        df_scatter["B"] = pd.to_numeric(df_scatter["B"], errors="coerce")
        df_scatter = df_scatter.dropna(subset=["T", "B", "category"])

        cats = ["non_int", "fail", "success"]
        df_cats = {c: df_scatter[df_scatter["category"] == c] for c in cats}

        # non_int
        ax_scatter.scatter(
            df_cats["non_int"]["T"],
            df_cats["non_int"]["B"],
            s=9,
            alpha=0.25,
            color=COLOURS["non_int"],
            linewidths=0,
            label="non-intelligent",
        )

        # fail
        ax_scatter.scatter(
            df_cats["fail"]["T"],
            df_cats["fail"]["B"],
            s=9,
            alpha=0.30,
            color=COLOURS["fail"],
            linewidths=0,
            label="failed intelligent",
        )

        # success
        ax_scatter.scatter(
            df_cats["success"]["T"],
            df_cats["success"]["B"],
            s=24,
            alpha=0.85,
            color=COLOURS["success"],
            linewidths=0,
            label="successful intelligent",
        )

        ax_scatter.set_xlabel(r"$T$ (lifetime)")
        ax_scatter.set_ylabel(r"$B$ (fragmentation count)")
        ax_scatter.set_title("(D) Tail dominance in $(T, B)$")
        # Subtle grid; allow both directions for readability here
        ax_scatter.grid(True, axis="both", alpha=0.18)
        leg = ax_scatter.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.98),
            frameon=False,
            fontsize=9,
        )

        # Annotate the sparse successful tail (if present)
        succ = df_scatter[df_scatter["category"] == "success"]
        if not succ.empty:
            ax_scatter.annotate(
                "Sparse high-T, high-B 'success' tail",
                xy=(0.80, 0.80),
                xycoords="axes fraction",
                xytext=(0.60, 0.92),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", linewidth=0.8, color="#222222"),
                fontsize=9,
                color="#222222",
            )
        any_panel = True
    except FileNotFoundError:
        print(f"[WARN] scatter panel skipped (missing CSV): {scatter_path}")
        ax_scatter.set_axis_off()
    except Exception as e:
        print(f"[WARN] scatter panel skipped (error): {e}")
        ax_scatter.set_axis_off()

    if not any_panel:
        print("[WARN] No panels could be plotted; skipping figure generation.")
        plt.close(fig)
        return

    fig.tight_layout()

    # Saving the figure
    out_dir = os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "Figure_1_Main.png")
    pdf_path = os.path.join(out_dir, "Figure_1_Main.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIGURE] Saved Figure 1 to {png_path} and {pdf_path}")

