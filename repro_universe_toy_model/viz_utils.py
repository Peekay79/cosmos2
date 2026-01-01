from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Colourblind-friendly palette (Okabe–Ito inspired)
COLOURS = {
    "non_int": "#999999",  # grey
    "fail": "#E69F00",  # orange
    "success": "#0072B2",  # blue
}


def _results_root() -> Path:
    return Path(__file__).resolve().parent / "results"


def _figures_dir() -> Path:
    d = _results_root() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _set_journal_style() -> None:
    # If seaborn is available, set a base theme first; then rcParams override wins.
    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid")
    except Exception:
        pass

    import matplotlib as mpl

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


def _ci95_bounds(df: pd.DataFrame, *, mean_col: str, std_col: str, n_col: str) -> tuple[pd.Series, pd.Series]:
    """
    Compute a simple 95% CI for the mean using normal approximation:
      mean ± 1.96 * (std / sqrt(n))

    This does not change any underlying statistics; it only derives plot bounds
    from already-computed summary columns.
    """
    mean = df[mean_col].astype(float)
    std = df[std_col].astype(float)
    n = df[n_col].astype(float)
    sem = std / np.sqrt(np.maximum(n, 1.0))
    half = 1.96 * sem
    return mean - half, mean + half


def generate_main_figure(*, mode: str = "full") -> tuple[Path, Path]:
    """
    Generate Paper 2 Figure 1 (Main): 2×2 panel summary of sweeps + scatter.

    Output paths (fixed):
      - repro_universe_toy_model/results/figures/Figure_1_Main.png
      - repro_universe_toy_model/results/figures/Figure_1_Main.pdf
    """
    if mode != "full":
        raise ValueError("generate_main_figure is intended for mode='full' only.")

    _set_journal_style()

    import matplotlib.pyplot as plt

    # ---- Load data (do not modify CSVs)
    root = _results_root()
    df_r = pd.read_csv(root / "sweep_r" / "sweep_r_summary.csv")
    df_s = pd.read_csv(root / "sweep_s" / "sweep_s_summary.csv")
    df_p = pd.read_csv(root / "sweep_pfail" / "sweep_pfail_summary.csv")
    df_scatter = pd.read_csv(root / "scatter_sample" / "scatter_r2.0_s0.5_pfail0.5.csv")

    # ---- Derive bounds for plotting only
    df_r = df_r.sort_values("r").copy()
    df_s = df_s.sort_values("s").copy()
    df_p = df_p.sort_values("p_fail").copy()

    df_r["rho_lower"], df_r["rho_upper"] = _ci95_bounds(df_r, mean_col="mean_rho_V", std_col="std_rho_V", n_col="n_seeds")
    df_s["rho_lower"], df_s["rho_upper"] = _ci95_bounds(df_s, mean_col="mean_rho_V", std_col="std_rho_V", n_col="n_seeds")
    df_p["rho_lower"], df_p["rho_upper"] = _ci95_bounds(df_p, mean_col="mean_rho_V", std_col="std_rho_V", n_col="n_seeds")

    # ---- Figure layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_r, ax_s, ax_pfail, ax_scatter = axes.flat

    # Light gridlines if seaborn isn't active
    for ax in (ax_r, ax_s, ax_pfail, ax_scatter):
        ax.grid(True, alpha=0.3)

    primary = COLOURS["success"]

    # (A) rho_V vs r
    ax_r.errorbar(
        df_r["r"],
        df_r["mean_rho_V"],
        yerr=[
            df_r["mean_rho_V"] - df_r["rho_lower"],
            df_r["rho_upper"] - df_r["mean_rho_V"],
        ],
        fmt="-o",
        capsize=3,
        color=primary,
        markerfacecolor="white",
        markeredgecolor=primary,
    )
    ax_r.set_xlabel(r"$r$ (cluster advantage)")
    ax_r.set_ylabel(r"$\rho_V$")
    ax_r.set_title(r"(A) Correlation vs cluster advantage $r$")

    r_crit: Optional[float] = None
    try:
        from .experiments import estimate_r_crit

        r_crit = estimate_r_crit(df_r["r"].to_numpy(dtype=float), df_r["mean_rho_V"].to_numpy(dtype=float))
    except Exception:
        r_crit = None

    if r_crit is None:
        ax_r.text(
            0.05,
            0.05,
            "No sign-flip in sampled range",
            transform=ax_r.transAxes,
            fontsize=9,
        )
    else:
        ax_r.axvline(r_crit, linestyle="--", color="black", linewidth=1.0, label=rf"$r^*\approx {r_crit:.3g}$")
        ax_r.legend(frameon=False, loc="best")

    # (B) rho_V vs s
    ax_s.errorbar(
        df_s["s"],
        df_s["mean_rho_V"],
        yerr=[
            df_s["mean_rho_V"] - df_s["rho_lower"],
            df_s["rho_upper"] - df_s["mean_rho_V"],
        ],
        fmt="-o",
        capsize=3,
        color=primary,
        markerfacecolor="white",
        markeredgecolor=primary,
    )
    ax_s.set_xlabel(r"$s$ (optimisation strength)")
    ax_s.set_ylabel(r"$\rho_V$")
    ax_s.set_title(r"(B) Correlation vs optimisation strength $s$")

    # (C) rho_V vs p_fail
    x = np.arange(len(df_p["p_fail"]))
    ax_pfail.bar(x, df_p["mean_rho_V"], color=COLOURS["non_int"], edgecolor="black", linewidth=0.6)
    ax_pfail.errorbar(
        x,
        df_p["mean_rho_V"],
        yerr=[
            df_p["mean_rho_V"] - df_p["rho_lower"],
            df_p["rho_upper"] - df_p["mean_rho_V"],
        ],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        zorder=3,
    )
    ax_pfail.set_xticks(x)
    ax_pfail.set_xticklabels([f"{v:.1f}" for v in df_p["p_fail"]])
    ax_pfail.set_xlabel(r"$p_{\mathrm{fail}}$")
    ax_pfail.set_ylabel(r"$\rho_V$")
    ax_pfail.set_title(r"(C) Robustness to failure probability")

    # (D) Tail-dominance scatter: plot categories separately
    for cat, label in [
        ("non_int", "Non-intelligent"),
        ("fail", "Failed intelligent"),
        ("success", "Successful intelligent"),
    ]:
        sub = df_scatter[df_scatter["category"] == cat]
        if sub.empty:
            continue
        ax_scatter.scatter(
            sub["T"],
            sub["B"],
            alpha=0.5 if cat != "success" else 0.8,
            s=8 if cat != "success" else 12,
            label=label,
            color=COLOURS.get(cat),
            edgecolors="none",
        )
    ax_scatter.set_xlabel(r"$T$ (lifetime)")
    ax_scatter.set_ylabel(r"$B$ (fragmentation count)")
    ax_scatter.set_title(r"(D) Tail dominance in $(T,B)$")
    ax_scatter.legend(frameon=False)

    # ---- Export
    fig.tight_layout()
    out_dir = _figures_dir()
    png_path = out_dir / "Figure_1_Main.png"
    pdf_path = out_dir / "Figure_1_Main.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIGURE] Saved Figure 1 to {png_path} and {pdf_path}")
    return png_path, pdf_path

