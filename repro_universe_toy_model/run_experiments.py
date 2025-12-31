from __future__ import annotations

import argparse

from . import config
from .experiments import export_scatter_sample, run_sweep_pfail, run_sweep_r, run_sweep_s
from .io_utils import ensure_results_layout, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vacuum-class toy model experiments.")
    p.add_argument(
        "--mode",
        choices=["full", "quick"],
        default="full",
        help="full: N_ENSEMBLE+N_SEEDS; quick: smaller dev run writing _quick outputs",
    )
    p.add_argument(
        "--only",
        choices=["sweep_r", "sweep_s", "sweep_pfail", "all"],
        default="all",
        help="Run only one sweep (or 'all' for the default full pipeline).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    only = args.only

    ensure_results_layout()
    logger = get_logger(mode=mode)

    if mode == "quick":
        n_ensemble = config.N_ENSEMBLE_QUICK
        n_seeds = config.N_SEEDS_QUICK
        n_scatter = min(5_000, config.N_ENSEMBLE_QUICK * 5)
    else:
        n_ensemble = config.N_ENSEMBLE
        n_seeds = config.N_SEEDS
        n_scatter = 50_000

    logger.info(f"RUN START mode={mode} n_ensemble={n_ensemble} n_seeds={n_seeds}")
    if only != "all":
        logger.info(f"RUN CONFIG only={only}")

    warn = logger.warning
    info = logger.info

    if only in ("all", "sweep_r"):
        run_sweep_r(
            mode=mode,
            n_ensemble=n_ensemble,
            n_seeds=n_seeds,
            logger_warn=warn,
            logger_info=info,
        )
    if only in ("all", "sweep_s"):
        run_sweep_s(
            mode=mode,
            n_ensemble=n_ensemble,
            n_seeds=n_seeds,
            logger_warn=warn,
            logger_info=info,
        )
    if only in ("all", "sweep_pfail"):
        run_sweep_pfail(
            mode=mode,
            n_ensemble=n_ensemble,
            n_seeds=n_seeds,
            logger_warn=warn,
            logger_info=info,
        )
    if only == "all":
        export_scatter_sample(
            mode=mode,
            n_scatter=n_scatter,
            logger_warn=warn,
            logger_info=info,
        )

    logger.info("RUN END")


if __name__ == "__main__":
    main()

