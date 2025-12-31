"""
Configuration for the vacuum-class toy model experiments.

Only numpy/pandas/tqdm/matplotlib are assumed available in the environment.
"""

# Ensemble sizes
N_ENSEMBLE = 100_000
N_SEEDS = 5

# Quick mode (dev / smoke test)
N_ENSEMBLE_QUICK = 1_000
N_SEEDS_QUICK = 1

# Baseline distributions
MU_T = 1.0
ALPHA_FC = 1.0
BETA_FC = 3.0

# Intelligence & lifetime modification
P_INT = 0.3
GAMMA_FAIL = 0.3
GAMMA_SUCC = 5.0

# Sweep ranges
R_VALUES = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
]
S_VALUES = [0.1 * i for i in range(11)]  # 0.0 → 1.0
P_FAIL_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

# Randomness
BASE_SEED = 12345

