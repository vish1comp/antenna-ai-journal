"""Central configuration: seeds, split, and optimizer/active-learning budgets."""
import os

SEED = 42
TEST_SIZE = 0.20

# Gaussian process is O(n^3); cap its training set (eval is still on full test).
GP_MAX_TRAIN = 1200

# Active-learning learning-curve budgets (clipped to available pool at runtime).
AL_INIT = 30
AL_SIZES = [40, 80, 160, 320, 640]

# Metaheuristic + Bayesian budgets for the surrogate-assisted optimization.
PSO = dict(n_particles=40, n_iters=120)
GA = dict(pop_size=60, n_gen=120)
BO = dict(n_init=12, n_iter=60)   # ~72 objective evals total vs. thousands for PSO/GA

# Uncertainty quantification + risk-averse (uncertainty-gated) design.
UQ = dict(n_models=6, alpha=0.10, kappa=1.5)   # 90% intervals; gate = mean + kappa*std
UQ_PSO = dict(n_particles=30, n_iters=60)      # lighter budget (ensemble evals are pricier)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
