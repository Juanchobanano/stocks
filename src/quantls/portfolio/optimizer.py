import logging

import cvxpy as cp
import pandas as pd

from src.quantls.config import Config

log = logging.getLogger(__name__)


def optimize_portfolio(alpha: pd.Series, cfg: Config) -> pd.Series:
    """
    Maximize alpha subject to:
      Σ w  = 0                          ← DollarNeutral
      Σ|w| ≤ max_gross_leverage         ← MaxGrossExposure
      |w_i| ≤ max_position_size         ← PositionConcentration

    Mirrors quantopian.optimize constraints from the original algorithm.
    """
    alpha = alpha.dropna()
    if alpha.empty:
        return pd.Series(dtype=float)

    n = len(alpha)
    w = cp.Variable(n)

    prob = cp.Problem(
        cp.Maximize(alpha.values @ w),
        [
            cp.sum(w) == 0,
            cp.norm1(w) <= cfg.max_gross_leverage,
            w >= -cfg.max_position_size,
            w <= cfg.max_position_size,
        ],
    )
    for solver in (cp.OSQP, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
            log.debug(f"Solver {solver} returned '{prob.status}', trying next.")
        except cp.SolverError as exc:
            log.debug(f"Solver {solver} raised SolverError: {exc}, trying next.")
    else:
        log.warning("All solvers failed — using zero weights.")
        return pd.Series(0.0, index=alpha.index)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        log.warning(f"Optimizer returned '{prob.status}' — using zero weights.")
        return pd.Series(0.0, index=alpha.index)

    return pd.Series(w.value, index=alpha.index)
