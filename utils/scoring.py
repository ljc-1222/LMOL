# utils/bradley_terry.py
from __future__ import annotations

from math import isclose
from typing import Sequence

import numpy as np


def bt_score(
    relations: Sequence[int],
    ref_scores: Sequence[float] | np.ndarray,
    *,
    delta: float = 0.3,
    k_scale: float = 1.0,
    search_space: np.ndarray | None = None,
    eps: float = 1e-8,
) -> float:
    r"""
    Bradley–Terry maximum-likelihood estimate for a single target score ``s_t``.

    Parameters
    ----------
    relations :
        Iterable of categorical labels in ``{0, 1, 2}``
           *0* → target < reference  
           *1* → target ≈ reference  
           *2* → target > reference
    ref_scores :
        Ground-truth scores ``s_{r_i}`` for each reference sample.
    delta :
        Positive threshold ``δ`` that defines the “≈” band.
    k_scale :
        Scaling factor ``k`` in Eq. (11).
    search_space :
        1-D NumPy array of candidate ``s_t`` values.
        Defaults to 200 points uniformly spaced in [1, 5].
    eps :
        Small constant added inside ``log`` to avoid ``log(0)``.

    Returns
    -------
    float
        The MLE \(\hat{s}_t\).

    Notes
    -----
    The cumulative probabilities are

    \[
        \begin{aligned}
          P(Y \le 0) &= \sigma\!\bigl(-\delta + k\,(s_{r_i}-s_t)\bigr),\\
          P(Y \le 1) &= \sigma\!\bigl(+\delta + k\,(s_{r_i}-s_t)\bigr),
        \end{aligned}
    \]

    where \(\sigma(x)=\frac{1}{1+e^{-x}}\).
    """
    # --------------------------- validation ---------------------------------
    R = np.asarray(relations, dtype=np.int8)
    s_ref = np.asarray(ref_scores, dtype=np.float32)

    if R.size == 0 or s_ref.size == 0 or R.size != s_ref.size:
        raise ValueError("`relations` and `ref_scores` must be non-empty and of equal length.")
    if not np.isin(R, [0, 1, 2]).all():
        raise ValueError("`relations` must contain only 0, 1, or 2.")

    if search_space is None:
        search_space = np.linspace(1.0, 5.0, 200, dtype=np.float32)

    # ---------------------- negative-log-likelihood -------------------------
    def neg_log_likelihood(st: float) -> float:
        diff = k_scale * (s_ref - st)                    # NOTE: s_ref − s_t  (sign fixed)

        cum_lo = 1.0 / (1.0 + np.exp(-(-delta + diff)))  # P(Y ≤ 0)
        cum_hi = 1.0 / (1.0 + np.exp(-( delta + diff)))  # P(Y ≤ 1)

        p = np.empty_like(cum_lo)
        mask0, mask1, mask2 = R == 0, R == 1, R == 2
        p[mask0] = cum_lo[mask0]
        p[mask1] = cum_hi[mask1] - cum_lo[mask1]
        p[mask2] = 1.0 - cum_hi[mask2]

        return -np.sum(np.log(p + eps))

    # ---------------------------- grid search ------------------------------
    nll = np.vectorize(neg_log_likelihood)(search_space)
    best_idx = int(np.argmin(nll))
    return float(search_space[best_idx])


# ------------------------------------------------------------------------- #
#                           self-contained test                             #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    TRUE_ST = 3.40
    REF_SCORES = np.array([1.80, 2.60, 3.15, 3.75, 4.50], dtype=np.float32)
    RELATIONS = [2, 2, 1, 0, 0]                       # generated from TRUE_ST

    estimate = bt_score(
        relations=RELATIONS,
        ref_scores=REF_SCORES,
        delta=0.3,
        k_scale=1.0,
        search_space=np.linspace(1.0, 5.0, 200),
    )

    print(f"estimated s_t = {estimate:.2f}  (ground truth {TRUE_ST})")
    assert isclose(estimate, TRUE_ST, abs_tol=0.05)
