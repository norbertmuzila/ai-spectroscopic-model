"""
Spectral unmixing - estimating what fraction of the sample each mineral is.

Real samples are almost never a single phase. A basalt is pyroxene plus olivine
plus plagioclase plus glass, and reporting only the best-matching endmember
throws away most of the answer.

Two mixing physics are solved and the better-fitting one is reported:

* **Areal** mixing is linear in reflectance. It applies when the phases are
  spatially separated at scales larger than the photon mean free path - a rock
  face with distinct patches.
* **Intimate** mixing is linear in single-scattering albedo, not reflectance.
  It applies to a powder, which is what a prepared sample cup contains. The
  distinction is not academic: a 10% magnetite content darkens an intimate
  mixture far more than 10% of the way to magnetite, so solving the linear
  problem in reflectance space would report perhaps 40% magnetite.

Constraints are physical: abundances are non-negative and sum to one. Subset
selection is greedy-forward with a fit-improvement threshold, which keeps the
solution sparse - adding endmembers always improves the fit numerically, and
without that threshold every solution degenerates into a blend of everything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from backend.library import synth


@dataclass
class UnmixResult:
    abundances: dict           # mineral -> fraction
    rmse: float
    mode: str                  # 'intimate' | 'areal'
    modelled: np.ndarray
    n_endmembers: int

    def as_dict(self) -> dict:
        return {
            "abundances": {k: round(float(v), 4) for k, v in self.abundances.items()},
            "rmse": round(float(self.rmse), 5),
            "mixing_model": self.mode,
            "n_endmembers": self.n_endmembers,
        }


def _fcls(A: np.ndarray, b: np.ndarray, sum_weight: float = 30.0):
    """
    Fully-constrained least squares: non-negative, sum-to-one.

    The sum-to-one constraint is imposed by augmenting the system with a heavily
    weighted row of ones, which is the standard FCLS formulation and keeps the
    problem solvable by ordinary NNLS.
    """
    n = A.shape[1]
    A_aug = np.vstack([A, sum_weight * np.ones((1, n))])
    b_aug = np.concatenate([b, [sum_weight]])
    x, _ = nnls(A_aug, b_aug)
    total = x.sum()
    if total > 1e-9:
        x = x / total
    return x


def _fit(endmembers: np.ndarray, target: np.ndarray, mode: str):
    if mode == "intimate":
        A = synth.reflectance_to_ssa(endmembers).T
        b = synth.reflectance_to_ssa(target)
        x = _fcls(A, b)
        modelled = synth.ssa_to_reflectance(A @ x)
    else:
        A = endmembers.T
        x = _fcls(A, target)
        modelled = A @ x
    rmse = float(np.sqrt(np.mean((modelled - target) ** 2)))
    return x, rmse, modelled


def unmix(target: np.ndarray,
          candidate_spectra: np.ndarray,
          candidate_names: list,
          *,
          max_endmembers: int = 4,
          min_improvement: float = 0.06,
          min_abundance: float = 0.03,
          seed_index: int | None = None) -> UnmixResult:
    """
    Greedy-forward subset selection followed by a constrained solve.

    ``min_improvement`` is the fractional RMSE reduction an extra endmember must
    deliver to be kept. Without it the solver happily reports five phases where
    two explain the data.

    ``seed_index`` forces an endmember into the solution as the first pick -
    the engine passes the identified mineral. Spectral twins (goethite against
    akaganeite, magnetite against ilmenite) sit within noise of each other in
    VNIR, so pure least squares picks between them essentially at random. Left
    unseeded, the report can name one mineral in the headline and a different
    one in the composition table, which is worse than useless to a reader. The
    seed keeps the two consistent by making the composition answer the question
    "given this identification, what else is present?"
    """
    target = np.asarray(target, dtype=np.float64)
    cand = np.asarray(candidate_spectra, dtype=np.float64)
    if cand.ndim == 1:
        cand = cand[None, :]

    best_overall = None
    for mode in ("intimate", "areal"):
        chosen: list = []
        prev_rmse = np.inf
        best_for_mode = None

        if seed_index is not None and 0 <= seed_index < cand.shape[0]:
            chosen = [int(seed_index)]
            x, prev_rmse, modelled = _fit(cand[chosen], target, mode)
            best_for_mode = (list(chosen), x, prev_rmse, modelled)

        for _ in range(min(max_endmembers, cand.shape[0]) - len(chosen)):
            best_add, best_rmse, best_x, best_model = None, np.inf, None, None
            for j in range(cand.shape[0]):
                if j in chosen:
                    continue
                trial = chosen + [j]
                x, rmse, modelled = _fit(cand[trial], target, mode)
                if rmse < best_rmse:
                    best_add, best_rmse, best_x, best_model = j, rmse, x, modelled
            if best_add is None:
                break
            if np.isfinite(prev_rmse) and best_rmse > prev_rmse * (1.0 - min_improvement):
                break
            chosen.append(best_add)
            prev_rmse = best_rmse
            best_for_mode = (list(chosen), best_x, best_rmse, best_model)

        if best_for_mode is None:
            continue
        if best_overall is None or best_for_mode[2] < best_overall[2]:
            best_overall = best_for_mode + (mode,)

    if best_overall is None:
        return UnmixResult({}, float("inf"), "areal", target.copy(), 0)

    chosen, x, rmse, modelled, mode = best_overall
    abund = {}
    for idx, frac in zip(chosen, x):
        if frac >= min_abundance:
            abund[candidate_names[idx]] = float(frac)
    total = sum(abund.values())
    if total > 1e-9:
        abund = {k: v / total for k, v in abund.items()}
    abund = dict(sorted(abund.items(), key=lambda kv: -kv[1]))

    return UnmixResult(abundances=abund, rmse=rmse, mode=mode,
                       modelled=modelled, n_endmembers=len(abund))
