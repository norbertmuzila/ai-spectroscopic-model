"""
Confidence calibration and conformal prediction.

This module is the reason the system's confidence numbers can be trusted.

A raw ensemble is over-confident: it will report 0.97 on cases it gets right
only 80% of the time. Reporting that number to a user is worse than useless,
because decisions get made on it. Two corrections are applied.

**Temperature scaling** (Guo et al. 2017) fits a single scalar T on held-out
data and divides the logits by it. It cannot change which mineral ranks first,
so it costs nothing in accuracy, and it makes the probability mean what it says:
of the identifications reported at 0.90, about 90% are correct.

**Conformal prediction** (Vovk; Romano et al. 2020) goes further and provides a
*distribution-free finite-sample guarantee*. Instead of one mineral it returns a
set - "Hematite or Goethite" - constructed so that the true mineral is in the
set at least 95% of the time, whatever the underlying model does. When the
spectrum is unambiguous the set has one member. When the instrument genuinely
cannot separate two phases the set has two, and that is the honest answer rather
than an arbitrary coin-flip presented as a result.

This is what "highest possible accuracy" actually looks like in practice: not a
claim of 100%, but a stated coverage level that is met, plus explicit abstention
when the evidence does not support a single answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
#  Temperature scaling
# ---------------------------------------------------------------------------
class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, probs: np.ndarray, y: np.ndarray,
            grid: np.ndarray | None = None) -> float:
        """Choose T minimising negative log-likelihood on held-out data."""
        logits = np.log(np.clip(probs, 1e-12, 1.0))
        if grid is None:
            grid = np.concatenate([np.linspace(0.30, 1.0, 36)[:-1],
                                   np.linspace(1.0, 6.0, 60)])
        best_t, best_nll = 1.0, np.inf
        n = y.size
        for t in grid:
            z = logits / t
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            nll = -np.mean(np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0)))
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        self.temperature = best_t
        return best_t

    def transform(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probs, 1e-12, 1.0)) / self.temperature
        logits -= logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
#  Conformal prediction
# ---------------------------------------------------------------------------
@dataclass
class ConformalPredictor:
    """
    Split-conformal prediction sets, in two flavours.

    ``lac`` - Least Ambiguous set-valued Classifier. The nonconformity score is
    ``1 - p_true``; the set is every class whose probability clears one global
    threshold. Among all methods achieving marginal coverage, LAC provably gives
    the *smallest average set size*, which is what makes it useful on a bench:
    a two-mineral answer is actionable, a thirty-mineral answer is not.

    ``aps`` - Adaptive Prediction Sets. The score is the cumulative probability
    mass ranked at or above the true class. Set size adapts to difficulty and
    conditional coverage is better, at the cost of larger sets overall.

    ``fit`` calibrates both and selects whichever gives the smaller mean set on
    the calibration split, since both satisfy the same marginal guarantee.

    Critically, ``predict_set`` applies **no size cap by default**. Truncating a
    conformal set to a convenient display length destroys the coverage guarantee
    outright - if the honest answer is "this measurement cannot narrow the
    mineral below twenty candidates", that is the answer, and the fix is a better
    measurement or a wider-range instrument, not a shorter list.
    """
    alpha: float = 0.05
    qhat: float = 1.0
    n_calibration: int = 0
    method: str = "lac"

    # -- scores --------------------------------------------------------
    @staticmethod
    def _aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = y.size
        order = np.argsort(-probs, axis=1)
        cumsum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
        rank_of_true = np.argmax(order == y[:, None], axis=1)
        return cumsum[np.arange(n), rank_of_true]

    @staticmethod
    def _lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1.0 - probs[np.arange(y.size), y]

    def _quantile(self, scores: np.ndarray) -> float:
        n = scores.size
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    # -- calibration ---------------------------------------------------
    def fit(self, probs: np.ndarray, y: np.ndarray,
            method: str | None = None) -> float:
        n = y.size
        if n == 0:
            self.qhat = 1.0
            return self.qhat
        self.n_calibration = int(n)

        candidates = [method] if method else ["lac", "aps"]
        best = None
        for m in candidates:
            scores = self._lac_scores(probs, y) if m == "lac" else self._aps_scores(probs, y)
            q = self._quantile(scores)
            probe = ConformalPredictor(alpha=self.alpha, qhat=q,
                                       n_calibration=n, method=m)
            sizes = [len(probe.predict_set(probs[i])[0]) for i in range(min(n, 1500))]
            mean_size = float(np.mean(sizes))
            if best is None or mean_size < best[2]:
                best = (m, q, mean_size)

        self.method, self.qhat, _ = best
        return self.qhat

    # -- prediction ----------------------------------------------------
    def predict_set(self, probs: np.ndarray, max_size: int | None = None):
        """
        Return (indices, cumulative_mass) of the prediction set.

        ``max_size`` is for display truncation only and voids the guarantee;
        leave it unset for any statistical use.
        """
        p = np.asarray(probs, dtype=np.float64).ravel()
        order = np.argsort(-p)

        if self.method == "lac":
            threshold = 1.0 - self.qhat
            chosen = [int(i) for i in order if p[i] >= threshold]
            if not chosen:                      # never return an empty set
                chosen = [int(order[0])]
        else:
            chosen, cum = [], 0.0
            for idx in order:
                chosen.append(int(idx))
                cum += float(p[idx])
                if cum >= self.qhat:
                    break

        if max_size is not None:
            chosen = chosen[:max_size]
        return chosen, float(sum(p[i] for i in chosen))


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, y: np.ndarray,
                               n_bins: int = 15) -> float:
    """
    ECE: mean gap between stated confidence and observed accuracy.

    A perfectly calibrated model scores 0. Anything under about 0.03 means the
    confidence figure on the report can be read at face value.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y.size
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


@dataclass
class CalibrationBundle:
    scaler: TemperatureScaler = field(default_factory=TemperatureScaler)
    conformal: ConformalPredictor = field(default_factory=ConformalPredictor)
    metrics: dict = field(default_factory=dict)

    def apply(self, probs: np.ndarray) -> np.ndarray:
        return self.scaler.transform(probs)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 temperature=self.scaler.temperature,
                 alpha=self.conformal.alpha,
                 qhat=self.conformal.qhat,
                 n_calibration=self.conformal.n_calibration,
                 method=str(self.conformal.method),
                 metrics=np.array([repr(self.metrics)], dtype=object))

    @classmethod
    def load(cls, path: Path) -> "CalibrationBundle":
        data = np.load(Path(path), allow_pickle=True)
        obj = cls()
        obj.scaler = TemperatureScaler(float(data["temperature"]))
        obj.conformal = ConformalPredictor(
            alpha=float(data["alpha"]),
            qhat=float(data["qhat"]),
            n_calibration=int(data["n_calibration"]),
            method=str(data["method"]) if "method" in data.files else "lac")
        try:
            obj.metrics = eval(str(data["metrics"][0]), {"__builtins__": {}}, {})
        except Exception:
            obj.metrics = {}
        return obj


def evaluate(probs: np.ndarray, y: np.ndarray, classes: list,
             conformal: ConformalPredictor | None = None,
             diagnosable: set | None = None) -> dict:
    """
    Accuracy, calibration and conformal coverage on a held-out set.

    ``diagnosable`` optionally names the minerals whose diagnostic bands lie
    inside the instrument's wavelength range. Overall top-1 accuracy across all
    92 minerals understates the system badly, because roughly a third of them
    are physically indistinguishable below 1000 nm - asking a VNIR instrument to
    separate gypsum from calcite from halite is asking it to read information
    that never reached the detector. The restricted figure is the one that
    describes what this instrument can actually do.
    """
    n = y.size
    pred = probs.argmax(axis=1)
    top1 = float((pred == y).mean())

    order = np.argsort(-probs, axis=1)
    top3 = float(np.mean([y[i] in order[i, :3] for i in range(n)]))
    top5 = float(np.mean([y[i] in order[i, :5] for i in range(n)]))

    out = {
        "n_test": int(n),
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "top5_accuracy": round(top5, 4),
        "expected_calibration_error": round(expected_calibration_error(probs, y), 4),
        "mean_confidence": round(float(probs.max(axis=1).mean()), 4),
    }

    if conformal is not None:
        covered, sizes = [], []
        for i in range(n):
            s, _ = conformal.predict_set(probs[i])
            covered.append(y[i] in s)
            sizes.append(len(s))
        out["conformal_method"] = conformal.method
        out["conformal_alpha"] = conformal.alpha
        out["conformal_target_coverage"] = round(1.0 - conformal.alpha, 3)
        out["conformal_empirical_coverage"] = round(float(np.mean(covered)), 4)
        out["conformal_mean_set_size"] = round(float(np.mean(sizes)), 3)
        out["conformal_singleton_rate"] = round(float(np.mean(np.array(sizes) == 1)), 4)

    # Where the model is weakest - useful for knowing which minerals to treat
    # with suspicion on this instrument.
    per_class = {}
    for ci, cname in enumerate(classes):
        m = y == ci
        if m.sum() == 0:
            continue
        per_class[cname] = round(float((pred[m] == ci).mean()), 3)
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:15]
    out["weakest_classes"] = dict(worst)
    out["per_class_accuracy"] = per_class

    if diagnosable:
        keep_ci = {i for i, c in enumerate(classes) if c in diagnosable}
        m = np.array([yi in keep_ci for yi in y])
        if m.sum() > 0:
            sub_pred, sub_y = pred[m], y[m]
            sub_order = order[m]
            out["diagnosable_subset"] = {
                "n_minerals": len(keep_ci),
                "n_test": int(m.sum()),
                "top1_accuracy": round(float((sub_pred == sub_y).mean()), 4),
                "top3_accuracy": round(float(np.mean(
                    [sub_y[i] in sub_order[i, :3] for i in range(sub_y.size)])), 4),
            }
    return out
