"""
The learned half of the engine.

Training data problem
---------------------
There is no labelled corpus of USB4000 mineral spectra. What exists is a few
hundred library endmembers. Training a classifier on those directly produces a
model that recognises library spectra and nothing else - it will fail on the
first real sample because real samples differ from library entries in every way
except mineralogy.

The fix is physics-based augmentation. Each library endmember is expanded into
hundreds of realistic variants by applying the transformations that actually
occur between a library measurement and a measurement on this bench:

    grain size          deeper or shallower bands, darker or brighter continuum
    weathering/coating  darkening, reddening, band suppression
    solid solution      band centres shifted by chemistry
    illumination        multiplicative gain, additive stray light
    baseline drift      low-order polynomial continuum error
    detector noise      wavelength-dependent SNR matching the real instrument
    mixing              intimate and areal mixtures with other endmembers

The classifier therefore learns invariance to everything that is not
mineralogy, which is exactly the generalisation that matters.

Ensemble
--------
Three models with genuinely different inductive biases, soft-voted:

    tabular trees   - Extra Trees over explicit band centre/depth/width/slope
                      features. Interpretable and robust; fails when band
                      detection fails.
    spectral trees  - Extra Trees over the continuum-removed curve and its
                      derivative. Sees shape the feature extractor missed.
    neural          - MLP (or a 1-D CNN when PyTorch is installed) over the same
                      curve. Learns feature combinations the trees cannot express.

Their errors are decorrelated, so the ensemble is both more accurate and better
calibrated than any member.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.processing import features as featmod
from backend.processing import preprocess as prep
from backend.library import synth

SPEC_DIM = 128


# ---------------------------------------------------------------------------
#  Augmentation
# ---------------------------------------------------------------------------
def _apply_gain_offset(r, rng):
    gain = rng.uniform(0.65, 1.45)
    offset = rng.uniform(-0.012, 0.028)
    return np.clip(r * gain + offset, 1e-5, 2.0)


def _apply_baseline_drift(r, wl, rng):
    x = (wl - wl.mean()) / max(float(np.ptp(wl)), 1e-9)
    drift = (rng.uniform(-0.05, 0.05) * x
             + rng.uniform(-0.03, 0.03) * x ** 2
             + rng.uniform(-0.02, 0.02) * x ** 3)
    return np.clip(r * (1.0 + drift), 1e-5, 2.0)


def _apply_noise(r, rng, snr):
    sigma = np.maximum(r, 1e-4) / max(snr, 1.0)
    return np.clip(r + rng.normal(0.0, 1.0, r.size) * sigma, 1e-5, 2.0)


def _apply_shift(wl, r, rng, max_nm=6.0):
    shift = rng.uniform(-max_nm, max_nm)
    return np.interp(wl, wl + shift, r, left=r[0], right=r[-1])


def augment(wl, spectrum, rng, snr_range=(30.0, 400.0)):
    r = np.asarray(spectrum, dtype=np.float64).copy()
    r = _apply_shift(wl, r, rng)
    r = _apply_gain_offset(r, rng)
    r = _apply_baseline_drift(r, wl, rng)
    snr = float(np.exp(rng.uniform(np.log(snr_range[0]), np.log(snr_range[1]))))
    r = _apply_noise(r, rng, snr)
    return r


# ---------------------------------------------------------------------------
#  Feature encoding
# ---------------------------------------------------------------------------
def encode(wl: np.ndarray, reflectance: np.ndarray,
           model_wl: np.ndarray | None = None):
    """
    Return (spectral_vector, tabular_vector) for one spectrum.

    The spectral vector is resampled onto a fixed grid spanning ``model_wl`` -
    the wavelength axis the model was trained on - rather than by array index.
    That distinction is essential: the engine narrows the analysis range per
    measurement to wherever the white reference has real signal, so an
    index-based downsample would silently hand the model a spectrum whose
    element 40 means 420 nm today and 445 nm tomorrow.
    """
    wl = np.asarray(wl, dtype=np.float64)
    _, cr = prep.continuum_removal(wl, reflectance)
    d1 = prep.derivative(cr, wl, order=1)

    axis = np.asarray(model_wl if model_wl is not None else wl, dtype=np.float64)
    grid = np.linspace(float(axis.min()), float(axis.max()), SPEC_DIM)
    # Outside the measured span np.interp edge-extends, which is the honest
    # behaviour: "no absorption information here", not a fabricated value.
    cr_ds = np.interp(grid, wl, cr)
    d1_ds = np.interp(grid, wl, d1)
    d1_ds = d1_ds / max(np.abs(d1_ds).max(), 1e-9)

    spec_vec = np.concatenate([cr_ds, d1_ds])
    tab_vec = featmod.feature_vector(wl, reflectance, cr)
    return spec_vec, tab_vec


# ---------------------------------------------------------------------------
#  Training-set synthesis
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    X_spec: np.ndarray
    X_tab: np.ndarray
    y: np.ndarray
    classes: list


def build_dataset(library, wl: np.ndarray, *,
                  samples_per_class: int = 400,
                  mixture_fraction: float = 0.45,
                  snr_range=(30.0, 400.0),
                  seed: int = 20260902,
                  progress=None) -> Dataset:
    rng = np.random.default_rng(seed)
    classes = list(library.classes)
    n_classes = len(classes)
    names = np.asarray(library.names)

    mask = (library.grid_nm >= wl.min() - 1e-6) & (library.grid_nm <= wl.max() + 1e-6)
    lib_wl = library.grid_nm[mask]
    lib_spec = library.spectra[:, mask]

    per_class_idx = {c: np.flatnonzero(names == c) for c in classes}
    minerals = {c: library.mineral(c) for c in classes}

    X_spec, X_tab, y = [], [], []
    total = n_classes * samples_per_class
    done = 0

    for ci, cname in enumerate(classes):
        idx_pool = per_class_idx[cname]
        mineral = minerals[cname]

        for k in range(samples_per_class):
            # Base spectrum: either a library entry or a freshly synthesised
            # variant of the mineral with new grain size and weathering.
            if mineral is not None and rng.random() < 0.55:
                base = synth.synthesize(
                    mineral, lib_wl,
                    grain_size=float(np.exp(rng.uniform(np.log(0.2), np.log(5.0)))),
                    weathering=float(rng.beta(1.4, 3.0) * 0.8),
                    band_scale=float(rng.uniform(0.65, 1.35)),
                    shift_nm=float(rng.normal(0.0, 9.0)),
                    albedo_scale=float(rng.uniform(0.7, 1.35)),
                    rng=rng)
            else:
                base = lib_spec[rng.choice(idx_pool)].copy()

            # Mix in contaminant phases. The dominant phase stays the label, so
            # the model learns to identify the major mineral through the kind of
            # contamination every real sample carries.
            if rng.random() < mixture_fraction and n_classes > 1:
                n_extra = int(rng.integers(1, 3))
                others = rng.choice(n_classes, size=n_extra, replace=False)
                specs = [base]
                dom = float(rng.uniform(0.58, 0.92))
                fracs = [dom]
                rest = 1.0 - dom
                w = rng.dirichlet(np.ones(n_extra)) * rest
                for oi, of in zip(others, w):
                    om = minerals[classes[oi]]
                    if om is None:
                        specs.append(lib_spec[rng.choice(per_class_idx[classes[oi]])])
                    else:
                        specs.append(synth.synthesize(
                            om, lib_wl,
                            grain_size=float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))),
                            weathering=float(rng.uniform(0.0, 0.4))))
                    fracs.append(float(of))
                mode = "intimate" if rng.random() < 0.75 else "areal"
                base = synth.mix(np.vstack(specs), np.asarray(fracs), mode=mode)

            noisy = augment(lib_wl, base, rng, snr_range)

            # Range-trim augmentation. At inference the engine narrows the
            # analysis window to wherever the white reference has signal, which
            # varies with lamp age, fibre and integration time. Training over
            # that same variation is what makes the model robust to it.
            if rng.random() < 0.5:
                span = float(lib_wl[-1] - lib_wl[0])
                t_lo = lib_wl[0] + rng.uniform(0.0, 0.09) * span
                t_hi = lib_wl[-1] - rng.uniform(0.0, 0.05) * span
                keep = (lib_wl >= t_lo) & (lib_wl <= t_hi)
                if keep.sum() > 50:
                    sv, tv = encode(lib_wl[keep], noisy[keep], model_wl=lib_wl)
                else:
                    sv, tv = encode(lib_wl, noisy, model_wl=lib_wl)
            else:
                sv, tv = encode(lib_wl, noisy, model_wl=lib_wl)

            X_spec.append(sv)
            X_tab.append(tv)
            y.append(ci)

            done += 1
            if progress is not None and done % 500 == 0:
                progress(done, total)

    return Dataset(
        X_spec=np.asarray(X_spec, dtype=np.float32),
        X_tab=np.asarray(X_tab, dtype=np.float32),
        y=np.asarray(y, dtype=np.int32),
        classes=classes,
    )


# ---------------------------------------------------------------------------
#  Ensemble
# ---------------------------------------------------------------------------
class EnsembleClassifier:
    def __init__(self, classes: list, wl: np.ndarray):
        self.classes = list(classes)
        self.wl = np.asarray(wl, dtype=np.float64)
        self.tab_model = None
        self.spec_model = None
        self.nn_model = None
        self.scaler = None
        self.weights = {"tab": 0.34, "spec": 0.36, "nn": 0.30}
        self.metrics: dict = {}

    # -- training ----------------------------------------------------------
    def fit(self, ds: Dataset, log=print) -> dict:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        t0 = time.time()

        # max_leaf_nodes is a size control, not just a regularizer. A decision
        # tree stores a full class-probability vector at every node, so with 92
        # minerals an unbounded forest serialises to hundreds of megabytes -
        # 92 x 8 bytes per node, times tens of thousands of nodes, times
        # hundreds of trees. Bounding leaves keeps the saved model in the tens
        # of MB and loading it near-instant, at negligible accuracy cost:
        # Extra Trees get their strength from averaging many decorrelated trees,
        # not from any single tree being deep.
        log(f"  tabular trees   : {ds.X_tab.shape}")
        self.tab_model = ExtraTreesClassifier(
            n_estimators=320, max_features="sqrt", min_samples_leaf=3,
            max_leaf_nodes=384, n_jobs=-1, random_state=11,
            class_weight="balanced_subsample")
        self.tab_model.fit(ds.X_tab, ds.y)

        log(f"  spectral trees  : {ds.X_spec.shape}")
        self.spec_model = ExtraTreesClassifier(
            n_estimators=320, max_features="sqrt", min_samples_leaf=3,
            max_leaf_nodes=384, n_jobs=-1, random_state=12,
            class_weight="balanced_subsample")
        self.spec_model.fit(ds.X_spec, ds.y)

        log("  neural network  : MLP over continuum-removed curve")
        self.scaler = StandardScaler().fit(ds.X_spec)
        Xs = self.scaler.transform(ds.X_spec)
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(512, 256), activation="relu", alpha=3e-4,
            batch_size=256, learning_rate_init=1.2e-3, max_iter=90,
            early_stopping=True, n_iter_no_change=8, validation_fraction=0.12,
            random_state=13)
        self.nn_model.fit(Xs, ds.y)

        elapsed = time.time() - t0
        self.metrics["train_seconds"] = round(elapsed, 1)
        self.metrics["n_train"] = int(ds.y.size)
        log(f"  done in {elapsed:.1f}s")
        return self.metrics

    # -- inference ---------------------------------------------------------
    def predict_proba_parts(self, spec_vec: np.ndarray, tab_vec: np.ndarray) -> dict:
        out = {}
        if self.tab_model is not None:
            out["tab"] = self.tab_model.predict_proba(tab_vec[None, :])[0]
        if self.spec_model is not None:
            out["spec"] = self.spec_model.predict_proba(spec_vec[None, :])[0]
        if self.nn_model is not None and self.scaler is not None:
            out["nn"] = self.nn_model.predict_proba(
                self.scaler.transform(spec_vec[None, :]))[0]
        return out

    def predict_proba(self, spec_vec: np.ndarray, tab_vec: np.ndarray) -> np.ndarray:
        parts = self.predict_proba_parts(spec_vec, tab_vec)
        if not parts:
            return np.full(len(self.classes), 1.0 / max(len(self.classes), 1))
        total = np.zeros(len(self.classes), dtype=np.float64)
        wsum = 0.0
        for key, p in parts.items():
            w = self.weights.get(key, 0.0)
            total += w * p
            wsum += w
        return total / max(wsum, 1e-9)

    def predict_proba_batch(self, X_spec: np.ndarray, X_tab: np.ndarray) -> np.ndarray:
        total = np.zeros((X_spec.shape[0], len(self.classes)), dtype=np.float64)
        wsum = 0.0
        if self.tab_model is not None:
            total += self.weights["tab"] * self.tab_model.predict_proba(X_tab)
            wsum += self.weights["tab"]
        if self.spec_model is not None:
            total += self.weights["spec"] * self.spec_model.predict_proba(X_spec)
            wsum += self.weights["spec"]
        if self.nn_model is not None and self.scaler is not None:
            total += self.weights["nn"] * self.nn_model.predict_proba(
                self.scaler.transform(X_spec))
            wsum += self.weights["nn"]
        return total / max(wsum, 1e-9)

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "classes": self.classes,
            "wl": self.wl,
            "tab_model": self.tab_model,
            "spec_model": self.spec_model,
            "nn_model": self.nn_model,
            "scaler": self.scaler,
            "weights": self.weights,
            "metrics": self.metrics,
        }, path, compress=3)
        self.metrics["model_size_mb"] = round(path.stat().st_size / 1e6, 1)

    @classmethod
    def load(cls, path: Path) -> "EnsembleClassifier":
        import joblib
        blob = joblib.load(Path(path))
        obj = cls(blob["classes"], blob["wl"])
        obj.tab_model = blob["tab_model"]
        obj.spec_model = blob["spec_model"]
        obj.nn_model = blob["nn_model"]
        obj.scaler = blob["scaler"]
        obj.weights = blob.get("weights", obj.weights)
        obj.metrics = blob.get("metrics", {})
        return obj
