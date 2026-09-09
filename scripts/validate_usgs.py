"""
Validate identification against USGS ground truth by leave-one-out.

    python scripts/validate_usgs.py
    python scripts/validate_usgs.py --noise 60 --json out.json

What this measures, and why it is the number that counts
--------------------------------------------------------
Every accuracy figure this project reported before now came from synthetic
spectra scored against the same forward model that generated them. That measures
self-consistency, not correctness, and it is exactly the kind of number that
looks healthy while bench results come back wrong.

This does something different. It takes a *real* USGS spectrum, hides it from the
library, degrades it the way this instrument would - truncate to the detector's
340-912 nm window, resample, add gain, offset, baseline drift and detector noise
- and asks the engine to name it. The answer is scored against the USGS mineral
name the file itself carries.

Only minerals with two or more library samples can be tested: with a single
sample, hiding it leaves nothing of that mineral to find, and the question stops
being meaningful. Roughly a hundred minerals qualify.

A result counted correct is one where the top-ranked mineral name equals the
USGS name of the hidden spectrum. Not a similar mineral, not the right group -
the same name, which is the standard the operator asked for.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config as cfg_mod                     # noqa: E402
from backend.library import store as libstore             # noqa: E402
from backend.models import degeneracy as degmod           # noqa: E402
from backend.models import matcher as matchmod            # noqa: E402
from backend.processing import preprocess as prep         # noqa: E402


def degrade(wl: np.ndarray, refl: np.ndarray, rng, snr: float) -> np.ndarray:
    """
    Turn a pristine library spectrum into something the bench would produce.

    Gain and offset stand in for illumination level and stray light, the
    low-order drift for an imperfect white reference, and the noise for the
    detector. Without this the test would be comparing laboratory spectra to
    laboratory spectra and would flatter the model.
    """
    r = np.asarray(refl, dtype=np.float64).copy()
    x = (wl - wl.mean()) / max(float(np.ptp(wl)), 1e-9)
    r = r * rng.uniform(0.75, 1.35) + rng.uniform(-0.010, 0.022)
    r = r * (1.0 + rng.uniform(-0.05, 0.05) * x
             + rng.uniform(-0.03, 0.03) * x ** 2)
    r = r + rng.normal(0.0, 1.0, r.size) * np.maximum(r, 1e-4) / max(snr, 1.0)
    return np.clip(r, 1e-5, 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=120.0,
                    help="SNR of the simulated measurement (lower = harder)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="degraded copies per held-out spectrum")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = cfg_mod.load_config()
    lo = float(cfg.get_path("instrument.wavelength_min_nm", 340.0))
    hi = float(cfg.get_path("instrument.wavelength_max_nm", 912.0))

    print("=" * 78)
    print("  Leave-one-out validation against USGS splib05a ground truth")
    print("=" * 78)

    lib = libstore.get_library()
    summary = lib.summary()
    if summary["sources"]["usgs"] == 0:
        print("\n  The library holds no measured USGS spectra, so there is no ground")
        print("  truth to validate against. Run:")
        print("      python scripts/fetch_usgs_splib05.py")
        return 1

    print(f"\n  library : {summary['n_entries']} spectra, {summary['n_classes']} minerals")
    print(f"  measured: {summary['sources']['usgs']} USGS  |  synthetic: "
          f"{summary['sources']['synthetic']}")
    print(f"  window  : {lo:.0f}-{hi:.0f} nm  ({lib.grid_nm.size} channels)")

    counts = Counter(lib.names)
    testable = [i for i, n in enumerate(lib.names) if counts[n] >= 2]
    if args.limit:
        testable = testable[: args.limit]
    minerals = sorted({lib.names[i] for i in testable})
    print(f"\n  {len(testable)} spectra across {len(minerals)} minerals have a second "
          f"sample and can be tested")
    print(f"  simulated measurement: SNR {args.noise:.0f}, {args.repeats} repeats each")

    matcher = matchmod.SpectralMatcher(lib)
    grid = lib.grid_nm
    rng = np.random.default_rng(args.seed)

    # Two minerals this window physically cannot separate are not an error to
    # be tuned away - naming either of them is the best any method could do with
    # the data. Scoring at group level as well as by exact name separates a
    # genuine mistake from a limit of the instrument.
    print("\n  building degeneracy map over the USGS library...")
    dmap = degmod.build(lib, lo, hi)
    ds = dmap.summary()
    print(f"  {ds['n_groups']} separable groups, {ds['n_singleton_groups']} minerals "
          f"uniquely identifiable, largest group {ds['largest_group']}")

    top1 = top3 = top5 = grouped = 0
    trials = 0
    per_mineral = defaultdict(lambda: [0, 0])
    confusions = Counter()
    t0 = time.time()

    for k, entry in enumerate(testable):
        truth = lib.names[entry]
        pristine = lib.spectra[entry]

        for _ in range(args.repeats):
            measured = degrade(grid, pristine, rng, args.noise)
            _, cr = prep.continuum_removal(grid, measured)
            results = matcher.match(grid, measured, cr, top_k=8,
                                    exclude_entries={entry})
            if not results:
                continue
            # Rank the way the engine reports, not by raw similarity. The
            # reported order applies the diagnostic-coverage penalty, which
            # pushes down minerals whose identifying bands lie outside this
            # window - and validating the unpenalised order would be scoring a
            # ranking the operator never sees.
            probs = matcher.scores_to_probabilities(results)
            ranked = sorted(probs, key=lambda k: -probs[k]) or [r.name for r in results]
            trials += 1
            hit = ranked[0] == truth
            top1 += hit
            top3 += truth in ranked[:3]
            top5 += truth in ranked[:5]
            # Correct at the level the measurement can resolve: the same mineral,
            # or one this window genuinely cannot tell it apart from.
            grouped += (hit or dmap.group_of(ranked[0]) == dmap.group_of(truth))
            per_mineral[truth][1] += 1
            per_mineral[truth][0] += hit
            if not hit:
                confusions[(truth, ranked[0])] += 1

        if (k + 1) % 40 == 0 or k + 1 == len(testable):
            pct = 100.0 * top1 / max(trials, 1)
            sys.stdout.write(f"\r  {k + 1}/{len(testable)} spectra   "
                             f"top-1 {pct:5.1f}%   ({time.time() - t0:.0f}s)")
            sys.stdout.flush()
    print()

    if trials == 0:
        print("\n  no trials ran")
        return 1

    print("\n" + "=" * 78)
    print("  RESULT - measured against real USGS spectra, not synthetic data")
    print("=" * 78)
    print(f"\n  trials              : {trials}")
    print(f"  top-1 accuracy      : {100.0 * top1 / trials:5.1f}%   "
          f"(exact USGS mineral name)")
    print(f"  top-3 accuracy      : {100.0 * top3 / trials:5.1f}%")
    print(f"  top-5 accuracy      : {100.0 * top5 / trials:5.1f}%")
    print(f"  group-level accuracy: {100.0 * grouped / trials:5.1f}%   "
          f"(correct mineral, or one indistinguishable from it here)")

    solid = [m for m, (h, n) in per_mineral.items() if n and h / n >= 0.8]
    never = [m for m, (h, n) in per_mineral.items() if n and h == 0]
    print(f"\n  minerals identified reliably (>=80%): {len(solid)} of {len(per_mineral)}")
    print(f"  minerals never identified           : {len(never)}")

    print("\n  Reliable in this wavelength window:")
    for m in sorted(solid)[:24]:
        h, n = per_mineral[m]
        print(f"    {100.0 * h / n:5.0f}%  {m}  ({n} trials)")

    if confusions:
        print("\n  Most common confusions (truth -> reported):")
        for (a, b), c in confusions.most_common(12):
            print(f"    {c:3d}x  {a}  ->  {b}")

    if args.json:
        out = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ground_truth": "USGS splib05a (Clark et al. 2003, OFR 03-395)",
            "method": "leave-one-out over library samples, simulated instrument view",
            "window_nm": [lo, hi],
            "snr": args.noise,
            "trials": trials,
            "top1": top1 / trials,
            "top3": top3 / trials,
            "top5": top5 / trials,
            "group_level": grouped / trials,
            "degeneracy": ds,
            "per_mineral": {m: {"hits": h, "trials": n} for m, (h, n) in per_mineral.items()},
            "confusions": [{"truth": a, "reported": b, "count": c}
                           for (a, b), c in confusions.most_common(60)],
        }
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n  written to {args.json}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
