"""
End-to-end self-test.

    python scripts/selftest.py                 # full check
    python scripts/selftest.py --quick         # skip report generation

Exercises the whole chain against the simulator: device open, references,
acquisition, preprocessing, quality gating, matching, learned inference,
fusion, calibration, unmixing, range audit, database write and report
generation. Run it after any change, and after installing the USGS library.

The mineral panel is deliberately mixed: some phases the USB4000 diagnoses
cleanly, some it can only partially constrain, and some it physically cannot
see at all. A pass means the system got the first group right *and* correctly
refused to over-claim on the last one.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config as cfg_mod                      # noqa: E402
from backend.db import Database                            # noqa: E402
from backend.hardware.device import open_device            # noqa: E402
from backend.models.engine import get_engine               # noqa: E402
from backend.reporting import report as repmod             # noqa: E402

# (mineral, expectation) where expectation is what a 350-1000 nm instrument
# should be able to conclude.
PANEL = [
    ("Hematite",              "identify"),
    ("Goethite",              "identify"),
    ("Native Sulfur",         "identify"),
    ("Almandine Garnet",      "identify"),
    ("Cinnabar",              "identify"),
    ("Olivine (Fo50)",        "constrain"),   # Fo50 vs forsterite is a solid-solution
                                              # call; the 1.05 um band sits at the
                                              # detector edge, so composition within
                                              # the olivine series is not resolvable
    ("Enstatite (LCP)",       "identify"),
    ("Malachite",             "identify"),
    ("Jarosite",              "constrain"),
    ("Nontronite",            "constrain"),
    ("Palagonite (Mars Soil Analog)", "constrain"),
    ("Martian Bright Dust",   "constrain"),
    ("Lunar Mare Regolith",   "constrain"),
    ("Magnetite",             "constrain"),
    ("Gypsum",                "refuse"),
    ("Calcite",               "refuse"),
    ("Kaolinite",             "refuse"),
    ("Halite",                "refuse"),
]

MIXTURES = [
    ({"Olivine (Fo50)": 0.6, "Augite (HCP)": 0.4}, "Martian basalt"),
    ({"Hematite": 0.35, "Basaltic Glass": 0.65}, "oxidised basalt"),
    ({"Goethite": 0.5, "Montmorillonite": 0.5}, "weathered clay"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip PDF generation")
    args = ap.parse_args()

    cfg = cfg_mod.load_config()
    cfg_mod.ensure_dirs()

    print("=" * 78)
    print("  AI Spectroscopic Model - self-test")
    print("=" * 78)

    print("\n[1] Opening simulator and taking references")
    dev = open_device(prefer_simulator=True)
    dev.set_integration_time(90)
    dev.set_averaging(12)
    dev.take_reference("dark")
    dev.take_reference("white")
    print(f"    {dev.model}  {dev.wavelengths().size} px  "
          f"{dev.wavelengths().min():.0f}-{dev.wavelengths().max():.0f} nm")

    print("\n[2] Loading engine")
    engine = get_engine()
    print(f"    learned model : {'loaded' if engine.model_loaded else 'NOT TRAINED (matcher only)'}")
    print(f"    library       : {engine.library.summary()['n_entries']} spectra, "
          f"{engine.library.summary()['n_classes']} minerals")
    if engine.model_metrics.get("test"):
        t = engine.model_metrics["test"]
        print(f"    validation    : top-1 {t['top1_accuracy'] * 100:.1f}%  "
              f"coverage {t.get('conformal_empirical_coverage', 0) * 100:.1f}%")

    db = Database(cfg.resolve("server.database"))
    session_id = db.create_session("Self-test", operator="selftest")

    def run(label):
        return engine.analyze(
            wavelength_nm=dev.wavelengths(), values=dev.acquire(),
            dark=dev.dark.counts, white=dev.white.counts,
            nonlinearity_coeffs=dev.nonlinearity_coeffs,
            sample_label=label,
            reference_age_minutes=dev.reference_age_minutes(),
            metadata={"source": "selftest", "ground_truth": dev.loaded_sample})

    print("\n[3] Single-phase panel")
    print(f"    {'truth':<32} {'reported':<28} {'conf':>6} {'set':>4} {'cov':>5}  verdict")
    print("    " + "-" * 92)

    passes = failures = 0
    for name, expect in PANEL:
        dev.load_sample(name)
        r = run(f"selftest {name}")
        db.save_analysis(r.as_dict(), session_id, source="selftest")
        i = r.identification
        pset = r.prediction_set["minerals"]
        cov = i["diagnostic_coverage"]

        if expect == "identify":
            ok = i["mineral"] == name and i["confidence"] >= 0.30
            why = "top-1 correct" if ok else "MISSED"
        elif expect == "constrain":
            # Correct at the level this instrument can actually resolve: either
            # the mineral itself, or its spectral-degeneracy group.
            siblings = set(i.get("indistinguishable_from") or [])
            ok = (i["mineral"] == name or name in pset or name in siblings)
            why = ("top-1 correct" if i["mineral"] == name
                   else "in group / prediction set" if ok else "MISSED")
        else:  # refuse
            ok = (i["status"] in ("inconclusive", "rejected_quality")
                  or cov < 0.5 or len(pset) > 3)
            why = "correctly not over-claimed" if ok else "OVER-CLAIMED"

        passes += ok
        failures += (not ok)
        print(f"    {'OK ' if ok else 'XX '}{name:<29} {str(i['mineral'])[:27]:<28} "
              f"{i['confidence'] * 100:5.1f}% {len(pset):>4} {cov * 100:4.0f}%  {why}")

    print("\n[4] Mixtures (does unmixing recover the endmembers?)")
    for mixture, desc in MIXTURES:
        dev.load_mixture(mixture)
        r = run(f"selftest {desc}")
        db.save_analysis(r.as_dict(), session_id, source="selftest")
        ab = r.composition["abundances"]
        i = r.identification
        pset = r.prediction_set["minerals"]
        sibs = set(i.get("indistinguishable_from") or [])
        truth = ", ".join(f"{k} {v * 100:.0f}%" for k, v in mixture.items())
        got = ", ".join(f"{k} {v * 100:.0f}%" for k, v in ab.items()) or "-"
        # Pass if any endmember is recovered - deliberately not "the one with the
        # largest mass fraction". In an intimate mixture a strong absorber
        # dominates the spectrum far beyond its abundance: 35% hematite in
        # basaltic glass produces a hematite spectrum, and reporting hematite is
        # correct spectroscopy, not an error. Mass fraction and spectral
        # dominance are different quantities.
        found = [k for k in mixture
                 if k == i["mineral"] or k in ab or k in pset or k in sibs]
        ok = len(found) >= 1
        passes += ok
        failures += (not ok)
        print(f"    {'OK ' if ok else 'XX '}{desc}")
        print(f"        truth  : {truth}")
        print(f"        ident  : {i['mineral']} ({i['confidence'] * 100:.0f}%, "
              f"{i['status']}), set of {len(pset)}")
        print(f"        found  : {', '.join(found) or 'none'}")
        print(f"        unmix  : {got}   ({r.composition['mixing_model']}, "
              f"RMSE {r.composition['rmse']})")

    print("\n[5] Quality gating (does a bad measurement get rejected?)")
    dev.load_sample("Hematite")
    dev.set_integration_time(4)          # far too short -> poor SNR
    dev.take_reference("dark"); dev.take_reference("white")
    dev.set_integration_time(4)
    r = run("selftest starved signal")
    low_ok = (not r.quality["passed"]) or r.quality["snr"] < 200
    print(f"    {'OK ' if low_ok else 'XX '}short integration -> SNR "
          f"{r.quality['snr']:.0f}, QC {'fail' if not r.quality['passed'] else 'pass'}")
    passes += low_ok; failures += (not low_ok)

    dev.set_integration_time(1500)       # far too long -> saturation
    r = run("selftest saturated")
    sat_ok = r.quality["saturated_fraction"] > 0.0 or not r.quality["passed"]
    print(f"    {'OK ' if sat_ok else 'XX '}long integration -> "
          f"{r.quality['saturated_fraction'] * 100:.1f}% saturated, "
          f"QC {'fail' if not r.quality['passed'] else 'pass'}")
    passes += sat_ok; failures += (not sat_ok)

    if not args.quick:
        print("\n[6] Report generation")
        dev.set_integration_time(90)
        dev.take_reference("dark"); dev.take_reference("white")
        dev.load_sample("Hematite")
        r = run("selftest report sample")
        d = r.as_dict()
        db.save_analysis(d, session_id, source="selftest")
        t0 = time.time()
        files = repmod.generate(d, cfg.resolve("server.reports_dir"))
        for kind, path in files.items():
            if path and Path(path).exists():
                size = Path(path).stat().st_size
                print(f"    OK  {kind:<7} {Path(path).name}  ({size / 1024:.0f} KB)")
                passes += 1
            else:
                print(f"    XX  {kind} MISSING")
                failures += 1
        print(f"    generated in {time.time() - t0:.1f}s")

        summary_pdf = repmod.generate_batch(
            [db.get_analysis(a["analysis_id"])
             for a in db.list_analyses(session_id, 100)][:10],
            cfg.resolve("server.reports_dir"), "selftest")
        ok = Path(summary_pdf).exists()
        print(f"    {'OK ' if ok else 'XX '}session summary  {Path(summary_pdf).name}")
        passes += ok; failures += (not ok)

    db.end_session(session_id)

    print("\n" + "=" * 78)
    print(f"  {passes} passed, {failures} failed")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
