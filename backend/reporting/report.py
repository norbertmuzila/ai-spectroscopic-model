"""
Analysis report generation.

Produces a self-contained PDF that states not just the answer but the evidence
and the limits: the spectrum and its continuum, the bands that were measured,
the ranked candidates with calibrated probabilities, the conformal prediction
set and its coverage level, the modelled composition, and - deliberately given
its own section - what the instrument's wavelength range could not determine.

A report that omits the last section is the one that gets someone into trouble,
because a 350-1000 nm instrument produces a perfectly plausible-looking answer
for minerals it fundamentally cannot see.

CSV and JSON exports are produced alongside for downstream use.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATUS_TEXT = {
    "identified": "IDENTIFIED",
    "ambiguous": "AMBIGUOUS - multiple minerals consistent with the data",
    "provisional_out_of_range": "PROVISIONAL - key evidence outside instrument range",
    "degenerate": "GROUP-LEVEL ONLY - this range cannot subdivide the spectral group",
    "inconclusive": "INCONCLUSIVE - confidence below reporting threshold",
    "rejected_quality": "REJECTED - measurement failed quality control",
}

STATUS_COLOR = {
    "identified": (0.05, 0.45, 0.22),
    "ambiguous": (0.72, 0.48, 0.05),
    "provisional_out_of_range": (0.72, 0.48, 0.05),
    "degenerate": (0.10, 0.40, 0.62),
    "inconclusive": (0.55, 0.20, 0.55),
    "rejected_quality": (0.68, 0.13, 0.13),
}


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------
def render_figure(result: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = result["spectrum"]
    wl = np.asarray(spec["wavelength_nm"], dtype=float)
    refl = np.asarray(spec["reflectance"], dtype=float)
    cont = np.asarray(spec["continuum"], dtype=float)
    cr = np.asarray(spec["continuum_removed"], dtype=float)
    modelled = np.asarray(spec.get("modelled") or [], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12})

    ax = axes[0]
    ax.plot(wl, refl, color="#1c4e80", lw=1.5, label="Measured reflectance")
    ax.plot(wl, cont, color="#9aa6b2", lw=1.0, ls="--", label="Convex-hull continuum")
    if modelled.size == wl.size and modelled.size:
        ax.plot(wl, modelled, color="#c8102e", lw=1.1, alpha=0.85,
                label="Unmixing model fit")
    ax.set_ylabel("Reflectance")
    ax.legend(loc="best", fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    ident = result["identification"]
    label = ident.get("headline") or ident.get("mineral") or "no identification"
    ax.set_title(f"{result.get('sample_label', 'Sample')}  -  {label}",
                 fontsize=10, loc="left")

    ax = axes[1]
    ax.plot(wl, cr, color="#0b6e4f", lw=1.4)
    ax.axhline(1.0, color="#9aa6b2", lw=0.8, ls=":")
    for band in result.get("bands", [])[:8]:
        c = band["centre_nm"]
        if wl.min() <= c <= wl.max():
            ax.axvline(c, color="#c8102e", lw=0.8, alpha=0.5)
            ax.annotate(f"{c:.0f}", (c, 1.005), fontsize=6.5, rotation=90,
                        ha="center", va="bottom", color="#c8102e")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Continuum removed")
    ax.grid(alpha=0.25, lw=0.5)

    # subplots_adjust rather than tight_layout: the shared-x gridspec already
    # fixes the vertical arrangement, and tight_layout fights it.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.095, hspace=0.12)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
#  PDF
# ---------------------------------------------------------------------------
def build_pdf(result: dict, out_path: Path, figure_path: Path | None = None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=4,
                        textColor=colors.HexColor("#12263f"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceBefore=10,
                        spaceAfter=4, textColor=colors.HexColor("#1c4e80"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8.6, leading=12,
                          alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=7.4, leading=10,
                           textColor=colors.HexColor("#5a6672"))

    ident = result["identification"]
    status = ident.get("status", "inconclusive")
    status_rgb = STATUS_COLOR.get(status, (0.3, 0.3, 0.3))
    status_col = colors.Color(*status_rgb)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"Spectral Analysis {result['analysis_id']}")
    flow = []

    flow.append(Paragraph("Spectral Mineral Identification Report", h1))
    flow.append(Paragraph(
        f"Analysis {result['analysis_id']} &nbsp;|&nbsp; {_fmt_time(result['timestamp'])}"
        f" &nbsp;|&nbsp; Ocean Optics USB4000", small))
    flow.append(Spacer(1, 6))

    # ---- headline ----------------------------------------------------
    conf = ident.get("confidence", 0.0)
    headline = [
        ["Sample", result.get("sample_label", "-")],
        ["Assessment", STATUS_TEXT.get(status, status)],
        ["Result", ident.get("headline") or ident.get("mineral") or "-"],
        ["Best-fit mineral", ident.get("mineral") or "-"],
        ["Indistinguishable from",
         ", ".join(ident.get("indistinguishable_from", [])[:10]) or "nothing - unique in range"],
        ["Calibrated confidence", f"{conf * 100:.1f}%"],
        ["Group / formula", f"{ident.get('group') or '-'}  /  {ident.get('formula') or '-'}"],
        ["Environments", ", ".join(ident.get("environments", [])) or "-"],
        ["Reference", f"{ident.get('reference_source') or '-'} "
                      f"({ident.get('reference_sample') or '-'})"],
        ["Spectral angle", f"{ident.get('sam_deg')} deg" if ident.get("sam_deg") is not None else "-"],
        ["Diagnostic coverage", f"{ident.get('diagnostic_coverage', 0) * 100:.0f}% of this "
                                f"mineral's diagnostic bands are inside the measured range"],
    ]
    t = Table(headline, colWidths=[42 * mm, 122 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5a6672")),
        ("TEXTCOLOR", (1, 1), (1, 1), status_col),
        ("FONTNAME", (1, 1), (1, 3), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e2e8ee")),
    ]))
    flow.append(t)

    # ---- figure ------------------------------------------------------
    if figure_path and Path(figure_path).exists():
        flow.append(Spacer(1, 8))
        flow.append(Image(str(figure_path), width=168 * mm, height=110 * mm))

    # ---- prediction set ---------------------------------------------
    pset = result.get("prediction_set", {})
    flow.append(Paragraph("Statistical confidence", h2))
    pset_names = pset.get("minerals", [])
    shown = ", ".join(pset_names[:12])
    if len(pset_names) > 12:
        shown += f", and {len(pset_names) - 12} more"
    flow.append(Paragraph(
        f"<b>{pset.get('coverage', 0.95) * 100:.0f}% prediction set "
        f"({len(pset_names)} mineral{'s' if len(pset_names) != 1 else ''}):</b> "
        f"{shown or '-'}<br/>"
        f"Method: {pset.get('method', '-')}, calibrated on "
        f"{pset.get('calibrated_on', 0)} held-out spectra. Constructed so the true mineral "
        f"falls inside this set at least {pset.get('coverage', 0.95) * 100:.0f}% of the time, "
        f"a distribution-free guarantee that does not depend on the model being good. "
        f"A single-member set is an unambiguous identification. A large set means the "
        f"measurement is under-determined - normally because the diagnostic evidence lies "
        f"outside the instrument's wavelength range, or SNR is too low - and is the correct "
        f"answer rather than a model failure.", body))

    val = result.get("engine", {}).get("validation", {})
    if val:
        flow.append(Paragraph(
            f"Model validation on held-out synthetic spectra: top-1 accuracy "
            f"{val.get('top1_accuracy', 0) * 100:.1f}%, top-3 "
            f"{val.get('top3_accuracy', 0) * 100:.1f}%, expected calibration error "
            f"{val.get('expected_calibration_error', 0):.3f}, empirical conformal coverage "
            f"{val.get('conformal_empirical_coverage', 0) * 100:.1f}% against a "
            f"{val.get('conformal_target_coverage', 0) * 100:.0f}% target.", small))

    # ---- candidates --------------------------------------------------
    flow.append(Paragraph("Ranked candidates", h2))
    rows = [["#", "Mineral", "Group", "Probability", "Diagnostic coverage"]]
    for i, c in enumerate(result.get("candidates", [])[:8], 1):
        rows.append([str(i), c["mineral"], c.get("group") or "-",
                     f"{c['probability'] * 100:.1f}%",
                     f"{c.get('diagnostic_coverage', 0) * 100:.0f}%"])
    t = Table(rows, colWidths=[8 * mm, 62 * mm, 34 * mm, 28 * mm, 32 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dde4ea")),
        ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    flow.append(t)

    flow.append(PageBreak())

    # ---- composition -------------------------------------------------
    comp = result.get("composition", {})
    flow.append(Paragraph("Modelled composition", h2))
    if comp.get("abundances"):
        rows = [["Phase", "Abundance", ""]]
        for name, frac in comp["abundances"].items():
            bar = "█" * max(1, int(round(frac * 28)))
            rows.append([name, f"{frac * 100:.1f}%", bar])
        t = Table(rows, colWidths=[62 * mm, 22 * mm, 80 * mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#1c4e80")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dde4ea")),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ]))
        flow.append(t)
        flow.append(Paragraph(
            f"Solved with the <b>{comp.get('mixing_model')}</b> mixing model "
            f"(residual RMSE {comp.get('rmse')}). Intimate mixing is solved in "
            f"single-scattering-albedo space because particulate mixtures are "
            f"non-linear in reflectance; opaque phases such as magnetite and ilmenite "
            f"darken a mixture far more than their abundance alone would suggest.", small))
    else:
        flow.append(Paragraph("No stable multi-phase solution was found; the sample is "
                              "consistent with a single dominant phase.", body))

    # ---- bands -------------------------------------------------------
    flow.append(Paragraph("Measured absorption features", h2))
    bands = result.get("bands", [])
    if bands:
        rows = [["Centre (nm)", "Depth", "Width (nm)", "Asymmetry"]]
        for b in sorted(bands, key=lambda x: -x["depth"])[:12]:
            rows.append([f"{b['centre_nm']:.1f}", f"{b['depth'] * 100:.2f}%",
                         f"{b['width_nm']:.0f}", f"{b['asymmetry']:+.2f}"])
        t = Table(rows, colWidths=[32 * mm, 28 * mm, 28 * mm, 28 * mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dde4ea")),
            ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
        ]))
        flow.append(t)
    else:
        flow.append(Paragraph("No absorption bands above the detection threshold.", body))

    # ---- interpretation ----------------------------------------------
    flow.append(Paragraph("Interpretation", h2))
    for line in result.get("interpretation", []):
        flow.append(Paragraph(f"• {line}", body))
        flow.append(Spacer(1, 1.5))

    # ---- limits ------------------------------------------------------
    audit = result.get("range_audit", {})
    flow.append(Paragraph("Instrument limits for this measurement", h2))
    lo, hi = audit.get("measured_range_nm", [0, 0])
    flow.append(Paragraph(
        f"Measured range <b>{lo:.0f}-{hi:.0f} nm</b>. Of the "
        f"{audit.get('library_minerals', 0)} minerals in the reference library, "
        f"<b>{audit.get('fully_diagnosable_here', 0)}</b> have all their diagnostic "
        f"absorption features inside this range, "
        f"<b>{audit.get('partially_diagnosable_here', 0)}</b> have some, and "
        f"<b>{audit.get('not_diagnosable_here', 0)}</b> have none. Minerals in the last "
        f"group cannot be confirmed by this instrument no matter how well their curves "
        f"correlate - hydrated phases, sulfates, carbonates and most clays are diagnosed "
        f"by vibrational features between 1400 and 2500 nm.", body))

    deg = audit.get("degeneracy", {})
    if deg.get("n_groups"):
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(
            f"Within this range those {audit.get('library_minerals', 0)} minerals collapse "
            f"into <b>{deg['n_groups']} spectrally separable groups</b>: "
            f"{deg['n_singleton_groups']} minerals are uniquely identifiable here, while "
            f"the largest indistinguishable group contains {deg['largest_group']} phases "
            f"that produce the same measurement below "
            f"{audit.get('measured_range_nm', [0, 0])[1]:.0f} nm. Where a result is "
            f"reported at group level it is because no analysis method could do better "
            f"with this data - the separating information is not present in it.", body))

    pset_groups = result.get("prediction_set", {}).get("groups", [])
    if len(pset_groups) > 1:
        rows = [["Consistent spectral group", "Phases in group"]]
        for g in pset_groups[:8]:
            rows.append([g["label"], str(g["n_in_group"])])
        t = Table(rows, colWidths=[120 * mm, 44 * mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 7.8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dde4ea")),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ]))
        flow.append(Spacer(1, 4))
        flow.append(t)

    blind = audit.get("candidates_with_evidence_outside_range", [])
    if blind:
        rows = [["Candidate", "Coverage", "Diagnostic bands outside range (nm)"]]
        for b in blind[:8]:
            rows.append([b["mineral"], f"{b['diagnostic_coverage'] * 100:.0f}%",
                         ", ".join(f"{v:.0f}" for v in b["bands_outside_range_nm"]) or "-"])
        t = Table(rows, colWidths=[58 * mm, 22 * mm, 84 * mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 7.6),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fff4e0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e8d9bd")),
        ]))
        flow.append(Spacer(1, 4))
        flow.append(t)

    # ---- QC ----------------------------------------------------------
    qc = result.get("quality", {})
    flow.append(Paragraph("Measurement quality", h2))
    rows = [
        ["Estimated SNR", f"{qc.get('snr', 0):.0f}"],
        ["Peak counts", f"{qc.get('max_counts', 0):.0f} / 65535"],
        ["Saturated channels", f"{qc.get('saturated_fraction', 0) * 100:.2f}%"],
        ["Dynamic range", f"{qc.get('dynamic_range', 0):.3f}"],
        ["QC outcome", "PASS" if qc.get("passed") else "FAIL"],
    ]
    t = Table(rows, colWidths=[42 * mm, 122 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5a6672")),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e2e8ee")),
    ]))
    flow.append(t)
    for w in qc.get("warnings", []):
        flow.append(Paragraph(f"⚠ {w}", small))
    for e in qc.get("errors", []):
        flow.append(Paragraph(f"✖ {e}", small))

    eng = result.get("engine", {})
    lib = eng.get("library", {})
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(
        f"Engine: physics matcher (SAM / SCM / SID / band fitting) fused with a "
        f"three-model learned ensemble at weights "
        f"{eng.get('fusion_weights', {}).get('matcher')} / "
        f"{eng.get('fusion_weights', {}).get('classifier')}; probabilities "
        f"temperature-scaled (T = {eng.get('temperature')}). Reference library: "
        f"{lib.get('n_entries', 0)} spectra over {lib.get('n_classes', 0)} minerals "
        f"({lib.get('sources', {}).get('usgs', 0)} measured USGS splib07a, "
        f"{lib.get('sources', {}).get('synthetic', 0)} modelled). "
        f"Processed in {eng.get('elapsed_ms', 0):.0f} ms.", small))

    doc.build(flow)
    return out_path


# ---------------------------------------------------------------------------
#  Exports
# ---------------------------------------------------------------------------
def write_json(result: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def write_csv(result: dict, path: Path) -> Path:
    """Spectrum as CSV, with the identification in a header comment block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = result["spectrum"]
    ident = result["identification"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# Analysis ID,{result['analysis_id']}\n")
        fh.write(f"# Timestamp,{_fmt_time(result['timestamp'])}\n")
        fh.write(f"# Sample,{result.get('sample_label', '')}\n")
        fh.write(f"# Mineral,{ident.get('mineral') or ''}\n")
        fh.write(f"# Confidence,{ident.get('confidence')}\n")
        fh.write(f"# Status,{ident.get('status')}\n")
        fh.write(f"# Prediction set,\"{'; '.join(result.get('prediction_set', {}).get('minerals', []))}\"\n")
        comp = result.get("composition", {}).get("abundances", {})
        if comp:
            fh.write("# Composition,\"" +
                     "; ".join(f"{k} {v * 100:.1f}%" for k, v in comp.items()) + "\"\n")
        w = csv.writer(fh)
        has_model = len(spec.get("modelled") or []) == len(spec["wavelength_nm"])
        header = ["wavelength_nm", "reflectance", "continuum", "continuum_removed"]
        if has_model:
            header.append("model_fit")
        w.writerow(header)
        for i in range(len(spec["wavelength_nm"])):
            row = [spec["wavelength_nm"][i], spec["reflectance"][i],
                   spec["continuum"][i], spec["continuum_removed"][i]]
            if has_model:
                row.append(spec["modelled"][i])
            w.writerow(row)
    return path


def generate(result: dict, reports_dir: Path) -> dict:
    """Write PDF + JSON + CSV for one analysis and return their paths."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    aid = result["analysis_id"]
    fig = reports_dir / f"{aid}_figure.png"
    try:
        render_figure(result, fig)
    except Exception as exc:
        print(f"[report] figure failed: {exc}")
        fig = None
    pdf = reports_dir / f"{aid}_report.pdf"
    build_pdf(result, pdf, fig)
    return {
        "pdf": str(pdf),
        "json": str(write_json(result, reports_dir / f"{aid}_data.json")),
        "csv": str(write_csv(result, reports_dir / f"{aid}_spectrum.csv")),
        "figure": str(fig) if fig else None,
    }


def generate_batch(results: list, reports_dir: Path, session_name: str = "session") -> Path:
    """One PDF covering a whole session, with a summary table up front."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_name)
    out = reports_dir / f"session_{safe}_summary.pdf"

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15,
                        textColor=colors.HexColor("#12263f"))
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=7.4, leading=10,
                           textColor=colors.HexColor("#5a6672"))

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=14 * mm,
                            rightMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    flow = [Paragraph(f"Session Summary - {session_name}", h1),
            Paragraph(f"{len(results)} analyses  |  generated {_fmt_time(__import__('time').time())}",
                      small),
            Spacer(1, 8)]

    rows = [["#", "Sample", "Mineral", "Conf.", "Status", "SNR", "Composition"]]
    for i, r in enumerate(results, 1):
        ident = r["identification"]
        comp = r.get("composition", {}).get("abundances", {})
        rows.append([
            str(i),
            (r.get("sample_label") or "-")[:26],
            (ident.get("mineral") or "-")[:24],
            f"{(ident.get('confidence') or 0) * 100:.0f}%",
            (ident.get("status") or "-").replace("_", " ")[:18],
            f"{r.get('quality', {}).get('snr', 0):.0f}",
            "; ".join(f"{k[:14]} {v * 100:.0f}%" for k, v in list(comp.items())[:3]) or "-",
        ])
    t = Table(rows, colWidths=[7 * mm, 34 * mm, 32 * mm, 13 * mm, 26 * mm, 12 * mm, 58 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dde4ea")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    flow.append(t)
    doc.build(flow)
    return out
