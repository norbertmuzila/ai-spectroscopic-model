"""
Download the USGS Digital Spectral Library splib05a as ground truth.

    python scripts/fetch_usgs_splib05.py            # download everything
    python scripts/fetch_usgs_splib05.py --limit 40 # quick partial run

Source: Clark, R.N., Swayze, G.A., Wise, R., Livo, K.E., Hoefen, T.M., Kokaly,
R.F., and Sutley, S.J., 2003, USGS Digital Spectral Library splib05a, USGS
Open-File Report 03-395. <https://pubs.usgs.gov/of/2003/ofr-03-395/datatable.html>

Why this matters
----------------
Without it the identification engine has nothing real to compare against. It
falls back to endmembers synthesised from published band parameters - physically
reasonable, but an idealised curve is not the mineral. Grain-size distribution,
crystallinity, trace substitution and the specific instrument all shape a real
spectrum in ways no analytic model reproduces, and matching a bench measurement
against a model rather than a measurement is exactly how a sample comes back as
the wrong mineral.

These files are also the authority for *naming*. The title line of each spectrum
carries the USGS mineral name, sample number and mineral group, and those are
what the system reports, so a result can be checked against the USGS entry it
came from rather than against a taxonomy invented here.

File format (documented in each file's own header):
    line 15   title, e.g. "Acmite NMNH133746 Pyroxene   W1R1Ba AREF"
    line 16   history
    line 17+  three columns: wavelength (um), reflectance, standard deviation
    ******    a deleted channel
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://pubs.usgs.gov/of/2003/ofr-03-395"
INDEX = f"{BASE}/datatable.html"
DEST = ROOT / "data" / "library" / "splib05a"

# The library is organised into chapters; these are the ones that describe
# solid materials a rock sample could be made of. Vegetation, liquids and
# man-made chapters are skipped - matching a mineral against a spectrum of
# lawn grass helps nobody.
WANTED_CHAPTERS = {"M", "S", "C", "A"}   # Minerals, Soils/mixtures, Coatings, Artificial
CHAPTER_NAMES = {
    "M": "Minerals", "S": "Soils and Mixtures", "C": "Coatings",
    "A": "Artificial", "L": "Liquids", "O": "Organics", "V": "Vegetation",
}

_print_lock = threading.Lock()


def fetch(url: str, timeout: float = 60.0, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "spectral-console/1.0 (mineral identification research)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def parse_index(html: str) -> list:
    """Every ASCII spectrum link on the data table, with its chapter."""
    out, seen = [], set()
    for href in re.findall(r'href=["\']([^"\']+\.asc)["\']', html, re.I):
        href = href.strip()
        m = re.match(r"ASCII/([A-Za-z])/(.+\.asc)$", href)
        if not m:
            continue
        chapter, fname = m.group(1).upper(), m.group(2)
        if chapter not in WANTED_CHAPTERS or href in seen:
            continue
        seen.add(href)
        out.append({"chapter": chapter, "file": fname, "href": href})
    return out


DELETED = re.compile(r"\*+")


# A data row begins with the wavelength field, which is either a decimal number
# or - in files where that channel was deleted - a run of asterisks. Both forms
# have to count as data: some spectra open with dozens of deleted channels, and
# a rule that only recognised numbers would skip past them and mistake the first
# surviving data row for the title.
DATA_ROW = re.compile(r"^\s*(?:[-+]?\d+\.\d+|\*{2,})")
# Extracts the wavelength when it is present.
WAVE = re.compile(r"^\s*([-+]?\d+\.\d+)")
# One field is either a run of asterisks (a deleted value) or a number.
FIELD = re.compile(r"\*{2,}|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_asc(text: str) -> dict | None:
    """
    Parse one splib05a ASCII spectrum.

    Returns wavelengths in nanometres (the file stores micrometres), reflectance,
    and the per-channel standard deviation the library ships. Deleted channels -
    written as a run of asterisks - are dropped rather than coerced to zero,
    because a zero would read as a total absorption band.
    """
    lines = text.splitlines()
    if len(lines) < 18:
        return None

    # Locate the layout by finding the first actual data row, then step back two
    # lines for title and history. A fixed offset is wrong for this library:
    # most files carry a 14-line preamble, but some have none at all and put the
    # title on line 1, and reading a fixed line then yields a row of numbers as
    # the mineral name.
    first_data = None
    for i, line in enumerate(lines):
        if DATA_ROW.match(line):
            first_data = i
            break
    if first_data is None or first_data < 2:
        return None

    start = first_data - 2
    title = lines[start].strip()
    history = lines[start + 1].strip()

    wl, refl, sd = [], [], []
    for line in lines[first_data:]:
        m = WAVE.match(line)
        if not m:
            continue          # wavelength deleted: the channel cannot be placed
        w = float(m.group(1))
        rest = line[m.end(1):]
        # The file is fixed-width, and a deleted value is a run of asterisks
        # that fills its whole field. When the wavelength happens to occupy its
        # field completely, the asterisks butt straight up against it with no
        # separating space, so split() fuses them into one token. Reading the
        # remainder of the line rather than splitting the whole of it keeps the
        # deleted marker attached to the column it actually belongs to.
        fields = FIELD.findall(rest)
        if not fields:
            continue
        if DELETED.search(fields[0]):
            continue                      # reflectance deleted: drop the channel
        try:
            r = float(fields[0])
        except ValueError:
            continue
        s = 0.0
        if len(fields) > 1 and not DELETED.search(fields[1]):
            try:
                s = float(fields[1])
            except ValueError:
                s = 0.0
        # splib also flags invalid data with a large negative sentinel.
        if r < -1.0 or w <= 0:
            continue
        wl.append(w * 1000.0)          # micrometres -> nanometres
        refl.append(r)
        sd.append(s)

    if len(wl) < 30:
        return None
    return {"title": title, "history": history,
            "wavelength_nm": wl, "reflectance": refl, "stddev": sd}


def parse_title(title: str) -> dict:
    """
    Split a splib05a title into its parts.

        "Acmite NMNH133746 Pyroxene   W1R1Ba AREF"
         name   sample     group      config  type

    The trailing tokens are the measurement configuration and the data type
    (AREF = absolute reflectance). What precedes them is the mineral name, its
    sample number and the group USGS assigns it - and those are the names the
    system reports, so a result traces back to a specific library entry.
    """
    t = " ".join(title.split())
    dtype = ""
    for suffix in ("AREF", "RREF", "TRAN", "REF"):
        if t.upper().endswith(suffix):
            dtype = suffix
            t = t[: -len(suffix)].strip()
            break

    tokens = t.split()
    config = ""
    if tokens and re.match(r"^[A-Z]\d[A-Z]\d[A-Za-z]{1,3}$", tokens[-1]):
        config = tokens.pop()

    # The sample number is usually the token carrying digits - NMNH133746,
    # GDS27, HS22.3B - but some are pure letters with a hyphen (SCF-NHJ,
    # BK-Cornell), so an all-caps hyphenated token counts too.
    def looks_like_sample(tok: str) -> bool:
        if tok[:1].isdigit():
            return False
        if re.search(r"\d", tok):
            return True
        return "-" in tok and tok.upper() == tok and len(tok) > 3

    sample, name_parts, group_parts = "", [], []
    for i, tok in enumerate(tokens):
        if looks_like_sample(tok):
            sample = tok
            name_parts = tokens[:i]
            group_parts = tokens[i + 1:]
            break
    if not sample:
        name_parts = tokens[:1]
        group_parts = tokens[1:]

    return {
        "name": " ".join(name_parts) or (tokens[0] if tokens else "Unknown"),
        "sample": sample,
        "group": " ".join(group_parts),
        "config": config,
        "type": dtype,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N spectra")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    raw_dir = DEST / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  USGS Digital Spectral Library splib05a")
    print("  Clark et al. 2003, USGS Open-File Report 03-395")
    print("=" * 74)

    print(f"\n[1/3] Index: {INDEX}")
    try:
        html = fetch(INDEX).decode("utf-8", "replace")
    except Exception as exc:
        print(f"  failed: {exc}")
        return 1
    entries = parse_index(html)
    if args.limit:
        entries = entries[: args.limit]
    by_chapter: dict = {}
    for e in entries:
        by_chapter.setdefault(e["chapter"], []).append(e)
    print(f"  {len(entries)} spectra to fetch")
    for ch, items in sorted(by_chapter.items()):
        print(f"    {ch} {CHAPTER_NAMES.get(ch, ch):<22} {len(items)}")

    print(f"\n[2/3] Downloading with {args.workers} workers")
    done = [0]
    results = []

    def work(entry):
        target = raw_dir / f"{entry['chapter']}_{entry['file']}"
        try:
            if target.exists() and not args.force and target.stat().st_size > 400:
                text = target.read_text(encoding="utf-8", errors="replace")
            else:
                text = fetch(f"{BASE}/{entry['href']}").decode("utf-8", "replace")
                target.write_text(text, encoding="utf-8")
        except Exception as exc:
            return {"error": str(exc), **entry}

        rec = parse_asc(text)
        if rec is None:
            return {"error": "unparseable", **entry}
        meta = parse_title(rec["title"])
        return {**entry, **rec, **meta}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, e): e for e in entries}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(entries):
                with _print_lock:
                    ok = sum(1 for x in results if "error" not in x)
                    sys.stdout.write(f"\r  {done[0]}/{len(entries)}  ok={ok}")
                    sys.stdout.flush()
    print()

    good = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    print(f"  parsed {len(good)}, failed {len(bad)}")
    if bad:
        for b in bad[:5]:
            print(f"    {b['file']}: {b['error'][:70]}")

    print("\n[3/3] Writing consolidated library")
    out = {
        "source": "USGS Digital Spectral Library splib05a",
        "citation": ("Clark, R.N., Swayze, G.A., Wise, R., Livo, K.E., Hoefen, T.M., "
                     "Kokaly, R.F., and Sutley, S.J., 2003, USGS Digital Spectral "
                     "Library splib05a: USGS Open-File Report 03-395."),
        "url": INDEX,
        "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_spectra": len(good),
        "spectra": [{
            "chapter": r["chapter"],
            "chapter_name": CHAPTER_NAMES.get(r["chapter"], r["chapter"]),
            "file": r["file"],
            "title": r["title"],
            "name": r["name"],
            "sample": r["sample"],
            "group": r["group"],
            "config": r["config"],
            "type": r["type"],
            "n_points": len(r["wavelength_nm"]),
            "range_nm": [round(min(r["wavelength_nm"]), 2),
                         round(max(r["wavelength_nm"]), 2)],
            "wavelength_nm": [round(v, 4) for v in r["wavelength_nm"]],
            "reflectance": [round(v, 6) for v in r["reflectance"]],
        } for r in good],
    }
    path = DEST / "splib05a.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    names = sorted({r["name"] for r in good})
    print(f"\n  {len(names)} distinct mineral names, {len(good)} spectra")
    print("  sample of names: " + ", ".join(names[:12]))
    print("\nNext:  python scripts/build_usgs_library.py")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
