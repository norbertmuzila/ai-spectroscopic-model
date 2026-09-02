"""
Fetch and install the USGS Spectral Library Version 7 (splib07a).

    python scripts/fetch_usgs_library.py                    # try to download
    python scripts/fetch_usgs_library.py --zip C:\\path\\to\\ASCIIdata_splib07a.zip
    python scripts/fetch_usgs_library.py --dir  C:\\path\\to\\ASCIIdata_splib07a

Why this matters
----------------
Until this runs, the system identifies minerals against *modelled* endmembers
generated from published band parameters. Those are physically sound and good
enough to build and validate the whole pipeline, but they are idealised. A real
hematite has a continuum shaped by its particular grain size distribution,
crystallinity and trace substitutions, and no analytic model reproduces that.
Measured references make field identifications materially more reliable, and
they are what lets the report cite a specific USGS sample as the basis for a
match.

Download notes
--------------
The USGS hosts this through ScienceBase and the direct file URL has changed
several times over the years. This script tries the URLs known to have worked;
if they all fail it prints manual instructions rather than guessing further.
Manual installation is a one-minute job and only has to happen once.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config as cfg_mod                      # noqa: E402

LANDING_PAGE = "https://www.sciencebase.gov/catalog/item/586e8c4ae4b0f5ce109fc9af"
DOI = "https://doi.org/10.3133/ds1035"

CANDIDATE_URLS = [
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Spectroscopy/splib07a/ASCIIdata_splib07a.zip",
    "https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/ASCIIdata_splib07a.zip",
    "https://crustal.usgs.gov/speclab/data/spectral.lib07/splib07a/ASCIIdata_splib07a.zip",
]


def try_download(dest: Path) -> Path | None:
    import urllib.error
    import urllib.request

    for url in CANDIDATE_URLS:
        print(f"  trying {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "spectro-model/1.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                with open(dest, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        if total:
                            sys.stdout.write(f"\r    {got / 1e6:.1f} / {total / 1e6:.1f} MB")
                            sys.stdout.flush()
            print("\n  downloaded")
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"    failed: {exc}")
    return None


def install_zip(zip_path: Path, target: Path) -> int:
    print(f"  extracting {zip_path.name}")
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if not member.lower().endswith(".txt"):
                continue
            # Flatten one level of the archive root but keep chapter folders,
            # so the reader's rglob finds both the wavelength files and the
            # per-chapter spectra.
            parts = Path(member).parts
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def install_dir(src: Path, target: Path) -> int:
    print(f"  copying from {src}")
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in src.rglob("*.txt"):
        rel = path.relative_to(src)
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Install the USGS splib07a spectral library")
    ap.add_argument("--zip", type=Path, help="path to an already-downloaded ASCIIdata_splib07a.zip")
    ap.add_argument("--dir", type=Path, help="path to an already-extracted ASCIIdata_splib07a folder")
    ap.add_argument("--no-download", action="store_true", help="skip the download attempt")
    args = ap.parse_args()

    cfg = cfg_mod.load_config()
    cfg_mod.ensure_dirs()
    target = cfg.resolve("library.usgs_dir")

    print("=" * 74)
    print("  USGS Spectral Library v7 (splib07a) installer")
    print("=" * 74)
    print(f"Target: {target}\n")

    n = 0
    if args.dir:
        n = install_dir(args.dir, target)
    elif args.zip:
        n = install_zip(args.zip, target)
    elif not args.no_download:
        print("Attempting download...")
        tmp = target.parent / "ASCIIdata_splib07a.zip"
        got = try_download(tmp)
        if got:
            n = install_zip(got, target)
            try:
                got.unlink()
            except OSError:
                pass

    if n == 0:
        print("\nNo spectra installed.\n")
        print("Manual installation (one minute, only needed once):")
        print(f"  1. Open  {LANDING_PAGE}")
        print(f"     (the data release DOI is {DOI})")
        print("  2. Download 'ASCIIdata_splib07a.zip' (about 250 MB).")
        print("  3. Run:")
        print(r"       python scripts/fetch_usgs_library.py --zip C:\path\to\ASCIIdata_splib07a.zip")
        print("\nThe system works without this - it falls back to modelled endmembers -")
        print("but measured references make field identifications materially stronger.")
        return 1

    print(f"\n  installed {n} files")

    print("\nRebuilding the spectral library index...")
    from backend.library import store as libstore
    lib = libstore.get_library(rebuild=True)
    summary = lib.summary()
    print(f"  {summary['n_entries']} spectra over {summary['n_classes']} minerals")
    print(f"  measured (USGS): {summary['sources']['usgs']}   "
          f"modelled: {summary['sources']['synthetic']}")
    matched = summary["classes_with_measured_reference"]
    print(f"\n  {len(matched)} minerals now have a measured reference:")
    for name in matched:
        print(f"    - {name}")
    print("\nNow retrain so the model learns from the measured spectra:")
    print("  python scripts/train.py")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
