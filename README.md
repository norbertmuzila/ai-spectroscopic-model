# AI Spectroscopic Model — Ocean Optics USB4000

An autonomous mineral-identification system for the Ocean Optics USB4000: live
acquisition, physics-based spectral matching fused with a trained ensemble,
calibrated confidence with a statistical coverage guarantee, spectral unmixing,
a live dashboard, and one-click PDF/CSV/JSON reports.

---

## Read this first — what your instrument can and cannot do

You asked for the highest possible accuracy across Martian and terrestrial
minerals. To deliver that honestly, two facts have to be on the table, because
they shape everything the software does.

### 1. Your unit is 195–913 nm. Most Mars minerals are diagnosed in SWIR.

Measured from the attached instrument (serial `USB4H06117`) rather than assumed:
**3648 pixels spanning 195.47–912.56 nm at 0.197 nm/pixel.** That is a UV-VIS
grating, not the 350–1000 nm VNIR configuration this project first assumed, and
the difference is not cosmetic.

The 912 nm ceiling is a hard optical limit, and a costly one. It sits right on
top of the Fe²⁺/Fe³⁺ band cluster at 0.92–1.05 µm, so it removes:

| Lost at 912 nm | Their diagnostic band |
|---|---|
| maghemite, ferrihydrite, lepidocrocite, schwertmannite | 0.93–0.97 µm |
| hypersthene, pigeonite | 0.92–0.94 µm |
| nontronite, chlorite, biotite, glauconite | 0.92 µm Fe³⁺ |
| Mars dark basaltic sand, lunar mare & highland regolith | 0.92–0.98 µm |
| **goethite** | 0.92 µm — drops from 100% to 50% coverage |
| **jarosite** | 67% → 33% coverage |

Counting across the whole 92-mineral library: **fully diagnosable falls from 33
to 23, and not-diagnosable-at-all rises from 41 to 56.**

What still works well is the visible region — electronic transitions below
0.9 µm: **hematite** (0.53, 0.86 µm), **enstatite** (0.905 µm, just inside),
**almandine garnet** (0.505/0.686/0.83 µm), **malachite, azurite, chrysocolla**
(0.77–0.80 µm Cu²⁺), **native sulfur, cinnabar, realgar, orpiment** (sharp
absorption edges), and **apatite** (REE lines).

The lower bound is set by your **light source**, not the detector. A
tungsten-halogen lamp is dark below ~340 nm; a deuterium or xenon source reaches
further into the UV. Measure it on your own bench:

```bash
python scripts/calibrate_range.py --write   # lamp on, white standard in place
python scripts/train.py                     # the model is tied to the grid
```

Silicon stops responding past about 1100 nm regardless — a property of the
semiconductor bandgap, not a setting — so no USB4000 reaches SWIR.

Mineral spectroscopy splits cleanly across that boundary:

| Feature type | Where | Minerals it identifies | USB4000 |
|---|---|---|---|
| Fe²⁺/Fe³⁺ crystal-field, charge transfer | 0.4–1.1 µm | olivine, pyroxene, hematite, goethite, jarosite, garnet, Cu minerals | **yes** |
| H₂O / OH overtones | 1.4, 1.9 µm | every hydrated phase | no |
| Al-OH, Mg-OH, Fe-OH | 2.16–2.35 µm | kaolinite, montmorillonite, nontronite, chlorite, serpentine | no |
| SO₄, CO₃ combinations | 2.1–2.53 µm | gypsum, kieserite, alunite, calcite, magnesite | no |

So on **your** instrument (340–912 nm):

**Strong performance** — hematite, nanophase hematite, almandine garnet,
malachite, azurite, chrysocolla, native sulfur, cinnabar, realgar, orpiment,
apatite, low-Ca pyroxene (enstatite, 0.905 µm), palagonite and Martian bright
dust. These are the visible-region electronic transitions, and this instrument
reads them well.

**Partial** — goethite and akaganeite (0.48 µm band visible, 0.92 µm at the
ceiling), jarosite (0.43 µm visible, 0.92 and 2.27 µm not), forsterite.

**Not possible** — everything diagnosed above 912 nm: all the hydrated phases,
sulfates, carbonates and clays as before, plus the 0.92–1.05 µm group listed in
the table above (maghemite, ferrihydrite, nontronite, olivine's main band, all
high-Ca pyroxene, the regolith analogs).

The engine knows this. Every candidate carries a **diagnostic coverage** score —
the fraction of that mineral's identifying bands inside your measured range. A
mineral scoring 0% is never reported as a confident identification no matter how
well its curve correlates, and the report devotes a section to what the
measurement could not determine. **A system that reported "Gypsum, 94%
confidence" from a 350–1000 nm spectrum would be lying to you**, and that is the
failure mode this design exists to prevent.

If SWIR mineralogy is essential to your work, the instrument to add is an
ASD-class or Spectral Evolution 350–2500 nm field spectrometer. The software
here is already built for it: set `wavelength_max_nm: 2500` in the config,
retrain, and the entire library unlocks. Until then, everything at 350–1000 nm
runs at full strength.

### 2. "100% accuracy" is not a thing anyone can deliver. Here is what replaces it.

No classifier is 100% accurate on real samples. Anyone claiming otherwise is
either overfitting to a test set or not measuring honestly. What *is* achievable
— and far more useful — is **calibrated confidence with a coverage guarantee**:

- **Temperature scaling** makes the confidence number mean what it says. When
  the system reports 90%, roughly 90% of such calls are correct.
- **Conformal prediction** returns a *set* — e.g. "Hematite or Goethite" —
  constructed so the true mineral is inside it **at least 95% of the time**,
  regardless of how good or bad the underlying model is. This is a
  distribution-free finite-sample guarantee, not an estimate.
- **Abstention.** Below a confidence floor the system reports `inconclusive`
  rather than guessing.

- **Spectral degeneracy groups.** Below 1000 nm, quartz, calcite, gypsum, halite
  and kaolinite are not merely *hard* to tell apart — they are the same
  measurement. The system derives these equivalence classes from the library
  itself and reports the group when it cannot subdivide it:

  > **Spectrally featureless, bright** — one of 31 phases including quartz,
  > calcite, gypsum, halite, kaolinite. SWIR coverage (1400–2500 nm) is required
  > to separate these.

  That is a usable finding. "Gypsum, 94% confident" would not be.

That is what "highest accuracy" looks like when built properly: a result you can
put in a report and defend.

---

## Quick start

```bash
pip install -r requirements.txt

python scripts/train.py            # ~10-20 min, builds and calibrates the model
python run.py                      # dashboard opens at http://127.0.0.1:8000
```

No hardware yet? Tick **Simulator mode** in the sidebar. The simulator models
the real signal chain — Planck lamp spectrum, grating efficiency, silicon QE,
Poisson shot noise, temperature-dependent dark current, stray light, ADC
saturation and cosmic-ray hits — so everything downstream is exercised for real.

### Recommended: install the USGS library

```bash
python scripts/fetch_usgs_library.py
python scripts/train.py            # retrain against the measured references
```

Without it the system matches against *modelled* endmembers built from published
band parameters. Those are physically sound and the pipeline works end-to-end,
but measured USGS spectra make field identification materially more reliable and
let the report cite a specific reference sample.

---

## Using it

### Direct mode (recommended)

1. **Connect** — the backend claims the USB4000 through `python-seabreeze`.
2. Set integration time so peak signal is 60–80% of full scale (never above 90%;
   saturation clips band depths and silently corrupts everything downstream).
3. **Take dark** with the light path blocked.
4. **Take white** on a Spectralon or PTFE standard.
   Both must be at the *same integration time* as the sample — the dashboard
   flags a mismatch in red because this is the most common way to get quietly
   wrong reflectance.
5. Label the sample and press **Run analysis**.
6. Press **PDF report**.

Because the USB4000 is uncooled, its dark current drifts with temperature.
References older than 30 minutes raise a warning; re-take them.

### SpectraSuite / OceanView mode

SpectraSuite is a closed 2009 Java application with no API, no plugin interface
and no IPC channel — nothing can call into it. What it does have is a
well-defined ASCII export, and that is the integration surface used here.

Set `spectrasuite.exclusive_device_mode: true` in `config/config.yaml`, then
drive acquisition from SpectraSuite as you do today and save (or auto-save) into
`data/incoming/`. Each new file is detected within a second, parsed, analysed
and pushed to the dashboard automatically. You can also drag any export in
through **Import file**.

Formats read: SpectraSuite tab-delimited, OceanView processed exports, bare
two-column CSV/TSV, and JCAMP-DX.

**The two modes are mutually exclusive.** The USB4000 can be claimed by only one
process at a time — SpectraSuite binds it through the Ocean Optics driver,
seabreeze binds it through libusb, and whichever starts first wins. This is a USB
driver constraint, not something software can work around.

Direct mode is the better one: it gives live streaming, enforced reference
validity and access to the factory nonlinearity coefficients.

---

## Architecture

```
Spectrometer ──┐
               ├─► preprocessing ─► quality gate ─┬─► physics matcher ──┐
SpectraSuite ──┘   dark/white         SNR         │   SAM SCM SID band  ├─► fusion
   export         nonlinearity      saturation    └─► ML ensemble ──────┘     │
                  despike            references       trees + trees + MLP     │
                  resample                                                    ▼
                  Savitzky-Golay                              temperature scaling
                  continuum removal                                     │
                                                              conformal prediction
                                                                        │
                                          ┌─────────────────────────────┤
                                          ▼                             ▼
                                   spectral unmixing            range audit
                                   (Hapke SSA space)      "what could this
                                          │               instrument not decide?"
                                          └──────────► dashboard + PDF report
```

### Preprocessing

Order matters, and getting it wrong destroys the features silently:
dark subtraction → nonlinearity correction → reflectance → despike → resample →
Savitzky-Golay → **continuum removal**.

Continuum removal is the critical step. Absolute reflectance depends on grain
size, packing, illumination angle and how level the sample cup is — none of which
say anything about mineralogy. Dividing by the convex hull leaves band position,
depth, width and asymmetry, which are what actually identify a mineral.

The engine also trims the analysis range to where the white reference carries
real signal (≥6% of its peak). Outside that, reflectance is noise divided by
noise and generates convincing-looking absorption bands that are pure artefact.

### Physics matcher

Four metrics, each insensitive to a different nuisance variable:

- **SAM** — spectral angle; invariant to illumination scaling.
- **SCM** — also removes additive offset (stray light, dark residual).
- **SID** — information divergence; catches subtle shape differences.
- **Band matching** — position and depth agreement, weighting position far more
  heavily because position is chemistry and depth is mostly grain size. Strong
  sample bands the candidate cannot explain count decisively against it.

All computed on the continuum-removed spectrum, masked to the measured range.

### Learned ensemble

There is no labelled corpus of USB4000 mineral spectra, so the training set is
built by **physics-based augmentation**: each endmember is expanded into hundreds
of realistic variants over grain size, weathering, solid-solution band shifts,
illumination gain, baseline drift, detector noise, wavelength-range trimming, and
intimate + areal mixtures. The model learns invariance to everything that is not
mineralogy.

Three models are soft-voted — Extra Trees on explicit band features, Extra Trees
on the continuum-removed curve, and an MLP on the same curve. Their errors are
decorrelated, so the ensemble is both more accurate and better calibrated than
any member.

Fusion with the matcher is **geometric** (log-linear), not an average: both
sources must support a hypothesis, so a mineral the physics rules out cannot be
resurrected by the classifier alone.

### Unmixing

Real samples are never single-phase. Both mixing physics are solved and the
better fit reported:

- **Areal** (patches larger than the photon mean free path) — linear in reflectance.
- **Intimate** (a powder, which is what a prepared sample is) — linear in
  single-scattering albedo, solved in Hapke *w*-space.

The distinction is not academic. 10% magnetite darkens an intimate mixture far
more than 10% of the way to magnetite; solving that in reflectance space would
report roughly 40% magnetite.

Subset selection is greedy-forward with a fit-improvement threshold, which keeps
solutions sparse — without it every answer degenerates into a blend of everything.

---

## Measured performance

From `scripts/train.py` on a held-out split that touched neither training nor
calibration (92 minerals, 350–1000 nm, modelled endmembers only — no USGS data
installed yet):

| Metric | Value |
|---|---|
| Conformal coverage (target 95%) | **95.5%** |
| Expected calibration error | **0.033** |
| Top-1, all 92 minerals | 36.1% |
| Top-1, the 51 minerals this range can diagnose | **58.1%** |
| Top-3, same 51 | **75.9%** |
| Model size / load | 52 MB |

And from `scripts/selftest.py`, an 18-mineral panel measured through the full
simulated instrument (**23/23 pass**):

| Sample | Reported | Confidence | Set size |
|---|---|---|---|
| Hematite | Hematite | 100% | 1 |
| Cinnabar | Cinnabar | 99.9% | 1 |
| Native sulfur | Native sulfur | 99.9% | 1 |
| Almandine garnet | Almandine garnet | 99.9% | 1 |
| Goethite | Goethite | 98.0% | 2 |
| Enstatite (LCP) | Enstatite (LCP) | 96.8% | 2 |
| Jarosite | Jarosite | 95.9% | 4 |
| Malachite | Malachite | 93.7% | 2 |
| Nontronite | Nontronite | 79.9% | 4 |
| Martian bright dust | Martian bright dust | 53.9% | 5 |
| Magnetite | *featureless very dark* group | 75.5% | 9 |
| Gypsum, calcite, kaolinite, halite | *featureless bright* group | ~20% | 15 |

The last two rows are the point. The system does not guess on samples whose
diagnostic evidence is outside its reach — it names the group it can defend and
says what instrument would settle it.

**The 36.1% vs 58.1% gap is not a tuning failure.** 41 of the 92 minerals have
no diagnostic feature below 1000 nm; no method can separate them here, because
the information never reached the detector.

Three real defects found and fixed during development, since they say something
about how the numbers should be read:

- Conformal sets were truncated to 8 members for display, which silently broke
  the coverage guarantee (67% actual against a 95% target). Sets are now
  untruncated.
- The classifier downsampled spectra by array index, so a measurement whose
  range had been trimmed was silently misaligned against the training grid.
- Band detection on dark samples was finding absorption features in **stray
  light**, not noise — reproducible across repeat measurements, so averaging
  could never remove it. The detection floor is now derived from the stray-light
  fraction and how much signal the sample returns.

## What the accuracy numbers mean

`scripts/train.py` prints top-1/top-3/top-5 accuracy, expected calibration error,
and empirical conformal coverage on a **held-out** split that touched neither
training nor calibration.

Be clear about what that measures: test spectra are synthetic, generated by the
same forward model as the training data. So top-1 accuracy measures how well the
model inverts that forward model under noise, mixing and grain-size variation.
It is a real and necessary number, but it is an **upper bound** on field
performance, because a real mineral is not exactly a sum of modified Gaussians.

The number that transfers to the field is the **conformal coverage**, which holds
regardless of model quality. And the way to get a truly authoritative figure is
to measure known standards on your own bench — once you have those, they become
the test set and the accuracy figure becomes the real one.

---

## Configuration

`config/config.yaml`. The most important entries:

| Key | Meaning |
|---|---|
| `instrument.wavelength_min_nm` / `max_nm` | **Set these to your unit's actual calibrated range.** They drive the range audit and every honesty check. |
| `instrument.default_integration_time_ms` | Starting integration time |
| `acquisition.min_snr_for_identification` | Below this, QC fails the measurement rather than identifying it |
| `acquisition.reference_ttl_minutes` | Age at which dark/white references are flagged stale |
| `spectrasuite.exclusive_device_mode` | `true` = SpectraSuite owns the device, backend ingests exported files |
| `spectrasuite.watch_dir` | Folder monitored for new exports |
| `model.conformal_alpha` | `0.05` → 95% coverage guarantee |
| `model.abstain_below` | Confidence floor below which the system reports `inconclusive` |
| `library.include_martian` / `lunar` / `terrestrial` | Which mineral families to load |

---

## Mineral coverage

92 minerals across 24 groups — 80 terrestrial, 56 Martian, 17 lunar.

**Mars-focused**: olivine (forsterite/Fo50/fayalite), pyroxenes (enstatite,
hypersthene, pigeonite, augite, diopside), plagioclase, hematite, nanophase
hematite, goethite, magnetite, maghemite, ferrihydrite, akaganeite,
schwertmannite, jarosite, kieserite, gypsum, bassanite, anhydrite, epsomite,
szomolnokite, copiapite, melanterite, alunite, Mg/Fe/Ca carbonates, nontronite,
saponite, montmorillonite, kaolinite, chlorite, serpentine, palygorskite,
opal/hydrated silica, basaltic glass, Mg-perchlorate, chlorides, native sulfur,
palagonite, Martian bright dust, Martian dark basaltic sand.

**Lunar**: anorthite, labradorite, LCP/HCP pyroxene, olivine, ilmenite,
Mg-spinel, chromite, apatite, mare regolith, highland regolith, agglutinate,
pyroclastic glass.

**Terrestrial**: the above plus quartz, feldspars, muscovite, illite, biotite,
talc, gibbsite, zeolite, amphiboles, epidote, garnets, tourmaline, barite,
fluorite, halite, sulfides (pyrite, chalcopyrite, galena, sphalerite, cinnabar,
realgar, orpiment), Cu minerals (malachite, azurite, chrysocolla).

`GET /api/minerals` lists each one with its diagnostic band positions and its
coverage score for *your* configured range — a direct answer to "what can this
instrument actually identify?"

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | Device, engine, library, validation metrics |
| `GET /api/minerals` | Full mineral list with per-mineral diagnostic coverage |
| `POST /api/device/connect` | Open hardware (or simulator) |
| `POST /api/device/settings` | Integration time, averaging, boxcar |
| `POST /api/device/reference/{dark\|white}` | Take a reference |
| `GET /api/device/preview` | One raw frame |
| `POST /api/measure` | Acquire + full analysis |
| `POST /api/import` | Upload and analyse a SpectraSuite/CSV export |
| `POST /api/stream/{start\|stop}` | Live streaming |
| `WS /ws/live` | Live spectra + analysis push |
| `GET /api/analyses` · `/{id}` | History |
| `POST /api/session` | Start a session |
| `GET /api/report/{id}/download?fmt=pdf\|csv\|json` | Download a report |
| `GET /api/report/session/{id}/download` | Session summary PDF |

Interactive docs at `/docs`.

---

## Layout

```
backend/
  config.py                 configuration + analysis grid
  main.py                   FastAPI: REST, WebSocket, static
  db.py                     SQLite sessions and analyses
  hardware/
    device.py               seabreeze USB4000 driver + reference management
    simulator.py            physically-modelled USB4000
    spectrasuite.py         export parser + folder watcher
  processing/
    preprocess.py           radiometry, despike, resample, continuum removal
    features.py             band centre/depth/width/asymmetry extraction
    quality.py              SNR, saturation, reference validity gating
  library/
    minerals.py             92-mineral knowledge base with MGM band parameters
    synth.py                MGM forward model + Hapke mixing
    usgs.py                 splib07a ASCII reader
    store.py                unified library with provenance
  models/
    matcher.py              SAM / SCM / SID / band matching
    classifier.py           augmentation + three-model ensemble
    unmixing.py             FCLS in reflectance and SSA space
    calibration.py          temperature scaling + conformal prediction
    engine.py               orchestration, fusion, range audit
  reporting/report.py       PDF + CSV + JSON
frontend/                   dashboard (no CDN dependencies — works offline)
scripts/
  fetch_usgs_library.py     install USGS splib07a
  train.py                  train + calibrate + evaluate
config/config.yaml
run.py
```

---

## Troubleshooting

**Start here:**

```bash
python scripts/diagnose.py
```

It checks the library, the USB backend, the bus, the Windows driver binding and
an actual open attempt, then prints one verdict and the specific fix. The
dashboard runs the same checks — the **Hardware** panel appears automatically
whenever a connect fails, and **Run hardware diagnostic** triggers it on demand.

**"No Ocean Optics device is attached"** — the spectrometer is not on the USB bus.
Windows keeps a record of every device ever plugged in, and Device Manager shows
those *remembered* entries exactly like live ones, which is why an unplugged
instrument so often looks like a driver fault. The diagnostic separates the two
explicitly. Check, in order: a data cable rather than a charge-only one; a port
directly on the machine rather than an unpowered hub (the USB4000 draws ~450 mA);
a different port. An `Unknown USB Device (Device Descriptor Request Failed)`
entry appearing when you plug it in is the signature of a failing cable or an
underpowered port.

**Device is on the bus but seabreeze cannot claim it** — something else holds it,
or the vendor driver is bound instead of WinUSB. Close SpectraSuite and OceanView
first; only one process can own the device. Then run `seabreeze_os_setup`, or use
[Zadig](https://zadig.akeo.ie/): *Options → List All Devices*, select
**Ocean Optics USB4000**, choose **WinUSB**, *Replace Driver*.

**"No backend available" from pyusb** — the libusb binary is missing.
`pip install libusb-package` vendors one. It is in `requirements.txt`.

**The dashboard will not let me measure** — measurement and reference endpoints
return `409` until a device is explicitly connected. This is deliberate: the
system will not substitute simulated data for a measurement you asked to take on
real hardware.

**Everything reads `inconclusive`** — usually no white reference, or one taken at
a different integration time. The References panel flags a mismatch in red.

**A hydrated mineral you know is present is never reported** — expected, and
correct. See the range table at the top. The report's *Instrument limits* section
names exactly which candidates had evidence outside your range.

**Model pill says `matcher only`** — `scripts/train.py` has not been run. The
physics matcher still works; you lose the learned ensemble and the conformal
guarantee.

---

## Running it

**Windows, one click:** double-click `START.bat`. It installs anything missing,
trains the model the first time, runs the hardware diagnostic, then opens
<http://127.0.0.1:8000>.

**Manually:**

```bash
pip install -r requirements.txt
python scripts/train.py     # once, ~4 minutes
python run.py
```

## The live address

The console is published at a permanent HTTPS address:

```
https://<machine>.<your-tailnet>.ts.net
user: spectro     password: in data/credentials.json
```

Your own address is printed by `scripts/setup_funnel.py`, stored in
`data/public_url.txt`, and shown in the dashboard header. It is deliberately not
recorded in this file: the console is reachable from the internet, and a public
repository is not the place to advertise the address of a live instrument.

It starts by itself at Windows sign-in, with no terminal window, and the address
does not change across reboots or network changes.

### How it is put together

| Piece | What it does |
|---|---|
| Scheduled Task "Spectral Console" | Launches `run.py` under `pythonw.exe` at sign-in. No console window; restarts up to 3 times on failure; 20 s delay so the USB stack settles first. |
| Tailscale Funnel | Publishes `127.0.0.1:8000` at a stable `*.ts.net` hostname with a Let's Encrypt certificate. Free, no domain to buy. |
| `backend/auth.py` | HTTP Basic or `X-Spectro-Token`, constant-time comparison. `/api/health` stays open as a liveness probe. |
| `data/credentials.json` | The password, ACL'd to your account. An unattended start has no environment to carry one, and a permanent address needs one that stays the same. |

Set it up on another machine with **`SETUP-PERMANENT-URL.bat`**, or step by step:

```bash
tailscale up --hostname=spectral-console
python scripts/setup_funnel.py                                  # publish
powershell -ExecutionPolicy Bypass -File scripts/install_autostart.ps1
```

### Why a tunnel and not a cloud host

The spectrometer is a USB device on this machine. A container in a datacentre
has no USB bus, so a cloud deployment could only ever run the simulator. A
tunnel publishes *this* instance, so the measurement still happens where the
hardware is. That is the only arrangement in which a public URL reads the real
instrument.

### Security, plainly

The address is public, and it is discoverable: Funnel hostnames get a Let's
Encrypt certificate, and every issued certificate is published in Certificate
Transparency logs that scanners read continuously. **An internet scanner probed
`/debug/default/view?panel=config` within minutes of this address going live**
and was refused with a 401. That is not hypothetical hardening - it happened
during setup, and it is why the password is mandatory rather than optional.

The API can start acquisitions, overwrite calibration references and read every
stored analysis, so:

- Do not remove the password. `setup_funnel.py` refuses to run without one.
- To change it: edit `data/credentials.json` and restart the task.
- To take the address down entirely: `tailscale funnel --https=443 off`.

### Known behaviour

When the machine's IP changes - moving between networks, a DHCP renewal -
Tailscale re-establishes its relay path and the public address is briefly
unreachable, typically under a minute. It recovers on its own; nothing needs
restarting. The local `127.0.0.1:8000` address is unaffected throughout.

Logs from the unattended process go to `data/console.log`.

## Deploying elsewhere

`deploy/DEPLOY.md` covers three options, but the headline is a constraint worth
stating plainly: **a hosted web app cannot read your spectrometer.** A container
in a datacentre has no USB bus and a browser tab cannot open one. So:

- **Hugging Face Spaces** or **Render** host the dashboard and engine running the
  simulator — free, permanent, good for sharing the interface and the model.
  `Dockerfile` and `deploy/render.yaml` are ready; the image trains the model at
  build time so the container starts instantly.
- **A tunnel** (`cloudflared tunnel --url http://127.0.0.1:8000`) gives a public
  URL that *does* reach the real hardware, because the app keeps running on your
  machine. Note the API has no authentication — fine for a short demo, put access
  control in front of anything long-lived.
