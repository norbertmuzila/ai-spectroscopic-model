---
title: USB4000 Spectral Console
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# USB4000 Spectral Console

Mineral identification from VNIR reflectance spectra: physics-based spectral
matching fused with a trained ensemble, conformal-calibrated confidence, and
spectral unmixing, over a 92-mineral library covering terrestrial, Martian and
lunar phases.

**This hosted instance runs the built-in instrument simulator.** A container in
a datacentre has no USB bus, so it cannot reach a real spectrometer — the
hardware path runs on the machine the USB4000 is plugged into. Everything else
is identical: same engine, same library, same calibration.

Pick a mineral under *Simulated sample*, press **Load into simulator**, then
**Run analysis**.
