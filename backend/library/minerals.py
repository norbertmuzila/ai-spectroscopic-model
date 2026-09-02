"""
Mineral knowledge base.

Each entry carries the *diagnostic absorption features* of a mineral, expressed
as Modified Gaussian Model (MGM) components (Sunshine, Pieters & Pratt 1990):

    ln R(x) = continuum(x) - sum_i  s_i * exp( -( 1/x - 1/c_i )^2 / (2 w_i^2) )

with x in micrometres. Electronic (crystal-field / charge-transfer) transitions
are genuinely Gaussian in *energy*, which is why the modified form is used
rather than a plain wavelength-space Gaussian.

Band centres follow the standard reflectance-spectroscopy literature: Burns
(1993) crystal field theory; Clark et al. (2007) USGS splib06/07; Sunshine &
Pieters (1993) pyroxene and olivine deconvolution; Bishop et al. (2008, 2019)
Mars mineral spectroscopy; Pieters (1986) and Pieters & Noble (2016) for lunar
space weathering.

These parameters serve two purposes.

1.  They synthesise a physically plausible endmember for any mineral with no
    measured library spectrum yet, so the system is useful before the USGS
    download completes.
2.  ``diagnostic`` marks which bands actually *identify* the mineral. The
    engine checks those centres against the instrument's wavelength range and
    refuses to claim confidence it cannot physically earn. This is what stops a
    350-1000 nm instrument from "confidently" reporting gypsum.

Field meanings
--------------
albedo   : continuum reflectance level near 1.0 um (0-1)
slope    : red slope of the continuum, per micrometre, in ln-reflectance
bands    : list of (centre_um, width_um, strength) MGM components
edge     : optional (lambda_um, width_um, depth) charge-transfer edge, modelled
           as a sigmoid drop toward shorter wavelengths
env      : environments in which the mineral is relevant
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

TERRESTRIAL = "terrestrial"
MARS = "mars"
LUNAR = "lunar"


@dataclass(frozen=True)
class Mineral:
    name: str
    group: str
    formula: str
    env: tuple
    albedo: float
    slope: float
    bands: tuple
    diagnostic: tuple = ()
    edge: tuple | None = None
    usgs_hint: str = ""
    notes: str = ""

    @property
    def diagnostic_centres(self) -> tuple:
        return self.diagnostic if self.diagnostic else tuple(b[0] for b in self.bands)


def M(name, group, formula, env, albedo, slope, bands, diagnostic=(), edge=None,
      usgs_hint="", notes="") -> Mineral:
    return Mineral(name, group, formula, tuple(env), albedo, slope, tuple(bands),
                   tuple(diagnostic), edge, usgs_hint, notes)


# =============================================================================
#  OLIVINE - three overlapping Fe2+ crystal-field bands (M1, M2, M1)
# =============================================================================
_OLIVINE = [
    M("Forsterite", "Olivine", "Mg2SiO4", (TERRESTRIAL, MARS, LUNAR), 0.52, 0.06,
      [(0.85, 0.055, 0.09), (1.045, 0.075, 0.30), (1.24, 0.085, 0.15)],
      diagnostic=(1.045, 0.85, 1.24), usgs_hint="olivine",
      notes="Mg-rich olivine; the composite 1.05 um triplet sits inside USB4000 range."),
    M("Olivine (Fo50)", "Olivine", "(Mg,Fe)2SiO4", (TERRESTRIAL, MARS), 0.34, 0.09,
      [(0.86, 0.058, 0.11), (1.055, 0.078, 0.38), (1.26, 0.090, 0.19)],
      diagnostic=(1.055, 0.86, 1.26), usgs_hint="olivine",
      notes="Intermediate olivine, dominant in Nili Fossae and Martian basalts."),
    M("Fayalite", "Olivine", "Fe2SiO4", (TERRESTRIAL, MARS), 0.17, 0.11,
      [(0.88, 0.060, 0.14), (1.08, 0.082, 0.46), (1.31, 0.095, 0.24)],
      diagnostic=(1.08, 0.88, 1.31), usgs_hint="fayalite",
      notes="Fe-rich olivine; band centres shift long with Fe content."),
]

# =============================================================================
#  PYROXENE - Band I near 0.9-1.05 um, Band II near 1.8-2.3 um. Centres migrate
#  long with Ca and Fe content (Adams 1974; Cloutis & Gaffey 1991).
# =============================================================================
_PYROXENE = [
    M("Enstatite (LCP)", "Pyroxene", "MgSiO3", (TERRESTRIAL, MARS, LUNAR), 0.42, 0.07,
      [(0.905, 0.048, 0.30), (1.82, 0.115, 0.28)],
      diagnostic=(0.905, 1.82), usgs_hint="enstatite",
      notes="Low-calcium pyroxene. Band I at 0.90 um is inside VNIR range."),
    M("Hypersthene", "Pyroxene", "(Mg,Fe)SiO3", (TERRESTRIAL, MARS, LUNAR), 0.30, 0.08,
      [(0.92, 0.050, 0.34), (1.88, 0.120, 0.30)],
      diagnostic=(0.92, 1.88), usgs_hint="hypersthene"),
    M("Pigeonite", "Pyroxene", "(Mg,Fe,Ca)SiO3", (MARS, LUNAR), 0.26, 0.08,
      [(0.94, 0.052, 0.33), (1.98, 0.130, 0.29)],
      diagnostic=(0.94, 1.98), usgs_hint="pigeonite",
      notes="Common in Martian shergottites and lunar mare basalts."),
    M("Augite (HCP)", "Pyroxene", "(Ca,Mg,Fe)2Si2O6", (TERRESTRIAL, MARS, LUNAR), 0.22, 0.09,
      [(1.02, 0.058, 0.31), (2.28, 0.150, 0.27)],
      diagnostic=(1.02, 2.28), usgs_hint="augite",
      notes="High-calcium pyroxene. Band I at 1.02 um separates it from LCP at 0.90 um."),
    M("Diopside", "Pyroxene", "CaMgSi2O6", (TERRESTRIAL, MARS), 0.34, 0.07,
      [(1.02, 0.055, 0.26), (2.30, 0.150, 0.24)],
      diagnostic=(1.02, 2.30), usgs_hint="diopside"),
    M("Hedenbergite", "Pyroxene", "CaFeSi2O6", (TERRESTRIAL,), 0.18, 0.09,
      [(1.05, 0.060, 0.34), (2.35, 0.155, 0.28)],
      diagnostic=(1.05, 2.35), usgs_hint="hedenbergite"),
]

# =============================================================================
#  FELDSPAR / SILICA / GLASS
# =============================================================================
_FELDSPAR = [
    M("Anorthite", "Feldspar", "CaAl2Si2O8", (TERRESTRIAL, MARS, LUNAR), 0.70, 0.02,
      [(1.26, 0.105, 0.055)],
      diagnostic=(1.26,), usgs_hint="anorthite",
      notes="Lunar highland rock-former. The weak 1.25 um Fe2+ band needs high SNR."),
    M("Labradorite", "Feldspar", "(Ca,Na)(Al,Si)4O8", (TERRESTRIAL, MARS, LUNAR), 0.66, 0.02,
      [(1.27, 0.110, 0.048)], diagnostic=(1.27,), usgs_hint="labradorite"),
    M("Albite", "Feldspar", "NaAlSi3O8", (TERRESTRIAL,), 0.78, 0.01,
      [(1.42, 0.030, 0.03)], diagnostic=(1.42,), usgs_hint="albite",
      notes="Essentially featureless in VNIR; identified by the absence of features."),
    M("Orthoclase", "Feldspar", "KAlSi3O8", (TERRESTRIAL,), 0.72, 0.02,
      [(1.41, 0.030, 0.035)], diagnostic=(1.41,), usgs_hint="orthoclase"),
    M("Buddingtonite", "Feldspar", "NH4AlSi3O8", (TERRESTRIAL, MARS), 0.62, 0.02,
      [(1.56, 0.020, 0.10), (2.02, 0.022, 0.14), (2.12, 0.020, 0.22)],
      diagnostic=(2.12, 2.02), usgs_hint="buddingtonite"),
    M("Quartz", "Silica", "SiO2", (TERRESTRIAL, MARS), 0.85, 0.00,
      [(1.41, 0.020, 0.02)], diagnostic=(), usgs_hint="quartz",
      notes="Featureless and bright in VNIR. Diagnostic Si-O features lie at 8-10 um."),
    M("Opal / Hydrated Silica", "Silica", "SiO2.nH2O", (TERRESTRIAL, MARS), 0.68, 0.02,
      [(1.41, 0.030, 0.14), (1.91, 0.040, 0.22), (2.21, 0.035, 0.10)],
      diagnostic=(1.91, 2.21, 1.41), usgs_hint="opal",
      notes="Gusev and Valles Marineris hydrated silica. Entirely SWIR-diagnostic."),
    M("Basaltic Glass", "Glass", "amorphous", (MARS, LUNAR, TERRESTRIAL), 0.14, 0.20,
      [(1.05, 0.180, 0.13)], diagnostic=(1.05,), usgs_hint="basaltic glass",
      notes="Broad shallow Fe2+ band with a strong red continuum slope."),
    M("Obsidian", "Glass", "amorphous", (TERRESTRIAL,), 0.09, 0.16,
      [(1.05, 0.190, 0.10)], diagnostic=(1.05,)),
]

# =============================================================================
#  IRON OXIDES AND OXYHYDROXIDES - the strongest VNIR discriminators, and the
#  group the USB4000 is genuinely excellent at.
# =============================================================================
_FE_OXIDE = [
    M("Hematite", "Fe-oxide", "Fe2O3", (TERRESTRIAL, MARS), 0.16, 0.02,
      [(0.53, 0.028, 0.16), (0.65, 0.022, 0.07), (0.86, 0.052, 0.40)],
      diagnostic=(0.86, 0.53), edge=(0.60, 0.075, 0.90), usgs_hint="hematite",
      notes="Crystalline red hematite: 0.86 um Fe3+ band plus a sharp 0.6 um edge. "
            "Fully resolvable by the USB4000, and a best-case target for it."),
    M("Nanophase Hematite (npOx)", "Fe-oxide", "Fe2O3 (np)", (MARS,), 0.22, 0.10,
      [(0.86, 0.110, 0.13)], diagnostic=(0.86,), edge=(0.63, 0.130, 0.70),
      usgs_hint="nanophase",
      notes="Martian dust pigment: broad weak 0.86 um band, strong red slope."),
    M("Goethite", "Fe-oxide", "alpha-FeO(OH)", (TERRESTRIAL, MARS), 0.19, 0.03,
      [(0.48, 0.030, 0.14), (0.65, 0.026, 0.10), (0.92, 0.060, 0.35)],
      diagnostic=(0.92, 0.48), edge=(0.55, 0.070, 0.85), usgs_hint="goethite",
      notes="The 0.92 um band against hematite's 0.86 um is the key VNIR discrimination."),
    M("Maghemite", "Fe-oxide", "gamma-Fe2O3", (TERRESTRIAL, MARS), 0.10, 0.05,
      [(0.50, 0.032, 0.09), (0.93, 0.100, 0.14)],
      diagnostic=(0.93,), edge=(0.60, 0.100, 0.60), usgs_hint="maghemite"),
    M("Magnetite", "Fe-oxide", "Fe3O4", (TERRESTRIAL, MARS, LUNAR), 0.045, -0.03,
      [], diagnostic=(), usgs_hint="magnetite",
      notes="Opaque, essentially featureless and very dark, with a slight blue slope."),
    M("Ilmenite", "Fe-oxide", "FeTiO3", (LUNAR, TERRESTRIAL, MARS), 0.055, -0.01,
      [], diagnostic=(), usgs_hint="ilmenite",
      notes="Lunar mare opaque. Dark and flat, identified by darkness rather than bands."),
    M("Ferrihydrite", "Fe-oxide", "Fe(OH)3", (TERRESTRIAL, MARS), 0.24, 0.06,
      [(0.48, 0.038, 0.10), (0.93, 0.115, 0.20)],
      diagnostic=(0.93,), edge=(0.58, 0.110, 0.72), usgs_hint="ferrihydrite"),
    M("Lepidocrocite", "Fe-oxide", "gamma-FeO(OH)", (TERRESTRIAL,), 0.21, 0.03,
      [(0.49, 0.030, 0.13), (0.65, 0.025, 0.09), (0.97, 0.065, 0.30)],
      diagnostic=(0.97,), edge=(0.56, 0.072, 0.82), usgs_hint="lepidocrocite"),
    M("Akaganeite", "Fe-oxide", "beta-FeO(OH,Cl)", (TERRESTRIAL, MARS), 0.20, 0.04,
      [(0.42, 0.028, 0.11), (0.65, 0.026, 0.08), (0.92, 0.070, 0.28)],
      diagnostic=(0.92, 0.42), edge=(0.54, 0.075, 0.80), usgs_hint="akaganeite",
      notes="Cl-bearing oxyhydroxide, reported at Gale crater."),
    M("Schwertmannite", "Fe-oxide", "Fe8O8(OH)6SO4", (TERRESTRIAL, MARS), 0.22, 0.04,
      [(0.48, 0.034, 0.12), (0.92, 0.080, 0.26)],
      diagnostic=(0.92,), edge=(0.56, 0.080, 0.78), usgs_hint="schwertmannite"),
    M("Chromite", "Spinel", "FeCr2O4", (TERRESTRIAL, LUNAR, MARS), 0.07, 0.02,
      [(0.50, 0.060, 0.06), (1.30, 0.200, 0.08)], diagnostic=(), usgs_hint="chromite"),
    M("Mg-Spinel", "Spinel", "MgAl2O4", (LUNAR, TERRESTRIAL), 0.48, 0.03,
      [(2.00, 0.180, 0.24), (1.30, 0.150, 0.07)],
      diagnostic=(2.00,), usgs_hint="spinel",
      notes="Lunar Mg-spinel exposures at Moscoviense. The diagnostic band is SWIR."),
]

# =============================================================================
#  SULFATES - overwhelmingly SWIR-diagnostic. Retained for completeness and for
#  SWIR spectra imported from other instruments.
# =============================================================================
_SULFATE = [
    M("Gypsum", "Sulfate", "CaSO4.2H2O", (TERRESTRIAL, MARS), 0.76, 0.01,
      [(1.45, 0.017, 0.22), (1.49, 0.015, 0.18), (1.75, 0.018, 0.11),
       (1.94, 0.022, 0.32), (2.21, 0.020, 0.13)],
      diagnostic=(1.94, 1.45, 1.49), usgs_hint="gypsum",
      notes="Olympia Undae and Gale vein material. No diagnostic feature below 1.0 um."),
    M("Bassanite", "Sulfate", "CaSO4.0.5H2O", (TERRESTRIAL, MARS), 0.72, 0.01,
      [(1.43, 0.016, 0.20), (1.77, 0.018, 0.12), (1.93, 0.022, 0.24)],
      diagnostic=(1.93, 1.43), usgs_hint="bassanite"),
    M("Anhydrite", "Sulfate", "CaSO4", (TERRESTRIAL, MARS), 0.80, 0.01,
      [(1.75, 0.016, 0.09), (2.48, 0.030, 0.14)],
      diagnostic=(2.48, 1.75), usgs_hint="anhydrite"),
    M("Kieserite", "Sulfate", "MgSO4.H2O", (MARS, TERRESTRIAL), 0.58, 0.02,
      [(1.63, 0.075, 0.16), (2.13, 0.035, 0.24), (2.40, 0.040, 0.19)],
      diagnostic=(2.13, 1.63), usgs_hint="kieserite",
      notes="Monohydrated Mg-sulfate, Valles Marineris interior layered deposits."),
    M("Epsomite", "Sulfate", "MgSO4.7H2O", (MARS, TERRESTRIAL), 0.70, 0.01,
      [(1.44, 0.020, 0.24), (1.75, 0.020, 0.10), (1.94, 0.026, 0.30)],
      diagnostic=(1.94, 1.44), usgs_hint="epsomite"),
    M("Szomolnokite", "Sulfate", "FeSO4.H2O", (MARS,), 0.34, 0.04,
      [(0.95, 0.090, 0.16), (2.10, 0.035, 0.20), (2.40, 0.040, 0.16)],
      diagnostic=(2.10, 0.95), usgs_hint="szomolnokite",
      notes="Fe2+ sulfate with a genuine VNIR 0.95 um handle, unusual for a sulfate."),
    M("Jarosite", "Sulfate", "KFe3(SO4)2(OH)6", (TERRESTRIAL, MARS), 0.32, 0.03,
      [(0.43, 0.026, 0.13), (0.65, 0.024, 0.06), (0.92, 0.070, 0.20),
       (1.47, 0.020, 0.12), (1.85, 0.025, 0.14), (2.27, 0.018, 0.24), (2.51, 0.030, 0.16)],
      diagnostic=(2.27, 0.92, 0.43), edge=(0.50, 0.055, 0.70), usgs_hint="jarosite",
      notes="Meridiani Planum. Carries both a VNIR Fe3+ signature and a SWIR OH/SO4 "
            "signature, so it is partially identifiable with the USB4000 alone."),
    M("Alunite", "Sulfate", "KAl3(SO4)2(OH)6", (TERRESTRIAL, MARS), 0.66, 0.01,
      [(1.43, 0.015, 0.18), (1.76, 0.018, 0.11), (2.17, 0.017, 0.32), (2.32, 0.020, 0.12)],
      diagnostic=(2.17, 1.76), usgs_hint="alunite"),
    M("Copiapite", "Sulfate", "Fe5(SO4)6(OH)2.20H2O", (TERRESTRIAL, MARS), 0.40, 0.03,
      [(0.43, 0.028, 0.14), (0.90, 0.075, 0.18), (1.45, 0.022, 0.16), (1.94, 0.026, 0.22)],
      diagnostic=(0.90, 1.94), edge=(0.51, 0.060, 0.72), usgs_hint="copiapite"),
    M("Melanterite", "Sulfate", "FeSO4.7H2O", (TERRESTRIAL, MARS), 0.42, 0.03,
      [(0.95, 0.110, 0.20), (1.45, 0.022, 0.20), (1.94, 0.028, 0.26)],
      diagnostic=(0.95, 1.94), usgs_hint="melanterite"),
]

# =============================================================================
#  CARBONATES
# =============================================================================
_CARBONATE = [
    M("Calcite", "Carbonate", "CaCO3", (TERRESTRIAL, MARS), 0.78, 0.01,
      [(1.88, 0.020, 0.06), (2.16, 0.020, 0.08), (2.34, 0.022, 0.26), (2.53, 0.028, 0.20)],
      diagnostic=(2.34, 2.53), usgs_hint="calcite"),
    M("Dolomite", "Carbonate", "CaMg(CO3)2", (TERRESTRIAL, MARS), 0.76, 0.01,
      [(2.32, 0.020, 0.25), (2.51, 0.026, 0.19)],
      diagnostic=(2.32, 2.51), usgs_hint="dolomite"),
    M("Magnesite", "Carbonate", "MgCO3", (TERRESTRIAL, MARS), 0.80, 0.01,
      [(2.30, 0.019, 0.28), (2.50, 0.025, 0.21)],
      diagnostic=(2.30, 2.50), usgs_hint="magnesite",
      notes="Nili Fossae Mg-carbonate deposits."),
    M("Siderite", "Carbonate", "FeCO3", (TERRESTRIAL, MARS), 0.36, 0.04,
      [(1.05, 0.090, 0.20), (1.26, 0.090, 0.10), (2.32, 0.020, 0.24), (2.51, 0.026, 0.18)],
      diagnostic=(2.32, 1.05), usgs_hint="siderite",
      notes="Fe2+ carbonate; the 1.05 um band is a real VNIR handle."),
    M("Rhodochrosite", "Carbonate", "MnCO3", (TERRESTRIAL,), 0.52, 0.02,
      [(0.55, 0.040, 0.12), (2.33, 0.020, 0.22), (2.52, 0.026, 0.17)],
      diagnostic=(2.33, 0.55), usgs_hint="rhodochrosite"),
]

# =============================================================================
#  PHYLLOSILICATES AND CLAYS
# =============================================================================
_CLAY = [
    M("Montmorillonite", "Phyllosilicate", "(Na,Ca)0.3(Al,Mg)2Si4O10(OH)2.nH2O",
      (TERRESTRIAL, MARS), 0.62, 0.02,
      [(1.41, 0.014, 0.22), (1.91, 0.020, 0.30), (2.21, 0.016, 0.24)],
      diagnostic=(2.21, 1.91, 1.41), usgs_hint="montmorillonite",
      notes="Al-smectite at Mawrth Vallis. Purely SWIR-diagnostic."),
    M("Nontronite", "Phyllosilicate", "Na0.3Fe2Si4O10(OH)2.nH2O", (TERRESTRIAL, MARS),
      0.36, 0.04,
      [(0.43, 0.030, 0.09), (0.92, 0.090, 0.13), (1.43, 0.015, 0.20),
       (1.91, 0.020, 0.24), (2.29, 0.016, 0.30), (2.40, 0.020, 0.12)],
      diagnostic=(2.29, 1.91, 0.92), edge=(0.55, 0.090, 0.55), usgs_hint="nontronite",
      notes="Fe-smectite, the most widespread Martian clay. Its 0.92 um Fe3+ band "
            "gives the USB4000 partial but not conclusive sensitivity."),
    M("Saponite", "Phyllosilicate", "Ca0.25Mg3(Si,Al)4O10(OH)2.nH2O", (TERRESTRIAL, MARS),
      0.56, 0.02,
      [(1.41, 0.014, 0.18), (1.91, 0.020, 0.22), (2.31, 0.016, 0.26), (2.39, 0.018, 0.12)],
      diagnostic=(2.31, 1.91), usgs_hint="saponite",
      notes="Mg-smectite of the Yellowknife Bay mudstone, Gale crater."),
    M("Kaolinite", "Phyllosilicate", "Al2Si2O5(OH)4", (TERRESTRIAL, MARS), 0.74, 0.01,
      [(1.395, 0.010, 0.20), (1.415, 0.010, 0.24), (2.165, 0.011, 0.24), (2.207, 0.011, 0.34)],
      diagnostic=(2.207, 2.165), usgs_hint="kaolinite",
      notes="The 2.16/2.21 um doublet is the classic kaolinite fingerprint."),
    M("Illite", "Phyllosilicate", "K(Al,Mg,Fe)2(Si,Al)4O10(OH)2", (TERRESTRIAL, MARS),
      0.58, 0.02,
      [(1.41, 0.014, 0.18), (1.91, 0.020, 0.16), (2.20, 0.015, 0.30),
       (2.35, 0.018, 0.10), (2.45, 0.020, 0.09)],
      diagnostic=(2.20,), usgs_hint="illite"),
    M("Muscovite", "Phyllosilicate", "KAl2(AlSi3O10)(OH)2", (TERRESTRIAL,), 0.64, 0.02,
      [(1.41, 0.013, 0.16), (2.20, 0.014, 0.34), (2.35, 0.017, 0.13), (2.44, 0.019, 0.11)],
      diagnostic=(2.20,), usgs_hint="muscovite"),
    M("Chlorite", "Phyllosilicate", "(Mg,Fe)6(Si,Al)4O10(OH)8", (TERRESTRIAL, MARS),
      0.30, 0.03,
      [(0.72, 0.070, 0.07), (0.92, 0.095, 0.12), (1.40, 0.016, 0.10),
       (2.25, 0.018, 0.18), (2.33, 0.018, 0.28)],
      diagnostic=(2.33, 2.25, 0.92), usgs_hint="chlorite"),
    M("Serpentine", "Phyllosilicate", "Mg3Si2O5(OH)4", (TERRESTRIAL, MARS), 0.48, 0.02,
      [(0.95, 0.100, 0.07), (1.39, 0.013, 0.14), (2.11, 0.016, 0.08), (2.32, 0.014, 0.30)],
      diagnostic=(2.32, 1.39), usgs_hint="serpentine",
      notes="Product of Nili Fossae serpentinisation."),
    M("Talc", "Phyllosilicate", "Mg3Si4O10(OH)2", (TERRESTRIAL,), 0.72, 0.01,
      [(1.39, 0.012, 0.16), (2.29, 0.013, 0.20), (2.31, 0.013, 0.26), (2.39, 0.016, 0.14)],
      diagnostic=(2.31, 2.29), usgs_hint="talc"),
    M("Prehnite", "Phyllosilicate", "Ca2Al2Si3O10(OH)2", (TERRESTRIAL, MARS), 0.62, 0.02,
      [(1.48, 0.016, 0.14), (2.33, 0.016, 0.26)],
      diagnostic=(2.33, 1.48), usgs_hint="prehnite"),
    M("Palygorskite", "Phyllosilicate", "(Mg,Al)2Si4O10(OH).4H2O", (TERRESTRIAL, MARS),
      0.60, 0.02,
      [(1.41, 0.015, 0.20), (1.91, 0.022, 0.26), (2.32, 0.016, 0.20)],
      diagnostic=(2.32, 1.91), usgs_hint="palygorskite"),
    M("Biotite", "Phyllosilicate", "K(Mg,Fe)3AlSi3O10(OH)2", (TERRESTRIAL,), 0.14, 0.06,
      [(0.92, 0.130, 0.14), (1.40, 0.016, 0.08), (2.25, 0.020, 0.12), (2.33, 0.020, 0.16)],
      diagnostic=(2.33, 0.92), usgs_hint="biotite"),
    M("Glauconite", "Phyllosilicate", "(K,Na)(Fe,Al,Mg)2(Si,Al)4O10(OH)2", (TERRESTRIAL,),
      0.26, 0.04,
      [(0.45, 0.035, 0.08), (0.93, 0.100, 0.16), (1.41, 0.015, 0.12), (2.30, 0.018, 0.20)],
      diagnostic=(2.30, 0.93), usgs_hint="glauconite"),
    M("Vermiculite", "Phyllosilicate", "(Mg,Fe,Al)3(Al,Si)4O10(OH)2.4H2O", (TERRESTRIAL,),
      0.44, 0.03,
      [(0.95, 0.110, 0.10), (1.41, 0.016, 0.20), (1.91, 0.022, 0.26), (2.31, 0.018, 0.16)],
      diagnostic=(1.91, 2.31), usgs_hint="vermiculite"),
    M("Gibbsite", "Hydroxide", "Al(OH)3", (TERRESTRIAL,), 0.78, 0.01,
      [(1.45, 0.014, 0.18), (2.27, 0.015, 0.24), (2.32, 0.016, 0.16)],
      diagnostic=(2.27, 1.45), usgs_hint="gibbsite"),
    M("Zeolite (Clinoptilolite)", "Zeolite", "(Na,K,Ca)Al2Si7O18.6H2O", (TERRESTRIAL, MARS),
      0.68, 0.01,
      [(1.41, 0.016, 0.20), (1.91, 0.024, 0.30), (2.25, 0.020, 0.08)],
      diagnostic=(1.91, 1.41), usgs_hint="clinoptilolite"),
]

# =============================================================================
#  HALIDES, PERCHLORATES AND EVAPORITES
# =============================================================================
_EVAPORITE = [
    M("Halite", "Halide", "NaCl", (TERRESTRIAL, MARS), 0.88, 0.00,
      [], diagnostic=(), usgs_hint="halite",
      notes="Bright and featureless; identified by a high flat albedo."),
    M("Sylvite", "Halide", "KCl", (TERRESTRIAL,), 0.86, 0.00, [], diagnostic=()),
    M("Fluorite", "Halide", "CaF2", (TERRESTRIAL,), 0.82, 0.00,
      [(0.58, 0.020, 0.05)], diagnostic=(), usgs_hint="fluorite"),
    M("Mg-Perchlorate", "Perchlorate", "Mg(ClO4)2.6H2O", (MARS,), 0.70, 0.01,
      [(1.45, 0.020, 0.22), (1.95, 0.026, 0.30), (2.45, 0.030, 0.12)],
      diagnostic=(1.95, 1.45), usgs_hint="perchlorate",
      notes="Detected by Phoenix and Curiosity. SWIR-only signature."),
    M("Martian Chloride Deposit", "Halide", "Cl-bearing", (MARS,), 0.30, -0.04,
      [], diagnostic=(),
      notes="Featureless with a slightly blue slope in THEMIS-class data."),
]

# =============================================================================
#  SULFIDES, NATIVE ELEMENTS AND ECONOMIC MINERALS
# =============================================================================
_SULFIDE = [
    M("Pyrite", "Sulfide", "FeS2", (TERRESTRIAL,), 0.11, 0.30,
      [], diagnostic=(), usgs_hint="pyrite",
      notes="Dark with a strong rising red slope through the VNIR."),
    M("Chalcopyrite", "Sulfide", "CuFeS2", (TERRESTRIAL,), 0.10, 0.26,
      [(0.60, 0.090, 0.05)], diagnostic=(), usgs_hint="chalcopyrite"),
    M("Galena", "Sulfide", "PbS", (TERRESTRIAL,), 0.09, 0.05, [], diagnostic=()),
    M("Sphalerite", "Sulfide", "ZnS", (TERRESTRIAL,), 0.30, 0.08,
      [(0.65, 0.100, 0.08)], diagnostic=(), usgs_hint="sphalerite"),
    M("Cinnabar", "Sulfide", "HgS", (TERRESTRIAL,), 0.42, 0.02,
      [], diagnostic=(0.60,), edge=(0.60, 0.020, 0.92), usgs_hint="cinnabar",
      notes="Extremely sharp red absorption edge at 0.60 um, a VNIR-perfect target."),
    M("Realgar", "Sulfide", "AsS", (TERRESTRIAL,), 0.50, 0.01,
      [], diagnostic=(0.56,), edge=(0.56, 0.022, 0.88), usgs_hint="realgar"),
    M("Orpiment", "Sulfide", "As2S3", (TERRESTRIAL,), 0.62, 0.01,
      [], diagnostic=(0.50,), edge=(0.50, 0.018, 0.90), usgs_hint="orpiment"),
    M("Native Sulfur", "Native", "S8", (TERRESTRIAL, MARS), 0.72, 0.00,
      [], diagnostic=(0.46,), edge=(0.46, 0.016, 0.95), usgs_hint="sulfur",
      notes="Razor-sharp absorption edge near 0.46 um. Found by Curiosity in 2024."),
    M("Malachite", "Carbonate", "Cu2CO3(OH)2", (TERRESTRIAL,), 0.34, -0.06,
      [(0.80, 0.230, 0.45), (2.28, 0.018, 0.22), (2.35, 0.018, 0.16)],
      diagnostic=(0.80, 2.28), usgs_hint="malachite",
      notes="The broad Cu2+ band near 0.8 um is strong and sits fully in VNIR range."),
    M("Azurite", "Carbonate", "Cu3(CO3)2(OH)2", (TERRESTRIAL,), 0.24, -0.10,
      [(0.77, 0.240, 0.50), (2.28, 0.018, 0.20)],
      diagnostic=(0.77,), usgs_hint="azurite"),
    M("Chrysocolla", "Silicate", "Cu2H2Si2O5(OH)4", (TERRESTRIAL,), 0.30, -0.08,
      [(0.80, 0.260, 0.42), (1.91, 0.026, 0.20), (2.28, 0.020, 0.14)],
      diagnostic=(0.80,), usgs_hint="chrysocolla"),
]

# =============================================================================
#  OTHER ROCK-FORMING AND ACCESSORY MINERALS
# =============================================================================
_OTHER = [
    M("Almandine Garnet", "Garnet", "Fe3Al2(SiO4)3", (TERRESTRIAL,), 0.22, 0.04,
      [(0.505, 0.014, 0.14), (0.573, 0.014, 0.10), (0.686, 0.020, 0.13),
       (0.83, 0.070, 0.28), (1.15, 0.090, 0.22), (1.30, 0.090, 0.18)],
      diagnostic=(0.505, 0.686, 0.83), usgs_hint="almandine",
      notes="Sharp Fe2+ spin-forbidden triplet in the visible, an excellent VNIR target."),
    M("Grossular Garnet", "Garnet", "Ca3Al2(SiO4)3", (TERRESTRIAL,), 0.56, 0.02,
      [(0.44, 0.030, 0.08), (1.00, 0.120, 0.06)], diagnostic=(), usgs_hint="grossular"),
    M("Hornblende", "Amphibole", "Ca2(Mg,Fe)4Al(Si7Al)O22(OH)2", (TERRESTRIAL,), 0.16, 0.05,
      [(0.70, 0.080, 0.08), (1.03, 0.110, 0.22), (2.32, 0.018, 0.20), (2.39, 0.020, 0.14)],
      diagnostic=(1.03, 2.32), usgs_hint="hornblende"),
    M("Actinolite", "Amphibole", "Ca2(Mg,Fe)5Si8O22(OH)2", (TERRESTRIAL,), 0.28, 0.03,
      [(1.02, 0.100, 0.18), (2.32, 0.016, 0.24), (2.39, 0.018, 0.16)],
      diagnostic=(2.32, 1.02), usgs_hint="actinolite"),
    M("Epidote", "Sorosilicate", "Ca2(Al,Fe)3(SiO4)3(OH)", (TERRESTRIAL,), 0.30, 0.03,
      [(0.45, 0.032, 0.10), (1.06, 0.130, 0.16), (2.25, 0.018, 0.18), (2.33, 0.018, 0.26)],
      diagnostic=(2.33, 1.06), usgs_hint="epidote"),
    M("Apatite", "Phosphate", "Ca5(PO4)3(F,Cl,OH)", (TERRESTRIAL, MARS, LUNAR), 0.68, 0.01,
      [(0.58, 0.008, 0.10), (0.74, 0.010, 0.12), (0.80, 0.010, 0.09)],
      diagnostic=(0.74, 0.58), usgs_hint="apatite",
      notes="Sharp REE (Nd, Pr) lines in the visible: narrow but very distinctive."),
    M("Barite", "Sulfate", "BaSO4", (TERRESTRIAL,), 0.80, 0.00,
      [(1.93, 0.024, 0.08), (2.46, 0.028, 0.10)], diagnostic=(), usgs_hint="barite"),
    M("Tourmaline", "Cyclosilicate", "Na(Mg,Fe)3Al6(BO3)3Si6O18(OH)4", (TERRESTRIAL,),
      0.20, 0.05,
      [(0.72, 0.100, 0.14), (1.15, 0.140, 0.10), (2.21, 0.020, 0.14)],
      diagnostic=(2.21, 0.72), usgs_hint="tourmaline"),
]

# =============================================================================
#  PLANETARY REGOLITH AND WEATHERING PRODUCTS
# =============================================================================
_REGOLITH = [
    M("Palagonite (Mars Soil Analog)", "Alteration", "altered basaltic glass", (MARS,),
      0.20, 0.22,
      [(0.90, 0.160, 0.10)], diagnostic=(0.90,), usgs_hint="palagonite",
      notes="Mauna Kea JSC Mars-1 analog: strong red slope with a weak broad ferric "
            "band. This is the most common Martian surface spectral signature."),
    M("Martian Bright Dust", "Regolith", "npOx + basalt + sulfate", (MARS,), 0.28, 0.26,
      [(0.86, 0.130, 0.09)], diagnostic=(0.86,),
      notes="Global dust unit: very red visible slope, weak 0.86 um band."),
    M("Martian Dark Basaltic Sand", "Regolith", "pyroxene + olivine + glass", (MARS,),
      0.10, 0.07,
      [(0.98, 0.140, 0.16), (1.95, 0.180, 0.12)], diagnostic=(0.98,),
      notes="Bagnold dunes-type material."),
    M("Lunar Mare Regolith", "Regolith", "cpx + ilmenite + agglutinate", (LUNAR,), 0.09, 0.24,
      [(0.98, 0.120, 0.13), (2.05, 0.180, 0.10)], diagnostic=(0.98,),
      notes="Space weathering darkens, reddens and suppresses bands (Pieters & Noble 2016)."),
    M("Lunar Highland Regolith", "Regolith", "plagioclase + opx", (LUNAR,), 0.28, 0.18,
      [(0.92, 0.110, 0.07), (1.26, 0.120, 0.04)], diagnostic=(0.92,)),
    M("Lunar Agglutinate", "Regolith", "impact glass + npFe0", (LUNAR,), 0.07, 0.34,
      [], diagnostic=(),
      notes="Nanophase iron gives an extreme red slope and near-total band suppression."),
    M("Lunar Pyroclastic Glass", "Glass", "Fe-Ti volcanic glass", (LUNAR,), 0.06, 0.20,
      [(1.05, 0.190, 0.11), (2.10, 0.220, 0.08)], diagnostic=(1.05,),
      notes="Orange and black beads from Taurus-Littrow."),
]

ALL_MINERALS = (
    _OLIVINE + _PYROXENE + _FELDSPAR + _FE_OXIDE + _SULFATE + _CARBONATE
    + _CLAY + _EVAPORITE + _SULFIDE + _OTHER + _REGOLITH
)

BY_NAME = {m.name: m for m in ALL_MINERALS}


def select(environments: Sequence[str]):
    """Minerals relevant to any of the given environments."""
    wanted = set(environments)
    return [m for m in ALL_MINERALS if wanted & set(m.env)]


def groups():
    return sorted({m.group for m in ALL_MINERALS})


def summary() -> dict:
    return {
        "n_minerals": len(ALL_MINERALS),
        "groups": groups(),
        "n_terrestrial": len(select([TERRESTRIAL])),
        "n_martian": len(select([MARS])),
        "n_lunar": len(select([LUNAR])),
    }
