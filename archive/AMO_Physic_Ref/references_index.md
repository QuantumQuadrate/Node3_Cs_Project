# SNAQ-Node-3 — AMO Physics Reference Index

Curated, **verified** citation notes for the SNAQ-Node-3 (single-Cs-133 atom,
ARTIQ-7, tweezer/FORT, MOT loading, microwave + 2-photon Raman control,
Ramsey/Rabi, remote-entanglement node) reference library.

> **Format note.** These are citation/reference notes with canonical download
> links, **not** binary PDFs. The advisor environment lacks Bash/curl, so raw
> PDF binaries cannot be reliably fetched/written. Each entry was located and
> verified via web search/fetch against publisher records (authors, venue, year,
> DOI / arXiv ID confirmed). Where a paper has a free preprint or an
> author-hosted PDF, the link is given so a human can grab the binary later.

Existing textbook PDFs already in this directory (not re-listed below):
`atomic physics 2026.05.09.pdf`, `optics 2023.03.30.pdf`,
`quantum information 2026.05.09.pdf`.

---

## 1. MOT cooling / laser cooling & trapping

### 1.1 Raab, Prentiss, Cable, Chu & Pritchard (1987) — first MOT
- **Citation:** E. L. Raab, M. Prentiss, A. Cable, S. Chu, D. E. Pritchard,
  "Trapping of Neutral Sodium Atoms with Radiation Pressure," *Phys. Rev. Lett.*
  **59**, 2631–2634 (1987).
- **DOI:** 10.1103/PhysRevLett.59.2631
- **URL:** https://link.aps.org/doi/10.1103/PhysRevLett.59.2631
- **Covers:** The original demonstration of the magneto-optical trap (MOT):
  three orthogonal retroreflected, red-detuned beams plus a quadrupole magnetic
  field produce position- and velocity-dependent restoring + damping forces.
  Trapped ~10^7 Na atoms, sub-mK, in a sub-0.5 mm cloud.
- **Relevance to SNAQ-Node-3:** Foundational physics of the MOT loading stage
  that feeds the Cs tweezer/FORT — explains why beam balance, detuning sign
  (red of resonance), and the quadrupole field are required to confine and cool.

### 1.2 Dalibard & Cohen-Tannoudji (1989) — sub-Doppler / polarization-gradient cooling
- **Citation:** J. Dalibard, C. Cohen-Tannoudji, "Laser cooling below the Doppler
  limit by polarization gradients: simple theoretical models," *J. Opt. Soc. Am.
  B* **6**, 2023–2045 (1989).
- **DOI:** 10.1364/JOSAB.6.002023
- **URL (author-hosted PDF):** https://pro.college-de-france.fr/jean.dalibard/publi2/josaB_89.pdf
- **Covers:** The two canonical sub-Doppler mechanisms — lin⊥lin (Sisyphus) and
  σ+σ− polarization-gradient cooling — that reach temperatures far below the
  Doppler limit by coupling internal-state optical pumping to atomic motion.
- **Relevance to SNAQ-Node-3:** The physics behind the PGC / molasses recool
  stages used to cool the single Cs atom in the tweezer (and to re-cool after
  imaging before coherent operations). Anchors why low-intensity, larger
  red-detuning is the correct PGC parameter regime.

### 1.3 Metcalf & van der Straten — *Laser Cooling and Trapping* (textbook)
- **Citation:** H. J. Metcalf, P. van der Straten, *Laser Cooling and Trapping*,
  Graduate Texts in Contemporary Physics, Springer-Verlag, New York (1999).
  ISBN 978-0-387-98728-6.
- **DOI:** 10.1007/978-1-4612-1470-0
- **URL:** https://link.springer.com/book/10.1007/978-1-4612-1470-0
- **Covers:** Comprehensive graduate textbook: optical forces, optical molasses,
  Doppler and sub-Doppler theory, magnetic and optical traps, atomic optics,
  toward BEC. The standard pedagogical anchor for laser cooling.
- **Relevance to SNAQ-Node-3:** General-purpose reference for the laser-cooling
  and trapping timescales/temperatures cited when reviewing MOT and molasses
  phases of a sequence.

---

## 2. Optical dipole traps / FORT / optical tweezers (single-atom)

### 2.1 Grimm, Weidemüller & Ovchinnikov (2000) — dipole-trap review
- **Citation:** R. Grimm, M. Weidemüller, Y. B. Ovchinnikov, "Optical Dipole Traps
  for Neutral Atoms," *Adv. At. Mol. Opt. Phys.* **42**, 95–170 (2000).
- **DOI:** 10.1016/S1049-250X(08)60186-X
- **arXiv:** physics/9902072
- **URL:** https://arxiv.org/abs/physics/9902072
- **Covers:** The definitive review of the optical dipole force: AC-Stark / light
  shift, red- vs blue-detuned traps, trap depth and scattering-rate scaling
  (U ∝ I/Δ, Γ_sc ∝ I/Δ²), Gaussian-beam (FORT) geometry, heating mechanisms.
- **Relevance to SNAQ-Node-3:** Direct theory for the Cs FORT/tweezer trap depth,
  the U/Δ vs scattering-rate trade-off, and why the FORT depth cap (see
  `HARDWARE_SAFETY.md`, FORT cap 0.25) bounds achievable confinement. Use for
  any quantitative trap-depth / scattering-rate argument.

### 2.2 Schlosser, Reymond, Protsenko & Grangier (2001) — collisional blockade / single-atom loading
- **Citation:** N. Schlosser, G. Reymond, I. Protsenko, P. Grangier,
  "Sub-poissonian loading of single atoms in a microscopic dipole trap,"
  *Nature* **411**, 1024–1027 (2001).
- **DOI:** 10.1038/35082512
- **URL:** https://www.nature.com/articles/35082512
- **Open-access record:** https://iogs.hal.science/hal-00554215
- **Covers:** Demonstrated that a sub-micron dipole trap loads **0 or 1** atom but
  never 2 — light-assisted two-body collisions eject pairs (the "collisional
  blockade" regime), giving sub-Poissonian (≈0.5 probability) single-atom loading.
- **Relevance to SNAQ-Node-3:** This *is* the mechanism by which Node-3 prepares a
  single Cs atom in the tweezer. Explains the ~50% loading statistics, why a light
  field (and correct detuning) is needed during loading, and the deterministic-
  loading test logic.

### 2.3 Kaufman & Ni (2021) — tweezer-array review
- **Citation:** A. M. Kaufman, K.-K. Ni, "Quantum science with optical tweezer
  arrays of ultracold atoms and molecules," *Nat. Phys.* **17**, 1324–1333 (2021).
- **DOI:** 10.1038/s41567-021-01357-2
- **URL:** https://www.nature.com/articles/s41567-021-01357-2
- **Note:** No verified arXiv preprint ID found; cited by DOI (confirmed against
  Nature Physics and ADS 2021NatPh..17.1324K).
- **Covers:** Modern review of single-atom/-molecule optical-tweezer control:
  imaging, cooling to the motional ground state, rearrangement, coherence,
  and applications to quantum information, simulation and metrology.
- **Relevance to SNAQ-Node-3:** State-of-the-art context for the tweezer platform —
  imaging/recool, qubit coherence, and the path from a single node to arrays.

---

## 3. Atom–photon entanglement, remote entanglement & quantum networking

### 3.1 Blinov, Moehring, Duan & Monroe (2004) — atom–photon entanglement
- **Citation:** B. B. Blinov, D. L. Moehring, L.-M. Duan, C. Monroe, "Observation
  of entanglement between a single trapped atom and a single photon," *Nature*
  **428**, 153–157 (2004).
- **DOI:** 10.1038/nature02377
- **URL:** https://www.nature.com/articles/nature02377
- **Covers:** First observation of entanglement between an internal (spin) state
  of a single trapped ion and the polarization of a spontaneously emitted photon —
  the matter–light interface underlying photonic networking.
- **Relevance to SNAQ-Node-3:** The atom–photon entanglement primitive is the
  building block of a networking node; the spin–photon mapping is exactly what a
  Cs node would emit to entangle with a remote node.

### 3.2 Moehring et al. (2007) — remote entanglement of two atomic qubits
- **Citation:** D. L. Moehring, P. Maunz, S. Olmschenk, K. C. Younge,
  D. N. Matsukevich, L.-M. Duan, C. Monroe, "Entanglement of single-atom quantum
  bits at a distance," *Nature* **449**, 68–71 (2007).
- **DOI:** 10.1038/nature06118
- **URL:** https://www.nature.com/articles/nature06118
- **Covers:** Heralded entanglement between two trapped-ion qubits ~1 m apart via
  two-photon interference + coincidence detection (DLCZ-type herald). Demonstrates
  measurement-based remote entanglement between independent matter qubits.
- **Relevance to SNAQ-Node-3:** The canonical protocol for entangling two separated
  nodes — exactly the multi-node networking goal of SNAQ-Node-3. Defines the
  herald/coincidence logic an experiment node must implement.

### 3.3 Hofmann et al. (2012) — heralded entanglement of distant neutral atoms
- **Citation:** J. Hofmann, M. Krug, N. Ortegel, L. Gérard, M. Weber,
  W. Rosenfeld, H. Weinfurter, "Heralded Entanglement Between Widely Separated
  Atoms," *Science* **337**, 72–75 (2012).
- **DOI:** 10.1126/science.1221856
- **URL:** https://www.science.org/doi/10.1126/science.1221856
- **Author-hosted PDF:** https://xqp.physik.uni-muenchen.de/publications/files/articles_2012/science_337_72.pdf
- **Covers:** Heralded entanglement between the spins of two single **neutral**
  ⁸⁷Rb atoms 20 m apart, each in its own dipole trap, via atom–photon entanglement
  + photonic Bell-state measurement. Closest analog to a neutral-atom network node.
- **Relevance to SNAQ-Node-3:** Most directly on-point for a **neutral-atom**
  (single-atom-in-dipole-trap) networking node like Node-3: shows the full chain
  from single-atom spin readout to heralded distant entanglement.

### 3.4 Kimble (2008) — "The quantum internet" (vision/review)
- **Citation:** H. J. Kimble, "The quantum internet," *Nature* **453**, 1023–1030
  (2008).
- **DOI:** 10.1038/nature07127
- **arXiv:** 0806.4195
- **URL:** https://arxiv.org/abs/0806.4195
- **Covers:** Conceptual framework for quantum networks: nodes (matter qubits) +
  quantum channels (photons), quantum interconnects, entanglement distribution and
  state teleportation across a network.
- **Relevance to SNAQ-Node-3:** The architectural "why" of a node-based experiment —
  frames where SNAQ-Node-3 sits in a multi-node quantum-network picture.

---

## 4. Cesium-specific atomic data (Node-3 species)

### 4.1 Steck — *Cesium D Line Data*
- **Citation:** D. A. Steck, "Cesium D Line Data," available online at
  http://steck.us/alkalidata (revision 2.x.x; originally 23 Jan 1998).
- **Canonical URL:** https://steck.us/alkalidata/cesiumnumbers.pdf
- **Covers:** Authoritative compilation of ¹³³Cs physical/optical constants:
  D1 (6²S₁/₂→6²P₁/₂) and D2 (6²S₁/₂→6²P₃/₂) line data, hyperfine structure and
  the 9.192 631 770 GHz clock splitting, linewidths, saturation intensities,
  transition dipole moments, light-shift coefficients.
- **Relevance to SNAQ-Node-3:** The number source for Cs-133: hyperfine clock
  splitting behind the microwave 00 transition and the 2-photon Raman pathway,
  Γ and I_sat for cooling/imaging, and dipole moments for FORT light-shift /
  trap-depth calculations. Frequently the reference to settle any Cs constant.

---

## Modern / recent (trapping & cooling)

Recent, widely-cited work (2012–2020) that is standard reading in the current
single-atom / optical-tweezer / sub-Doppler-cooling community. Added 2026-06-08;
each entry located and cross-checked via web search/fetch against publisher pages
(Science/AAAS, Nature Physics, APS) and arXiv. **DOIs and arXiv IDs below were
confirmed present in those records — none invented.** Entries marked
**[arXiv ✓]** have a verified open-access preprint that can be downloaded.

### M.1 Endres et al. (2016) — atom-by-atom assembly of defect-free 1D arrays  **[arXiv ✓]**
- **Citation:** M. Endres, H. Bernien, A. Keesling, H. Levine, E. R. Anschuetz,
  A. Krajenbrink, C. Senko, V. Vuletić, M. Greiner, M. D. Lukin, "Atom-by-atom
  assembly of defect-free one-dimensional cold atom arrays," *Science* **354**,
  1024–1027 (2016).
- **DOI:** 10.1126/science.aah3752
- **arXiv:** 1607.03044
- **URL:** https://www.science.org/doi/10.1126/science.aah3752 ·
  https://arxiv.org/abs/1607.03044
- **Covers:** Real-time imaging + feedback rearrangement of ~100 optical tweezers
  to deterministically build defect-free 1D arrays of >50 single atoms in <400 ms,
  replacing lost atoms from a reservoir. Establishes the measure-and-rearrange
  paradigm for filling stochastically loaded tweezers.
- **Relevance to SNAQ-Node-3:** The canonical method for turning ~50% collisional-
  blockade loading (Schlosser 2.2) into deterministic single-atom occupancy — the
  scaling path from one Cs tweezer node to a filled array.

### M.2 Barredo et al. (2016) — atom-by-atom assembler of arbitrary 2D arrays  **[arXiv ✓]**
- **Citation:** D. Barredo, S. de Léséleuc, V. Lienhard, T. Lahaye, A. Browaeys,
  "An atom-by-atom assembler of defect-free arbitrary two-dimensional atomic
  arrays," *Science* **354**, 1021–1023 (2016).
- **DOI:** 10.1126/science.aah3778
- **arXiv:** 1607.03042
- **URL:** https://www.science.org/doi/10.1126/science.aah3778 ·
  https://arxiv.org/abs/1607.03042
- **Covers:** A moving optical tweezer rearranges atoms from a half-filled,
  stochastically loaded array into arbitrary 2D target geometries of up to ~50
  single atoms with filling fractions up to ~98%. The 2D companion to Endres
  M.1, published back-to-back.
- **Relevance to SNAQ-Node-3:** Same defect-free-assembly motivation as M.1, in
  arbitrary 2D geometry — the reference for scaling a single-node tweezer to a
  programmable 2D register.

### M.3 Kaufman, Lester & Regal (2012) — single-atom ground-state cooling in a tweezer  **[arXiv ✓]**
- **Citation:** A. M. Kaufman, B. J. Lester, C. A. Regal, "Cooling a Single Atom
  in an Optical Tweezer to Its Quantum Ground State," *Phys. Rev. X* **2**, 041014
  (2012).
- **DOI:** 10.1103/PhysRevX.2.041014
- **arXiv:** 1209.2087
- **URL:** https://link.aps.org/doi/10.1103/PhysRevX.2.041014 ·
  https://arxiv.org/abs/1209.2087
- **Covers:** Raman sideband cooling of a single neutral atom (⁸⁷Rb) in a tightly
  focused optical tweezer to ~90% 3D motional-ground-state occupation, with
  sideband spectroscopy as thermometry. Shows a bare tweezer gives enough
  confinement (resolved sidebands) for efficient sideband cooling.
- **Relevance to SNAQ-Node-3:** The reference technique for getting a single Cs
  atom near the motional ground state in the FORT — sets expectations for
  trap-frequency / sideband-resolution requirements and the recool needed to keep
  motional coherence high before Ramsey/Rabi sequences.

### M.4 Grier et al. (2013) — Λ-enhanced D1 gray molasses (sub-Doppler cooling)  **[arXiv ✓]**
- **Citation:** A. T. Grier, I. Ferrier-Barbut, B. S. Rem, M. Delehaye,
  L. Khaykovich, F. Chevy, C. Salomon, "Λ-enhanced sub-Doppler cooling of lithium
  atoms in D1 gray molasses," *Phys. Rev. A* **87**, 063411 (2013).
- **DOI:** 10.1103/PhysRevA.87.063411
- **arXiv:** 1304.6971
- **URL:** https://link.aps.org/doi/10.1103/PhysRevA.87.063411 ·
  https://arxiv.org/abs/1304.6971
- **Covers:** Bichromatic blue-detuned D1 gray-molasses cooling, with an extra
  threefold temperature drop when the two-photon Raman condition between hyperfine
  ground states is met (long-lived Λ-system dark-state coherence). Reaches ~60 µK
  while keeping all atoms; solves the OBEs for a Λ three-level model to identify
  the cooling force near Raman resonance.
- **Relevance to SNAQ-Node-3:** The modern theory/method of Λ-enhanced (dark-state)
  sub-Doppler cooling — the same blue-detuned-D1 / two-photon-Raman dark-state
  physics that underlies advanced alkali molasses recool; useful context when
  reasoning about Cs D1-line cooling stages beyond plain lin⊥lin PGC (Dalibard 1.2).

### M.5 Browaeys & Lahaye (2020) — review: many-body physics with single Rydberg atoms  **[arXiv ✓]**
- **Citation:** A. Browaeys, T. Lahaye, "Many-body physics with individually
  controlled Rydberg atoms," *Nat. Phys.* **16**, 132–142 (2020).
- **DOI:** 10.1038/s41567-019-0733-z
- **arXiv:** 2002.07413
- **URL:** https://www.nature.com/articles/s41567-019-0733-z ·
  https://arxiv.org/abs/2002.07413
- **Covers:** Review of the optical-tweezer-array + Rydberg platform: single-atom
  trapping and rearrangement, mapping of Rydberg interactions onto quantum spin
  models, and many-body quantum-simulation results. Companion-in-spirit to the
  Kaufman & Ni tweezer review (2.3), with the interaction/quantum-simulation focus.
- **Relevance to SNAQ-Node-3:** State-of-the-art context for where a single-atom
  tweezer node sits in the broader neutral-atom array program — the bridge from
  one controlled atom to interacting, individually addressed arrays.

### M.6 Cooper et al. (2018) — alkaline-earth atoms in tweezers (narrow-line cooling/imaging)  **[arXiv ✓]**
- **Citation:** A. Cooper, J. P. Covey, I. S. Madjarov, S. G. Porsev,
  M. S. Safronova, M. Endres, "Alkaline-Earth Atoms in Optical Tweezers,"
  *Phys. Rev. X* **8**, 041055 (2018).
- **DOI:** 10.1103/PhysRevX.8.041055
- **arXiv:** 1810.06537
- **URL:** https://link.aps.org/doi/10.1103/PhysRevX.8.041055 ·
  https://arxiv.org/abs/1810.06537
- **Covers:** Single-shot imaging and narrow-line (intercombination-line) cooling
  of individual ⁸⁸Sr atoms in 515.2 nm tweezers, including a narrow-line Sisyphus
  cooling mechanism that cools during imaging, extended to uniform 121-site 2D
  arrays. Foundational for magic-wavelength tweezer arrays and tweezer-clock work.
- **Relevance to SNAQ-Node-3:** Reference for simultaneous cool-while-image and
  magic/clock-magic trapping strategies — the conceptual model for why a recool
  scheme that runs concurrently with (or immediately after) imaging preserves
  fidelity and reduces heating before coherent operations.

### M.7 Saskin et al. (2019) — narrow-line cooling & imaging of Yb in a tweezer array  **[arXiv ✓]**
- **Citation:** S. Saskin, J. T. Wilson, B. Grinkemeyer, J. D. Thompson,
  "Narrow-Line Cooling and Imaging of Ytterbium Atoms in an Optical Tweezer
  Array," *Phys. Rev. Lett.* **122**, 143002 (2019).
- **DOI:** 10.1103/PhysRevLett.122.143002
- **arXiv:** 1810.10517
- **URL:** https://link.aps.org/doi/10.1103/PhysRevLett.122.143002 ·
  https://arxiv.org/abs/1810.10517
- **Covers:** Trapping single ¹⁷¹/¹⁷⁴Yb atoms in 532 nm magic-wavelength tweezers,
  using the narrow ¹S₀–³P₁ intercombination line for both cooling and imaging; the
  low Doppler temperature allows near-saturation imaging with very high single-atom
  detection fidelity. Establishes Yb (alkaline-earth-like) as a tweezer-array
  qubit/clock platform.
- **Relevance to SNAQ-Node-3:** Companion to M.6 for a second species — reinforces
  the narrow-line cool-and-image paradigm and high-fidelity single-atom detection
  that any single-atom node (including Cs) benchmarks against.

---

## Verification status
The original 11 entries (sections 1–4) were located and cross-checked via web
search/fetch against publisher pages (APS, Optica, Springer, Nature, Science,
arXiv, ADS) on 2026-06-08. DOIs and arXiv IDs shown were confirmed present in
those records. No DOI/arXiv ID was invented; where an arXiv preprint could not be
verified (Kaufman & Ni 2021) the entry is cited by confirmed DOI only and that is
noted inline.

The 7 **Modern / recent (trapping & cooling)** entries (M.1–M.7) were added
2026-06-08, each verified the same way against publisher pages and arXiv; all 7
have a confirmed arXiv preprint (marked **[arXiv ✓]**) in addition to a confirmed
DOI.
