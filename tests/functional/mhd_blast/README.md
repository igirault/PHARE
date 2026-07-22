# Magnetized blast — low-β failure evidence

Reproduces Wu & Shu 2018 (DOI 10.1137/18M1168042), Example 4.4 (originally Balsara
& Spicer 1999), to show PHARE ideal-MHD produces negative pressure at low plasma-β.
PHARE has no positivity-preserving limiter and no pressure floor
(`to_primitive_converter.hpp` computes `p = (γ-1)(etot - 0.5ρv² - 0.5b²)` with no
clamp), so once the total-energy update drives the reconstructed thermal pressure
negative the sound speed `sqrt(γp/ρ)` becomes NaN and the run aborts.

## Setup

Domain `[0,1]²`, ρ=1, γ=1.4, uniform `Bx=Ba`, pressure `pe` inside `r<0.1`
(center `(0.5,0.5)`), `pa=0.1` outside, `v=0`. PHARE units (Heaviside-Lorentz,
μ₀=1): magnetic pressure `B²/2`, so `β = 2·pa/Ba²` and `Ba = sqrt(2·pa/β)`.
320×320, single uniform level (`max_mhd_level=1`), WENOZ reconstruction (no
limiter) + SSPRK4_5 time integration, Rusanov flux, 10 MPI ranks.

## Observed outcomes

| case      | pe   | β        | Ba       | tmax  | outcome                       | crash time |
|-----------|------|----------|----------|-------|-------------------------------|------------|
| reference | 1e3  | 2.5      | 0.2828   | 0.01  | completed (stable)            | —          |
| blast1    | 1e3  | 2.51e-4  | 28.2279  | 0.01  | NaN / negative pressure       | t = 1.2e-5 |
| blast2    | 1e4  | 2.51e-6  | 282.2787 | 0.001 | NaN / negative pressure       | t = 4e-7   |

Both low-β runs abort at level 0 with
`NaN detected in MHD field ... at time <t>` (`solver_mhd.hpp:394`), surfaced as a
caught `SolverMHD::advanceLevel` exception; neither prints `completed`. The
reference case (β≈2.5) runs cleanly to `t=0.01`.

For reference, the paper's limiter-free 3rd-order DG fails at t≈2.85e-4 (blast1)
and t≈1.2e-5 (blast2); PHARE fails ~20–30× earlier under the same initial
conditions.

## Run

    phare-mpi -n 10 python tests/functional/mhd_blast/blast_reference.py
    phare-mpi -n 10 python tests/functional/mhd_blast/blast_blast1.py
    phare-mpi -n 10 python tests/functional/mhd_blast/blast_blast2.py

Output: `phare_outputs/blast/<case>/*.vtkhdf` (view in ParaView).

## Conclusion

As β decreases (magnetic energy ≫ internal energy), the ideal-MHD total-energy
update yields p<0 → NaN → crash, and lower β crashes earlier (blast2 ≪ blast1),
isolating β as the cause. The moderate-β reference is stable. PHARE's ideal-MHD
solver cannot handle low plasma-β without a positivity-preserving mechanism.
