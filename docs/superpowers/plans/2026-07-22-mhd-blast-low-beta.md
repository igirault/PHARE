# Low-β Magnetized Blast Evidence Case — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a magnetized blast test case (Wu & Shu 2018, Ex. 4.4) that demonstrates PHARE ideal-MHD failing at low plasma-β via negative-pressure crash.

**Architecture:** One shared module `blast.py` (`config(label)` + `run_case(label)`) plus three thin per-case runner scripts, each launched as its own MPI test. Separate processes because a low-β crash aborts the whole MPI run and must not block the other cases. Uniform single-level grid, high-order solver (SSPRK4_5 + WENOZ, no limiter), `pharevtkhdf` output for ParaView.

**Tech Stack:** Python (`pyphare.pharein` DSL), MPI via `phare_mpi_python3_exec`, existing MHD C++ solver. A new solver permutation is added → one `cpp_*` module rebuild.

## Global Constraints

- Unit convention: Heaviside-Lorentz, μ₀=1, magnetic pressure `B²/2`, `β = 2p/B²`. Field expressed natively: `Ba = np.sqrt(2*pa/beta)`. No `√(4π)` in code.
- Fixed physical params for all cases: `pa = 0.1`, `rho = 1.0`, `gamma = 1.4`, circle radius `r0 = 0.1`, domain `[0,1]²` with circle centered at `(0.5, 0.5)`, `v = 0`, `B = (Ba, 0, 0)`.
- Grid: `cells = (320, 320)`, `max_nbr_levels = 1`, `max_mhd_level = 1`, no `refinement` kwarg (uniform single level).
- Solver: `reconstruction="WENOZ"`, `limiter="None"` (string), `riemann="Rusanov"`, `mhd_timestepper="SSPRK4_5"`, `hall=False`, `res=False`, `hyper_res=False`, `eta=0.0`, `nu=0.0`, `resistivity=0.0`, `hyper_resistivity=0.0`, `model_options=["MHDModel"]`.
- Required permutation (2D, interp 1, refined 4): `2,1,4,SSPRK4_5,WENOZ,None,Rusanov,false,false,false` in `res/sim/all.txt`.
- Diagnostics format: `pharevtkhdf`. Quantities `B`, `rho`, `V`, `P`.
- Cases (`pa=0.1`): reference `pe=1e3, beta=2.5, tmax=0.01, dt=2e-5`; blast1 `pe=1e3, beta=2.51e-4, tmax=0.01, dt=2e-6`; blast2 `pe=1e4, beta=2.51e-6, tmax=0.001, dt=2e-7`.
- MPI procs per test: **10**.
- Test exec level: 101 (heavy; excluded from default CI — `blast1`/`blast2` are *expected* to crash and must not gate CI).
- Build: use `tools/cmake.sh` (not `uv run cmake --build`). Adding a permutation triggers a `cpp_*` module build; keep `-j` modest (≤8) to avoid OOM.
- `docs/` is git-excluded locally → commit plan/spec with `git add -f`. All other paths: stage explicitly, never `git add -A`.

---

### Task 1: Add the SSPRK4_5/WENOZ 2D permutation and rebuild

**Files:**
- Modify: `res/sim/all.txt` (add one permutation line in the 2D section)

**Interfaces:**
- Produces: a built `cpp_*` module supporting `(dim=2, interp=1, refined=4, SSPRK4_5, WENOZ, None, Rusanov, no Hall/res/hyper)`, selected at runtime by `simulator_id`.

- [ ] **Step 1: Add the permutation line**

In `res/sim/all.txt`, in the 2D functional-test block, add (next to the existing `2,1,4,TVDRK2,Linear,VanLeer,Rusanov,false,false,false` line):
```
2,1,4,SSPRK4_5,WENOZ,None,Rusanov,false,false,false
```

- [ ] **Step 2: Reconfigure and build the new module**

Run:
```bash
uv run cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DdevMode=ON >/dev/null
tools/cmake.sh -j 8
```
Expected: build completes; a new `pybindlibs/cpp_*` for the added permutation is produced (`ls build/**/cpp_*.so` grows by one vs before).

- [ ] **Step 3: Commit**

```bash
git add res/sim/all.txt
git commit -m "build(mhd): add 2D SSPRK4_5/WENOZ ideal permutation for blast case"
```

---

### Task 2: Shared blast module (`blast.py`) + reference case runs clean

**Files:**
- Create: `tests/functional/mhd_blast/blast.py`
- Create: `tests/functional/mhd_blast/blast_reference.py`

**Interfaces:**
- Produces: `config(label: str) -> ph.Simulation`, `run_case(label: str) -> None`, module dict `CASES: dict[str, dict]` with keys `pe, pa, beta, tmax, dt`.

- [ ] **Step 1: Write `blast.py`**

```python
#!/usr/bin/env python3
import os

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.simulator.simulator import Simulator, startMPI

os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()

# Wu & Shu 2018 (DOI 10.1137/18M1168042), Example 4.4 (Balsara & Spicer 1999).
# PHARE is Heaviside-Lorentz (mu0=1): magnetic pressure = B**2/2, beta = 2 p / B**2.
# The paper's Ba = 100/sqrt(4 pi) is only its CGS->normalized rescaling; expressed
# natively here as Ba = sqrt(2 pa / beta), which reproduces the paper's beta exactly.
CASES = {
    "reference": dict(pe=1e3, pa=0.1, beta=2.5, tmax=0.01, dt=2.0e-5),
    "blast1": dict(pe=1e3, pa=0.1, beta=2.51e-4, tmax=0.01, dt=2.0e-6),
    "blast2": dict(pe=1e4, pa=0.1, beta=2.51e-6, tmax=0.001, dt=2.0e-7),
}

RHO = 1.0
GAMMA = 1.4
R0 = 0.1
XC, YC = 0.5, 0.5


def config(label):
    case = CASES[label]
    pe, pa, beta = case["pe"], case["pa"], case["beta"]
    tmax, dt = case["tmax"], case["dt"]

    Ba = np.sqrt(2.0 * pa / beta)  # native field from target beta

    diag_dir = f"phare_outputs/blast/{label}"
    n_out = 5
    timestamps = np.arange(0, tmax + dt, tmax / n_out)

    cells = (320, 320)
    dl = (1.0 / cells[0], 1.0 / cells[1])

    sim = ph.Simulation(
        time_step=dt,
        final_time=tmax,
        cells=cells,
        dl=dl,
        max_mhd_level=1,
        max_nbr_levels=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        nesting_buffer=1,
        eta=0.0,
        nu=0.0,
        gamma=GAMMA,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        hall=False,
        res=False,
        hyper_res=False,
        model_options=["MHDModel"],
    )

    def r(x, y):
        return np.sqrt((x - XC) ** 2 + (y - YC) ** 2)

    def density(x, y):
        return RHO + 0.0 * x

    def vx(x, y):
        return 0.0 * x

    def vy(x, y):
        return 0.0 * x

    def vz(x, y):
        return 0.0 * x

    def bx(x, y):
        return Ba + 0.0 * x

    def by(x, y):
        return 0.0 * x

    def bz(x, y):
        return 0.0 * x

    def p(x, y):
        return np.where(r(x, y) < R0, pe, pa)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


def run_case(label):
    ph.global_vars.sim = None
    Ba = np.sqrt(2.0 * CASES[label]["pa"] / CASES[label]["beta"])
    if cpp.mpi_rank() == 0:
        print(f"[blast] case={label} beta={CASES[label]['beta']:.3e} Ba={Ba:.4f}", flush=True)
    sim = config(label)
    Simulator(sim).run().reset()
    ph.global_vars.sim = None
    if cpp.mpi_rank() == 0:
        print(f"[blast] case={label} completed to t={CASES[label]['tmax']}", flush=True)
```

- [ ] **Step 2: Write `blast_reference.py`**

```python
#!/usr/bin/env python3
from pyphare.simulator.simulator import startMPI

from tests.functional.mhd_blast.blast import run_case

if __name__ == "__main__":
    startMPI()
    run_case("reference")
```

- [ ] **Step 3: Run the reference case and verify it completes**

Run (from repo root):
```bash
pkill -x mpirun || true
phare-mpi -n 10 python tests/functional/mhd_blast/blast_reference.py
```
Expected: prints `case=reference beta=2.500e+00 Ba=0.2828`, runs to `t=0.01`, prints `completed to t=0.01`, no exception. Dumps under `phare_outputs/blast/reference/*.vtkhdf`.

- [ ] **Step 4: Commit**

```bash
git add tests/functional/mhd_blast/blast.py tests/functional/mhd_blast/blast_reference.py
git commit -m "test(mhd): blast module + reference (moderate-beta) case"
```

---

### Task 3: Low-β cases (`blast1`, `blast2`) — verify negative-pressure crash

**Files:**
- Create: `tests/functional/mhd_blast/blast_blast1.py`
- Create: `tests/functional/mhd_blast/blast_blast2.py`

**Interfaces:**
- Consumes: `run_case` from `blast.py` (Task 2).

- [ ] **Step 1: Write `blast_blast1.py`**

```python
#!/usr/bin/env python3
from pyphare.simulator.simulator import startMPI

from tests.functional.mhd_blast.blast import run_case

if __name__ == "__main__":
    startMPI()
    run_case("blast1")
```

- [ ] **Step 2: Write `blast_blast2.py`**

```python
#!/usr/bin/env python3
from pyphare.simulator.simulator import startMPI

from tests.functional.mhd_blast.blast import run_case

if __name__ == "__main__":
    startMPI()
    run_case("blast2")
```

- [ ] **Step 3: Run blast1 and confirm it crashes on negative pressure**

Run:
```bash
pkill -x mpirun || true
phare-mpi -n 10 python tests/functional/mhd_blast/blast_blast1.py 2>&1 | tee /tmp/blast1.log
pkill -x mpirun || true
```
Expected: prints `case=blast1 beta=2.510e-04 Ba=28.2094`, then aborts before `t=0.01` with a negative-pressure / NaN error (emergency dump under `phare_outputs/blast/blast1/`). Record the reported crash time from the log. Does NOT print `completed`.

- [ ] **Step 4: Run blast2 and confirm earlier crash**

Run:
```bash
pkill -x mpirun || true
phare-mpi -n 10 python tests/functional/mhd_blast/blast_blast2.py 2>&1 | tee /tmp/blast2.log
pkill -x mpirun || true
```
Expected: prints `case=blast2 beta=2.510e-06 Ba=282.0947`, aborts earlier than blast1 (paper: t≈1.2e-5). Record crash time. Does NOT print `completed`.

- [ ] **Step 5: Commit**

```bash
git add tests/functional/mhd_blast/blast_blast1.py tests/functional/mhd_blast/blast_blast2.py
git commit -m "test(mhd): low-beta blast1/blast2 crash cases"
```

---

### Task 4: Evidence README

**Files:**
- Create: `tests/functional/mhd_blast/README.md`

**Interfaces:**
- Consumes: crash times observed in Task 3 (`/tmp/blast1.log`, `/tmp/blast2.log`).

- [ ] **Step 1: Write `README.md` with the observed-outcomes table**

Fill `<...>` from the Task 3 logs. Template:

```markdown
# Magnetized blast — low-β failure evidence

Reproduces Wu & Shu 2018 (DOI 10.1137/18M1168042), Example 4.4 (Balsara & Spicer 1999),
to show PHARE ideal-MHD produces negative pressure at low plasma-β (no positivity-
preserving limiter, no pressure floor).

Setup: domain [0,1]², ρ=1, γ=1.4, uniform Bx=Ba, pressure pe inside r<0.1 (center
(0.5,0.5)), pa=0.1 outside, v=0. PHARE units: β = 2 pa / Ba². 320×320, single level,
WENOZ reconstruction (no limiter) + SSPRK4_5, Rusanov, 10 MPI ranks.

| case      | pe   | β        | Ba       | tmax  | outcome           | crash time |
|-----------|------|----------|----------|-------|-------------------|------------|
| reference | 1e3  | 2.5      | 0.2828   | 0.01  | completed         | —          |
| blast1    | 1e3  | 2.51e-4  | 28.21    | 0.01  | negative pressure | <blast1>   |
| blast2    | 1e4  | 2.51e-6  | 282.1    | 0.001 | negative pressure | <blast2>   |

Run:
    phare-mpi -n 10 python tests/functional/mhd_blast/blast_reference.py
    phare-mpi -n 10 python tests/functional/mhd_blast/blast_blast1.py
    phare-mpi -n 10 python tests/functional/mhd_blast/blast_blast2.py

Output: phare_outputs/blast/<case>/*.vtkhdf (view in ParaView).

Conclusion: as β decreases (magnetic energy ≫ internal energy), the total-energy
update yields P<0 → NaN sound speed → crash. Lower β crashes earlier, isolating β
as the cause. Reference (β~2.5) is stable.
```

- [ ] **Step 2: Commit**

```bash
git add tests/functional/mhd_blast/README.md
git commit -m "docs(mhd): blast low-beta evidence README with observed outcomes"
```

---

### Task 5: CTest registration

**Files:**
- Create: `tests/functional/mhd_blast/CMakeLists.txt`
- Modify: `res/cmake/test.cmake` (add subdirectory, alphabetical before `mhd_harris` at line 73)

**Interfaces:**
- Consumes: `blast_reference.py`, `blast_blast1.py`, `blast_blast2.py` (Tasks 2–3).

- [ ] **Step 1: Write `tests/functional/mhd_blast/CMakeLists.txt`**

```cmake
cmake_minimum_required (VERSION 3.20.1)

project(test-mhd-blast)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)

  ## These tests use dump diagnostics so require HighFive!
  ## Heavy level 101: blast1/blast2 are EXPECTED to crash (negative pressure at
  ## low plasma-beta) and must not gate default CI.
  phare_mpi_python3_exec(101 10 test-mhd-blast-reference blast_reference.py ${CMAKE_CURRENT_BINARY_DIR})
  phare_mpi_python3_exec(101 10 test-mhd-blast-blast1 blast_blast1.py ${CMAKE_CURRENT_BINARY_DIR})
  phare_mpi_python3_exec(101 10 test-mhd-blast-blast2 blast_blast2.py ${CMAKE_CURRENT_BINARY_DIR})
endif()
```

- [ ] **Step 2: Add subdirectory in `res/cmake/test.cmake`**

Insert immediately before the `add_subdirectory(tests/functional/mhd_harris)` line (currently line 73):
```cmake
  add_subdirectory(tests/functional/mhd_blast)
```

- [ ] **Step 3: Reconfigure and confirm the tests register at high exec level**

Run:
```bash
uv run cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DdevMode=ON -DPHARE_EXEC_LEVEL_MAX=101 >/dev/null
(cd build && uv run ctest -N | grep mhd-blast)
```
Expected: three tests — `test-mhd-blast-reference`, `test-mhd-blast-blast1`, `test-mhd-blast-blast2`.

- [ ] **Step 4: Confirm they are excluded at the default exec level**

Run:
```bash
uv run cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DdevMode=ON >/dev/null
(cd build && uv run ctest -N | grep -c mhd-blast || true)
```
Expected: `0` (default `PHARE_EXEC_LEVEL_MAX=10` excludes level-101 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/functional/mhd_blast/CMakeLists.txt res/cmake/test.cmake
git commit -m "test(mhd): register blast low-beta cases at heavy exec level"
```

---

## Self-Review

**Spec coverage:**
- Reference + failing + extreme configs → Tasks 2, 3. ✓
- PHARE-native `Ba=√(2pa/β)`, no 4π → `blast.py` `config()`. ✓
- Uniform single level → Global Constraints + `config()`. ✓
- SSPRK4_5 / WENOZ / limiter None / nprocs 10 (user directive) → Global Constraints, permutation Task 1, `config()`, runners. ✓
- `pharevtkhdf` dumps, no matplotlib plotting → `config()` diag_options; README → ParaView. ✓
- Failure detection = crash + crash time → Task 3 steps 3–4, README table. ✓
- Heavy exec level, excluded from CI → Task 5. ✓
- MPI runs → separate runner per case, `phare_mpi_python3_exec` 10 procs. ✓

**Placeholder scan:** README `<blast1>`/`<blast2>` are intentional fill-from-observation slots (crash times unknowable until run), not plan placeholders. All code steps show full code. ✓

**Type consistency:** `config(label)`, `run_case(label)`, `CASES` keys `pe/pa/beta/tmax/dt` consistent across `blast.py` and all runners. ✓

## Open risks (verify during execution)

1. **Crash vs silent NaN:** whether negative pressure surfaces as a clean abort/emergency-dump or a silent NaN that runs to `tmax`. If `blast1` prints `completed`, inspect the final `P` dump in ParaView for NaN/negative values; if needed lower β further. Evidence claim then shifts from "crash" to "produces unphysical negative pressure" — still a low-β failure.
2. **Build OOM:** adding the permutation builds a `cpp_*` module; if `tools/cmake.sh -j 8` OOMs, drop to `-j 6`.
3. **New permutation is high-order:** WENOZ+SSPRK4_5 may itself be more or less robust than a 2nd-order scheme at low β; the reference case at β~2.5 confirms the scheme is otherwise stable, so a blast1/blast2 crash is attributable to β, not the discretization order.
