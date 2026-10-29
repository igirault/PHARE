# VTKHDF writer: one write per rank vs one write per patch

Branch `bench/vtkhdf-io` is [PR #1505](https://github.com/PHAREHUB/PHARE/pull/1505) at `56a74b2a0` plus one commit, `8ecb4edf1`, that changes how each rank writes its field data:

- PR #1505 (`pr`): each rank converts its patches of a level into one buffer, components interleaved, and writes it with a single `H5Dwrite` (`VTKFileWriter::writeLevel`).
- this branch (`pp`): each patch is converted into a one-patch buffer and written straight away (`VTKFileWriter::writePatch`), through a dataset handle opened once per level.

The file layout and content are the same in both, so the two builds write bit-identical files and only the time differs. Since `56a74b2a0`, the PR gained `27f2f96e2` and `0e0ecefaa`, which drop a documentation note and respell the chunk default; neither changes what is written or how.

This directory holds the case and the job scripts that produced the results below. A standalone benchmark of the same write patterns, without PHARE, is at https://gist.github.com/igirault/77043e3e85105b0e3cbd7e5e243d244e.

## The case

`vtkhdf_dump.py` sets up a 3D MHD run (Linear reconstruction, VanLeer limiter, Rusanov, TVDRK2, one level) with the diagnostics B, rho, rhoV, Etot and V in VTKHDF format. It initialises the simulation, then triggers `E2E_DUMPS` dumps directly with `dump_diagnostics`, without advancing the solver, and times each dump between two MPI barriers. Its inputs are environment variables:

| Variable | Meaning |
|---|---|
| `E2E_CELLS` | cells per direction, e.g. `384,384,384` |
| `E2E_LARGEST` | `largest_patch_size`; it sets the number of patches per rank |
| `E2E_DUMPS` | number of dumps |
| `E2E_DIR` | diagnostics directory |

## Reproduce on Adastra

The scripts assume Adastra GENOA nodes (192 cores) with CCE and `cray-hdf5-parallel`, and `$SCRATCHDIR` for the output. Submit them from this directory and pass your project with `-A`.

1. Two checkouts, one per build:

   ```bash
   git clone -b bench/vtkhdf-io https://github.com/igirault/PHARE.git PHARE-pp
   git clone https://github.com/igirault/PHARE.git PHARE-pr && git -C PHARE-pr checkout 56a74b2a0
   ```

   CMake clones SAMRAI, HighFive, pybind11 and cppdict into `subprojects/` at configure time. Adastra has no outbound internet access, so clone on a machine that has it, run CMake there once to fetch `subprojects/`, and copy both checkouts to Adastra (for example with `rsync`).

2. A Python virtual environment with PHARE's `requirements.txt` and `h5py`, given as `PHARE_VENV`.

3. Build both, with identical flags: Release, `-march=native`, `PHARE_DIAG_DOUBLES=1`, only the permutation in `permutations.txt`.

   ```bash
   cd PHARE-pp/tools/bench/vtkhdf
   export PHARE_VENV=<venv> PHARE_PR=<path>/PHARE-pr PHARE_PP=<path>/PHARE-pp
   BUILD_SRCS="$PHARE_PR $PHARE_PP" sbatch -A <account> -J build build.sbatch
   ```

4. Run the two passes, the second in reverse order so that drift on the shared file system does not favour one build, then compare the kept output:

   ```bash
   A=$(E2E_CELLS=384,384,384 E2E_DUMPS=6 STEP_MIN=10 sbatch --parsable -A <account> -N 8 -t 00:59:00 -J ppcmp_a \
       compare.sbatch pr:24 pp:24 pr:12 pp:12 pr:8 pp:8 pr:6 pp:6)
   B=$(E2E_CELLS=384,384,384 E2E_DUMPS=6 STEP_MIN=10 sbatch --parsable -A <account> -N 8 -t 00:59:00 -J ppcmp_b \
       --dependency=afterany:$A \
       compare.sbatch pp:6 pr:6 pp:8 pr:8 pp:12 pr:12 pp:24 pr:24 pr:6:keep:2 pp:6:keep:2)
   sbatch -A <account> -t 00:30:00 -J diff_pp --dependency=afterany:$B \
       diff_pair.sbatch $SCRATCHDIR/vtkhdf_bench/$B/pr_p6 $SCRATCHDIR/vtkhdf_bench/$B/pp_p6
   ```

   Each `compare.sbatch` argument is `build:largest[:keep[:dumps]]`. For each one the script stripes a fresh output directory with `lfs setstripe -c $STRIPE -S 1M` (default 2 stripes), runs `vtkhdf_dump.py` on all ranks of the job, reads the number of patches back from the file, and deletes the output unless `keep` is given. `diff_pair.sbatch` runs `h5diff` on each pair of files, then deletes both directories.

## Results

Adastra, jobs 5776164 (build) to 5776167 (h5diff), 29 October 2026. 1536 ranks (8 nodes), 384³ cells, 5 files in double precision, 2 stripes. Times are the mean of dumps 2–6, averaged over the two passes:

| Largest patch | Patches per rank | Data per dump | Writes per rank per dump (pr / pp) | `pr` | `pp` | Difference |
|---|---|---|---|---|---|---|
| 24³ | 4 | 5.5 GiB | 5 / 20 | 2.72 s | 2.98 s | +0.26 s (+10%) |
| 12³ | 27 | 6.2 GiB | 5 / 133 | 2.84 s | 3.31 s | +0.47 s (+17%) |
| 8³ | 72 | 6.7 GiB | 5 / 360 | 3.18 s | 3.44 s | +0.25 s (+8%) |
| 6³ | 192 | 7.7 GiB | 5 / 960 | 3.18 s | 4.08 s | +0.89 s (+28%) |

- `pp` was slower than `pr` in all 8 pairs run in the same pass, by 0.04 to 1.19 s.
- Repeat runs of the same build and case differed by up to 0.7 s on the shared scratch, so the two middle rows are only roughly measured. At 192 patches per rank, both `pp` runs (4.04 and 4.11 s) were slower than both `pr` runs (2.85 and 3.52 s).
- The data per dump grows with the patch count because nodes on patch boundaries are written by each patch that holds them.
- At 192 patches per rank, the files of the two builds are bit-identical (`h5diff`, 5 files, 2 dumps).

The raw job output is in `results/`, with the account-specific paths replaced by `$SCRATCHDIR`.
