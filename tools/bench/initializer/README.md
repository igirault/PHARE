# Benchmark: filling a field from a user-specified function

How much does it cost to fill a field on a grid from a **Python** user function, compared to
evaluating the same function in **plain C++**, and how much of that cost is removable?

This matters wherever a field is (re-)stamped from a user function more than once — in particular a
time-dependent external field `B0(x, t)` re-evaluated every timestep, but also plain initialization
of every field / particle-initializer profile on every patch.

The benchmark is standalone: it builds without configuring PHARE (header-only core only), embeds
CPython, and reproduces `pyphare`'s wrapper layer verbatim.

## Build & run

```bash
source ~/.zshrc.phare        # any env with a python that has numpy
cmake -S tools/bench/initializer -B /tmp/bench_field_init -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/bench_field_init -j
/tmp/bench_field_init/bench_field_init --benchmark_min_time=0.2s
```

`googlebench`, `pybind11` and `cppdict` are taken from `<phare>/subprojects` when they are there and
cloned into the build tree otherwise; use `-DPHARE_SUBPROJECTS=/path/to/subprojects` to point
elsewhere (needed when building from a worktree that has no `subprojects/`).

The binary first fills the field with every variant and compares checksums, so a variant that
computes the wrong thing fails loudly instead of benchmarking fast garbage.

Whatever else is printed, a pivoted summary is written at the end — one row per variant, one column
per grid size, each cell the time and its signed difference from the plain C++ reference:

```
  2d / trivial
  cells                       64               128               256               512
  nodes                     4.7k             17.6k             67.9k            266.8k
  --------------------------------------------------------------------------------------------
  cpp_direct                1.52              5.65              22.1              93.6
  cpp_via_vectors           10.1   +563%      38.5   +581%     158.3   +616%      1554  +1561%
  py_now                   194.5 +12678%     731.7 +12853%      3176 +14263%     16514 +17547%
  py_np                     11.9   +679%      42.6   +655%     165.5   +648%      1607  +1617%
  py_np_cached              3.38   +122%      8.92    +58%      31.4    +42%     123.3    +32%
  py_np_cached_flat         2.13    +40%      4.34    -23%      15.3    -31%      60.9    -35%
  py_np_out                 1.25    -18%      2.53    -55%      7.58    -66%      27.8    -70%
```

Pass `--summary-only` to drop googlebench's own per-case lines and print just that. It composes with
the usual flags:

```bash
bench_field_init --summary-only --benchmark_min_time=0.2s       # quick pass
bench_field_init --summary-only --benchmark_filter='2d/trivial' # one group
bench_field_init --summary-only --benchmark_repetitions=5       # the summary uses the means
bench_field_init --benchmark_format=json --benchmark_out=a.json # machine readable, for diffing runs
```

## What is measured

One field (`HybridQuantity::Scalar::Bx`, dual in x / primal elsewhere) is filled over its whole AMR
ghost box — exactly what `FieldUserFunctionInitializer::initialize` does.

| variant | what it does |
|---|---|
| `cpp_direct` | plain C++: one pass, coordinates computed per node, no intermediate vectors — the reference |
| `cpp_via_vectors` | plain C++ but through the same index/coordinate-vector machinery the Python path builds, to separate that machinery from the binding layer |
| `py_now` | **the current path**: `initializer::InitFunction<dim>` = `std::function<shared_ptr<Span<double>>(std::vector<double> const&...)>` bound to a Python callable, with `py_fn_wrapper`/`fn_wrapper` (`np.asarray`, `makePyArrayWrapper`) on the Python side |
| `py_np` | **improvement 2**: coordinates handed over as zero-copy `py::array_t` views onto the existing `std::vector` buffers, returned ndarray read in place |
| `py_np_cached` | **+ improvement 1**: the index/coordinate vectors are built once and reused (models a field re-stamped every step; the first stamp still pays for them) |
| `py_np_cached_flat` | **+** contiguous scatter (`std::copy`) instead of per-node `field(AMRToLocal(Point{...}))` |
| `py_np_out` | **+** the field's own buffer handed over as a writable `out` array, filled in place: no result allocation and no copy back. Costs an API change — user functions write into a buffer instead of returning one |

Two user functions: `trivial` returns a scalar (so the measurement is *pure binding-layer overhead*),
`sincos` is a representative numpy expression (`sin(2πx)·cos(2πy)…`).

## Results

gcc `-O3`, Python 3.14 + numpy, one pinned core, `--benchmark_min_time=0.2s --benchmark_repetitions=3`.
Times in µs per field fill, percentages relative to `cpp_direct`.

### 2D, `trivial` — pure overhead

| cells | nodes | `cpp_direct` | `cpp_via_vectors` | `py_now` | `py_np` | `py_np_cached` | `py_np_cached_flat` | `py_np_out` |
|---|---|---|---|---|---|---|---|---|
| 64  | 4.7 k   | 1.52 | 10.1 | 194   | 11.9 | 3.38 | 2.13 | 1.25 |
| 128 | 17.6 k  | 5.65 | 38.5 | 732   | 42.6 | 8.92 | 4.34 | 2.53 |
| 256 | 67.9 k  | 22.1 | 158  | 3176  | 166  | 31.4 | 15.3 | 7.58 |
| 512 | 266.8 k | 93.6 | 1554 | 16514 | 1607 | 123  | 60.9 | 27.8 |

At 256²: the current path is **+14263 %** over a plain C++ fill, `py_np_cached_flat` is **−31 %**,
`py_np_out` **−66 %** (numpy's vectorized fill straight into the field beats per-node `Point`
construction).

### 2D, `sincos` — with real work in the user function

| cells | `cpp_direct` | `cpp_via_vectors` | `py_now` | `py_np` | `py_np_cached` | `py_np_cached_flat` | `py_np_out` |
|---|---|---|---|---|---|---|---|
| 64  | 63.5 | 71.6 | 259   | 74.2 | 65.7 | 64.2 | 63.9 |
| 128 | 236  | 267  | 985   | 274  | 240  | 235  | 234  |
| 256 | 907  | 1037 | 3955  | 1070 | 936  | 916  | 904  |
| 512 | 3558 | 5147 | 22122 | 6044 | 3792 | 3688 | 3603 |

At 256²: current path **+336 %**; `py_np_cached_flat` **+1.0 %**; `py_np_out` **−0.4 %** — i.e.
calling into Python becomes free relative to the arithmetic the user asked for.

### 3D, `sincos`

| cells | nodes | `cpp_direct` | `cpp_via_vectors` | `py_now` | `py_np` | `py_np_cached` | `py_np_cached_flat` | `py_np_out` |
|---|---|---|---|---|---|---|---|---|
| 16 | 8.4 k   | 168  | 190  | 705   | 199   | 175  | 170  | 169  |
| 32 | 48.0 k  | 961  | 1085 | 4332  | 1128  | 999  | 974  | 960  |
| 64 | 319.1 k | 6382 | 9129 | 37907 | 10021 | 6917 | 6688 | 6539 |

## Reading it

- The overhead of the current binding layer is ~**46 ns per node per component**, independent of what
  the user function computes. It is two full copies of every coordinate array:
  `std::vector<double>` → `py::list` (one `PyFloat` per node, done by `pybind11/stl.h`) →
  `np.asarray` → ndarray.
- **Zero-copy numpy views (`py_np`) remove ~95 % of it** on their own — that is the one change that
  matters. It is a change to `initializer::InitFunction`'s parameter type plus the matching
  `py_fn_wrapper`, and it speeds up ordinary initialization too, not only re-stamping.
- **Caching the index/coordinate vectors (`py_np_cached`)** then removes what is left, but only for
  fields stamped more than once (the `cpp_via_vectors` column is what that caching saves).
- The contiguous scatter is a small extra, worth taking since it falls out of the same rewrite.
- **`py_np_out` measures what the return path still costs**: even with zero-copy inputs, the user
  function allocates an ndarray, numpy fills it, and the result is copied into the field. Filling the
  field's buffer in place instead halves the remaining time for a trivial function (15.3 → 7.58 µs at
  256²) and is worth ~1 % once the user function does real work. That gain buys an API change —
  user functions would take a destination array and write into it rather than return one — so it is
  a separate decision from the zero-copy inputs, which need no change to user scripts.

## Note

The target is deliberately *not* registered in `res/cmake/bench.cmake`: it needs neither SAMRAI nor
any built PHARE library, and `add_phare_cpp_benchmark` would link `phare_simulator` and register it
as a ctest. Add `add_subdirectory(tools/bench/initializer)` there if that is wanted anyway.
