#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)

default_refs=(master=2316025c9 pr=56f812769 iter=HEAD)
default_ratios="pr/master iter/pr"

usage() {
    cat << EOF
usage: $(basename "$0") [label=ref ...]

Compiles ${here#"$repo"/}/bench_field_box.cpp from this checkout against the
src/ tree of each git ref, runs the binaries interleaved on one pinned core and
prints the median time of each benchmark.

default refs: ${default_refs[*]}

environment (default):
  CXXS        compilers                          (g++ clang++)
  ARCH_FLAGS  target flags                       (-march=native)
  CPU         core the runs are pinned to        (2)
  ROUNDS      interleaved rounds                 (3)
  REPS        repetitions per round              (5)
  MIN_TIME    minimum seconds per repetition     (0.1)
  FILTER      --benchmark_filter regex           (all benchmarks)
  RATIOS      ratio columns as num/den labels    ($default_ratios)
  WORK        work directory, kept afterwards    (mktemp -d)
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi

refs=("$@")
[[ ${#refs[@]} -gt 0 ]] || refs=("${default_refs[@]}")
read -r -a cxxs <<< "${CXXS:-g++ clang++}"
read -r -a arch <<< "${ARCH_FLAGS--march=native}"
read -r -a ratios <<< "${RATIOS:-$default_ratios}"
cpu=${CPU:-2}
rounds=${ROUNDS:-3}
reps=${REPS:-5}
min_time=${MIN_TIME:-0.1}
work=${WORK:-$(mktemp -d)}
rm -rf "$work/json"
mkdir -p "$work/trees" "$work/bin" "$work/json"

dependency() {
    local name=$1 url=$2
    if [[ -d $repo/subprojects/$name ]]; then
        echo "$repo/subprojects/$name"
        return
    fi
    [[ -d $work/$name ]] || git clone -q --depth 1 "$url" "$work/$name"
    echo "$work/$name"
}

cppdict=$(dependency cppdict https://github.com/LaboratoryOfPlasmaPhysics/cppdict)
gbench=$(dependency googlebench https://github.com/google/benchmark)
gbench_lib=$work/gbench-build/src/libbenchmark.a
if [[ ! -f $gbench_lib ]]; then
    echo "building google benchmark from $gbench"
    cmake -S "$gbench" -B "$work/gbench-build" -DCMAKE_BUILD_TYPE=Release \
        -DBENCHMARK_ENABLE_TESTING=OFF -DBENCHMARK_ENABLE_GTEST_TESTS=OFF \
        -DBENCHMARK_ENABLE_WERROR=OFF > "$work/gbench-build.log" 2>&1
    cmake --build "$work/gbench-build" --target benchmark -j 8 >> "$work/gbench-build.log" 2>&1
fi

labels=()
for spec in "${refs[@]}"; do
    if [[ $spec != *=* ]]; then
        usage >&2
        exit 1
    fi
    label=${spec%%=*}
    ref=${spec#*=}
    if ! sha=$(git -C "$repo" rev-parse --verify --quiet "$ref^{commit}"); then
        echo "unknown ref: $ref" >&2
        exit 1
    fi
    labels+=("$label")
    tree=$work/trees/$label
    rm -rf "$tree"
    mkdir -p "$tree"
    git -C "$repo" archive "$sha" src tests | tar -x -C "$tree"
    for cxx in "${cxxs[@]}"; do
        echo "building $label (${sha:0:9}) with $cxx"
        "$cxx" -O3 -DNDEBUG ${arch[@]+"${arch[@]}"} -std=gnu++20 -pthread -DPHARE_HAS_HIGHFIVE=0 \
            -I"$tree/src" -I"$tree" -I"$cppdict/include" -I"$gbench/include" \
            "$here/bench_field_box.cpp" -o "$work/bin/$(basename "$cxx")--$label" "$gbench_lib"
    done
done

pin=()
if command -v taskset > /dev/null; then
    pin=(taskset -c "$cpu")
fi
bench_args=(--benchmark_repetitions="$reps" --benchmark_min_time="${min_time}s" --benchmark_out_format=json)
if [[ -n ${FILTER:-} ]]; then
    bench_args+=(--benchmark_filter="$FILTER")
fi

for round in $(seq "$rounds"); do
    for cxx in "${cxxs[@]}"; do
        for label in "${labels[@]}"; do
            name=$(basename "$cxx")--$label
            log=$work/json/$name--$round.log
            echo "round $round/$rounds: $name"
            if ! ${pin[@]+"${pin[@]}"} "$work/bin/$name" "${bench_args[@]}" \
                --benchmark_out="$work/json/$name--$round.json" > /dev/null 2> "$log"; then
                cat "$log" >&2
                exit 1
            fi
        done
    done
done

compilers=()
for cxx in "${cxxs[@]}"; do
    compilers+=("$(basename "$cxx")")
done
python3 "$here/compare.py" "$work/json" --compilers "${compilers[@]}" --labels "${labels[@]}" \
    --ratios "${ratios[@]}"
echo
echo "raw results: $work/json"
