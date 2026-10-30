import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("json_dir", type=Path)
parser.add_argument("--compilers", nargs="+", required=True)
parser.add_argument("--labels", nargs="+", required=True)
parser.add_argument("--ratios", nargs="*", default=[])
args = parser.parse_args()

samples = defaultdict(list)
cells = {}
for path in sorted(args.json_dir.glob("*.json")):
    compiler, label, _ = path.stem.split("--")
    for bench in json.loads(path.read_text())["benchmarks"]:
        if bench.get("run_type") != "iteration":
            continue
        name = re.sub(r"/repeats:\d+$", "", bench["run_name"])
        samples[compiler, name, label].append(bench["cpu_time"])
        if "cells" in bench:
            cells[name] = int(bench["cells"])

ratios = [r.split("/") for r in args.ratios]
ratios = [(a, b) for a, b in ratios if a in args.labels and b in args.labels]


def median(compiler, name, label):
    xs = samples.get((compiler, name, label))
    return statistics.median(xs) if xs else None


def cell(compiler, name, label):
    xs = samples.get((compiler, name, label))
    if not xs:
        return "—"
    med = statistics.median(xs)
    q = statistics.quantiles(xs, n=4) if len(xs) > 1 else [med] * 3
    return f"{med:,.0f} ±{100 * (q[2] - q[0]) / med:.0f}%"


def ratio(compiler, name, a, b):
    num, den = median(compiler, name, a), median(compiler, name, b)
    return "—" if num is None or den is None else f"{num / den:.2f}"


for compiler in args.compilers:
    names = sorted(
        {k[1] for k in samples if k[0] == compiler}, key=lambda n: (n.split("<")[0], n)
    )
    if not names:
        continue
    print(f"\n### {compiler}: median cpu time [ns] (±IQR/median)\n")
    header = ["benchmark", "cells", *args.labels, *[f"{a}/{b}" for a, b in ratios]]
    print("| " + " | ".join(header) + " |")
    print("|---|" + "---:|" * (len(header) - 1))
    for name in names:
        row = [name.removeprefix("BM_"), str(cells.get(name, ""))]
        row += [cell(compiler, name, label) for label in args.labels]
        row += [ratio(compiler, name, a, b) for a, b in ratios]
        print("| " + " | ".join(row) + " |")
