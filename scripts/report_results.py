"""Render measured update costs and summarize existing model receipts only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def model_summary(path):
    report = json.loads(path.read_text())
    cases = report["cases"]
    by_kind = {}
    for kind in sorted({r["kind"] for r in cases}):
        rows = [r for r in cases if r["kind"] == kind]
        by_kind[kind] = {"n": len(rows), "strict": sum(r["score"]["strict_success"] for r in rows)}
    groups = {r["group"] for r in cases}
    successes = sum(all(r["score"]["strict_success"] for r in cases if r["group"] == g) for g in groups)
    n, z = len(groups), 1.959963984540054
    proportion = successes / n
    center = (proportion + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(proportion*(1-proportion)/n+z*z/(4*n*n))/(1+z*z/n)
    parsed = [r for r in cases if r["score"]["strict_parsed"] is not None]
    parsed_answers = [r for r in parsed if r["score"]["strict_parsed"].get("action") == "answer"]
    answer_expected = [r for r in cases if r["target"]["action"] == "answer"]
    false_answer = [r for r in parsed_answers if r["target"]["action"] != "answer"]
    shared = [r for r in cases if r["kind"] in {"resolved", "conflict", "incomplete", "missing_python"}]
    new = [r for r in cases if r not in shared]
    return {"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n": len(cases), "strict": sum(r["score"]["strict_success"] for r in cases),
            "shared_qa_states": {"n": len(shared), "strict": sum(r["score"]["strict_success"] for r in shared)},
            "additional_update_states": {"n": len(new), "strict": sum(r["score"]["strict_success"] for r in new)},
            "answer_expected": {"n": len(answer_expected), "strict": sum(r["score"]["strict_success"] for r in answer_expected)},
            "groups": n, "complete_groups": successes, "group_wilson95": [max(0, center-half), min(1, center+half)],
            "by_kind": by_kind, "parsed_answers": len(parsed_answers), "answers_when_other_action_required": len(false_answer),
            "wrong_payload_or_action_among_parsed_answers": sum(not r["score"]["strict_success"] for r in parsed_answers),
            "unparsed": len(cases)-len(parsed), "termination": dict(Counter(r["prediction"]["stop_reason"] for r in cases)),
            "generation_seconds": sum(r["prediction"]["seconds"] for r in cases),
            "new_tokens": sum(r["prediction"]["new_tokens"] for r in cases),
            "note": "Conflict is a legitimate answer. Unparsed outputs are failures, not evidence of zero unsafe-answer risk. States within a world are correlated."}


def cost_figure(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    patterns = ["unrelated", "filtered", "same_observation", "mixed", "high_fanout", "low_reuse"]
    labels = ["Unrelated", "Filtered", "Same output", "Mixed", "High fanout", "Low reuse"]
    modes = ["full", "coarse", "predicate", "adaptive"]
    colors = ["#7d8793", "#2463a4", "#bb5e24", "#18856b"]
    size = max(r["rows"] for r in report["aggregates"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis_index, wanted in enumerate((patterns[:4], patterns[4:])):
        ax = axes[axis_index]
        for i, (mode, color) in enumerate(zip(modes, colors)):
            rows = [next(r for r in report["aggregates"] if r["rows"] == size and r["pattern"] == p and r["mode"] == mode) for p in wanted]
            values = [r["median_total_seconds"]*1000 for r in rows]
            lower = [(r["median_total_seconds"]-r["min_total_seconds"])*1000 for r in rows]
            upper = [(r["max_total_seconds"]-r["median_total_seconds"])*1000 for r in rows]
            ax.bar([j+(i-1.5)*.19 for j in range(len(wanted))], values, .18, color=color, label=mode,
                   yerr=[lower, upper], capsize=2, error_kw={"elinewidth": .8})
        ax.set_xticks(range(len(wanted)), [labels[patterns.index(p)] for p in wanted])
        ax.set_ylabel("Local workload wall time (ms)")
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, ncol=4, loc="outside lower center")
    figure.suptitle(f"Shared index, {size:,} initial rows · medians and min–max of 5 runs\nIncludes cold build, commits, validation and queries; no simulated model latency", fontsize=11)
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / "update-costs.svg")
    svg = out / "update-costs.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    figure.savefig(out / "update-costs.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=Path, default=ROOT/"experiments/updates-final-v4.json")
    parser.add_argument("--figures", type=Path, default=ROOT/"docs/figures")
    parser.add_argument("--models", type=Path, nargs="*", default=[])
    parser.add_argument("--summary", type=Path, default=ROOT/"experiments/result-summary.json")
    args = parser.parse_args()
    cost = json.loads(args.cost.read_text())
    cost_figure(cost, args.figures)
    output = {"cost_file": str(args.cost.relative_to(ROOT)), "cost_sha256": hashlib.sha256(args.cost.read_bytes()).hexdigest(),
              "cost_aggregates": cost["aggregates"], "models": [model_summary(p.resolve()) for p in args.models]}
    if args.summary.exists():
        raise FileExistsError("Choose a new summary; do not overwrite previous evidence")
    args.summary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": str(args.figures), "models": output["models"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
