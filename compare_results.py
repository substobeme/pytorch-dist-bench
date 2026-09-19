#!/usr/bin/env python3
"""
Compare two pytorch-dist-bench result directories and flag regressions.

Reads JSON files from baseline and test directories, matches them by
filename, extracts p50_us timing metrics, and reports % change.
Exits non-zero if any regression exceeds the threshold — suitable for CI.

Inspired by PyTorch's benchmarks/distributed/ddp/diff.py but adapted
for the pytorch-dist-bench JSON format.

Usage:
  python compare_results.py results/baseline/ results/test/
  python compare_results.py results/baseline/ results/test/ --threshold 10
"""

import argparse
import json
import math
import os
import sys


def load_json(path):
    with open(path) as f:
        return json.load(f)


LABEL_KEYS = ("section", "topology", "collective", "op", "routing",
              "model", "param_name", "name")
VALUE_KEYS = ("nelems", "seq_len", "num_tokens", "num_layers", "batch_size",
              "num_microbatches", "dtype", "hidden")


def entry_key(entry):
    """Build a hashable key from identifying fields of a result entry."""
    parts = []
    for key in LABEL_KEYS + VALUE_KEYS:
        if key in entry:
            parts.append((key, entry[key]))
    return tuple(parts)


def extract_label(entry):
    """Build a human-readable label for a result entry."""
    parts = []
    for key in LABEL_KEYS:
        if key in entry:
            parts.append(str(entry[key]))
    for key in VALUE_KEYS:
        if key in entry:
            parts.append(f"{key}={entry[key]}")
    return "  ".join(parts) if parts else "unknown"


def find_p50_metrics(entry, prefix=""):
    """Recursively find all p50_us values in a result entry.

    Yields (metric_path, value) tuples. Walks the full tree so it
    handles both shallow nesting (stats.p50_us) and deep nesting
    (dist.gpu_event.p50_us).
    """
    if not isinstance(entry, dict):
        return
    if "p50_us" in entry:
        yield (prefix.rstrip("."), entry["p50_us"])
    for k, v in entry.items():
        if isinstance(v, dict):
            yield from find_p50_metrics(v, f"{prefix}{k}.")


def compare_alpha_beta(baseline, test, threshold):
    """Compare alpha-beta model fits between baseline and test.

    Returns (comparisons, regressions, improvements) for alpha (latency)
    and beta (bandwidth) per collective.
    """
    b_ab = baseline.get("alpha_beta", {})
    t_ab = test.get("alpha_beta", {})

    comparisons = []
    regressions = 0
    improvements = 0

    for coll in b_ab:
        if coll not in t_ab:
            continue

        b_alpha = b_ab[coll].get("alpha_us", 0)
        t_alpha = t_ab[coll].get("alpha_us", 0)
        if b_alpha > 0:
            pct = (t_alpha - b_alpha) / b_alpha * 100
            flag = ""
            if pct > threshold:
                flag = "REGRESSION"
                regressions += 1
            elif pct < -threshold:
                flag = "IMPROVED"
                improvements += 1
            comparisons.append(
                (coll, "alpha_us (latency)", b_alpha, t_alpha, pct, flag))

        b_beta = b_ab[coll].get("beta_gbps", 0)
        t_beta = t_ab[coll].get("beta_gbps", 0)
        if b_beta > 0:
            pct = (t_beta - b_beta) / b_beta * 100
            flag = ""
            if pct < -threshold:
                flag = "REGRESSION"
                regressions += 1
            elif pct > threshold:
                flag = "IMPROVED"
                improvements += 1
            comparisons.append(
                (coll, "beta_gbps (bandwidth)", b_beta, t_beta, pct, flag))

    return comparisons, regressions, improvements


def compare_file(baseline_path, test_path, threshold):
    """Compare two JSON result files.

    Returns (comparisons, regressions, improvements, missing): every p50_us
    metric (and pass/fail for correctness entries) present in both files,
    and the labels of baseline entries absent from the test file.
    """
    baseline = load_json(baseline_path)
    test = load_json(test_path)

    # Result files are paired by name; a misnamed file would otherwise be
    # compared across dtypes without warning.
    for key in ("benchmark", "dtype"):
        if baseline.get(key) != test.get(key):
            raise ValueError(
                f"{key} differs: baseline={baseline.get(key)!r} "
                f"test={test.get(key)!r}")

    b_results = baseline.get("results", [])
    t_results = test.get("results", [])

    comparisons = []
    regressions = 0
    improvements = 0
    missing = []

    t_by_key = {}
    for t_entry in t_results:
        t_by_key[entry_key(t_entry)] = t_entry

    for b_entry in b_results:
        t_entry = t_by_key.get(entry_key(b_entry))
        label = extract_label(b_entry)
        if t_entry is None:
            missing.append(label)
            continue

        # Correctness entries (bench_verify): pass -> fail is a regression.
        if "passed" in b_entry and "passed" in t_entry:
            b_ok, t_ok = bool(b_entry["passed"]), bool(t_entry["passed"])
            flag = ""
            if b_ok and not t_ok:
                flag = "REGRESSION"
                regressions += 1
            elif t_ok and not b_ok:
                flag = "IMPROVED"
                improvements += 1
            comparisons.append((label, "passed", float(b_ok), float(t_ok),
                                0.0, flag))
            continue

        b_metrics = dict(find_p50_metrics(b_entry))
        t_metrics = dict(find_p50_metrics(t_entry))

        for metric_name in b_metrics:
            if metric_name not in t_metrics:
                continue
            b_val = b_metrics[metric_name]
            t_val = t_metrics[metric_name]
            if b_val <= 0 or math.isnan(b_val) or math.isnan(t_val):
                continue

            pct = (t_val - b_val) / b_val * 100
            flag = ""
            if pct > threshold:
                flag = "REGRESSION"
                regressions += 1
            elif pct < -threshold:
                flag = "IMPROVED"
                improvements += 1

            comparisons.append((label, metric_name, b_val, t_val, pct, flag))

    ab_comps, ab_regs, ab_imps = compare_alpha_beta(baseline, test, threshold)
    comparisons.extend(ab_comps)
    regressions += ab_regs
    improvements += ab_imps

    return comparisons, regressions, improvements, missing


def main():
    parser = argparse.ArgumentParser(
        description="Compare pytorch-dist-bench results and flag regressions")
    parser.add_argument("baseline", help="Baseline results directory")
    parser.add_argument("test", help="Test results directory")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold %% (default: 5.0)")
    args = parser.parse_args()

    if not os.path.isdir(args.baseline):
        print(f"Error: {args.baseline} is not a directory")
        sys.exit(2)
    if not os.path.isdir(args.test):
        print(f"Error: {args.test} is not a directory")
        sys.exit(2)

    baseline_files = {f for f in os.listdir(args.baseline) if f.endswith(".json")}
    test_files = {f for f in os.listdir(args.test) if f.endswith(".json")}

    matched = sorted(baseline_files & test_files)
    only_baseline = sorted(baseline_files - test_files)
    only_test = sorted(test_files - baseline_files)

    if not matched:
        print("No matching JSON files found between directories.")
        sys.exit(2)

    total_regressions = 0
    total_improvements = 0
    total_comparisons = 0
    total_missing = 0
    no_overlap = 0
    errors = 0

    print(f"\n{'=' * 90}")
    print(f"Comparing: {args.baseline} (baseline) vs {args.test} (test)")
    print(f"Threshold: +/-{args.threshold}%")
    print(f"{'=' * 90}")

    for filename in matched:
        b_path = os.path.join(args.baseline, filename)
        t_path = os.path.join(args.test, filename)

        try:
            comparisons, regs, imps, missing = compare_file(
                b_path, t_path, args.threshold)
        except Exception as e:
            print(f"\n=== {filename} === ERROR: {e}")
            errors += 1
            continue

        if not comparisons and not missing:
            # Both files exist but no entry keys or metrics lined up: a
            # schema change, not a clean pass.
            print(f"\n=== {filename} === WARNING: no comparable metrics")
            no_overlap += 1
            continue

        print(f"\n=== {filename} ===")

        for label, metric, b_val, t_val, pct, flag in comparisons:
            flag_str = f"  {flag}" if flag else ""
            if metric == "passed":
                b_s, t_s = ("pass" if b_val else "FAIL"), ("pass" if t_val else "FAIL")
                print(f"  {label:<50s} {metric:<20s} {b_s:>8} -> {t_s:>8}{flag_str}")
                continue
            sign = "+" if pct >= 0 else ""
            print(
                f"  {label:<50s} {metric:<20s}"
                f" {b_val:>8.1f} -> {t_val:>8.1f}"
                f"  ({sign}{pct:.1f}%){flag_str}"
            )
        for label in missing:
            print(f"  {label:<50s} MISSING in test")

        total_regressions += regs
        total_improvements += imps
        total_comparisons += len(comparisons)
        total_missing += len(missing)

    unchanged = total_comparisons - total_regressions - total_improvements

    print(f"\n{'=' * 90}")
    print(f"Summary: {total_comparisons} metrics compared")
    if total_regressions:
        print(f"  {total_regressions} REGRESSIONS (>{args.threshold}% slower)")
    if total_improvements:
        print(f"  {total_improvements} improvements (<-{args.threshold}% faster)")
    print(f"  {unchanged} unchanged (within +/-{args.threshold}%)")

    # Anything the baseline has that the test run lacks means the test run
    # did not complete; a gate must not pass on a partial comparison.
    incomplete = []
    if only_baseline:
        incomplete.append(f"{len(only_baseline)} baseline file(s) missing from "
                          f"test: {', '.join(only_baseline)}")
    if total_missing:
        incomplete.append(f"{total_missing} baseline entr(y/ies) missing from test")
    if no_overlap:
        incomplete.append(f"{no_overlap} file pair(s) with no comparable metrics")
    if errors:
        incomplete.append(f"{errors} file pair(s) could not be compared")
    if total_comparisons == 0:
        incomplete.append("no metrics compared")
    for line in incomplete:
        print(f"  INCOMPLETE: {line}")
    if only_test:
        print(f"  (new in test, not compared: {', '.join(only_test)})")
    print(f"{'=' * 90}\n")

    # 1: regression; 2: comparison incomplete or inconsistent; 0: clean.
    if total_regressions > 0:
        sys.exit(1)
    sys.exit(2 if incomplete else 0)


if __name__ == "__main__":
    main()
