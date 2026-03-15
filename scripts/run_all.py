#!/usr/bin/env python3
"""Run Fortran and Rust BLAS benchmarks + accuracy tests, then generate HTML report."""
import subprocess
import json
import glob
import os
import sys
import argparse
import re

import pandas as pd
from jinja2 import Environment, FileSystemLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FORTRAN_DIR = os.path.join(PROJECT_ROOT, "fortran_blas")
RUST_DIR = os.path.join(PROJECT_ROOT, "rust_blas")
REPORT_OUT = os.path.join(PROJECT_ROOT, "report.html")


def run(cmd, cwd=None, shell=False, capture=True):
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"  $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    r = subprocess.run(
        cmd,
        shell=shell,
        cwd=cwd,
        capture_output=capture,
        text=True,
    )
    return r.returncode, r.stdout, r.stderr


# ── Step 1: Fortran bench ──────────────────────────────────────────────────

def run_fortran_bench():
    print("\n[1/4] Building Fortran benchmark...")
    rc, out, err = run("make -C {} -s".format(FORTRAN_DIR), shell=True)
    if rc != 0:
        print("  ERROR building Fortran:", err)
        sys.exit(1)

    print("[1/4] Running Fortran benchmark...")
    rc, out, err = run(
        os.path.join(FORTRAN_DIR, "bench"),
        cwd=FORTRAN_DIR,
    )
    if rc != 0:
        print("  ERROR running Fortran bench:", err)
        sys.exit(1)

    try:
        rows = json.loads(out)
    except json.JSONDecodeError as e:
        print("  ERROR parsing Fortran JSON:", e)
        print("  Output was:", out[:500])
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["source"] = "fortran"
    # Normalize: strip d/s precision prefix so routine names match Rust bench keys
    # e.g. 'dasum' → 'asum', 'sasum' → 'asum' (precision column already set)
    def _strip_prec_prefix(row):
        r = row["routine"]
        if row["precision"] == "f64" and r.startswith("d"):
            return r[1:]
        if row["precision"] == "f32" and r.startswith("s"):
            return r[1:]
        return r
    df["routine"] = df.apply(_strip_prec_prefix, axis=1)
    print(f"  Fortran: {len(df)} benchmark rows")
    return df


# ── Step 2: Rust accuracy tests ────────────────────────────────────────────

def run_rust_accuracy():
    print("\n[2/4] Building Rust library...")
    cmd = ". ~/.cargo/env && cargo build --release -q"
    rc, out, err = run(cmd, cwd=RUST_DIR, shell=True)
    if rc != 0:
        print("  ERROR building Rust:", err)
        sys.exit(1)

    print("[2/4] Running Rust accuracy tests...")
    cmd = ". ~/.cargo/env && cargo test write_accuracy_report -- --nocapture"
    rc, out, err = run(cmd, cwd=RUST_DIR, shell=True)
    if rc != 0:
        print("  ERROR running accuracy tests:", err)
        print("  stdout:", out[-1000:])
        sys.exit(1)

    accuracy_path = "/tmp/accuracy_results.json"
    with open(accuracy_path) as f:
        accuracy_rows = json.load(f)
    print(f"  Accuracy: {len(accuracy_rows)} routines tested")
    return accuracy_rows


# ── Step 3: Rust benchmarks ────────────────────────────────────────────────

def load_rust_bench_data():
    """Parse criterion estimates from the 'new' subdirectory on disk."""
    # Criterion stores latest results in target/criterion/<group>/<n>/new/estimates.json
    pattern = os.path.join(RUST_DIR, "target", "criterion", "*", "*", "new", "estimates.json")
    files = glob.glob(pattern)
    print(f"  Found {len(files)} criterion estimate files")

    criterion_dir = os.path.join(RUST_DIR, "target", "criterion")
    rows = []
    for fpath in files:
        rel = os.path.relpath(fpath, criterion_dir)
        parts = rel.split(os.sep)
        # Expected structure: <group_dir>/<n>/new/estimates.json
        # e.g. parts = ["level1_dasum", "512", "new", "estimates.json"]
        if len(parts) < 4:
            continue

        group_dir = parts[0]   # e.g. "level1_dasum"
        param = parts[1]       # e.g. "512"

        try:
            n = int(param)
        except ValueError:
            continue

        with open(fpath) as f:
            est = json.load(f)
        mean_ns = est.get("mean", {}).get("point_estimate")
        std_ns = est.get("std_dev", {}).get("point_estimate")
        if mean_ns is None:
            continue

        # Parse routine key: "level1_dasum" → "dasum", "level2_dgemv" → "dgemv"
        m = re.match(r'^level\d+_(.+)$', group_dir)
        if not m:
            continue
        routine_key = m.group(1)

        if routine_key.startswith("d"):
            precision = "f64"
            routine = routine_key[1:]
        elif routine_key.startswith("s"):
            precision = "f32"
            routine = routine_key[1:]
        else:
            precision = "f64"
            routine = routine_key

        rows.append({
            "routine": routine,
            "precision": precision,
            "n": n,
            "mean_ns": mean_ns,
            "std_ns": std_ns if std_ns else 0.0,
            "source": "rust",
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["routine", "precision", "n", "mean_ns", "std_ns", "source"]
    )
    print(f"  Rust: {len(df)} benchmark rows")
    return df


def run_rust_bench():
    print("\n[3/4] Running Rust benchmarks (this may take a few minutes)...")
    cmd = ". ~/.cargo/env && cargo bench --bench blas_bench"
    rc, out, err = run(cmd, cwd=RUST_DIR, shell=True)
    if rc != 0:
        print("  WARNING: cargo bench returned non-zero:", rc)
        print("  stderr:", err[-500:])
        # Continue anyway — estimates may still be on disk

    return load_rust_bench_data()


# ── Step 4: Generate report ────────────────────────────────────────────────

def build_chart_data(bench_df):
    """Build chart_data dict: routine -> {labels, fortran, rust}"""
    chart_data = {}
    if bench_df.empty:
        return chart_data

    # Use f64 precision only for charts
    df64 = bench_df[bench_df["precision"] == "f64"].copy()
    routines = sorted(df64["routine"].unique())
    # Only include routines that have both Fortran and Rust data
    paired = [r for r in routines
              if not df64[(df64["routine"] == r) & (df64["source"] == "fortran")].empty
              and not df64[(df64["routine"] == r) & (df64["source"] == "rust")].empty]

    for routine in paired:
        rdf = df64[df64["routine"] == routine]
        sizes = sorted(rdf["n"].unique())
        labels = [f"n={n}" for n in sizes]
        fortran_vals = []
        rust_vals = []
        for n in sizes:
            f_row = rdf[(rdf["source"] == "fortran") & (rdf["n"] == n)]
            r_row = rdf[(rdf["source"] == "rust") & (rdf["n"] == n)]
            fortran_vals.append(round(float(f_row["mean_ns"].values[0]), 1) if not f_row.empty else None)
            rust_vals.append(round(float(r_row["mean_ns"].values[0]), 1) if not r_row.empty else None)
        chart_data[routine] = {
            "labels": labels,
            "fortran": fortran_vals,
            "rust": rust_vals,
        }
    return chart_data


def build_accuracy_lookup(accuracy_rows):
    """Build dict keyed by (routine, precision, n) → accuracy entry."""
    lookup = {}
    for r in accuracy_rows:
        key = (r["routine"], r["precision"], int(r["n"]))
        lookup[key] = r
    return lookup


def build_speedup_rows(bench_df, accuracy_lookup=None):
    """Build speedup table rows for f64, enriched with accuracy data where available."""
    rows = []
    if bench_df.empty:
        return rows
    if accuracy_lookup is None:
        accuracy_lookup = {}
    df64 = bench_df[bench_df["precision"] == "f64"].copy()
    routines = sorted(df64["routine"].unique())
    sizes = sorted(df64["n"].unique())
    for routine in routines:
        for n in sizes:
            f_row = df64[(df64["routine"] == routine) & (df64["source"] == "fortran") & (df64["n"] == n)]
            r_row = df64[(df64["routine"] == routine) & (df64["source"] == "rust") & (df64["n"] == n)]
            if f_row.empty or r_row.empty:
                continue
            f_ns = float(f_row["mean_ns"].values[0])
            r_ns = float(r_row["mean_ns"].values[0])
            speedup = r_ns / f_ns if f_ns > 0 else float("inf")
            # Accuracy for f64 at this (routine, n)
            acc = accuracy_lookup.get((routine, "f64", int(n)))
            rows.append({
                "routine": routine,
                "n": n,
                "fortran_ns": round(f_ns, 1),
                "rust_ns": round(r_ns, 1),
                "speedup": round(speedup, 3),
                # accuracy fields (None when not available)
                "max_rel_error": acc["max_rel_error"] if acc else None,
                "threshold": acc["threshold"] if acc else None,
                "acc_passed": acc["passed"] if acc else None,
            })
    return rows


def build_speedup_rows_f32(bench_df, accuracy_lookup=None):
    """Same as build_speedup_rows but for f32 precision."""
    rows = []
    if bench_df.empty:
        return rows
    if accuracy_lookup is None:
        accuracy_lookup = {}
    df32 = bench_df[bench_df["precision"] == "f32"].copy()
    routines = sorted(df32["routine"].unique())
    sizes = sorted(df32["n"].unique())
    for routine in routines:
        for n in sizes:
            f_row = df32[(df32["routine"] == routine) & (df32["source"] == "fortran") & (df32["n"] == n)]
            r_row = df32[(df32["routine"] == routine) & (df32["source"] == "rust") & (df32["n"] == n)]
            if f_row.empty or r_row.empty:
                continue
            f_ns = float(f_row["mean_ns"].values[0])
            r_ns = float(r_row["mean_ns"].values[0])
            speedup = r_ns / f_ns if f_ns > 0 else float("inf")
            acc = accuracy_lookup.get((routine, "f32", int(n)))
            rows.append({
                "routine": routine,
                "n": n,
                "fortran_ns": round(f_ns, 1),
                "rust_ns": round(r_ns, 1),
                "speedup": round(speedup, 3),
                "max_rel_error": acc["max_rel_error"] if acc else None,
                "threshold": acc["threshold"] if acc else None,
                "acc_passed": acc["passed"] if acc else None,
            })
    return rows


def generate_report(accuracy_rows, bench_df):
    print("\n[4/4] Generating HTML report...")
    accuracy_lookup = build_accuracy_lookup(accuracy_rows)
    chart_data = build_chart_data(bench_df)
    speedup_rows = build_speedup_rows(bench_df, accuracy_lookup)
    speedup_rows_f32 = build_speedup_rows_f32(bench_df, accuracy_lookup)
    bench_rows = bench_df.to_dict("records") if not bench_df.empty else []

    # Summary counts
    n_acc_total = len(accuracy_rows)
    n_acc_pass = sum(1 for r in accuracy_rows if r.get("passed"))

    env = Environment(loader=FileSystemLoader(SCRIPT_DIR))
    template = env.get_template("report_template.html")

    html = template.render(
        accuracy_rows=accuracy_rows,
        bench_rows=bench_rows,
        chart_data=chart_data,
        speedup_rows=speedup_rows,
        speedup_rows_f32=speedup_rows_f32,
        n_acc_total=n_acc_total,
        n_acc_pass=n_acc_pass,
    )

    with open(REPORT_OUT, "w") as f:
        f.write(html)
    print(f"  Report written to {REPORT_OUT}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run BLAS benchmarks and generate report")
    parser.add_argument("--skip-fortran", action="store_true", help="Skip Fortran bench (use empty df)")
    parser.add_argument("--skip-rust-bench", action="store_true",
                        help="Skip running cargo bench; use cached criterion data on disk")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip running accuracy tests; use cached /tmp/accuracy_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Bentley Systems — BLAS Performance & Accuracy Pipeline")
    print("=" * 60)

    if args.skip_fortran:
        print("\n[1/4] Skipping Fortran benchmark (--skip-fortran)")
        fortran_df = pd.DataFrame(columns=["routine", "precision", "n", "mean_ns", "std_ns", "source"])
    else:
        fortran_df = run_fortran_bench()

    if args.skip_accuracy:
        print("\n[2/4] Loading cached accuracy results from /tmp/accuracy_results.json...")
        with open("/tmp/accuracy_results.json") as f:
            accuracy_rows = json.load(f)
        print(f"  Accuracy: {len(accuracy_rows)} routines tested (cached)")
    else:
        accuracy_rows = run_rust_accuracy()

    if args.skip_rust_bench:
        print("\n[3/4] Loading cached Rust benchmark data...")
        rust_df = load_rust_bench_data()
    else:
        rust_df = run_rust_bench()

    bench_df = pd.concat([fortran_df, rust_df], ignore_index=True) if not rust_df.empty else fortran_df

    generate_report(accuracy_rows, bench_df)
    print("\nDone! Open report.html in your browser.")


if __name__ == "__main__":
    main()
