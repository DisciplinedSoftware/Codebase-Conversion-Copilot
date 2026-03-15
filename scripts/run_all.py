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

_NAVY   = "#002E5D"
_BLUE   = "#0071CE"
_ORANGE = "#F58220"
_GREY   = "#6B7280"
_LGREY  = "#E5E7EB"
_TEXT   = "#1F2937"


def _fmt_ns(v):
    """Format nanoseconds for SVG axis labels."""
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.1f}s"
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}ms"
    if v >= 1_000:
        return f"{v/1_000:.1f}µs"
    return f"{v:.0f}ns"


def make_svg_chart(labels, fortran_vals, rust_vals, width=340, height=230):
    """Generate a self-contained SVG grouped bar chart (no JS required)."""
    ML, MR, MT, MB = 68, 10, 12, 50   # margins: left, right, top, bottom
    cw = width - ML - MR              # chart area width
    ch = height - MT - MB             # chart area height

    all_vals = [v for v in (list(fortran_vals) + list(rust_vals))
                if v is not None and v > 0]
    if not all_vals:
        return (f'<svg viewBox="0 0 {width} {height}" '
                f'xmlns="http://www.w3.org/2000/svg" '
                f'style="width:100%;height:100%">'
                f'<text x="{width//2}" y="{height//2}" text-anchor="middle" '
                f'fill="{_GREY}" font-size="12">No data</text></svg>')

    max_val = max(all_vals)
    # Round max_val up to a "nice" number
    import math
    exp = 10 ** math.floor(math.log10(max_val))
    max_val = math.ceil(max_val / exp) * exp

    n_groups = len(labels)
    group_w = cw / n_groups
    bar_w = group_w * 0.32
    gap   = group_w * 0.04

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:100%;font-family:\'Segoe UI\',Arial,sans-serif">',
    ]

    # Grid lines + Y-axis ticks (5 levels)
    n_ticks = 4
    for i in range(n_ticks + 1):
        y = MT + ch - ch * i / n_ticks
        val = max_val * i / n_ticks
        # gridline
        if i > 0:
            parts.append(
                f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+cw}" y2="{y:.1f}" '
                f'stroke="{_LGREY}" stroke-width="0.8" stroke-dasharray="4,3"/>'
            )
        # tick label
        parts.append(
            f'<text x="{ML-5}" y="{y+3.5:.1f}" text-anchor="end" '
            f'font-size="9" fill="{_GREY}">{_fmt_ns(val)}</text>'
        )

    # Axes
    parts.append(
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+ch}" '
        f'stroke="{_NAVY}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{ML}" y1="{MT+ch}" x2="{ML+cw}" y2="{MT+ch}" '
        f'stroke="{_NAVY}" stroke-width="1.5"/>'
    )

    # Bars + x-labels
    for gi, (label, fv, rv) in enumerate(zip(labels, fortran_vals, rust_vals)):
        cx = ML + gi * group_w + group_w / 2

        # Fortran bar
        if fv is not None and fv > 0:
            bh = ch * min(fv, max_val) / max_val
            bx = cx - bar_w - gap / 2
            by = MT + ch - bh
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" '
                f'height="{bh:.1f}" fill="{_BLUE}" rx="2">'
                f'<title>Fortran {label}: {_fmt_ns(fv)}</title></rect>'
            )

        # Rust bar
        if rv is not None and rv > 0:
            bh = ch * min(rv, max_val) / max_val
            bx = cx + gap / 2
            by = MT + ch - bh
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" '
                f'height="{bh:.1f}" fill="{_ORANGE}" rx="2">'
                f'<title>Rust {label}: {_fmt_ns(rv)}</title></rect>'
            )

        # x-axis label (may need wrapping for large n)
        lbl_text = label if len(label) <= 12 else label.replace("n=", "")
        parts.append(
            f'<text x="{cx:.1f}" y="{MT+ch+13}" text-anchor="middle" '
            f'font-size="9" fill="{_TEXT}">{lbl_text}</text>'
        )

    # Legend
    lx, ly = ML + 4, height - 10
    parts += [
        f'<rect x="{lx}" y="{ly-8}" width="10" height="10" fill="{_BLUE}" rx="2"/>',
        f'<text x="{lx+13}" y="{ly}" font-size="10" fill="{_TEXT}">Fortran</text>',
        f'<rect x="{lx+63}" y="{ly-8}" width="10" height="10" fill="{_ORANGE}" rx="2"/>',
        f'<text x="{lx+76}" y="{ly}" font-size="10" fill="{_TEXT}">Rust</text>',
    ]

    parts.append("</svg>")
    return "\n".join(parts)


def build_svg_charts(bench_df):
    """Build dict: routine -> SVG string (f64 only, only paired routines)."""
    svg_charts = {}
    if bench_df.empty:
        return svg_charts

    df64 = bench_df[bench_df["precision"] == "f64"].copy()
    routines = sorted(df64["routine"].unique())
    paired = [r for r in routines
              if not df64[(df64["routine"] == r) & (df64["source"] == "fortran")].empty
              and not df64[(df64["routine"] == r) & (df64["source"] == "rust")].empty]

    for routine in paired:
        rdf = df64[df64["routine"] == routine]
        # Only keep sizes present in BOTH sources
        f_ns = set(df64[(df64["routine"] == routine) & (df64["source"] == "fortran")]["n"].tolist())
        r_ns = set(df64[(df64["routine"] == routine) & (df64["source"] == "rust")]["n"].tolist())
        sizes = sorted(f_ns & r_ns)
        if not sizes:
            continue
        labels = [f"n={n:,}" for n in sizes]
        fortran_vals = []
        rust_vals = []
        for n in sizes:
            f_row = rdf[(rdf["source"] == "fortran") & (rdf["n"] == n)]
            r_row = rdf[(rdf["source"] == "rust") & (rdf["n"] == n)]
            fortran_vals.append(round(float(f_row["mean_ns"].values[0]), 1) if not f_row.empty else None)
            rust_vals.append(round(float(r_row["mean_ns"].values[0]), 1) if not r_row.empty else None)
        svg_charts[routine] = make_svg_chart(labels, fortran_vals, rust_vals)

    return svg_charts


def build_accuracy_lookup(accuracy_rows):
    """Build exact + fallback accuracy lookups.

    Returns:
        exact:    (routine, precision, n) -> row
        fallback: (routine, precision)    -> best row (highest tested n)
    """
    exact, fallback = {}, {}
    for r in accuracy_rows:
        key = (r["routine"], r["precision"], int(r["n"]))
        exact[key] = r
        fb_key = (r["routine"], r["precision"])
        if fb_key not in fallback or int(r["n"]) > int(fallback[fb_key]["n"]):
            fallback[fb_key] = r
    return exact, fallback


def build_speedup_rows(bench_df, accuracy_lookup=None):
    """Build speedup table rows for f64, enriched with accuracy data where available."""
    rows = []
    if bench_df.empty:
        return rows
    exact_lookup, fallback_lookup = accuracy_lookup if accuracy_lookup else ({}, {})
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
            # Accuracy: exact match first, then fallback to best tested n
            acc = (exact_lookup.get((routine, "f64", int(n)))
                   or fallback_lookup.get((routine, "f64")))
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


def build_speedup_rows_f32(bench_df, accuracy_lookup=None):
    """Same as build_speedup_rows but for f32 precision."""
    rows = []
    if bench_df.empty:
        return rows
    exact_lookup, fallback_lookup = accuracy_lookup if accuracy_lookup else ({}, {})
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
            acc = (exact_lookup.get((routine, "f32", int(n)))
                   or fallback_lookup.get((routine, "f32")))
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
    exact_lookup, fallback_lookup = build_accuracy_lookup(accuracy_rows)
    accuracy_lookup = (exact_lookup, fallback_lookup)
    svg_charts = build_svg_charts(bench_df)
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
        svg_charts=svg_charts,
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
