#!/usr/bin/env python3
"""Table of both crossings by instance, with graph properties.

    python src/make_crossings_table.py --tex report/tables/crossings.tex

The default --glob also picks up dncc10, which is the same graph as
contiguous10 with identical results. The paper's table leaves it out;
reproduce.py passes the ten sweeps explicitly.

Options:
    --glob  Sweep files to read.
    --tex   Write the LaTeX table here.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.find_crossings import analyse
from src import instances_ext


def cell(v, tex=True):
    if v is None or "error" in v or v.get("first") is None:
        return "---" if tex else "none"
    if v.get("interval"):
        lo, hi = v["interval"]
        return f"$[{lo:.2f}, {hi:.2f}]$" if tex else f"[{lo:.2f}, {hi:.2f}]"
    c = v["first"]
    s = f"[{c['lo_eta']:.2f}, {c['hi_eta']:.2f}]"
    return f"${s}$\\,\\textsuperscript{{g}}" if tex else s + " g"


def collect(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files matched {pattern}")
        sys.exit(1)
    rows, seen = [], {}
    for f in files:
        r = analyse(f)
        tag = r["instance"]
        if tag in seen:
            print(f"  WARNING: {tag} appears twice; keeping the wider grid")
            print(f"           {seen[tag]['source']} and {os.path.basename(f)}")
            if r["n_eta"] <= seen[tag]["n_eta"]:
                continue
            rows = [x for x in rows if x["instance"] != tag]
        r["source"] = os.path.basename(f)
        d = instances_ext.describe(tag)
        r["graph"] = d
        seen[tag] = r
        rows.append(r)
    rows.sort(key=lambda r: ((r["graph"] or {}).get("n_wards") or r["n_wards"] or 0,
                             r["instance"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(
        "evaluation_dengue", "exp_efficacy*_full.csv"))
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()

    rows = collect(a.glob)

    print("=" * 96)
    print("EFFICACY CROSSINGS BY INSTANCE")
    print("=" * 96)
    print(f"  {'instance':<15}{'wards':>6}{'deg med':>9}{'isol':>6}{'pop CV':>9}"
          f"{'concentration':>18}{'immediacy':>18}")
    missing_graph = []
    for r in rows:
        g = r["graph"]
        if g is None:
            missing_graph.append(r["instance"])
        wards = (g or {}).get("n_wards") or r["n_wards"] or "?"
        deg = f"{g['deg_median']:.1f}" if g else "?"
        iso = f"{g['isolated']}" if g else "?"
        cv = f"{g['pop_cv']:.3f}" if g else "?"
        print(f"  {r['instance']:<15}{wards:>6}{deg:>9}{iso:>6}{cv:>9}"
              f"{cell(r['concentration'], False):>18}"
              f"{cell(r['delay'], False):>18}")

    print("\n  'g' marks a grid bracket rather than a CI-based interval.")
    if missing_graph:
        print("\n  graph file not found for: " + ", ".join(missing_graph))
        print("  those rows carry '?' instead of measured properties.")

    if not a.tex:
        print("\n  --tex not given; no table written.")
        return

    L = [r"\begin{table}[htbp]", "\t" + r"\centering\small",
         "\t" + r"\caption{Efficacy at which each crossing occurs, across instances. "
         r"\emph{Concentration} is where a front-loaded campaign begins to beat "
         r"spreading the same expenditure; \emph{immediacy} is where starting a "
         r"week later begins to raise the realised peak. Intervals account for "
         r"the bootstrap confidence intervals where the sweep provides them; "
         r"cells marked \textsuperscript{g} give the adjacent grid points "
         r"instead. The sensitivity and seeding tables use adjacent grid points "
         r"throughout, which for dncc gives $[0.54, 0.56]$ for immediacy. "
         r"Median degree, isolated wards and the coefficient of variation of "
         r"ward populations are measured from each graph.}",
         "\t" + r"\label{tab:crossings}",
         "\t" + r"\begin{tabular}{@{}lrrrrcc@{}}", "\t\t" + r"\toprule",
         "\t\t" + r"Instance & Wards & Med.\ deg. & Isol. & Pop.\ CV & "
         r"Concentration & Immediacy \\", "\t\t" + r"\midrule"]
    for r in rows:
        g = r["graph"]
        wards = (g or {}).get("n_wards") or r["n_wards"] or ""
        deg = f"{g['deg_median']:.1f}" if g else r"\needdeg"
        iso = f"{g['isolated']}" if g else ""
        cv = f"{g['pop_cv']:.3f}" if g else ""
        L.append(f"\t\t{r['instance'].replace('_', ' ')} & {wards} & {deg} & {iso} & "
                 f"{cv} & {cell(r['concentration'])} & {cell(r['delay'])} \\\\")
    L += ["\t\t" + r"\bottomrule", "\t" + r"\end{tabular}", r"\end{table}"]

    os.makedirs(os.path.dirname(a.tex) or ".", exist_ok=True)
    if os.path.exists(a.tex):
        print(f"\n{a.tex} already exists; not replacing it. Pass another --tex path.")
        return
    open(a.tex, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {a.tex}")


if __name__ == "__main__":
    main()
