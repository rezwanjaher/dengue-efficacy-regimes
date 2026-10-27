#!/usr/bin/env python3
"""Build every figure and the evaluation tables.

    python src/make_figures.py --out report

Options:
    --dir          directory holding the evaluation CSVs Default
                   evaluation_dengue.
    --out          output directory; figures land in <out>/figures/ Default
                   report.
    --version_tag  File prefix of the results. Default v18.
    --wide         Draw figures at full text width instead of panel width.
    --autofit      Derive each drawn width from its include width.
    --compile      Also build the report PDF.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("report figures", "make_report_panels.py",
     ["--no-compile"]),
    ("result figures", "make_result_figures_panels.py",
     ["--force"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="evaluation_dengue",
                    help="directory holding the evaluation CSVs")
    ap.add_argument("--out", default="report",
                    help="output directory; figures land in <out>/figures/")
    ap.add_argument("--version_tag", default="v18")
    ap.add_argument("--wide", action="store_true",
                    help='Draw figures at full text width instead of panel width.')
    ap.add_argument("--autofit", action="store_true",
                    help='Derive each drawn width from its include width.')
    ap.add_argument("--compile", action="store_true",
                    help='Also build the report PDF.')
    a = ap.parse_args()

    geometry = ([] if not a.wide else ["--wide"]) + \
               ([] if not a.autofit else ["--autofit"])

    failures = []
    for title, script, extra in STEPS:
        path = os.path.join(HERE, script)
        if not os.path.exists(path):
            print(f"\n!! {script} not found beside this file; skipping {title}")
            failures.append(script)
            continue
        args = [sys.executable, path, "--dir", a.dir, "--out", a.out]
        if script == "make_report_panels.py":
            args += ["--version_tag", a.version_tag]
            extra = [] if a.compile else extra
        args += extra + geometry
        print("\n" + "=" * 72)
        print(f"{title}:  {' '.join(args[1:])}")
        print("=" * 72)
        r = subprocess.run(args)
        if r.returncode != 0:
            failures.append(script)

    figdir = os.path.join(a.out, "figures")
    n = len([f for f in os.listdir(figdir)]) if os.path.isdir(figdir) else 0
    print("\n" + "=" * 72)
    print(f"{n} files in {figdir}")
    if failures:
        print("failed: " + ", ".join(failures))
        sys.exit(1)
    print("If any NOTE lines appeared above, a figure's drawn size does not")
    print("match its include width. Each note gives the size to set in")
    print("figstyle.py, or rerun with --autofit.")


if __name__ == "__main__":
    main()
