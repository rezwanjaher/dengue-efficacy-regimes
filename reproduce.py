#!/usr/bin/env python3
"""
reproduce.py -- rebuild the paper's tables and figures from the shipped results,
and check every table against the copy the paper uses.

    python reproduce.py

Run from the repository root. Takes a few minutes on a laptop; no GPU.

What it does:
  1. Writes everything into a new folder, reproduced/ . Nothing shipped is changed.
     If reproduced/ already exists it stops; delete it to run again.
  2. Builds the tables and figures from evaluation_dengue/ with the same scripts
     that built the paper's.
  3. Compares each table the paper includes with report/tables/, and reports
     SAME or DIFFERENT for each.

The crossings table lists ten instances. The 10-ward instance (dncc10) is the
same graph as contiguous10, with identical results, so its sweep is left out.

The schedule tables (schedules_dncc, schedules_dncc10, start_week) come from
simulations run by src/exp_schedules.py, not from a results file. Step 2 takes
them from report/tables/ so that the combined schedule table can be built; the
README shows how to rerun the simulations.
"""
import difflib, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "reproduced")
TAB = os.path.join(OUT, "report", "tables")
REF = os.path.join(ROOT, "report", "tables")
PY = sys.executable

PAPER_TABLES = ["convergence_dncc", "crossings_ec", "delay_ratios_ec", "headline_V_both",
                "headline_peak_both", "protocol_matrix", "schedules_both", "sens_etaS_compact",
                "sens_price_cm", "start_week", "threshold_summary_ec", "wards"]
FROM_SIMULATION = ["schedules_dncc", "schedules_dncc10", "start_week"]
CROSSING_INSTANCES = ["contiguous10", "dispersed10", "random10", "contiguous20", "dispersed20",
                      "random20", "contiguous30", "dispersed30", "random30", "dncc"]
SURVEILLANCE = [("eta_S", "V", "sens_surveillance"), ("eta_S", "peak", "sens_surveillance_peak"),
                ("surv_cost", "V", "sens_survcost"), ("surv_cost", "peak", "sens_survcost_peak"),
                ("eta_C", "V", "sens_etaC")]


def run(title, args, cwd=ROOT):
    print("\n" + "=" * 72 + f"\n{title}\n  " + " ".join(args) + "\n" + "=" * 72, flush=True)
    r = subprocess.run([PY] + args, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"\nSTOPPED: '{title}' failed (exit {r.returncode}). See the output above.")


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists. Delete it to run again. Nothing was done.")
    t0 = time.time()
    os.makedirs(TAB)
    src = lambda name: os.path.join(ROOT, "src", name)

    run("Figures and the evaluation tables",
        [src("make_figures.py"), "--out", os.path.join(OUT, "report")])
    sweeps = os.path.join(OUT, "crossings_input")
    os.makedirs(sweeps)
    for name in CROSSING_INSTANCES:
        shutil.copy2(os.path.join(ROOT, "evaluation_dengue", f"exp_efficacy_{name}_full.csv"), sweeps)
    run("Crossings by instance",
        [src("make_crossings_table.py"), "--glob", os.path.join(sweeps, "exp_efficacy_*_full.csv"),
         "--tex", os.path.join(TAB, "crossings.tex")])
    run("Ward listing",
        [src("make_ward_table.py"), "--out", os.path.join(TAB, "wards.tex")])
    for param, metric, name in SURVEILLANCE:
        run(f"Surveillance sensitivity: {param}, {metric}",
            [src("make_surveillance_table.py"), "--param", param, "--metric", metric,
             "--tex", os.path.join(TAB, name + ".tex")])
    for name in FROM_SIMULATION:
        shutil.copy2(os.path.join(REF, name + ".tex"), os.path.join(TAB, name + ".tex"))
    print(f"\ncopied from report/tables (simulation output): {', '.join(FROM_SIMULATION)}")
    run("Combined appendix tables", [src("compact_tables.py")], cwd=OUT)

    print("\n" + "=" * 72 + "\nCHECK: each table the paper includes, against report/tables/\n" + "=" * 72)
    bad = 0
    for name in PAPER_TABLES:
        new_p, ref_p = os.path.join(TAB, name + ".tex"), os.path.join(REF, name + ".tex")
        if not os.path.exists(new_p):
            print(f"  MISSING    {name}.tex was not produced")
            bad += 1
            continue
        a = open(ref_p, encoding="utf-8").read().replace("\r\n", "\n").splitlines()
        b = open(new_p, encoding="utf-8").read().replace("\r\n", "\n").splitlines()
        if a == b:
            print(f"  SAME       {name}.tex")
        else:
            bad += 1
            print(f"  DIFFERENT  {name}.tex")
            for line in list(difflib.unified_diff(a, b, "paper", "reproduced", lineterm="", n=0))[:12]:
                print("      " + line)
    print(f"\n{len(PAPER_TABLES) - bad} of {len(PAPER_TABLES)} tables match the paper's copies. "
          f"Figures are in reproduced/report/figures/. Time: {(time.time() - t0) / 60:.1f} min.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
