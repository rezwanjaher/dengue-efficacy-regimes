#!/usr/bin/env python3
"""Headroom and capture: the true-state LP against blanket adulticide, and the
share of that gap reached by policies that infer the state.

    python src/capture_curve.py --out evaluation_dengue/capture_curve.csv

Options:
    --dir        Folder with the evaluation results. Default
                 evaluation_dengue.
    --mult       Budget multiplier to read. Default 1.0.
    --objective  Objective column, if detection picks the wrong one.
    --out        Output path.
    --scale      divide V by this when printing (default: billions) Default
                 1000000000.0.
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voi_curve import usable, pick, OBJ_CANDIDATES

BLANKET = "H1_Blanket_Fogging"
ORACLE = "H5_MyopicLP_Oracle"
INFORMED = ["H2_Test_and_Target", "H4_Greedy_Belief", "H5_MyopicLP_Belief"]
SEPARATE = ["RL_Agent"]
BASE = "H0_Do_Nothing"

SHORT = {"H2_Test_and_Target": "test-and-target",
         "H4_Greedy_Belief": "greedy belief",
         "H5_MyopicLP_Belief": "LP belief"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="evaluation_dengue")
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--objective", default=None)
    ap.add_argument("--out", default=os.path.join("evaluation_dengue",
                                                  "capture_curve.csv"))
    ap.add_argument("--scale", type=float, default=1e9,
                    help="divide V by this when printing (default: billions)")
    a = ap.parse_args()

    runs = [r for r in (usable(p) for p in
                        sorted(glob.glob(os.path.join(a.dir, "*.meta.json")))) if r]
    if not runs:
        print(f"no usable runs in {a.dir}. Run src/eta_inventory.py first.")
        sys.exit(1)
    byeta = {}
    for r in sorted(runs, key=lambda r: r["name"]):
        byeta.setdefault(round(r["eta"], 4), r)
    runs = [byeta[k] for k in sorted(byeta)]

    d0 = pd.read_csv(runs[0]["csv"])
    cols = list(d0.columns)
    pol = pick(cols, ["policy", "Policy"], contains=("policy",))
    mult = pick(cols, ["budget_mult", "budget_multiplier"], contains=("mult",))
    cfg = pick(cols, ["config", "instance"], contains=("config",))
    obj = a.objective or pick(cols, OBJ_CANDIDATES, contains=("objective",))
    if pol is None or mult is None or obj is None:
        print("could not find policy, multiplier or objective column. Present:")
        for c in cols:
            print("  " + c)
        sys.exit(1)

    present = set(d0[pol].astype(str))
    wanted = [BLANKET, ORACLE, BASE] + INFORMED + SEPARATE
    absent = [p for p in wanted if p not in present]
    if absent:
        print("not found in the file: " + ", ".join(absent))
        print("policies present:")
        for p in sorted(present):
            print("  " + p)
        if BLANKET in absent or ORACLE in absent:
            sys.exit(1)
        print()

    print(f"objective {obj} (lower is better)   budget x{a.mult:g}   "
          f"{len(runs)} efficacies\n")

    out = []
    for r in runs:
        d = pd.read_csv(r["csv"])
        d = d[abs(d[mult].astype(float) - a.mult) < 1e-9]
        for inst, g in ([(None, d)] if not cfg else d.groupby(cfg)):
            def V(name):
                rows = g[g[pol].astype(str) == name]
                return float(rows[obj].astype(float).mean()) if len(rows) else float("nan")

            vb, vo, v0 = V(BLANKET), V(ORACLE), V(BASE)
            inf = {p: V(p) for p in INFORMED if p in present}
            best_p = min(inf, key=lambda p: inf[p]) if inf else None
            best = inf[best_p] if best_p else float("nan")
            headroom = vb - vo
            cap = (vb - best) / headroom if headroom else float("nan")
            row = {"instance": inst or "all", "eta_F": r["eta"],
                   "V_blanket": vb, "V_oracle": vo, "headroom": headroom,
                   "best_informed": best, "best_policy": best_p,
                   "capture": cap, "V_H0": v0,
                   "headroom_over_H0": headroom / v0 if v0 else float("nan"),
                   "source": r["name"]}
            for p in INFORMED + SEPARATE:
                if p in present:
                    row["V_" + p] = V(p)
            out.append(row)

    df = pd.DataFrame(out).sort_values(["instance", "eta_F"])
    s = a.scale

    for inst, g in df.groupby("instance"):
        print("=" * 86)
        print(f"CAPTURE OF THE INFORMATION GAP   instance {inst}")
        print("=" * 86)
        print(f"  {'eta_F':>7}{'blanket':>11}{'oracle':>11}{'headroom':>11}"
              f"{'best inf.':>11}{'capture':>10}  best policy")
        for _, r in g.iterrows():
            print(f"  {r.eta_F:>7.2f}{r.V_blanket / s:>11.3f}{r.V_oracle / s:>11.3f}"
                  f"{r.headroom / s:>11.3f}{r.best_informed / s:>11.3f}"
                  f"{r.capture:>10.2f}  {SHORT.get(r.best_policy, r.best_policy)}")
        pos = g[g.capture > 0]
        print()
        if not len(pos):
            print("  capture is negative or zero at every efficacy on this instance.")
            print("  No policy that infers the state beats spraying everywhere.")
            print("  The gap is a ceiling, not an investment case.")
        else:
            print("  capture is positive at eta_F "
                  + ", ".join(f"{e:.2f}" for e in pos.eta_F))
            print("  those are the points where inferred information paid for itself.")
        print()

    if any(c in df.columns for c in ("V_RL_Agent",)):
        print("  RL_Agent is reported in the CSV but excluded from best_informed:")
        print("  it was trained at one efficacy, so its figures elsewhere are")
        print("  behavioural observations, not like-for-like comparisons.\n")

    if a.out:
        if os.path.exists(a.out):
            print(f"{a.out} already exists; not replacing it.")
            return
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
