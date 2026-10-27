#!/usr/bin/env python3
"""Value of ward-level information against efficacy: the myopic LP on the true
state against the same LP on its belief.

    python src/voi_curve.py --out evaluation_dengue/voi_curve.csv

Options:
    --dir        Folder with the evaluation results. Default
                 evaluation_dengue.
    --mult       budget multiplier to read (default 1.0) Default 1.0.
    --objective  name of the objective column, if detection picks wrong
    --out        write the curve to this CSV; refuses to replace one
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

WANT = {"free_heuristics": True, "survey_share": 0.25, "lambda_sigma": 0.8}

ORACLE = "H5_MyopicLP_Oracle"
BELIEF = "H5_MyopicLP_Belief"
BASE = "H0_Do_Nothing"

OBJ_CANDIDATES = ["V", "v", "objective", "obj", "V_mean", "mean_V",
                  "discounted_objective", "cum_disc_cost"]


def usable(meta_path):
    try:
        m = json.load(open(meta_path))
    except Exception:
        return None
    r = m.get("resolved", {}) or {}
    if r.get("eta_F_override") is None:
        return None
    if not (r.get("free_heuristics") is WANT["free_heuristics"]
            and r.get("survey_share") == WANT["survey_share"]
            and r.get("lambda_sigma") == WANT["lambda_sigma"]):
        return None
    d = os.path.dirname(meta_path)
    csv = os.path.join(d, m.get("csv") or "")
    if not os.path.exists(csv):
        return None
    return {"eta": float(r["eta_F_override"]), "csv": csv,
            "name": os.path.basename(csv)[:-4]}


def pick(cols, candidates, contains=()):
    for c in candidates:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    for c in cols:
        if any(t in c.lower() for t in contains):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="evaluation_dengue")
    ap.add_argument("--mult", type=float, default=1.0,
                    help="budget multiplier to read (default 1.0)")
    ap.add_argument("--objective", default=None,
                    help="name of the objective column, if detection picks wrong")
    ap.add_argument("--out", default=os.path.join("evaluation_dengue",
                                                   "voi_curve.csv"),
                    help="write the curve to this CSV; refuses to replace one")
    a = ap.parse_args()

    runs = [r for r in (usable(p) for p in
                        sorted(glob.glob(os.path.join(a.dir, "*.meta.json")))) if r]
    if not runs:
        print(f"no usable runs found in {a.dir}. Run src/eta_inventory.py.")
        sys.exit(1)

    byeta = {}
    for r in sorted(runs, key=lambda r: r["name"]):
        byeta.setdefault(round(r["eta"], 4), r)
    dropped = len(runs) - len(byeta)
    runs = [byeta[k] for k in sorted(byeta)]

    d0 = pd.read_csv(runs[0]["csv"])
    cols = list(d0.columns)

    pol = pick(cols, ["policy", "Policy"], contains=("policy",))
    mult = pick(cols, ["budget_mult", "budget_multiplier"], contains=("mult",))
    cfg = pick(cols, ["config", "instance"], contains=("config",))
    obj = a.objective or pick(cols, OBJ_CANDIDATES, contains=("objective",))

    missing = [n for n, c in (("policy", pol), ("budget multiplier", mult),
                              ("objective", obj)) if c is None]
    if missing:
        print("could not find: " + ", ".join(missing))
        print("\ncolumns present in " + os.path.basename(runs[0]["csv"]) + ":")
        for c in cols:
            print("  " + c)
        print("\nre-run with --objective <column> once you know which it is.")
        sys.exit(1)

    print(f"objective column: {obj}    policy: {pol}    multiplier: {mult}")
    if cfg:
        print(f"instance column : {cfg}")
    if dropped:
        print(f"{dropped} duplicate run(s) at an efficacy already covered were skipped")
    print()

    out = []
    for r in runs:
        d = pd.read_csv(r["csv"])
        d = d[abs(d[mult].astype(float) - a.mult) < 1e-9]
        groups = [(None, d)] if not cfg else list(d.groupby(cfg))
        for inst, g in groups:
            vals = {}
            for label, name in (("oracle", ORACLE), ("belief", BELIEF),
                                ("base", BASE)):
                rows = g[g[pol].astype(str) == name]
                vals[label] = float(rows[obj].astype(float).mean()) \
                    if len(rows) else float("nan")
                vals[label + "_n"] = len(rows)
            gap = vals["belief"] - vals["oracle"]
            frac = gap / vals["base"] if vals["base"] else float("nan")
            ratio = (vals["belief"] / vals["oracle"]) if vals["oracle"] else float("nan")
            out.append({"instance": inst or "all", "eta_F": r["eta"],
                        "V_oracle": vals["oracle"], "V_belief": vals["belief"],
                        "gap": gap, "gap_over_H0": frac, "belief_over_oracle": ratio,
                        "n_oracle": vals["oracle_n"], "n_belief": vals["belief_n"],
                        "n_base": vals["base_n"], "source": r["name"]})

    df = pd.DataFrame(out).sort_values(["instance", "eta_F"])

    missing_rows = df[(df.n_oracle == 0) | (df.n_belief == 0) | (df.n_base == 0)]
    if len(missing_rows):
        print("WARNING: some policy rows were not found. Policies present:")
        for p in sorted(set(d0[pol].astype(str))):
            print("  " + p)
        print()

    for inst, g in df.groupby("instance"):
        print("=" * 78)
        print(f"VALUE OF INFORMATION   instance {inst}   budget x{a.mult:g}")
        print("=" * 78)
        print(f"  {'eta_F':>7}{'V oracle':>16}{'V belief':>16}{'gap':>16}"
              f"{'gap / H0':>11}{'bel/orac':>10}")
        for _, r in g.iterrows():
            print(f"  {r.eta_F:>7.2f}{r.V_oracle:>16,.0f}{r.V_belief:>16,.0f}"
                  f"{r.gap:>16,.0f}{r.gap_over_H0:>10.2%}"
                  f"{r.belief_over_oracle:>10.2f}")
        neg = g[g.gap < 0]
        if len(neg):
            print(f"\n  WARNING: gap is negative at eta_F "
                  + ", ".join(f"{e:.2f}" for e in neg.eta_F)
                  + ".\n  The objective is a cost, so that says the belief rule beat")
            print("  the oracle. Check before reporting any of this.")
        lo, hi = g.gap_over_H0.idxmin(), g.gap_over_H0.idxmax()
        print(f"\n  highest at eta_F {g.loc[hi].eta_F:.2f} ({g.loc[hi].gap_over_H0:.2%}), "
              f"lowest at {g.loc[lo].eta_F:.2f} ({g.loc[lo].gap_over_H0:.2%}), "
              f"ratio {g.loc[hi].gap_over_H0 / g.loc[lo].gap_over_H0:.1f}x")
        mono = g.sort_values("eta_F").gap_over_H0.to_numpy()
        import numpy as _np
        if not _np.all(_np.diff(mono) <= 1e-12):
            turns = [float(g.sort_values("eta_F").eta_F.iloc[i + 1])
                     for i in range(len(mono) - 1) if mono[i + 1] > mono[i]]
            print("  not monotone: it rises again at eta_F "
                  + ", ".join(f"{t:.2f}" for t in turns))
        print()

    if a.out:
        if os.path.exists(a.out):
            print(f"{a.out} already exists; not replacing it.")
            return
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
