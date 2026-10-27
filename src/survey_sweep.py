#!/usr/bin/env python3
"""Effect of the surveillance cap on the belief-based policies at one efficacy.

    python src/survey_sweep.py --eta 0.45 --out evaluation_dengue/survey_sweep.csv

Options:
    --dir        Folder with the evaluation results. Default
                 evaluation_dengue.
    --eta        Adulticide efficacy. Default 0.45.
    --mult       Budget multiplier to read. Default 1.0.
    --objective  Objective column, if detection picks the wrong one.
    --out        Output path.
    --scale      Divide V by this when printing. Default 1000000000.0.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voi_curve import pick, OBJ_CANDIDATES

BLANKET = "H1_Blanket_Fogging"
ORACLE = "H5_MyopicLP_Oracle"
BELIEF = ["H5_MyopicLP_Belief", "H4_Greedy_Belief", "H2_Test_and_Target"]
SHORT = {"H5_MyopicLP_Belief": "LP belief", "H4_Greedy_Belief": "greedy belief",
         "H2_Test_and_Target": "test-and-target",
         BLANKET: "blanket", ORACLE: "LP oracle"}


def runs_at(d, eta, sigma=0.8):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.meta.json"))):
        try:
            m = json.load(open(p))
        except Exception:
            continue
        r = m.get("resolved", {}) or {}
        if r.get("eta_F_override") is None:
            continue
        if abs(float(r["eta_F_override"]) - eta) > 1e-9:
            continue
        if not r.get("free_heuristics") or r.get("lambda_sigma") != sigma:
            continue
        csv = os.path.join(d, m.get("csv") or "")
        if not os.path.exists(csv):
            continue
        share = float(r.get("survey_share", 1.0))
        name = os.path.basename(csv)[:-4]
        if share in out:
            print(f"  note: two runs at survey share {share:g} "
                  f"({out[share]['name']}, {name}); using the first")
            continue
        out[share] = {"csv": csv, "name": name}
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="evaluation_dengue")
    ap.add_argument("--eta", type=float, default=0.45)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--objective", default=None)
    ap.add_argument("--out", default=os.path.join("evaluation_dengue",
                                                  "survey_sweep.csv"))
    ap.add_argument("--scale", type=float, default=1e9)
    a = ap.parse_args()

    R = runs_at(a.dir, a.eta)
    if len(R) < 2:
        print(f"found {len(R)} run(s) at eta_F={a.eta:.2f}. Need at least two "
              f"survey shares to draw a curve.")
        sys.exit(1)

    first = pd.read_csv(next(iter(R.values()))["csv"])
    cols = list(first.columns)
    pol = pick(cols, ["policy"], contains=("policy",))
    mult = pick(cols, ["budget_mult", "budget_multiplier"], contains=("mult",))
    cfg = pick(cols, ["config", "instance"], contains=("config",))
    obj = a.objective or pick(cols, OBJ_CANDIDATES, contains=("objective",))
    spend = pick(cols, ["spendshare_survey"], contains=("spendshare_survey",))
    cov = pick(cols, ["cov_vector_control"], contains=("cov_vector_control",))
    if pol is None or mult is None or obj is None:
        print("could not find policy, multiplier or objective column. Present:")
        for c in cols:
            print("  " + c)
        sys.exit(1)

    rows = []
    for share, r in R.items():
        d = pd.read_csv(r["csv"])
        d = d[abs(d[mult].astype(float) - a.mult) < 1e-9]
        for inst, g in ([(None, d)] if not cfg else d.groupby(cfg)):
            for p in BELIEF + [BLANKET, ORACLE]:
                sel = g[g[pol].astype(str) == p]
                if not len(sel):
                    continue
                rows.append({
                    "instance": inst or "all", "eta_F": a.eta,
                    "survey_share": share, "policy": p,
                    "V": float(sel[obj].astype(float).mean()),
                    "survey_spend_share": (float(sel[spend].astype(float).mean())
                                           if spend else float("nan")),
                    "fog_coverage": (float(sel[cov].astype(float).mean())
                                     if cov else float("nan")),
                    "source": r["name"]})
    df = pd.DataFrame(rows)
    s = a.scale

    for inst, gi in df.groupby("instance"):
        print("=" * 84)
        print(f"SURVEY-SHARE SWEEP   instance {inst}   eta_F {a.eta:.2f}   "
              f"budget x{a.mult:g}")
        print("=" * 84)
        shares = sorted(gi.survey_share.unique())
        hdr = "".join(f"{x:>10.2f}" for x in shares)
        print(f"  {'V, billions':<20}{hdr}")
        for p in BELIEF + [BLANKET, ORACLE]:
            gp = gi[gi.policy == p].set_index("survey_share")
            if not len(gp):
                continue
            line = "".join(f"{gp.V.get(x, float('nan')) / s:>10.2f}" for x in shares)
            print(f"  {SHORT.get(p, p):<20}{line}")

        print(f"\n  {'survey share of spend':<20}{hdr}")
        for p in BELIEF:
            gp = gi[gi.policy == p].set_index("survey_share")
            if not len(gp):
                continue
            line = "".join(f"{gp.survey_spend_share.get(x, float('nan')):>10.3f}"
                           for x in shares)
            print(f"  {SHORT.get(p, p):<20}{line}")

        print(f"\n  {'fogging coverage':<20}{hdr}")
        for p in BELIEF:
            gp = gi[gi.policy == p].set_index("survey_share")
            if not len(gp):
                continue
            line = "".join(f"{gp.fog_coverage.get(x, float('nan')):>10.3f}"
                           for x in shares)
            print(f"  {SHORT.get(p, p):<20}{line}")

        print()
        for p in BELIEF:
            gp = gi[gi.policy == p].sort_values("survey_share")
            if len(gp) < 2:
                continue
            v = gp.V.to_numpy()
            if (v[1:] >= v[:-1] - 1e-9).all():
                verdict = "rises throughout: more surveillance costs more"
            elif (v[1:] <= v[:-1] + 1e-9).all():
                verdict = "falls throughout: more surveillance helps"
            else:
                verdict = "not monotone; read the row"
            print(f"  {SHORT.get(p, p):<20}{verdict}")
        print()

    print("  A rising V means surveillance is drawing budget away from the")
    print("  coverage that suppresses transmission. Belief accuracy is not")
    print("  logged, so this shows displaced spend, not a belief that failed")
    print("  to improve. Do not claim the second from this table.")

    if a.out:
        if os.path.exists(a.out):
            print(f"\n{a.out} already exists; not replacing it.")
            return
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
