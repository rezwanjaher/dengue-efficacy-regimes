#!/usr/bin/env python3
"""Locate the concentration and immediacy crossings in an efficacy sweep. Used
by make_crossings_table.py, or on its own:

    python src/find_crossings.py --glob "evaluation_dengue/exp_efficacy_*_full.csv"

Options:
    --glob  Sweep files to read.
    --tex   also write a LaTeX table to this path
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

COLS = {
    "concentration": ("front_vs_sustained", None, None),
    "delay":         ("delay_1wk", "ci_lo", "ci_hi"),
}


def crossings(x, y, lo=None, hi=None, thresh=1.0):
    out = []
    for i in range(len(y) - 1):
        if y[i] < thresh <= y[i + 1]:
            span = y[i + 1] - y[i]
            frac = (thresh - y[i]) / span if abs(span) > 1e-12 else 0.5
            out.append(dict(lo_eta=x[i], hi_eta=x[i + 1],
                            interp=x[i] + frac * (x[i + 1] - x[i])))
    return out


def interval(x, y, lo, hi, thresh=1.0):
    if lo is None or hi is None:
        return None
    below = [x[i] for i in range(len(x)) if hi[i] < thresh]
    above = [x[i] for i in range(len(x)) if lo[i] > thresh]
    if not below or not above:
        return None
    return (max(below), min(a for a in above if a > max(below))) \
        if any(a > max(below) for a in above) else None


def analyse(path):
    d = pd.read_csv(path).sort_values("eta_F")
    cfg = str(d.config.iloc[0]) if "config" in d.columns else os.path.basename(path)
    x = d.eta_F.to_numpy(float)
    res = {"instance": cfg, "n_eta": len(x),
           "eta_min": x.min(), "eta_max": x.max(),
           "trials": int(d.trials.iloc[0]) if "trials" in d.columns else None,
           "n_wards": int(d.n_wards.iloc[0]) if "n_wards" in d.columns else None}
    for name, (yc, lc, hc) in COLS.items():
        if yc not in d.columns:
            res[name] = {"error": f"column {yc} absent"}
            continue
        y = d[yc].to_numpy(float)
        lo = d[lc].to_numpy(float) if lc and lc in d.columns else None
        hi = d[hc].to_numpy(float) if hc and hc in d.columns else None
        cs = crossings(x, y, lo, hi)
        res[name] = {
            "crossings": cs,
            "first": cs[0] if cs else None,
            "extra": cs[1:],
            "interval": interval(x, y, lo, hi),
            "has_ci": lo is not None,
            "monotone": bool(np.all(np.diff(y) >= -1e-9)),
        }
    return res


def fmt(c):
    if c is None:
        return "no crossing in the swept range"
    return f"[{c['lo_eta']:.2f}, {c['hi_eta']:.2f}]  interp {c['interp']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join("evaluation_dengue",
                                                   "exp_efficacy*.csv"))
    ap.add_argument("--tex", default=None,
                    help="also write a LaTeX table to this path")
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        print(f"no files matched {a.glob}")
        sys.exit(1)

    print("=" * 92)
    print("EFFICACY CROSSINGS")
    print("  concentration : front_vs_sustained rises through 1.0")
    print("                  (below it, spreading a budget beats concentrating it)")
    print("  delay         : delay_1wk rises through 1.0")
    print("                  (below it, starting later gives a lower peak)")
    print("=" * 92)

    rows = []
    for f in files:
        r = analyse(f)
        rows.append(r)
        print(f"\n  {r['instance']}   {r['n_wards'] or '?'} wards, "
              f"{r['n_eta']} efficacies over [{r['eta_min']:.2f}, {r['eta_max']:.2f}], "
              f"{r['trials'] or '?'} trials")
        print(f"    source {os.path.basename(f)}")
        for name in ("concentration", "delay"):
            v = r[name]
            if "error" in v:
                print(f"    {name:<15} {v['error']}")
                continue
            print(f"    {name:<15} {fmt(v['first'])}")
            if v["interval"]:
                print(f"    {'':<15} interval accounting for CIs: "
                      f"[{v['interval'][0]:.2f}, {v['interval'][1]:.2f}]")
            elif v["has_ci"]:
                print(f"    {'':<15} intervals overlap 1.0 throughout; "
                      f"no CI-based bracket")
            if v["extra"]:
                print(f"    {'':<15} ALSO re-crosses at "
                      + ", ".join(fmt(c) for c in v["extra"]))
            if not v["monotone"]:
                print(f"    {'':<15} note: this quantity is not monotone in eta_F")

    print("\n" + "=" * 92)
    print("READ IT")
    print("=" * 92)
    print("  Quote the CI-based interval where one exists; it is the honest")
    print("  statement. Quote the grid bracket otherwise, and say it is grid-limited.")
    print("  The interpolated value assumes local linearity and should not be")
    print("  quoted alone.")
    print("  A quantity flagged non-monotone needs a sentence in the text, or")
    print("  finer grid and more trials to establish whether the wobble is real.")

    if a.tex:
        os.makedirs(os.path.dirname(a.tex) or ".", exist_ok=True)
        L = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Efficacy at which each crossing occurs, by instance. "
             r"\emph{Concentration} is where a front-loaded campaign begins to "
             r"beat spreading the same expenditure; \emph{delay} is where "
             r"starting a week later begins to raise the realised peak. Brackets "
             r"are the adjacent grid points; where bootstrap intervals permit a "
             r"wider statement it is given instead.}",
             r"\label{tab:crossings}",
             r"\begin{tabular}{@{}lrrcc@{}}", r"\toprule",
             r"Instance & Wards & Median degree & Concentration & Delay \\",
             r"\midrule"]
        for r in rows:
            def cell(name):
                v = r[name]
                if "error" in v or v["first"] is None:
                    return "---"
                if v["interval"]:
                    return f"$[{v['interval'][0]:.2f}, {v['interval'][1]:.2f}]$"
                c = v["first"]
                return f"$[{c['lo_eta']:.2f}, {c['hi_eta']:.2f}]$"
            L.append(f"{r['instance'].replace('_',' ')} & {r['n_wards'] or ''} & "
                     f"\\needdeg & {cell('concentration')} & {cell('delay')} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(a.tex, "w").write("\n".join(L) + "\n")
        print(f"\nwrote {a.tex}")
        print("  NOTE: median degree is left as \\needdeg -- it is a property of the")
        print("  graph, not of the sweep. Fill it from make_instances.py output.")


if __name__ == "__main__":
    main()
