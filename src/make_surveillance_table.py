#!/usr/bin/env python3
r"""Surveillance sensitivity tables, from the logs of sweep_surveillance.py.

    python src/make_surveillance_table.py --param eta_S --metric V \
        --tex report/tables/sens_surveillance.tex

Options:
    --dir     Folder with the evaluation results.
    --param   restrict to one parameter
    --eta-f   Restrict to one pinned efficacy, e.g. 0.45.
    --metric  V, the discounted objective, or peak, the mean peak. One of: V,
              peak. Default V.
    --tex     Write the LaTeX table here.
"""
import argparse
import glob
import os
import re

UNTREATED = "H0_Do_Nothing"
BLANKET = "H1_Blanket_Fogging"
PRETTY = {"eta_S": "surveillance detection $\\eta_{S}$",
          "surv_cost": "the surveillance unit cost, BDT per person",
          "eta_C": "case-management efficacy $\\eta_{C}$"}

BELIEF = ["H2_Test_and_Target", "H4_Greedy_Belief", "H5_MyopicLP_Belief"]
ORACLE = "H5_MyopicLP_Oracle"

SECTIONS = {
    "V":    "CUMULATIVE DISCOUNTED ECONOMIC COST",
    "peak": "BUDGET SWEEP",
}
ROW = re.compile(r"^([A-Za-z0-9_]+)\s+((?:-?[\d,]+(?:\.\d+)?\s*\*?\s*)+)$")
FNAME = re.compile(r"^(?P<param>[A-Za-z_]+)_(?P<value>[\d.]+)"
                   r"(?:_etaF(?P<eta>[\d.]+))?\.log$")


WANT_LAMBDA = "lambda_sigma=0.8"
WANT_MODELS = "DENGUE_DHAKA_V18"


def protocol_problem(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.read(4000)
    if WANT_LAMBDA not in head:
        m = re.search(r"lambda_sigma=([\d.]+)", head)
        return f"run at lambda_sigma={m.group(1) if m else '?'}, paper uses 0.8"
    if WANT_MODELS not in head:
        return "not run with the V18 checkpoints"
    return None


def parse(path, metric="V"):
    SECTION = SECTIONS[metric]
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\r\n") for l in f]
    try:
        start = next(i for i, l in enumerate(lines) if SECTION in l)
    except StopIteration:
        return None
    out, started = {}, False
    for l in lines[start:start + 60]:
        t = l.strip()
        if set(t) == {"="}:
            continue
        if t.startswith("---"):
            if started:
                break
            started = True
            continue
        if not started:
            continue
        m = ROW.match(l.strip())
        if m:
            out[m.group(1)] = float(m.group(2).split()[0].replace(",", "").rstrip("*"))
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("evaluation_dengue", "sweeps_ls08"))
    ap.add_argument("--param", default=None, help="restrict to one parameter")
    ap.add_argument("--eta-f", dest="eta_f", default=None,
                    help='Restrict to one pinned efficacy, e.g. 0.45.')
    ap.add_argument("--metric", choices=["V", "peak"], default="V",
                    help='V, the discounted objective, or peak, the mean peak.')
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.dir, "*.log"))):
        m = FNAME.match(os.path.basename(path))
        if not m:
            continue
        if a.param and m.group("param") != a.param:
            continue
        if a.eta_f and (m.group("eta") or "") != a.eta_f:
            continue
        why = protocol_problem(path)
        if why:
            print(f"  skipped {os.path.basename(path)}: {why}")
            continue
        d = parse(path, a.metric)
        if not d or BLANKET not in d or UNTREATED not in d:
            print(f"  skipped {os.path.basename(path)}: no objective table")
            continue
        avail = {k: d[k] for k in BELIEF if k in d}
        if not avail:
            print(f"  skipped {os.path.basename(path)}: no belief policy")
            continue
        best = min(avail, key=avail.get)
        rows.append(dict(metric=a.metric, param=m.group("param"), value=float(m.group("value")),
                         eta_f=m.group("eta") or "default",
                         untreated=d[UNTREATED], blanket=d[BLANKET],
                         best=avail[best], best_policy=best,
                         oracle=d.get(ORACLE),
                         gap=(d[BLANKET] - avail[best]) / d[UNTREATED]))
    if not rows:
        print(f"no usable logs in {a.dir}")
        return
    rows.sort(key=lambda r: (r["param"], r["eta_f"], r["value"]))

    print("=" * 86)
    print(f"SURVEILLANCE SENSITIVITY, appropriated budget, metric = {a.metric}")
    print("  gap = (uniform - best inferring policy) / untreated, negative "
          "means uniform wins")
    print("=" * 86)
    print(f"  {'param':>10}{'value':>8}{'eta_F':>8}{'uniform':>10}"
          f"{'best':>10}{'gap':>9}   best policy")
    for r in rows:
        print(f"  {r['param']:>10}{r['value']:>8}{r['eta_f']:>8}"
              + (f"{r['blanket']:>10.3f}{r['best']:>10.3f}"
                 if a.metric == "V" else
                 f"{r['blanket']:>10,.0f}{r['best']:>10,.0f}")
              + f"{r['gap']:>+8.2%}"
              f"   {r['best_policy'].replace('_',' ')}")

    if a.tex:
        os.makedirs(os.path.dirname(a.tex) or ".", exist_ok=True)
        if os.path.exists(a.tex):
            print(f"\n{a.tex} exists; not replacing it.")
            return
        L = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Sensitivity of the information result to "
             + PRETTY.get(rows[0]["param"], rows[0]["param"])
             + r", at the appropriated budget. Columns give the "
             + ("discounted objective in billions BDT" if a.metric == "V"
                else "mean peak infections")
             + r". \emph{Gap} is uniform spraying less the best policy that "
               r"must infer ward state, as a share of untreated burden; "
               r"negative means uniform spraying wins.}",
             r"\label{tab:sens-" + rows[0]["param"].replace("_", "-")
             + ("" if a.metric == "V" else "-peak") + "}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"$\eta_{F}$ & Value & Uniform & Best inferring & Gap \\",
             r"\midrule"]
        num = "{:.3f}" if a.metric == "V" else "{:,.0f}"
        for r in rows:
            L.append(f"{r['eta_f']} & {r['value']} & "
                     + num.format(r['blanket']) + " & " + num.format(r['best'])
                     + f" & {r['gap']*100:+.2f}\\% \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(a.tex, "w").write("\n".join(L) + "\n")
        print(f"\nwrote {a.tex}")


if __name__ == "__main__":
    main()
