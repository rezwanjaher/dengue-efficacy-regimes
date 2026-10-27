#!/usr/bin/env python3
"""Schedule shape at a fixed budget: sustained coverage against full-coverage
campaigns started in different weeks, across the efficacy grid.

    python src/exp_schedules.py --all-instances --tex-dir report/tables

Options:
    --instance        Instance name.
    --all-instances   Run every instance.
    --archetype_file  Parameter file.
    --archetype       Parameter set. Default Dengue_Dhaka_Advanced.
    --lambda_sigma    Risk-field dispersion. Default 0.8.
    --horizon         Season length in weeks. Default 20.
    --burst           campaign length T Default 6.
    --max-delay       Longest start delay tried, in weeks. Default 4.
    --trials          Epidemic seeds per point. Default 40.
    --eta-grid        Adulticide efficacies to sweep.
    --csv             Output CSV.
    --tex-dir         Write the LaTeX tables to this folder.
"""
import argparse
import copy
import csv
import datetime as _dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage_threshold import make_env, GRAPH_CONFIGS
from src.exp_all import run_traj

RUN_UTC = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
SCRIPT_VERSION = "1.0"

FULL_GRID = ("0.10,0.20,0.30,0.40,0.42,0.44,0.46,0.48,0.50,"
             "0.52,0.54,0.56,0.58,0.60,0.62,0.75,0.90")


def schedules(H, T, max_delay):
    out = [("sustained", [T / H] * H)]
    for tau in range(max_delay + 1):
        s = [0.0] * H
        for w in range(tau, min(tau + T, H)):
            s[w] = 1.0
        spent = sum(s)
        if abs(spent - T) > 1e-9:
            continue
        out.append((f"campaign@{tau}", s))
    half = T // 2
    if half > 0 and T == 2 * half:
        s = [0.0] * H
        mid = H // 2
        for w in range(0, half):
            s[w] = 1.0
        for w in range(mid, min(mid + half, H)):
            s[w] = 1.0
        if abs(sum(s) - T) < 1e-9:
            out.append(("split", s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    ap.add_argument("--all-instances", dest="all_instances", action="store_true")
    ap.add_argument("--archetype_file",
                    default=os.path.join("configs", "archetypes.json"))
    ap.add_argument("--archetype", default="Dengue_Dhaka_Advanced")
    ap.add_argument("--lambda_sigma", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--burst", type=int, default=6, help="campaign length T")
    ap.add_argument("--max-delay", dest="max_delay", type=int, default=4)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--eta-grid", dest="eta_grid", default=FULL_GRID)
    ap.add_argument("--csv", default=os.path.join("evaluation_dengue",
                                                  "exp_schedules.csv"))
    ap.add_argument("--tex-dir", dest="tex_dir", default=None,
                    help='Write the LaTeX tables to this folder.')
    a = ap.parse_args()

    if a.all_instances:
        names = sorted(GRAPH_CONFIGS)
    elif a.instance:
        if a.instance not in GRAPH_CONFIGS:
            print(f"unknown instance '{a.instance}'. Known: "
                  + ", ".join(sorted(GRAPH_CONFIGS)))
            sys.exit(1)
        names = [a.instance]
    else:
        print("pass --instance NAME or --all-instances")
        sys.exit(1)

    if os.path.exists(a.csv):
        print(f"{a.csv} already exists; not replacing it. Pass --csv with a "
              f"new name.")
        sys.exit(1)

    H, T = a.horizon, a.burst
    grid = [float(x) for x in a.eta_grid.split(",")]
    scheds = schedules(H, T, a.max_delay)
    rows = []

    for name in names:
        cfg = GRAPH_CONFIGS[name]
        env = make_env(a, cfg)
        env.reset(seed=0)
        base = env.ep_params["action_semantics"]["vector_control"]["efficacy"]

        print("=" * 92)
        print(f"SCHEDULE SHAPE AT FIXED BUDGET   {name}")
        print(f"  calibrated efficacy {base}; {T}-week campaign in a {H}-week "
              f"season; {a.trials} seeds")
        print(f"  every schedule spends {T} week-equivalents of full coverage")
        print("  policy-free: coverage written into the dynamics")
        print("=" * 92)
        print(f"  {'eta_F':>6}" + "".join(f"{s:>13}" for s, _ in scheds)
              + f"{'best':>14}")

        for eta in grid:
            p = copy.deepcopy(env.ep_params)
            p["action_semantics"]["vector_control"]["efficacy"] = eta
            untreated = float(np.mean([
                run_traj(env, s, H, [0.0] * H, ("vector_control",), p)["peak"]
                for s in range(a.trials)]))

            vals = {}
            for label, sched in scheds:
                pk = np.array([
                    run_traj(env, s, H, sched, ("vector_control",), p)["peak"]
                    for s in range(a.trials)])
                vals[label] = float(pk.mean())
                rows.append(dict(config=name, eta_F=eta, schedule=label,
                                 peak=round(vals[label], 1),
                                 untreated=round(untreated, 1),
                                 share_untreated=round(vals[label] / untreated, 5),
                                 sd=round(float(pk.std(ddof=1)), 1),
                                 trials=a.trials, horizon=H, campaign_weeks=T,
                                 script_version=SCRIPT_VERSION, run_utc=RUN_UTC))
            best = min(vals, key=vals.get)
            print(f"  {eta:>6.2f}"
                  + "".join(f"{vals[s]/untreated:>13.4f}" for s, _ in scheds)
                  + f"{best:>14}")
        print()

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows -> {a.csv}")

    if a.tex_dir:
        os.makedirs(a.tex_dir, exist_ok=True)
        labels = [lab for lab, _ in scheds]
        pretty = {"sustained": "sustained"}
        for lab in labels:
            if lab.startswith("campaign@"):
                pretty[lab] = f"week {int(lab.split('@')[1]) + 1}"
        pretty.setdefault("split", "split")

        for name in names:
            sub = [r for r in rows if r["config"] == name]
            etas = sorted({r["eta_F"] for r in sub})
            L = [r"\begin{table}[htbp]", r"\centering\small",
                 r"\caption{Mean peak as a share of the untreated peak at the "
                 r"same efficacy, " + name + r", every schedule at the same "
                 r"budget of six week-equivalents of full coverage. Campaign "
                 r"columns give the week the campaign starts; \emph{split} runs "
                 r"half at the start and half at mid-season.}",
                 r"\label{tab:schedules-" + name + "}",
                 r"\begin{tabular}{@{}l" + "c" * len(labels) + "@{}}",
                 r"\toprule",
                 "$\\eta_{F}$ & " + " & ".join(pretty[l] for l in labels)
                 + r" \\", r"\midrule"]
            for e in etas:
                row = {r["schedule"]: r["share_untreated"] for r in sub
                       if r["eta_F"] == e}
                L.append(f"{e:.2f} & "
                         + " & ".join(f"{row[l]:.4f}" for l in labels) + r" \\")
            L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            path = os.path.join(a.tex_dir, f"schedules_{name}.tex")
            open(path, "w").write("\n".join(L) + "\n")
            print(f"  wrote {path}")

        camp = [l for l in labels if l.startswith("campaign@")]
        etas = sorted({r["eta_F"] for r in rows})

        def cells_for(e):
            out = []
            for n in names:
                row = {r["schedule"]: r["share_untreated"] for r in rows
                       if r["config"] == n and r["eta_F"] == e}
                if not row:
                    out.append("")
                    continue
                best = min(row, key=row.get)
                out.append(str(int(best.split("@")[1]) + 1)
                           if best in camp else "--")
            return out

        half = (len(etas) + 1) // 2
        left, right = etas[:half], etas[half:]
        head = "$\\eta_{F}$ & " + " & ".join(
            f"{GRAPH_CONFIGS[n].get('n_wards', n)} wards" for n in names)
        spec = "l" + "c" * len(names)
        L = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Start week of the campaign leaving the lowest mean "
             r"peak, at a fixed budget. A dash marks an efficacy at which no "
             r"concentrated schedule is best.}",
             r"\label{tab:start-week}",
             r"\begin{tabular}{@{}" + spec + r"@{\qquad}" + spec + r"@{}}",
             r"\toprule", head + " & " + head + r" \\", r"\midrule"]
        for i in range(half):
            a_cells = [f"{left[i]:.2f}"] + cells_for(left[i])
            if i < len(right):
                b_cells = [f"{right[i]:.2f}"] + cells_for(right[i])
            else:
                b_cells = [""] * (1 + len(names))
            L.append(" & ".join(a_cells + b_cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        path = os.path.join(a.tex_dir, "start_week.tex")
        open(path, "w").write("\n".join(L) + "\n")
        print(f"  wrote {path}")
    print("\nREADING IT")
    print("  Every column is the mean peak as a share of the untreated peak at")
    print("  the same efficacy, so lower is better and the columns are")
    print("  comparable across efficacies and instances. Every schedule costs")
    print("  the same, so a difference between columns is the value of timing")
    print("  alone.")


if __name__ == "__main__":
    main()
