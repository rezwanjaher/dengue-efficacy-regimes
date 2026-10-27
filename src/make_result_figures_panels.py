#!/usr/bin/env python3
"""Figures for the crossings and the value of information. Called by
make_figures.py, or on its own:

    python src/make_result_figures_panels.py --dir evaluation_dengue --out report --force

Options:
    --dir              Folder with the evaluation results. Default
                       evaluation_dengue.
    --out              Output folder. Default report.
    --force            replace figures that already exist
    --wide, --autofit  Figure geometry, as in figstyle.py.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import (apply_style, configure, add_geometry_args, save_figure,
                      panel_legend, panel_title, panel_xlabel, panel_ylabel,
                      thin_ticks, differentiate, fig_cfg)

apply_style(True)

SHORT = {"H5_MyopicLP_Belief": "Myopic LP (belief)",
         "H4_Greedy_Belief": "Greedy (belief)",
         "H2_Test_and_Target": "Test-and-target",
         "H1_Blanket_Fogging": "Blanket fogging",
         "H5_MyopicLP_Oracle": "Myopic LP (oracle)"}

MIN_HEADROOM_SHARE = 0.005

FIGS = []


def save(name, out, caption, label, force):
    path = os.path.join(out, "figures", f"{name}.pdf")
    if os.path.exists(path) and not force:
        plt.close()
        print(f"  {path} exists; not replacing it (pass --force)")
        return
    for ax in plt.gcf().get_axes():
        differentiate(ax)
        thin_ticks(ax)
    include = save_figure(name, path)
    FIGS.append((f"figures/{name}", caption, label, include))
    print(f"  wrote {path}")


def read(path):
    if not os.path.exists(path):
        print(f"  {path} not found; skipping this figure")
        return None
    return pd.read_csv(path)


def fig_crossings(d, out, force):
    try:
        from src.make_crossings_table import collect
    except Exception as e:
        print(f"  cannot import make_crossings_table ({e}); skipping crossings")
        return
    try:
        rows = collect(os.path.join(d, "exp_efficacy*_full.csv"))
    except SystemExit:
        print("  no full-grid efficacy sweeps found; skipping crossings")
        return

    pts = []
    for r in rows:
        g = r.get("graph") or {}
        wards = g.get("n_wards") or r.get("n_wards")
        if not wards:
            continue
        for kind in ("concentration", "delay"):
            v = r.get(kind) or {}
            if "error" in v or v.get("first") is None:
                continue
            c = v["first"]
            pts.append({"instance": r["instance"], "wards": wards,
                        "deg": g.get("deg_median", np.nan), "kind": kind,
                        "lo": c["lo_eta"], "hi": c["hi_eta"],
                        "mid": 0.5 * (c["lo_eta"] + c["hi_eta"])})
    if not pts:
        print("  no crossings located; skipping crossings figure")
        return
    p = pd.DataFrame(pts)

    fig, ax = plt.subplots()
    for kind, marker, lab in (("delay", "o", "immediacy crossing"),
                              ("concentration", "s", "concentration crossing")):
        g = p[p.kind == kind]
        if not len(g):
            continue
        err = np.vstack([g["mid"] - g["lo"], g["hi"] - g["mid"]])
        ax.errorbar(g.wards, g["mid"], yerr=err, fmt=marker, ms=5, lw=0,
                    elinewidth=1.2, capsize=2.5, label=lab)
    weak = p[(p.deg <= 1) & (p.kind == "delay")]
    for _, r in weak.iterrows():
        ax.annotate(r["instance"], (r.wards, r["mid"]), textcoords="offset points",
                    xytext=(6, -3), fontsize=7, color="0.35")
    panel_xlabel(ax, "wards in the instance")
    panel_ylabel(ax, r"adulticide efficacy $\eta_F$")
    panel_legend(ax)
    save("crossings", out,
         "Efficacy at which each crossing occurs, against instance size. Bars "
         "span the adjacent grid points. Labelled instances are the two whose "
         "graphs are effectively disconnected.", "crossings", force)


def fig_voi(d, out, force):
    v = read(os.path.join(d, "voi_curve.csv"))
    if v is None:
        return
    fig, ax = plt.subplots()
    for inst, g in v.groupby("instance"):
        g = g.sort_values("eta_F")
        ax.plot(g.eta_F, 100 * g.gap_over_H0, marker="o", ms=4, lw=2,
                label=f"{inst}")
    ax.axvspan(0.40, 0.62, color="0.85", zorder=0)
    ax.set_yscale("log")
    panel_xlabel(ax, r"adulticide efficacy $\eta_F$")
    panel_ylabel(ax, "gap as \\% of untreated burden")
    panel_legend(ax)
    save("voi_curve", out,
         "Gap between a myopic LP on the true ward state and the same rule on "
         "its inferred state, as a share of untreated burden. The shaded band "
         "marks the range over which the gap falls monotonically; outside it "
         "the relationship reverses on both instances.", "voi_curve", force)


def fig_capture(d, out, force):
    c = read(os.path.join(d, "capture_curve.csv"))
    if c is None:
        return
    if "headroom_over_H0" in c.columns:
        keep = c[c.headroom_over_H0 >= MIN_HEADROOM_SHARE]
        dropped = len(c) - len(keep)
    else:
        keep, dropped = c, 0
    if not len(keep):
        print("  every capture point has negligible headroom; skipping figure")
        return

    fig, ax = plt.subplots()
    for inst, g in keep.groupby("instance"):
        g = g.sort_values("eta_F")
        ax.plot(g.eta_F, g.capture, marker="o", ms=4, lw=2, label=f"{inst}")
    ax.axhline(0.0, color="0.3", lw=1.2)
    ax.axhline(1.0, color="0.3", lw=1.0, ls=":")
    panel_xlabel(ax, r"adulticide efficacy $\eta_F$")
    panel_ylabel(ax, "fraction of the gap captured")
    panel_legend(ax)
    note = ("Zero marks a policy that only ties an information-free campaign; "
            "one marks a policy that reaches the LP on the true state. "
            "Negative values mean inferred information cost more than ignoring "
            "it.")
    if dropped:
        note += (f" {dropped} point(s) are omitted where the gap itself is "
                 f"within noise, which leaves the fraction undefined.")
    save("capture_curve", out,
         "Fraction of the information gap reached by the best policy that must "
         "infer ward state. " + note, "capture_curve", force)


def fig_information(d, out, force):
    c = read(os.path.join(d, "capture_curve.csv"))
    sw = read(os.path.join(d, "survey_sweep.csv"))
    if c is None or sw is None:
        print("  need both capture_curve.csv and survey_sweep.csv; "
              "skipping the information figure")
        return

    if "headroom_over_H0" in c.columns:
        keep = c[c.headroom_over_H0 >= MIN_HEADROOM_SHARE]
        dropped = len(c) - len(keep)
    else:
        keep, dropped = c, 0
    if not len(keep):
        print("  every capture point has negligible headroom; skipping figure")
        return

    insts = sorted(sw.instance.unique())
    eta = float(sw.eta_F.iloc[0])
    size = fig_cfg("information")["size"]
    fig, axes = plt.subplots(1, 1 + len(insts), figsize=size,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for inst, g in keep.groupby("instance"):
        g = g.sort_values("eta_F")
        ax.plot(g.eta_F, g.capture, label=f"{inst}")
    ax.axhline(0.0, color="0.3", lw=1.2)
    ax.axhline(1.0, color="0.3", lw=1.0, ls=":")
    panel_xlabel(ax, r"adulticide efficacy $\eta_F$")
    panel_ylabel(ax, "fraction of the gap captured")
    panel_title(ax, "gap captured")
    differentiate(ax)
    thin_ticks(ax)
    ax.legend(fontsize=plt.rcParams["legend.fontsize"])

    for ax, inst in zip(axes[1:], insts):
        g = sw[sw.instance == inst]
        for pol, gp in g.groupby("policy"):
            gp = gp.sort_values("survey_share")
            flat = gp.V.nunique() == 1
            ax.plot(gp.survey_share, gp.V / 1e9,
                    ls="--" if flat else "-", label=SHORT.get(pol, pol))
        panel_title(ax, inst)
        panel_xlabel(ax, "surveillance cap")
        differentiate(ax)
        thin_ticks(ax)
    panel_ylabel(axes[1], "objective $V$, billions BDT")

    n_panels = len(axes)
    x_mid = (1.0 / n_panels + 1.0) / 2.0
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(x_mid, -0.02), bbox_transform=fig.transFigure,
               ncol=3, frameon=False,
               fontsize=plt.rcParams["legend.fontsize"])

    note = ("Left: fraction of the information gap reached by the best policy "
            "that must infer ward state; zero marks a policy that only ties an "
            "information-free campaign, one marks a policy that reaches the LP "
            "on the true state, and negative values mean inferred information "
            "cost more than ignoring it.")
    if dropped:
        note += (f" {dropped} points are omitted where the gap itself is "
                 f"within noise, which leaves the fraction undefined.")
    note += (f" Centre and right: objective against the share of the weekly "
             f"budget surveillance may claim, at $\\eta_F = {eta:.2f}$, with "
             f"the policies that do not survey shown dashed and therefore flat.")
    save("information", out, note, "information", force)


def fig_survey(d, out, force):
    s = read(os.path.join(d, "survey_sweep.csv"))
    if s is None:
        return
    insts = sorted(s.instance.unique())
    fig, axes = plt.subplots(1, len(insts),
                             figsize=fig_cfg("survey_sweep")["size"],
                             sharey=False)
    axes = np.atleast_1d(axes)
    eta = float(s.eta_F.iloc[0])
    for ax, inst in zip(axes, insts):
        g = s[s.instance == inst]
        for p, gp in g.groupby("policy"):
            gp = gp.sort_values("survey_share")
            flat = gp.V.nunique() == 1
            ax.plot(gp.survey_share, gp.V / 1e9, marker="o", ms=4, lw=2,
                    ls="--" if flat else "-", label=SHORT.get(p, p))
        panel_title(ax, inst)
        panel_xlabel(ax, "surveillance cap")
    panel_ylabel(axes[0], "objective $V$, billions BDT")
    panel_legend(axes[-1])
    save("survey_sweep", out,
         f"Objective against the share of the weekly budget surveillance may "
         f"claim, at $\\eta_F = {eta:.2f}$. Dashed lines are the policies that "
         f"do not survey and are therefore unaffected. Every policy that infers "
         f"ward state gets worse as the cap rises.", "survey_sweep", force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="evaluation_dengue")
    ap.add_argument("--out", default="report")
    ap.add_argument("--force", action="store_true",
                    help="replace figures that already exist")
    add_geometry_args(ap)
    a = ap.parse_args()
    configure(a)

    print(f"reading {a.dir}/  writing {a.out}/figures/")
    fig_crossings(a.dir, a.out, a.force)
    fig_voi(a.dir, a.out, a.force)
    fig_information(a.dir, a.out, a.force)
    fig_capture(a.dir, a.out, a.force)
    fig_survey(a.dir, a.out, a.force)

    if not FIGS:
        print("\nnothing written.")
        return
    print("\n" + "=" * 72)
    print("PASTE INTO THE MANUSCRIPT")
    print("=" * 72)
    for f, cap, lab, iw in FIGS:
        print(r"\begin{figure}[htbp]\centering")
        print(rf"  \includegraphics[width={iw}\textwidth]{{report/{f}}}")
        print(rf"  \caption{{{cap}}}\label{{fig:{lab}}}")
        print(r"\end{figure}")
    print("=" * 72)


if __name__ == "__main__":
    main()
