#!/usr/bin/env python3
"""Tables and figures from the evaluation results. Called by make_figures.py,
or on its own:

    python src/make_report_panels.py --dir evaluation_dengue --out report --no-compile

Options:
    --version_tag      File prefix of the results, e.g. v18. Default v18.
    --dir              Folder with the evaluation results. Default
                       evaluation_dengue.
    --out              Output folder. Default report.
    --no-compile       Skip building the PDF.
    --paper            Manuscript mode: short captions and only the outputs
                       the paper uses.
    --wide, --autofit  Figure geometry, as in figstyle.py.
"""
import argparse, os, sys, glob, subprocess
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import (apply_style, configure, add_geometry_args, save_figure,
                      panel_legend, panel_title, panel_xlabel, panel_ylabel,
                      thin_ticks, differentiate)

apply_style(True)

SHORT = {
    "H0_Do_Nothing": "Do nothing", "H1_Blanket_Fogging": "Blanket fogging",
    "H2_Test_and_Target": "Test-and-target", "H3_Random_Larviciding": "Random larviciding",
    "H4_Greedy_Belief": "Greedy (belief)", "H4_Greedy_Oracle": "Greedy (oracle)",
    "H5_MyopicLP_Belief": "Myopic LP (belief)", "H5_MyopicLP_Oracle": "Myopic LP (oracle)",
    "RL_Agent": "RL agent",
}
ORDER = ["H0_Do_Nothing", "H1_Blanket_Fogging", "H2_Test_and_Target",
         "H3_Random_Larviciding", "H4_Greedy_Belief", "H5_MyopicLP_Belief",
         "H4_Greedy_Oracle", "H5_MyopicLP_Oracle", "RL_Agent"]
BELIEF = [p for p in ORDER if "Oracle" not in p]
GRAPHS = {}


def discover_graphs(frames):
    out = {}
    for df in frames:
        if df is None or "config" not in df.columns:
            continue
        for cfg, grp in df.groupby("config"):
            if "n_wards" in grp.columns and grp.n_wards.notna().any():
                out[cfg] = f"{int(grp.n_wards.dropna().iloc[0])} wards"
            else:
                out.setdefault(cfg, str(cfg))
    return dict(sorted(out.items(), key=lambda kv: -int(kv[1].split()[0])
                       if kv[1].split()[0].isdigit() else 0))
CELLS = [("paced", "paced", "uncapped"), ("paced_cap25", "paced", "capped"),
         ("free", "free", "uncapped"), ("free_cap25", "free", "capped")]

NOTES, TEX_FILES, FIG_FILES = [], [], []


def note(s):
    NOTES.append(s)
    print("  " + s)


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def order_policies(idx):
    return [p for p in ORDER if p in list(idx)] + [p for p in idx if p not in ORDER]


def write_table(name, df, caption, label, out, note_txt=None,
                index_name="Policy", float_fmt="{:.3f}"):
    path = os.path.join(out, "tables", f"{name}.tex")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ncol = len(df.columns)
    body = []
    body.append(r"\begin{table}[htbp]")
    body.append(r"\centering\small")
    body.append(rf"\caption{{{caption}}}")
    body.append(rf"\label{{tab:{label}}}")
    body.append(r"\begin{tabular}{l" + "r" * ncol + "}")
    body.append(r"\toprule")
    body.append(esc(index_name) + " & " + " & ".join(esc(c) for c in df.columns) + r" \\")
    body.append(r"\midrule")
    for i, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, float) and np.isfinite(v):
                cells.append(float_fmt.format(v))
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("--")
            else:
                cells.append(esc(v))
        cells = [c.replace("nan", "--") for c in cells]
        body.append(esc(SHORT.get(i, i)) + " & " + " & ".join(cells) + r" \\")
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    if note_txt:
        body.append(rf"\\[2pt]\parbox{{0.92\textwidth}}{{\footnotesize {note_txt}}}")
    body.append(r"\end{table}")
    open(path, "w").write("\n".join(body) + "\n")
    TEX_FILES.append(f"tables/{name}")
    return path


def save_fig(name, out, caption, label):
    for ax in plt.gcf().get_axes():
        differentiate(ax)
        thin_ticks(ax)
    path = os.path.join(out, "figures", f"{name}.pdf")
    include = save_figure(name, path)
    FIG_FILES.append((f"figures/{name}", caption, label, include))
    return path


def load(d, tag, stem, episodes=False):
    suff = "_episodes" if episodes else ""
    for cand in (f"{tag}_{stem}{suff}.csv", f"{tag.upper()}_{stem}{suff}.csv",
                 f"{stem}{suff}.csv"):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def sec_headline(cells, out):
    made = 0
    for metric, col, cap, fmt in [
            ("peak", "mean_peak", "Mean peak infections", "{:,.0f}"),
            ("V", "cum_disc_cost", "Discounted objective $V$ (billions BDT, lower is better)",
             "{:.3f}")]:
        for g, gl in GRAPHS.items():
            key = "free_cap25" if "free_cap25" in cells else list(cells)[0]
            df = cells.get(key)
            if df is None:
                continue
            s = df[(df.config == g)]
            s = s[(s.seed.isna()) | (s.seed.astype(str) == "mean")]
            if s.empty:
                continue
            piv = s.pivot_table(index="policy", columns="budget_mult", values=col)
            if metric == "V":
                piv = piv / 1e9
            piv = piv.loc[order_policies(piv.index)]
            piv.columns = [f"$\\times${c:g}" for c in piv.columns]
            write_table(f"headline_{metric}_{g}", piv,
                        f"{cap}, {gl}. Matched protocol (baselines free, surveillance capped).",
                        f"headline_{metric}_{g}", out, float_fmt=fmt,
                        note_txt="Every policy faces the same budget and the same "
                                 "evaluation seeds. Oracle rows see the true state "
                                 "for free; they are references, not competitors or bounds.")
            made += 1
    if not made:
        note("no protocol CSVs found -- headline tables skipped")


def sec_protocol_matrix(eps, out):
    rows = []
    for g, gl in GRAPHS.items():
        for stem, pace, cap in CELLS:
            d = eps.get(stem)
            if d is None:
                continue
            s = d[(d.config == g) & (d.budget_mult == 1.0)]
            if s.empty:
                continue
            m = s.groupby("policy").cum_disc_cost.mean()
            if not {"H5_MyopicLP_Belief", "RL_Agent", "H0_Do_Nothing"} <= set(m.index):
                continue
            rows.append(dict(graph=gl, pacing=pace, survey=cap,
                             vo=(m["H5_MyopicLP_Belief"] - m["RL_Agent"]) / m["H0_Do_Nothing"]))
    if not rows:
        note("protocol matrix skipped (need the four *_episodes.csv)")
        return
    r = pd.DataFrame(rows)
    piv = r.pivot_table(index=["graph", "survey"], columns="pacing", values="vo")
    piv.index = [f"{a}, surveillance {b}" for a, b in piv.index]
    piv.columns = [f"baselines {c}" for c in piv.columns]
    write_table("protocol_matrix", piv,
                "Measured advantage of the learned controller over the myopic LP at the appropriated "
                "budget, under four evaluation protocols, as a fraction of "
                "untreated burden.",
                "protocol_matrix", out, index_name="Instance",
                note_txt="Same agent, same baselines, same data. Only the rules the "
                         "baselines play under differ. The matched comparison, "
                         "reported in the paper, is baselines free with surveillance "
                         "capped; the others show how far the measurement moves when "
                         "the baseline is paced or allowed to spend everything on "
                         "surveillance.")

    fig, ax = plt.subplots()
    for i, (idx, row) in enumerate(piv.iterrows()):
        ax.bar(np.arange(len(row)) + i * 0.35, row.values, 0.33, label=idx)
    ax.set_xticks(np.arange(len(piv.columns)) + 0.18)
    ax.set_xticklabels(piv.columns)
    panel_ylabel(ax, "measured advantage")
    panel_legend(ax)
    panel_title(ax, "How much of the measured advantage is the evaluation protocol")
    save_fig("protocol_matrix", out,
             "Measured advantage of the learned policy over the myopic baseline, "
             "at the appropriated budget, under four evaluation protocols. "
             "\\textbf{How to read:} taller means the learned policy looks better. "
             "The agent, the baselines and the disease are identical in all four "
             "bars -- only the rules the baseline plays under change. `Paced' means "
             "the baseline must spend in equal weekly instalments; `uncapped' means "
             "it may pour the whole budget into surveying instead of spraying. The "
             "gap between the tallest and shortest bar is measurement, not "
             "performance.", "protocol_matrix")


def sec_budget_curves(cells, out):
    for g, gl in GRAPHS.items():
        df = cells.get("free_cap25")
        if df is None:
            continue
        s = df[df.config == g]
        s = s[(s.seed.isna()) | (s.seed.astype(str) == "mean")]
        if s.empty:
            continue
        for col, ylab, logy, nm in [("mean_peak", "mean peak infections", True, "peak"),
                                    ("cum_disc_cost", "objective $V$ (BDT)", False, "V"),
                                    ("budget_util", "fraction of budget spent", False, "util")]:
            REFERENCE = ["H0_Do_Nothing", "H3_Random_Larviciding"]
            excluded = {}
            for rp in REFERENCE:
                r = s[s.policy == rp]
                if len(r):
                    excluded[SHORT.get(rp, rp)] = (float(r[col].min()),
                                                   float(r[col].max()))
            fig, ax = plt.subplots()
            for p in order_policies(s.policy.unique()):
                if p in REFERENCE:
                    continue
                q = s[s.policy == p].sort_values("budget_mult")
                if q.empty:
                    continue
                style = dict(lw=2.2, marker="o", ms=3.5)
                if p == "RL_Agent":
                    style.update(color="k", lw=2.8, zorder=5)
                elif "Oracle" in p:
                    style.update(ls="--", alpha=0.55)
                ax.plot(q.budget_mult, q[col], label=SHORT.get(p, p), **style)
            shown = s[~s.policy.isin(REFERENCE)][col]
            shown = shown[shown > 0]
            span = (shown.max() / shown.min()) if len(shown) and shown.min() > 0 else 1.0
            if span > 8.0:
                ax.set_yscale("log")
                scale_note = " The vertical scale is logarithmic."
            else:
                scale_note = (f" The vertical scale is linear: the policies shown "
                              f"differ by at most {span:.1f}$\\times$, and a "
                              f"logarithmic axis would hide that.")
            panel_xlabel(ax, "budget (multiple of the city's actual season budget)")
            panel_ylabel(ax, ylab)
            panel_title(ax, f"{ylab} vs budget, {gl}")
            panel_legend(ax)
            KEY = {
 "peak": "\\textbf{How to read:} lower is better. The horizontal axis is how much "
         "money the city has, as a multiple of its real season budget, so $\\times$1 "
         "is the actual budget. The black line is the learned policy. Dashed lines "
         "are oracle policies that see the true infection state for free -- they are "
         "references, not bounds and not fair competitors. Watch "
         "where the lines converge: past that budget, method choice stops mattering.",
 "V": "\\textbf{How to read:} lower is better. This counts illness and money "
      "together. A line that RISES with budget is a policy being charged for "
      "spending, which is the objective problem, not the policy failing.",
 "util": "\\textbf{How to read:} 1.0 means the policy spent everything it was "
         "given. Because the budget is ring-fenced, money left unspent helps "
         "nobody, so a low line is unused capacity rather than thrift."}[nm]
            if excluded and nm != "util":
                bits = []
                for nmx, (lo, hi) in excluded.items():
                    if nm == "peak":
                        v = f"{lo:,.0f}" if abs(hi - lo) < 1 else f"{lo:,.0f} to {hi:,.0f}"
                    else:
                        v = (f"{lo/1e9:.2f}" if abs(hi - lo) < 1e7
                             else f"{lo/1e9:.2f} to {hi/1e9:.2f}") + " billion"
                    bits.append(f"{nmx.lower()} gives {v}")
                ref = (" Two reference policies are left off the chart -- "
                       + "; ".join(bits) + " -- because they are there to show what "
                       "failure looks like, not to compete, and plotting them "
                       "squashes the real contenders into an unreadable band. Both "
                       "remain in every table.")
            else:
                ref = ""
            save_fig(f"budget_{nm}_{g}", out,
                     f"{ylab} against budget, {gl}. All policies face the same "
                     f"budget, the same disease, and the same evaluation seeds."
                     + ref + scale_note + " " + KEY, f"budget_{nm}_{g}")


def sec_learning(d, out):
    lc = None
    for c in ("learning_curve.csv",):
        p = os.path.join(d, c)
        if os.path.exists(p):
            lc = pd.read_csv(p)
    if lc is None:
        note("learning_curve.csv not found -- convergence figures skipped")
        return
    cfg_col = "config" if "config" in lc.columns else None
    step_col = next((c for c in ("steps", "step", "timesteps") if c in lc.columns), None)
    val_col = next((c for c in ("V", "cum_disc_cost", "value") if c in lc.columns), None)
    seed_col = next((c for c in ("seed", "train_seed") if c in lc.columns), None)
    if not all([cfg_col, step_col, val_col, seed_col]):
        note(f"learning_curve.csv has unexpected columns {list(lc.columns)} -- skipped")
        return
    for g, gl in GRAPHS.items():
        s = lc[lc[cfg_col] == g]
        if s.empty:
            continue
        seeds = sorted(s[seed_col].unique())
        steps = np.sort(s[step_col].unique())
        gap = float(np.median(np.diff(steps))) if len(steps) > 1 else 5e5
        spread = 0.55 * gap
        fig, ax = plt.subplots()
        for i, sd in enumerate(seeds):
            q = s[s[seed_col] == sd].sort_values(step_col)
            off = (i - (len(seeds) - 1) / 2.0) * (spread / max(len(seeds), 1))
            ax.plot((q[step_col] + off) / 1e6, q[val_col] / 1e9,
                    marker="o", ms=3.5, lw=1.8, alpha=0.9, label=f"seed {sd}")
        for x in steps:
            ax.axvline(x / 1e6, color="0.85", lw=0.6, zorder=0)
        panel_xlabel(ax, "training steps (millions)")
        panel_ylabel(ax, "objective $V$ (billions BDT, lower is better)")
        panel_title(ax, f"Training convergence, {gl}")
        panel_legend(ax)
        save_fig(f"convergence_{g}", out,
                 f"Training progress, {gl}, one line per training seed. "
                 f"\\textbf{{How to read:}} lower is better, and the horizontal axis "
                 f"is how long the policy trained. A line still falling at the right "
                 f"edge means the run stopped too early and its results understate "
                 f"the method. Flat tails mean training had finished. Lines that end "
                 f"close together mean the result does not depend on the random "
                 f"seed. Seeds are nudged sideways at each checkpoint so they can be "
                 f"told apart; the faint vertical lines mark the true training step "
                 f"all four share.",
                 f"convergence_{g}")

        piv = s.pivot_table(index=seed_col, columns=step_col, values=val_col) / 1e9
        piv.columns = [f"{c/1e6:g}M" for c in piv.columns]
        write_table(f"convergence_{g}", piv,
                    f"Objective $V$ (billions BDT) against training steps, {gl}.",
                    f"convergence_{g}", out, index_name="Seed")


def sec_sweeps(d, tag, out):
    ct = load(d, tag, "coverage_threshold")
    tm = load(d, tag, "threshold_timing")
    hz = load(d, tag, "threshold_horizon")

    if ct is not None:
        for g, gl in GRAPHS.items():
            for mode, xl, nm in [("sustained", "coverage held all season", "sustained"),
                                 ("burst", "weeks of full coverage from week 1", "burst")]:
                s = ct[(ct.config == g) & (ct["mode"] == mode)]
                if s.empty:
                    continue
                fig, ax = plt.subplots()
                for ch in s.channel.unique():
                    q = s[s.channel == ch].sort_values("u" if mode == "sustained" else "weeks")
                    x = q.u if mode == "sustained" else q.weeks
                    ax.plot(x, q.peak, marker="o", ms=3.5, lw=2, label=ch.replace("_", " "))
                ax.set_yscale("log")
                panel_xlabel(ax, xl)
                panel_ylabel(ax, "peak infections")
                panel_title(ax, f"{'Sustained' if mode=='sustained' else 'Front-loaded'} "
                             f"control, {gl}")
                panel_legend(ax)
                extra = ("\\textbf{How to read:} lower is better. Each line is one "
                         "kind of intervention. Moving right means doing more of it. "
                         "A line that flattens has stopped buying anything: past that "
                         "point extra effort is wasted. A line that never comes down "
                         "to the others cannot solve the problem however much is "
                         "spent." if nm == "sustained" else
                         "\\textbf{How to read:} lower is better. Moving right means "
                         "spraying at full strength for more weeks at the start of "
                         "the season, then stopping. The elbow is the number of weeks "
                         "that actually finishes the job; weeks beyond it change "
                         "nothing.")
                save_fig(f"sweep_{nm}_{g}", out,
                         f"Peak infections against {xl}, {gl}. No policy is involved "
                         f"here -- coverage is written straight into the disease "
                         f"model, so nothing about the agent, the baselines or the "
                         f"budget code can affect it. " + extra, f"sweep_{nm}_{g}")

        TOL = 1.01
        rows = []
        for g, gl in GRAPHS.items():
            for ch in sorted(ct.channel.unique()):
                bur = ct[(ct.config == g) & (ct.channel == ch)
                         & (ct["mode"] == "burst") & (ct.weeks > 0)
                         ].sort_values("weeks")
                sus = ct[(ct.config == g) & (ct.channel == ch)
                         & (ct["mode"] == "sustained") & (ct.u > 0)
                         ].sort_values("u")
                if bur.empty or sus.empty:
                    continue

                floor = float(bur.peak.min())
                reached = bur[bur.peak <= floor * TOL]
                kstar = int(reached.weeks.iloc[0])
                front = float(reached.spend_x_budget.iloc[0])
                flat = kstar < int(bur.weeks.max())

                match = sus[sus.peak <= floor * TOL]
                sustained = float(match.spend_x_budget.iloc[0]) if len(match) else np.nan

                rows.append({
                    "Instance": gl,
                    "Channel": ch.replace("_", " "),
                    "Weeks to floor": (f"{kstar}" if flat else f"$>{kstar - 1}$"),
                    "Peak at floor": floor,
                    "Campaign cost": front,
                    "Sustained cost": sustained,
                })
        r = pd.DataFrame(rows)
        r["Ratio"] = r["Sustained cost"] / r["Campaign cost"]
        r["Peak at floor"] = r["Peak at floor"].map(lambda v: f"{v:,.0f}")
        r["Channel"] = r["Channel"].replace({"both 50 50": "adulticide + larviciding",
                                             "fogging": "adulticide"})
        r = r.set_index("Instance")
        write_table("threshold_summary", r,
                    "Cost of reaching the same peak by each route, as a multiple "
                    "of the season budget.",
                    "threshold_summary", out, index_name="Instance",
                    note_txt="\\emph{Weeks to floor} is the shortest week-1 "
                             "campaign whose peak is within 1\\% of the lowest "
                             "reached at full coverage; $>11$ means it was still "
                             "falling at the longest swept. \\emph{Sustained cost} "
                             "is the cheapest all-season coverage reaching that "
                             "same peak, blank where none does.",
                    float_fmt="{:.2f}")

    if tm is not None and "start_week" in tm.columns:
        t = tm[tm["mode"] == "timing"] if "timing" in set(tm["mode"]) else tm.dropna(subset=["start_week"])
        for g, gl in GRAPHS.items():
            s = t[t.config == g]
            if s.empty:
                continue
            fig, ax = plt.subplots()
            for ch in s.channel.unique():
                q = s[s.channel == ch].sort_values("start_week")
                ax.plot(q.start_week, q.peak, marker="o", ms=3.5, lw=2,
                        label=ch.replace("_", " "))
            ax.set_yscale("log")
            panel_xlabel(ax, "week the campaign starts")
            panel_ylabel(ax, "peak infections")
            panel_title(ax, f"Cost of starting late, {gl} (identical spend at every start)")
            panel_legend(ax)
            save_fig(f"timing_{g}", out,
                     f"The cost of starting late, {gl}. \\textbf{{How to read:}} lower "
                     f"is better. Every point spends exactly the same money on exactly "
                     f"the same campaign; the only thing that changes is which week it "
                     f"begins. A rising line means delay alone costs cases, with no "
                     f"saving to show for it. The vertical scale is logarithmic, so a "
                     f"straight rising line means each week of delay multiplies the "
                     f"peak rather than adding to it.", f"timing_{g}")

            fog = s[s.channel == "fogging"].sort_values("start_week")
            if len(fog):
                base = fog.peak.iloc[0]
                tab = pd.DataFrame({
                    "Peak": fog.peak.values,
                    "Relative to starting in week 1": fog.peak.values / base,
                }, index=[f"week {int(w)}" for w in fog.start_week])
                write_table(f"timing_{g}", tab,
                            f"Effect of delay at constant spend, adulticide, {gl}.",
                            f"timing_{g}", out, index_name="Campaign starts",
                            float_fmt="{:,.1f}")

    if hz is not None:
        h = hz[hz["mode"] == "horizon"] if "horizon" in set(hz.get("mode", [])) else hz
        for g, gl in GRAPHS.items():
            s = h[(h.config == g)]
            if s.empty or "weeks" not in s.columns:
                continue
            piv = s[s.u == 1.0].pivot_table(index="channel", columns="weeks", values="peak")
            if piv.empty:
                continue
            piv.columns = [f"{int(c)} weeks" for c in piv.columns]
            write_table(f"horizon_{g}", piv,
                        f"Peak infections at full coverage as the season is lengthened, {gl}.",
                        f"horizon_{g}", out, index_name="Channel", float_fmt="{:,.0f}",
                        note_txt="If a channel does not improve with more time, its "
                                 "failure is structural rather than a matter of the "
                                 "season being too short.")


def sec_objective(d, out):
    p = os.path.join(d, "constrained_objective.csv")
    if not os.path.exists(p):
        note("constrained_objective.csv not found -- objective comparison skipped")
        return
    c = pd.read_csv(p)
    for g in c.config.unique():
        s = c[c.config == g]
        rows = []
        for m in sorted(s.budget_mult.unique()):
            q = s[(s.budget_mult == m)]
            q = q[q.policy != "H0_Do_Nothing"]
            if q.empty:
                continue
            bw = q.loc[q.V.idxmin()]
            bc = q.loc[q.burden.idxmin()]
            rows.append({
                "Budget": f"$\\times${m:g}",
                "Weighted objective picks": SHORT.get(bw.policy, bw.policy),
                "and it spends": bw.util,
                "Constrained objective picks": SHORT.get(bc.policy, bc.policy),
                "and it spends ": bc.util,
            })
        if rows:
            r = pd.DataFrame(rows).set_index("Budget")
            write_table(f"objective_{g}", r,
                        f"Which policy each objective selects, {GRAPHS.get(g, g)}.",
                        f"objective_{g}", out, index_name="Budget",
                        note_txt="The weighted objective prices an unspent taka the same "
                                 "as a taka of averted illness. The city's budget is "
                                 "ring-fenced, so an unspent taka is worth nothing. "
                                 "\\emph{Spends} is the fraction of the allocated budget used.")

        fig, ax = plt.subplots()
        for pol in order_policies(s.policy.unique()):
            if pol == "H0_Do_Nothing":
                continue
            q = s[s.policy == pol].sort_values("budget_mult")
            ax.plot(q.budget_mult, q.util, marker="o", ms=3.5, lw=2,
                    color="k" if pol == "RL_Agent" else None,
                    ls="--" if "Oracle" in pol else "-",
                    label=SHORT.get(pol, pol))
        panel_xlabel(ax, "budget (multiple of the actual season budget)")
        panel_ylabel(ax, "fraction of the budget actually spent")
        ax.set_ylim(0, 1.05)
        panel_title(ax, f"Budget actually used, {GRAPHS.get(g, g)}")
        panel_legend(ax)
        save_fig(f"utilisation_{g}", out,
                 f"How much of the allocated budget each policy actually spends, "
                 f"{GRAPHS.get(g, g)}. \\textbf{{How to read:}} 1.0 means all of it. "
                 f"There is no reward for underspending -- the budget is ring-fenced, "
                 f"so unspent money simply disappears. A policy sitting low on this "
                 f"chart is declining resources it was given, which the weighted "
                 f"objective nevertheless scores as a success.", f"utilisation_{g}")


def sec_surveillance(eps, out):
    made = 0
    for g, gl in GRAPHS.items():
        rows = {}
        for stem, pace, cap in CELLS:
            d = eps.get(stem)
            if d is None or "spendshare_survey" not in d.columns:
                continue
            s = d[(d.config == g) & (d.budget_mult == 1.0)]
            if s.empty:
                continue
            m = s.groupby("policy").spendshare_survey.mean()
            col = f"{pace}, {cap}"
            for pol in BELIEF + ["RL_Agent"]:
                if pol in m.index and pol not in ("H0_Do_Nothing",
                                                 "H1_Blanket_Fogging",
                                                 "H3_Random_Larviciding"):
                    rows.setdefault(pol, {})[col] = float(m[pol])
        if not rows:
            continue
        df = pd.DataFrame(rows).T
        df = df.reindex([p for p in ORDER if p in df.index])
        want = [f"{pace}, {cap}" for _, pace, cap in CELLS]
        df = df[[c for c in want if c in df.columns]]
        write_table(f"surveillance_share_{g}", df,
                    f"Share of each policy's spending that goes to surveillance "
                    f"rather than to intervention, {gl}, at the appropriated budget.",
                    f"surveillance_share_{g}", out,
                    note_txt="Columns are the four evaluation protocols. "
                             "\\emph{Paced} means the baseline must spend in equal "
                             "weekly instalments; \\emph{uncapped} means it may put "
                             "the whole weekly pool into surveying. A figure near 1.0 "
                             "means the policy is surveying and doing nothing else, so "
                             "any comparison against it measures the learned policy "
                             "against inaction rather than against a real alternative. "
                             "Policies that never survey are omitted. The learned "
                             "agent is unaffected by these protocols and is shown as "
                             "a reference.")
        made += 1
    if not made:
        note("surveillance-share tables skipped (need the four *_episodes.csv)")


PAPER_ONLY = {
    "headline_V_dncc": "Discounted objective $V$ (billions BDT) by policy and budget, 54 wards.",
    "headline_V_dncc10": "Discounted objective $V$ (billions BDT) by policy and budget, 10 wards.",
    "protocol_matrix": "Measured advantage of the learned controller over the myopic LP under four evaluation protocols.",
    "threshold_summary": "Cost of reaching the same peak by each route, as a multiple of the season budget.",
    "surveillance_share_dncc": "Share of spending directed to surveillance, 54 wards.",
    "surveillance_share_dncc10": "Share of spending directed to surveillance, 10 wards.",
    "convergence_dncc": "Objective against training steps by seed, 54 wards.",
    "convergence_dncc10": "Objective against training steps by seed, 10 wards.",
    "delay_ratios": "Realised peak relative to a week-1 start, at identical expenditure.",
    "prereg": "Pre-registered test of the delay proposition.",
    "sweep_sustained_dncc": "Peak infections against sustained coverage, 54 wards.",
    "sweep_burst_dncc": "Peak infections against campaign length from week 1, 54 wards.",
    "budget_V_dncc": "Objective against budget, 54 wards.",
    "budget_V_dncc10": "Objective against budget, 10 wards.",
    "channels_dncc": "Weekly infections under no intervention, larviciding and adulticide.",
}


def sec_exp_all(d, out):
    import glob

    rows = []
    for p in sorted(glob.glob(os.path.join(d, "exp_delay_*.csv"))):
        try:
            rows.append(pd.read_csv(p))
        except Exception:
            pass
    if rows:
        dl = pd.concat(rows, ignore_index=True)
        piv = dl.pivot_table(index="start_week", columns=["config", "forcing"],
                             values="ratio")
        lo = dl.pivot_table(index="start_week", columns=["config", "forcing"],
                            values="ci_lo")
        hi = dl.pivot_table(index="start_week", columns=["config", "forcing"],
                            values="ci_hi")
        disp = pd.DataFrame(index=piv.index)
        for c in piv.columns:
            disp[f"{c[0]}, {c[1].replace('constant_', 'const ')}"] = [
                f"{piv.loc[i, c]:.2f} [{lo.loc[i, c]:.2f}, {hi.loc[i, c]:.2f}]"
                if pd.notna(piv.loc[i, c]) else "--" for i in piv.index]
        disp.index = [f"week {int(i)}" for i in disp.index]
        write_table("delay_ratios", disp,
                    "Realised peak relative to a week-1 start, at identical total "
                    "expenditure. Geometric mean of paired episode ratios with "
                    "95\\% bootstrap intervals.",
                    "delay_ratios", out, index_name="Campaign starts",
                    note_txt="Ratios are paired within evaluation seed. No control "
                             "threshold enters the calculation.")

        fig, ax = plt.subplots()
        for c in piv.columns:
            lab = f"{c[0]}, {'seasonal' if c[1]=='seasonal' else 'constant forcing'}"
            ax.plot(piv.index, piv[c], marker="o", ms=3.5, lw=2, label=lab,
                    ls="-" if c[1] == "seasonal" else "--")
        panel_xlabel(ax, "week the campaign starts")
        panel_ylabel(ax, "peak relative to a week-1 start")
        ax.set_yscale("log")
        panel_legend(ax)
        save_fig("delay_ratios", out,
                 "Realised peak relative to a week-1 start, at identical total "
                 "expenditure, for both instances under seasonal and constant "
                 "forcing.", "delay_ratios")

    rows = []
    for p in sorted(glob.glob(os.path.join(d, "exp_prereg_*.csv"))):
        try:
            rows.append(pd.read_csv(p))
        except Exception:
            pass
    if rows:
        pr = pd.concat(rows, ignore_index=True)
        fig, ax = plt.subplots()
        for cfg, g in pr.groupby("config"):
            g = g.sort_values("start_week")
            ax.plot(g.start_week, g.log_peak, marker="o", ms=3.5, lw=2, label=str(cfg))
            tau = float(g.pred_tau_star.iloc[0])
            ax.axvline(tau, ls=":", lw=1.2, color="0.5")
        panel_xlabel(ax, "week the campaign starts")
        panel_ylabel(ax, "log realised peak")
        panel_legend(ax)
        save_fig("prereg", out,
                 "Log realised peak against start week under constant forcing. "
                 "Dotted lines mark the breakpoint predicted before the runs.",
                 "prereg")

    for p in sorted(glob.glob(os.path.join(d, "exp_channels_*.csv"))):
        try:
            ch = pd.read_csv(p)
        except Exception:
            continue
        cfg = str(ch.config.iloc[0]) if "config" in ch.columns else "instance"
        fig, ax = plt.subplots()
        for col, lab in (("I_none", "no intervention"),
                         ("I_larviciding", "larviciding"),
                         ("I_adulticide", "adulticide")):
            if col in ch.columns:
                ax.plot(ch.week, ch[col], marker="o", ms=3, lw=2, label=lab)
        ax.set_yscale("log")
        panel_xlabel(ax, "week")
        panel_ylabel(ax, "infections")
        panel_legend(ax)
        save_fig(f"channels_{cfg}", out,
                 f"Weekly infections under no intervention, full larviciding and "
                 f"full adulticide from week~1, {cfg}.", f"channels_{cfg}")


def emit_manuscript_snippet(out):
    print("\n" + "=" * 72)
    print("PASTE INTO THE MANUSCRIPT")
    print("=" * 72)
    print("% tables")
    for t in TEX_FILES:
        print(f"\\input{{report/{t}}}")
    print("\n% figures")
    for f, cap, lab, iw in FIG_FILES:
        print(f"\\begin{{figure}}[htbp]\\centering")
        print(f"  \\includegraphics[width=0.86\\textwidth]{{report/{f}}}")
        print(f"  \\caption{{{cap}}}\\label{{fig:{lab}}}")
        print(f"\\end{{figure}}")
    print("=" * 72)

def build_main(out, tag):
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs,graphicx,amsmath,longtable}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}",
        rf"\title{{Results digest \\ \large run tag \texttt{{{esc(tag)}}}}}",
        r"\date{\today}", r"\begin{document}\maketitle",
        r"\tableofcontents\clearpage",
    ]
    if NOTES:
        lines += [r"\section*{Build notes}", r"\begin{itemize}"]
        lines += [rf"\item {esc(n)}" for n in NOTES]
        lines += [r"\end{itemize}", r"\clearpage"]

    lines.append(r"\section{Tables}")
    for t in TEX_FILES:
        lines.append(rf"\input{{{t}}}")
    lines.append(r"\clearpage\section{Figures}")
    for f, cap, lab, iw in FIG_FILES:
        lines += [r"\begin{figure}[htbp]\centering",
                  rf"\includegraphics[width={iw}\textwidth]{{{f}}}",
                  rf"\caption{{{cap}}}\label{{fig:{lab}}}",
                  r"\end{figure}"]
    lines.append(r"\end{document}")
    p = os.path.join(out, "main.tex")
    open(p, "w").write("\n".join(lines) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version_tag", default="v18",
                    help='File prefix of the results, e.g. v18.')
    ap.add_argument("--dir", default="evaluation_dengue")
    ap.add_argument("--out", default="report")
    ap.add_argument("--no-compile", action="store_true")
    add_geometry_args(ap)
    ap.add_argument("--paper", action="store_true",
                    help='Manuscript mode: short captions and only the outputs the paper uses.')
    a = ap.parse_args()
    configure(a)
    tag = a.version_tag

    os.makedirs(a.out, exist_ok=True)
    print(f"reading {a.dir}/  tag '{tag}'")

    cells, eps = {}, {}
    for stem, _, _ in CELLS:
        c, e = load(a.dir, tag, stem), load(a.dir, tag, stem, episodes=True)
        if c is not None:
            cells[stem] = c
        if e is not None:
            eps[stem] = e
        if c is None and e is None:
            note(f"{tag}_{stem}.csv not found")
    print(f"found {len(cells)} summary and {len(eps)} per-episode files")

    GRAPHS.update(discover_graphs(list(cells.values()) + list(eps.values())))
    if not GRAPHS:
        for stem in ("coverage_threshold", "threshold_timing"):
            GRAPHS.update(discover_graphs([load(a.dir, tag, stem)]))
    if not GRAPHS:
        print("could not identify any instance in the data -- nothing to report")
        return
    print("instances found in the data: "
          + ", ".join(f"{k} ({v})" for k, v in GRAPHS.items()))

    print("building tables and figures")
    sec_headline(cells, a.out)
    sec_protocol_matrix(eps, a.out)
    sec_budget_curves(cells, a.out)
    sec_learning(a.dir, a.out)
    sec_sweeps(a.dir, tag, a.out)
    sec_objective(a.dir, a.out)
    sec_surveillance(eps, a.out)
    sec_exp_all(a.dir, a.out)

    if a.paper:
        keep_t = [t for t in TEX_FILES if os.path.basename(t) in PAPER_ONLY]
        keep_f = [(f, PAPER_ONLY.get(os.path.basename(f), c), l, iw)
                  for f, c, l, iw in FIG_FILES
                  if os.path.basename(f) in PAPER_ONLY]
        dropped = len(TEX_FILES) - len(keep_t) + len(FIG_FILES) - len(keep_f)
        TEX_FILES[:] = keep_t
        FIG_FILES[:] = keep_f
        print(f'  paper mode: kept {len(keep_t)} tables and {len(keep_f)} figures, dropped {dropped}')


    print(f"  {len(TEX_FILES)} tables, {len(FIG_FILES)} figures")
    main_tex = build_main(a.out, tag)
    print(f"wrote {main_tex}")

    if a.paper:
        emit_manuscript_snippet(a.out)

    if a.no_compile:
        return
    try:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                           cwd=a.out, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        pdf = os.path.join(a.out, "main.pdf")
        print(f"compiled {pdf}" if os.path.exists(pdf)
              else "pdflatex ran but produced no PDF -- check report/main.log")
    except FileNotFoundError:
        print("pdflatex not found -- tables and figures are still in " + a.out)


if __name__ == "__main__":
    main()
