#!/usr/bin/env python3
"""Combine generated tables into the appendix versions, without retyping any
number. Run from the folder that holds report/tables; reproduce.py does this
from reproduced/:

    cd reproduced
    python ../src/compact_tables.py
"""
import os, re, sys
from decimal import Decimal, ROUND_HALF_UP

D = os.path.join("report", "tables")


def read(name):
    t = open(os.path.join(D, name + ".tex"), encoding="utf-8").read()
    cap = re.search(r"\\caption\{(.*?)\}\n\\label", t, re.S).group(1)
    label = re.search(r"\\label\{(.*?)\}", t).group(1)
    spec = re.search(r"\\begin\{tabular\}\{(.*?)\}\n", t).group(1)
    head = t.split("\\toprule\n")[1].split("\\midrule")[0].strip()
    body = t.split("\\midrule\n", 1)[1].split("\\bottomrule")[0]
    rows = [r.strip() for r in body.strip().split("\n") if r.strip() and r.strip() != "\\midrule"]
    note = re.search(r"\\parbox\{[^}]*\}\{\\footnotesize (.*?)\}\n\\end\{table\}", t, re.S)
    return dict(cap=cap, label=label, spec=spec, head=head, rows=rows,
                note=note.group(1) if note else None)


def write(name, text):
    p = os.path.join(D, name + ".tex")
    if os.path.exists(p):
        print(f"SKIPPED {p}: it exists. Delete it and rerun to rebuild it.")
        return
    open(p, "w", encoding="utf-8").write(text)
    print("wrote", p)


def panels(a, b, out, caption, la="$54$ wards", lb="$10$ wards"):
    A, B = read(a), read(b)
    if A["head"] != B["head"] or A["spec"] != B["spec"]:
        sys.exit(f"{a} and {b} have different columns; not merged.")
    ncol = A["head"].count("&") + 1
    s = ["\\begin{table}[htbp]", "\\centering\\small", "\\caption{" + caption + "}",
         "\\label{" + A["label"] + "}\\label{" + B["label"] + "}",
         "\\begin{tabular}{" + A["spec"] + "}", "\\toprule", A["head"], "\\midrule",
         f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{la}}}}} \\\\"] + A["rows"] + [
         "\\midrule", f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{lb}}}}} \\\\"] + B["rows"] + [
         "\\bottomrule", "\\end{tabular}"]
    note = A["note"]
    if note:
        s.append("\\\\[2pt]\\parbox{0.92\\textwidth}{\\footnotesize " + note + "}")
    s.append("\\end{table}\n")
    write(out, "\n".join(s))


def cells(row):
    return [c.strip() for c in row.rstrip("\\").strip().rstrip("\\").split("&")]


def pivot(tabs, out, caption, valname):
    blocks, etas_all = [], None
    labels = []
    for name, title in tabs:
        T = read(name); labels.append(T["label"])
        data = [cells(r) for r in T["rows"]]
        etas = []
        for e, *_ in data:
            if e not in etas: etas.append(e)
        vals = []
        for _, v, *_ in data:
            if v not in vals: vals.append(v)
        uni = {}
        for e, v, u, b, g in data:
            if uni.setdefault(e, u) != u:
                sys.exit(f"{name}: uniform value differs within eta {e}; not pivoted.")
        look = {(e, v): (b, g) for e, v, u, b, g in data}
        if etas_all is None: etas_all = etas
        elif etas != etas_all: sys.exit(f"{name}: efficacies differ between panels.")
        blocks.append((title, vals, uni, look))
    eh = [(e if e != "default" else "0.62") for e in etas_all]
    n = 1 + 2 * len(etas_all)
    s = ["\\begin{table}[htbp]", "\\centering\\footnotesize", "\\caption{" + caption + "}",
         "".join("\\label{" + l + "}" for l in labels),
         "\\begin{tabular}{@{}l" + "rr" * len(etas_all) + "@{}}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\eta_{{F}}={h}$}}" for h in eh) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(etas_all))),
         valname + " & " + " & ".join(["Best & Gap"] * len(etas_all)) + " \\\\", "\\midrule"]
    for k, (title, vals, uni, look) in enumerate(blocks):
        if k: s.append("\\midrule")
        s.append(f"\\multicolumn{{{n}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        s.append("Uniform & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{uni[e]}}}" for e in etas_all) + " \\\\")
        for v in vals:
            r = [v]
            for e in etas_all:
                b, g = look.get((e, v), ("--", "--"))
                r += [b, g.replace("\\%", "")]
            s.append(" & ".join(r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}\n"]
    write(out, "\n".join(s))


def capspan(t):
    i = t.index("\\caption{") + len("\\caption{"); d = 1; j = i
    while d:
        d += {"{": 1, "}": -1}.get(t[j], 0); j += 1
    return i, j - 1


def recaption(name, out, caption, swaps=()):
    t = open(os.path.join(D, name + ".tex"), encoding="utf-8").read()
    i, j = capspan(t)
    t = t[:i] + caption + t[j:]
    for old, new in swaps:
        if t.count(old) != 1:
            sys.exit(f"{name}: expected text not found exactly once: {old[:60]}...")
        t = t.replace(old, new)
    write(out, t)

def num(cell, name):
    s = re.sub(r"[$\\{}\s]", "", cell).replace(",", "")
    try:
        return Decimal(s)
    except Exception:
        sys.exit(f"{name}: cannot read a number from cell {cell!r}. Nothing written.")


def tidy_threshold(name, out):
    t = open(os.path.join(D, name + ".tex"), encoding="utf-8").read()
    head, rest = t.split("\\midrule\n", 1)
    body, tail = rest.split("\\bottomrule", 1)
    rows = []
    for r in body.strip().split("\n"):
        c = [x.strip() for x in r.rstrip().rstrip("\\").split("&")]
        if len(c) != 7:
            sys.exit(f"{name}: row with {len(c)} cells, expected 7: {r}")
        c[1] = {"both 50 50": "both, half each"}.get(c[1], c[1])
        c[3] = f"{num(c[3], name).quantize(Decimal('1'), ROUND_HALF_UP):,}".replace(",", "{,}")
        if c[6].strip("$ "):
            c[6] = f"{num(c[6], name):.2f}"
        rows.append(" & ".join(c) + " \\\\")
    head = head.replace("\\begin{tabular}{lrrrrrr}", "\\begin{tabular}{@{}llrrrrr@{}}")
    write(out, head + "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule" + tail)


def main():
    if not os.path.isdir(D):
        sys.exit("Run from the manuscript folder (report/tables not found).")
    note = ("Every policy faces the same budget and the same evaluation seeds. "
            "Oracle rows see the true state for free; they are references, not competitors or bounds.")
    panels("headline_V_dncc", "headline_V_dncc10", "headline_V_both",
           "Discounted objective $V$ (billions BDT, lower is better) by budget multiplier. "
           "Matched protocol (baselines free, surveillance capped).")
    panels("headline_peak_dncc", "headline_peak_dncc10", "headline_peak_both",
           "Mean peak infections by budget multiplier. Matched protocol (baselines free, surveillance capped).")
    panels("schedules_dncc", "schedules_dncc10", "schedules_both",
           "Mean peak as a share of the untreated peak at the same efficacy. "
           "Every schedule spends six week-equivalents of full coverage.")
    gap = " \\emph{Best} and \\emph{Gap} are defined in the text."
    pivot([("sens_surveillance", "Discounted objective $V$, billions BDT"),
           ("sens_surveillance_peak", "Mean peak infections")], "sens_etaS_compact",
          "Sensitivity of the information result to surveillance detection $\\eta_{S}$." + gap,
          "$\\eta_{S}$")
    pivot([("sens_survcost", "Survey price (BDT per person): objective $V$, billions BDT"),
           ("sens_survcost_peak", "Survey price (BDT per person): mean peak infections"),
           ("sens_etaC", "Case-management efficacy $\\eta_{C}$: objective $V$, billions BDT")],
          "sens_price_cm",
          "Sensitivity of the information result to the survey price and to case-management efficacy." + gap,
          "Value")
    recaption("delay_ratios", "delay_ratios_ec",
              "Peak relative to a week-1 start, at equal total spend. "
              "Geometric mean of paired ratios, with $95\\%$ bootstrap intervals.",
              [("\\begin{tabular}{lrrrr}", "\\begin{tabular}{@{}lrrrr@{}}"),
               ("Campaign starts & dncc, const 0.6 & dncc, seasonal & dncc10, const 0.6 & dncc10, seasonal \\\\",
                " & \\multicolumn{2}{c}{$54$ wards} & \\multicolumn{2}{c}{$10$ wards} \\\\\n"
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
                "Campaign starts & constant & seasonal & constant & seasonal \\\\"),
               ("Ratios are paired within evaluation seed.",
                "Constant forcing is held at $0.6$. Ratios are paired within evaluation seed.")])
    recaption("crossings", "crossings_ec",
              "Efficacy at which each crossing occurs, by instance.")
    tidy_threshold("threshold_summary", "threshold_summary_ec")


if __name__ == "__main__":
    main()
