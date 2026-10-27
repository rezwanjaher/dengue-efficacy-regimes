#!/usr/bin/env python3
"""Table of the 54 wards: primary thana, census population, degree and
membership of the 10-ward instance.

    python src/make_ward_table.py --out report/tables/wards.tex

Options:
    --graph        Graph file.
    --sub          10-ward graph file.
    --crosswalk    Ward-by-thana census crosswalk.
    --corporation  Corporation code. Default DNCC.
    --no-thana     omit the thana column
    --out          Output path.
"""
import argparse, collections, csv, json, os, re


def ward_number(name):
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else None


def load_crosswalk(path, corporation):
    agg = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("corporation") != corporation:
                continue
            agg[r["ward"].strip()].append(
                (r["thana"].strip(), int(r["ward_thana_pop"])))
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default=os.path.join("data", "dncc_graph.json"))
    p.add_argument("--sub", default=os.path.join("data", "dncc10_graph.json"))
    p.add_argument("--crosswalk",
                   default=os.path.join("t", "ward_thana_crosswalk_bbs2022.csv"))
    p.add_argument("--corporation", default="DNCC")
    p.add_argument("--no-thana", dest="thana", action="store_false",
                   help="omit the thana column")
    p.add_argument("--out", default=os.path.join("report", "tables", "wards.tex"))
    a = p.parse_args()

    g = json.load(open(a.graph))
    names, pops = g["ward_names"], g["N_initial"]
    deg = {i: len(v) for i, (k, v) in enumerate(sorted(
        g["neighbors"].items(), key=lambda kv: int(kv[0])))}

    sub = set()
    if os.path.exists(a.sub):
        sub = {ward_number(n) for n in json.load(open(a.sub))["ward_names"]}

    cross, n_split, biggest_split = {}, 0, None
    if a.thana:
        if not os.path.exists(a.crosswalk):
            print(f"  crosswalk not found at {a.crosswalk} -- run with --crosswalk "
                  f"or --no-thana")
            return
        agg = load_crosswalk(a.crosswalk, a.corporation)
        bad = []
        for nm, pop in zip(names, pops):
            key = nm.replace(f"{a.corporation}_", "")
            parts = agg.get(key)
            if parts is None:
                bad.append(f"{nm}: absent from crosswalk")
                continue
            if sum(q for _, q in parts) != int(pop):
                bad.append(f"{nm}: graph {int(pop):,} vs crosswalk "
                           f"{sum(q for _, q in parts):,}")
            parts = sorted(parts, key=lambda t: -t[1])
            cross[nm] = parts
            if len(parts) > 1:
                n_split += 1
                if biggest_split is None or len(parts) > len(biggest_split[1]):
                    biggest_split = (key, parts)
        if bad:
            print("  REFUSING TO WRITE -- crosswalk does not reproduce the graph:")
            for b in bad[:10]:
                print("   ", b)
            return
        print(f"  crosswalk reproduces all {len(names)} ward populations exactly")

    rows = []
    for i, (nm, pop) in enumerate(zip(names, pops)):
        w = ward_number(nm)
        parts = cross.get(nm, [])
        rows.append(dict(ward=w, thana=(parts[0][0] if parts else ""),
                         split=len(parts) > 1, pop=int(pop), deg=deg.get(i, 0),
                         sub="$\\bullet$" if w in sub else ""))
    rows.sort(key=lambda r: (r["ward"] is None, r["ward"]))

    ncol = 2 if a.thana else 3
    per = -(-len(rows) // ncol)
    panels = [rows[i * per:(i + 1) * per] for i in range(ncol)]
    hdr = ["Ward"] + (["Thana"] if a.thana else []) + ["Population", "Deg.", ""]
    spec = "r" + ("l" if a.thana else "") + "rrc"

    o = [r"\begin{table}[htbp]", r"\centering\footnotesize",
         r"\caption{Wards of the Dhaka North City Corporation. Population is the "
         r"2022 census figure carried per ward by the graph; \emph{Thana} is the "
         r"thana holding the largest share of the ward's population; \emph{Deg.} "
         r"is the number of adjacent wards under the administrative-contiguity "
         r"rule; $\bullet$ marks membership of the 10-ward subgraph.}",
         r"\label{tab:wards}",
         r"\begin{tabular}{@{}" + "@{\\hspace{1.2em}}".join([spec] * ncol) + r"@{}}",
         r"\toprule",
         " & ".join([" & ".join(hdr)] * ncol) + r" \\", r"\midrule"]
    for i in range(per):
        cells = []
        for pan in panels:
            if i < len(pan):
                r = pan[i]
                c = [str(r["ward"])]
                if a.thana:
                    c.append(r["thana"] + (r"\textsuperscript{*}" if r["split"] else ""))
                c += [f"{r['pop']:,}", str(r["deg"]), r["sub"]]
            else:
                c = [""] * len(hdr)
            cells.append(" & ".join(c))
        o.append(" & ".join(cells) + r" \\")
    o.append(r"\bottomrule")
    o.append(r"\end{tabular}")

    tot = sum(r["pop"] for r in rows)
    subtot = sum(r["pop"] for r in rows if r["sub"])
    note = (rf"Total population {tot:,} across {len(rows)} wards; the subgraph "
            rf"covers {subtot:,} residents in {len(sub)} wards. ")
    if a.thana:
        note += (rf"\textsuperscript{{*}}{n_split} wards span more than one thana "
                 r"and are listed under the thana holding the largest share")
        if biggest_split:
            k, parts = biggest_split
            note += (" (" + k.replace("Ward No-", "ward ") + " divides between "
                     + ", ".join(t for t, _ in parts) + ")")
        note += ". "
    note += (r"The cantonment ward is excluded and edges crossing to Dhaka South "
             r"are removed, so each corporation is modelled as a closed system.")
    o.append(r"\\[2pt]\parbox{0.95\textwidth}{\footnotesize " + note + "}")
    o.append(r"\end{table}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(o) + "\n")
    print(f"wrote {a.out}  ({len(rows)} wards, {len(sub)} in subgraph"
          + (f", {n_split} spanning multiple thanas)" if a.thana else ")"))


if __name__ == "__main__":
    main()
