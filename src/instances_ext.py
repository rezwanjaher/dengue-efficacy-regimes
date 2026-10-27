#!/usr/bin/env python3
"""Registry of the ten instances. Measures each one from its graph file.
Used by exp_multi.py and make_crossings_table.py.

    python src/instances_ext.py
"""
import json
import os

import numpy as np

DEFAULT_ECON = "DNCC_Seasonal_Budget"

KINDS = ("contiguous", "random", "dispersed")
SIZES = (10, 20, 30)

TAGS = [f"{k}{n}" for n in SIZES for k in KINDS]


def graph_path(tag, datadir="data"):
    return os.path.join(datadir, f"{tag}_graph.json")


def describe(tag, datadir="data"):
    path = graph_path(tag, datadir)
    if not os.path.exists(path):
        return None
    g = json.load(open(path))
    nb = {int(k): list(v) for k, v in g["neighbors"].items()}
    deg = [len(nb.get(i, [])) for i in range(int(g["num_sites"]))]
    pop = np.asarray(g["N_initial"], dtype=float)
    top3 = float(np.sort(pop)[::-1][:3].sum() / pop.sum())
    return {
        "tag": tag,
        "graph": path,
        "n_wards": int(g["num_sites"]),
        "population": float(pop.sum()),
        "edges": sum(deg) // 2,
        "deg_median": float(np.median(deg)),
        "deg_mean": float(np.mean(deg)),
        "isolated": int(sum(1 for d in deg if d == 0)),
        "top3_share": top3,
        "top3_vs_uniform": top3 / (3.0 / len(pop)),
        "pop_cv": float(pop.std() / pop.mean()),
    }


def registry(datadir="data", econ=None):
    out = {}
    for tag in TAGS:
        d = describe(tag, datadir)
        if d is None:
            continue
        out[tag] = dict(graph=d["graph"], econ=econ or DEFAULT_ECON,
                        n_wards=d["n_wards"])
    return out


def missing(datadir="data"):
    return [t for t in TAGS if describe(t, datadir) is None]


HEAD = (f"  {'instance':<16}{'wards':>6}{'population':>13}{'edges':>7}"
        f"{'deg med':>9}{'isol':>6}{'top3':>8}{'pop CV':>9}")


def line(d):
    return (f"  {d['tag']:<16}{d['n_wards']:>6}{d['population']:>13,.0f}"
            f"{d['edges']:>7}{d['deg_median']:>9.1f}{d['isolated']:>6}"
            f"{d['top3_share']:>8.3f}{d['pop_cv']:>9.3f}")


def main():
    print("=" * 80)
    print("INSTANCE REGISTRY  (measured from data/*.json)")
    print("=" * 80)
    print(HEAD)
    any_found = False
    for tag in TAGS:
        d = describe(tag)
        if d is None:
            print(f"  {tag:<16}{'NOT BUILT':>6}")
            continue
        any_found = True
        print(line(d))
    if not any_found:
        print("\n  Nothing built yet. Run:  python t/make_instances.py --write")
        return
    print("\n  pop CV is the coefficient of variation of ward populations. Unlike the")
    print("  top-three share it carries no built-in trend in ward count, so it is the")
    print("  concentration measure to regress a crossing on across sizes.")
    print("\n  Median degree rises with ward count in this family, so ward count and")
    print("  coupling are still correlated across the nine. Comparisons WITHIN a")
    print("  ward count are clean; comparisons across ward counts are not.")


if __name__ == "__main__":
    main()
