#!/usr/bin/env python3
"""Sensitivity of the policy results to the surveillance parameters. Each value
runs the harness on a separate copy of the configuration; the original is not
changed.

    python src/sweep_surveillance.py --param eta_S
    python src/sweep_surveillance.py --param surv_cost
    python src/sweep_surveillance.py --param eta_C

Options:
    --param    Parameter to sweep.
    --values   comma-separated override
    --extra    arguments passed through to run_all_evals Default --dncc.
    --logdir   Folder for the sweep logs.
    --eta-f    Pin adulticide efficacy for every run.
    --fix      Pin another parameter, as path=value.
    --dry-run  Print the runs without running them.
"""
import argparse
import datetime as _dt
import json
import os
import shutil
import sys

ORIG_CONFIG = os.path.join("configs", "archetypes.json")
SWEEP_DIR = os.path.join("configs", "sweeps")
GROUP, NAME = "Urban Public Health", "Dengue_Dhaka_Advanced"

PARAMS = {
    "eta_S":     ("action_semantics.survey.efficacy",
                  [0.30, 0.50, 0.72, 0.90, 1.00]),
    "surv_cost": ("action_semantics.survey.cost",
                  [0.5, 1.25, 2.5, 5.0, 10.0]),
    "eta_C":     ("action_semantics.case_management.efficacy",
                  [0.0, 0.25, 0.50, 0.75, 1.00]),
}

MATCHED = ["--free-heuristics", "--survey-share", "0.25",
           "--lambda_sigma", "0.8", "--version_tag", "V18"]


def node_and_key(tree, path):
    node = tree
    for k in path.split(".")[:-1]:
        node = node[k]
    return node, path.split(".")[-1]


def make_config(path, value, dest, fixes=()):
    with open(ORIG_CONFIG, encoding="utf-8") as f:
        d = json.load(f)
    bio = d["archetype_groups"][GROUP][NAME]["biological_facts"]
    node, key = node_and_key(bio, path)
    old = node.get(key)
    node[key] = value
    for fpath, fval in fixes:
        fnode, fkey = node_and_key(bio, fpath)
        fnode[fkey] = fval
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", choices=sorted(PARAMS), required=True)
    ap.add_argument("--values", default=None, help="comma-separated override")
    ap.add_argument("--extra", default="--dncc",
                    help="arguments passed through to run_all_evals")
    ap.add_argument("--logdir", default=os.path.join("evaluation_dengue", "sweeps_ls08"))
    ap.add_argument("--eta-f", dest="eta_f", type=float, default=None,
                    help='Pin adulticide efficacy for every run.')
    ap.add_argument("--fix", action="append", default=[],
                    help='Pin another parameter, as path=value.')
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(ORIG_CONFIG):
        sys.exit(f"{ORIG_CONFIG} not found; run from the project root.")

    path, values = PARAMS[a.param]
    if a.values:
        values = [float(x) for x in a.values.split(",")]

    fixes = []
    if a.eta_f is not None:
        fixes.append(("action_semantics.vector_control.efficacy", a.eta_f))
    for item in a.fix:
        if "=" not in item:
            sys.exit(f"--fix needs path=value, got '{item}'")
        k, v = item.split("=", 1)
        fixes.append((k.strip(), float(v)))

    os.makedirs(SWEEP_DIR, exist_ok=True)
    os.makedirs(a.logdir, exist_ok=True)

    before = os.path.getmtime(ORIG_CONFIG), os.path.getsize(ORIG_CONFIG)

    print(f"parameter     : {a.param}  ->  biological_facts.{path}")
    print(f"values        : {values}")
    print(f"config copies : {SWEEP_DIR}")
    print(f"originals     : never written to")
    if fixes:
        for k, v in fixes:
            print(f"pinned        : {k} = {v}")
    else:
        print("pinned        : nothing; runs at the calibrated efficacy 0.62, "
              "which is OUTSIDE\n                the band the targeting claim "
              "concerns. Pass --eta-f 0.45.")
    print()

    prepared = []
    for v in values:
        tag = f"{a.param}_{v}" + (f"_etaF{a.eta_f}" if a.eta_f is not None else "")
        dest = os.path.join(SWEEP_DIR, f"archetypes_{tag}.json")
        try:
            old = make_config(path, v, dest, fixes)
        except KeyError:
            sys.exit(f"'{path}' not found in biological_facts for {NAME}.")
        prepared.append((v, dest))
        print(f"  {a.param} = {v:<6} (was {old})  ->  {dest}")

    if a.dry_run:
        print(f"\nharness would run: run_all_evals "
              f"{' '.join(MATCHED + a.extra.split())}")
        print("dry run; the config copies above were written, nothing else ran.")
        return

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import run_all_evals as H

    argv0 = sys.argv[0]
    for v, cfg in prepared:
        print("\n" + "=" * 70)
        print(f"{a.param} = {v}")
        print("=" * 70)
        H.EVAL_ARCHETYPE_FILE = cfg
        sys.argv = [argv0] + MATCHED + a.extra.split()
        log = os.path.join(a.logdir, os.path.basename(cfg)
                           .replace("archetypes_", "").replace(".json", ".log"))
        out = sys.stdout
        try:
            with open(log, "w", encoding="utf-8") as f:
                sys.stdout = f
                H.main()
        except SystemExit:
            pass
        except Exception as e:
            sys.stdout = out
            print(f"  harness raised: {type(e).__name__}: {e}")
            print(f"  see {log}")
            continue
        finally:
            sys.stdout = out
        print(f"  done -> {log}")

    after = os.path.getmtime(ORIG_CONFIG), os.path.getsize(ORIG_CONFIG)
    print(f"\n{ORIG_CONFIG} unchanged: {before == after}")
    print("\nNEXT")
    print("  Open the log for the published value first and check it")
    print("  reproduces Section 6.6. Only then read the other logs.")


if __name__ == "__main__":
    main()
