#!/usr/bin/env python3
r"""Run the exp_all.py efficacy sweep on every instance.

    python src/exp_multi.py efficacy --all-instances --eta-grid \
        "0.10,0.20,0.30,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.75,0.90"

Options:
    mode              same modes as exp_all.py
    --instance        Instance name, as listed by --list.
    --all-instances   Run every instance in turn.
    --list            show which instances are available and stop
    --archetype_file  Parameter file.
    --archetype       Parameter set. Default Dengue_Dhaka_Advanced.
    --econ            Override the economic scenario.
    --lambda_sigma    Risk-field dispersion. Default 0.8.
    --horizon         Season length in weeks. Default 20.
    --trials          Epidemic seeds per point. Default 40.
    --burst           Campaign length in weeks. Default 6.
    --coverage        Campaign coverage. Default 1.0.
    --max-start       Latest start week tried. Default 12.
    --seasonal        Use seasonal forcing.
    --season          Constant seasonal forcing level. Default 0.6.
    --fit-start       First week of the growth-rate fit. Default 2.
    --fit-end         Last week of the growth-rate fit. Default 6.
    --eta-grid        Adulticide efficacies to sweep.
    --max-k           Default 12.
    --csv             Output path.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage_threshold import GRAPH_CONFIGS, make_env
from src.exp_all import (mode_selfcheck, mode_delay, mode_prereg,
                         mode_channels, mode_efficacy)
from src import instances_ext

MODES = {"selfcheck": mode_selfcheck, "delay": mode_delay,
         "prereg": mode_prereg, "channels": mode_channels,
         "efficacy": mode_efficacy}

FULL_GRID = ("0.10,0.20,0.30,0.40,0.42,0.44,0.46,0.48,0.50,"
             "0.52,0.54,0.56,0.58,0.60,0.62,0.75,0.90")


def default_csv(mode, name, args):
    if mode == "delay":
        suffix = "seasonal" if args.seasonal else f"const{args.season:g}"
        stem = f"exp_delay_{name}_{suffix}_full"
    else:
        stem = f"exp_{mode}_{name}_full"
    return os.path.join("evaluation_dengue", f"{stem}.csv")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", choices=sorted(MODES),
                   help="same modes as exp_all.py")
    p.add_argument("--instance", default=None,
                   help='Instance name, as listed by --list.')
    p.add_argument("--all-instances", dest="all_instances", action="store_true",
                   help='Run every instance in turn.')
    p.add_argument("--list", action="store_true",
                   help="show which instances are available and stop")
    p.add_argument("--archetype_file",
                   default=os.path.join("configs", "archetypes.json"))
    p.add_argument("--archetype", default="Dengue_Dhaka_Advanced")
    p.add_argument("--econ", default=None,
                   help='Override the economic scenario.')
    p.add_argument("--lambda_sigma", type=float, default=0.8)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--burst", type=int, default=6)
    p.add_argument("--coverage", type=float, default=1.0)
    p.add_argument("--max-start", dest="max_start", type=int, default=12)
    p.add_argument("--seasonal", action="store_true")
    p.add_argument("--season", type=float, default=0.60)
    p.add_argument("--fit-start", dest="fit_start", type=int, default=2)
    p.add_argument("--fit-end", dest="fit_end", type=int, default=6)
    p.add_argument("--eta-grid", dest="eta_grid", default=FULL_GRID,
                   help='Adulticide efficacies to sweep.')
    p.add_argument("--max-k", dest="max_k", type=int, default=12)
    p.add_argument("--csv", default=None,
                   help='Output path.')
    return p


def show_list(new):
    print("=" * 80)
    print("AVAILABLE INSTANCES")
    print("=" * 80)
    print("  existing:  " + ", ".join(k for k in GRAPH_CONFIGS if k not in new))
    built = sorted(new)
    print("  built:     " + (", ".join(built) if built else "none"))
    gone = instances_ext.missing()
    if gone:
        print("  not built: " + ", ".join(gone))
        print("\n  build them with:  python t/make_instances.py --write")
    if built:
        print()
        print(instances_ext.HEAD)
        for tag in built:
            print(instances_ext.line(instances_ext.describe(tag)))


def run_one(name, args):
    cfg = GRAPH_CONFIGS[name]
    print("\n" + "#" * 80)
    print(f"# {args.mode}   instance {name}   graph {cfg['graph']}")
    print(f"# econ {cfg['econ']}   lambda_sigma {args.lambda_sigma}")
    d = instances_ext.describe(name)
    if d:
        print(f"# {d['n_wards']} wards, {d['edges']} edges, median degree "
              f"{d['deg_median']:.1f}, {d['isolated']} isolated, "
              f"pop CV {d['pop_cv']:.3f}")
    print("#" * 80)

    env = make_env(args, cfg)
    env.reset(seed=0)

    if args.mode != "selfcheck":
        print("\nrunning self-check first\n")
        if not mode_selfcheck(env, args, name, cfg):
            print("\nself-check failed -- refusing to run the experiment")
            sys.exit(1)
        print()

    MODES[args.mode](env, args, name, cfg)


def main():
    args = build_parser().parse_args()

    new = instances_ext.registry(econ=args.econ)
    for k, v in new.items():
        GRAPH_CONFIGS.setdefault(k, v)
        if args.econ:
            GRAPH_CONFIGS[k]["econ"] = args.econ

    if args.list:
        show_list(new)
        return
    if args.mode is None:
        print("pass a mode, or --list. See --help.")
        sys.exit(1)

    if args.all_instances:
        names = sorted(new)
        if not names:
            print("no generalisation instances built. "
                  "Run: python t/make_instances.py --write")
            sys.exit(1)
        if args.csv:
            print("--csv cannot be combined with --all-instances")
            sys.exit(1)
    elif args.instance:
        names = [args.instance]
        if args.instance not in GRAPH_CONFIGS:
            print(f"unknown instance '{args.instance}'. Known: "
                  + ", ".join(sorted(GRAPH_CONFIGS)))
            sys.exit(1)
    else:
        print("pass --instance NAME or --all-instances")
        sys.exit(1)

    targets, skipped = {}, []
    for n in names:
        path = args.csv or default_csv(args.mode, n, args)
        if args.mode != "selfcheck" and os.path.exists(path):
            if args.all_instances:
                skipped.append((n, path))
                continue
            print(f"output already exists: {path}")
            print("results files are never replaced. Pass --csv with a new name.")
            sys.exit(1)
        targets[n] = path

    if skipped:
        print("already done, skipping:")
        for n, path in skipped:
            print(f"  {n:<16} {path}")
        print()
    if not targets:
        print("nothing left to run.")
        return

    for n in list(targets):
        args.csv = targets[n]
        run_one(n, args)

    print("\n" + "=" * 80)
    print("wrote:")
    for n in targets:
        print(f"  {targets[n]}")
    print("\nnext:  python src/find_crossings.py "
          "--glob \"evaluation_dengue/exp_efficacy*_full.csv\"")


if __name__ == "__main__":
    main()
