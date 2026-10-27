#!/usr/bin/env python3
"""Policy-free runs: fixed coverage schedules written directly into the
dynamics, with no agent, belief or allocator. Used by the timing experiments.

    python src/coverage_threshold.py --all

Options:
    --all             Both instances.
    --dncc10          10-ward instance only.
    --dncc            54-ward instance only.
    --archetype_file  Parameter file.
    --archetype       Parameter set. Default Dengue_Dhaka_Advanced.
    --lambda_sigma    Risk-field dispersion. Default 0.8.
    --horizon         Season length in weeks. Default 20.
    --trials          epidemic seeds per point Default 8.
    --control         peak below this = controlled Default 5000.0.
    --timing          Sweep the START week of a fixed-length full-coverage
                      burst.
    --burst_len       Burst length for the timing sweep, in weeks (default 7).
                      Default 7.
    --horizons        Comma-separated horizons for the long-horizon sweep.
    --csv             Output CSV.
"""
#              full-coverage fogging controls the outbreak for 0.78x the real
#              budget. But "spend early" and "spend in exactly the right 7-week
#              window" are very different operational asks. This sweeps the START
#              week of a fixed-length burst. If control degrades sharply with a
#              late start, the finding is fragile and needs stating that way.
#              inside 20 weeks -- it only reached 53,000 peak at u=1.0 sustained
#              on 54 wards, against 3,000 for fogging. That contradicts the WHO
#              position on source reduction and physics.py's own comment calling
#              larviciding the decisive lever. The likely explanation is that
#              larviciding acts on carrying capacity, so it suppresses BIRTHS and
#              takes generations to bite, while fogging kills the adults already
#              flying. This sweeps longer horizons to separate "does not work"
#              from "too slow for a 20-week season" -- a much more defensible and
#              more useful statement.
#             Sweeps k. The action traces showed front-loading beats spreading at
#             equal total spend, so u* is NOT a single number -- it depends on the
#             temporal profile. Comparing total spend at each strategy's control
#             point is the direct test of that.

import argparse
import csv
import datetime as _dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_VERSION = "1.0"
RUN_UTC = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

GRAPH_CONFIGS = {
    "dncc10": dict(graph="data/dncc10_graph.json", econ="DNCC10_Seasonal_Budget", n_wards=10),
    "dncc":   dict(graph="data/dncc_graph.json",   econ="DNCC_Seasonal_Budget",   n_wards=54),
}


def make_env(args, cfg):
    from src.env.smip_env import SMIPEnv
    return SMIPEnv(archetype_filepath=args.archetype_file,
                   species_group="Urban Public Health", species_name=args.archetype,
                   economic_scenarios=[cfg["econ"]], lambda_sigma=args.lambda_sigma,
                   graph_filepath_override=cfg["graph"], force_T_HORIZON=args.horizon)


def run_fixed(env, seed, horizon, schedule, channels):
    from src.env.physics import run_dynamics_step
    env.reset(seed=seed)
    true_H = {k: v.copy() for k, v in env.true_H.items()}
    true_M = {k: v.copy() for k, v in env.true_M.items()}
    rng = np.random.default_rng(seed)
    n = env.num_sites
    pop = np.asarray(env.dhaka_populations, dtype=float)
    aps = env.ep_params["action_semantics"]
    cw = int(env.calendar_week)
    peak, cum_inf, spend = 0.0, 0.0, 0.0

    for w in range(horizon):
        u = float(schedule[w])
        acts = {ch: np.full(n, u if ch in channels else 0.0) for ch in
                ("vector_control", "case_management", "larviciding")}
        for ch in channels:
            spend += u * float(pop.sum()) * float(aps[ch]["cost"])
        S0 = true_H["S"].copy()
        true_H, true_M = run_dynamics_step(true_H, true_M, acts, cw, rng,
                                           env.ep_params, env.neighbors)
        cum_inf += float(np.sum(np.maximum(0.0, S0 - true_H["S"])))
        peak = max(peak, float(true_H["I"].sum()))
        cw += 1
    return peak, cum_inf, spend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--dncc10", action="store_true")
    p.add_argument("--dncc", action="store_true")
    p.add_argument("--archetype_file", default=os.path.join("configs", "archetypes.json"))
    p.add_argument("--archetype", default="Dengue_Dhaka_Advanced")
    p.add_argument("--lambda_sigma", type=float, default=0.8)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--trials", type=int, default=8, help="epidemic seeds per point")
    p.add_argument("--control", type=float, default=5000.0, help="peak below this = controlled")
    p.add_argument("--timing", action="store_true",
                   help="Sweep the START week of a fixed-length full-coverage burst.")
    p.add_argument("--burst_len", type=int, default=7,
                   help='Burst length for the timing sweep, in weeks (default 7).')
    p.add_argument("--horizons", default=None,
                   help='Comma-separated horizons for the long-horizon sweep.')
    p.add_argument("--csv", default=os.path.join("evaluation_dengue", "coverage_threshold.csv"))
    args = p.parse_args()

    wanted = ["dncc10", "dncc"] if args.all else \
             [k for k, f in (("dncc10", args.dncc10), ("dncc", args.dncc)) if f]
    if not wanted:
        print("pass --all, --dncc or --dncc10")
        return

    U = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.00]
    KS = list(range(0, 13))
    CHANSETS = [("fogging", ("vector_control",)),
                ("larviciding", ("larviciding",)),
                ("both_50_50", ("vector_control", "larviciding"))]

    rows = []
    for name in wanted:
        cfg = GRAPH_CONFIGS[name]
        env = make_env(args, cfg)
        env.reset(seed=0)
        pop = float(np.sum(env.dhaka_populations))
        budget = float(env.total_episode_budget)
        print("\n" + "=" * 92)
        print(f"COVERAGE THRESHOLD  {name} ({cfg['n_wards']} wards)  "
              f"pop {pop:,.0f}  season budget {budget:,.0f} BDT")
        print("=" * 92)

        for cname, chans in CHANSETS:
            unit = sum(float(env.ep_params["action_semantics"][c]["cost"]) for c in chans)
            print(f"\n--- {cname}  (unit cost {unit:.1f} BDT/person/week) ---")

            print(f"\n  SUSTAINED: constant u for {args.horizon} weeks")
            print(f"    {'u':>6}{'peak':>12}{'controlled':>12}{'spend (M)':>12}{'x budget':>10}")
            ustar = None
            for u in U:
                sched = [u] * args.horizon
                res = [run_fixed(env, s, args.horizon, sched, chans) for s in range(args.trials)]
                pk = float(np.mean([r[0] for r in res]))
                sp = float(np.mean([r[2] for r in res]))
                ok = pk < args.control
                if ok and ustar is None:
                    ustar = u
                print(f"    {u:>6.2f}{pk:>12,.0f}{('YES' if ok else 'no'):>12}"
                      f"{sp/1e6:>12.1f}{sp/budget:>10.2f}")
                rows.append(dict(config=name, channel=cname, mode="sustained", u=u, weeks=args.horizon,
                                 peak=round(pk, 1), cum_infections=round(float(np.mean([r[1] for r in res])), 1),
                                 spend=round(sp, 1), spend_x_budget=round(sp / budget, 4),
                                 controlled=int(ok), n_wards=cfg["n_wards"],
                                 script_version=SCRIPT_VERSION, run_utc=RUN_UTC))
            print(f"    -> critical sustained coverage u* = "
                  f"{('%.2f' % ustar) if ustar is not None else '> 1.00 (never controls)'}")

            print(f"\n  BURST: u = 1.0 for the first k weeks, then nothing")
            print(f"    {'k':>6}{'peak':>12}{'controlled':>12}{'spend (M)':>12}{'x budget':>10}")
            kstar = None
            for k in KS:
                sched = [1.0] * k + [0.0] * (args.horizon - k)
                res = [run_fixed(env, s, args.horizon, sched, chans) for s in range(args.trials)]
                pk = float(np.mean([r[0] for r in res]))
                sp = float(np.mean([r[2] for r in res]))
                ok = pk < args.control
                if ok and kstar is None:
                    kstar = k
                print(f"    {k:>6d}{pk:>12,.0f}{('YES' if ok else 'no'):>12}"
                      f"{sp/1e6:>12.1f}{sp/budget:>10.2f}")
                rows.append(dict(config=name, channel=cname, mode="burst", u=1.0, weeks=k,
                                 peak=round(pk, 1), cum_infections=round(float(np.mean([r[1] for r in res])), 1),
                                 spend=round(sp, 1), spend_x_budget=round(sp / budget, 4),
                                 controlled=int(ok), n_wards=cfg["n_wards"],
                                 script_version=SCRIPT_VERSION, run_utc=RUN_UTC))
            print(f"    -> shortest controlling burst k* = "
                  f"{('%d weeks' % kstar) if kstar is not None else '> %d (never controls)' % max(KS)}")

            if ustar is not None and kstar is not None:
                s_sus = ustar * pop * unit * args.horizon
                s_bur = 1.0 * pop * unit * kstar
                cheaper = "BURST" if s_bur < s_sus else "SUSTAINED"
                print(f"\n    total spend to control:  sustained {s_sus/1e6:,.1f}M   "
                      f"burst {s_bur/1e6:,.1f}M   -> {cheaper} is cheaper "
                      f"({abs(s_bur - s_sus)/max(s_sus, 1):.0%} difference)")
                print(f"    as a fraction of the real season budget:  "
                      f"sustained {s_sus/budget:.2f}x   burst {s_bur/budget:.2f}x")

    if args.timing:
        for name in wanted:
            cfg = GRAPH_CONFIGS[name]
            env = make_env(args, cfg); env.reset(seed=0)
            pop = float(np.sum(env.dhaka_populations))
            budget = float(env.total_episode_budget)
            print("\n" + "=" * 92)
            print(f"TIMING SWEEP  {name}  --  {args.burst_len}-week full-coverage burst, "
                  f"varying the START week")
            print("  Total spend is IDENTICAL at every start week. Only the timing changes,")
            print("  so any difference here is purely about when the money lands.")
            print("=" * 92)
            for cname, chans in CHANSETS:
                unit = sum(float(env.ep_params["action_semantics"][c]["cost"]) for c in chans)
                sp = args.burst_len * pop * unit
                print(f"\n  {cname}   (spend {sp/1e6:,.0f}M = {sp/budget:.2f}x budget, "
                      f"the same for every start week)")
                print(f"    {'start wk':>9}{'peak':>12}{'controlled':>12}")
                for st in range(0, max(1, args.horizon - args.burst_len + 1)):
                    sched = [0.0] * st + [1.0] * args.burst_len
                    sched = (sched + [0.0] * args.horizon)[:args.horizon]
                    res = [run_fixed(env, s, args.horizon, sched, chans)
                           for s in range(args.trials)]
                    pk = float(np.mean([r[0] for r in res]))
                    ok = pk < args.control
                    print(f"    {st+1:>9d}{pk:>12,.0f}{('YES' if ok else 'no'):>12}")
                    rows.append(dict(config=name, channel=cname, mode="timing", u=1.0,
                                     weeks=args.burst_len, start_week=st + 1,
                                     peak=round(pk, 1),
                                     cum_infections=round(float(np.mean([r[1] for r in res])), 1),
                                     spend=round(sp, 1), spend_x_budget=round(sp / budget, 4),
                                     controlled=int(ok), n_wards=cfg["n_wards"],
                                     script_version=SCRIPT_VERSION, run_utc=RUN_UTC))

    if args.horizons:
        HS = [int(h) for h in args.horizons.split(",")]
        for name in wanted:
            cfg = GRAPH_CONFIGS[name]
            print("\n" + "=" * 92)
            print(f"HORIZON SWEEP  {name}  --  does larviciding control given more time?")
            print("  Larviciding suppresses carrying capacity, so it acts on BIRTHS and takes")
            print("  generations to bite. Fogging kills the adults already flying. If")
            print("  larviciding controls at a longer horizon, the honest claim is 'too slow")
            print("  for a 20-week season', not 'does not work'.")
            print("=" * 92)
            for H in HS:
                a2 = argparse.Namespace(**vars(args)); a2.horizon = H
                env = make_env(a2, cfg); env.reset(seed=0)
                pop = float(np.sum(env.dhaka_populations))
                print(f"\n  horizon = {H} weeks")
                print(f"    {'channel':<14}{'u':>6}{'peak':>12}{'controlled':>12}{'spend (M)':>12}")
                for cname, chans in CHANSETS:
                    unit = sum(float(env.ep_params["action_semantics"][c]["cost"]) for c in chans)
                    for u in (0.5, 1.0):
                        sched = [u] * H
                        res = [run_fixed(env, s, H, sched, chans) for s in range(args.trials)]
                        pk = float(np.mean([r[0] for r in res]))
                        sp = u * pop * unit * H
                        ok = pk < args.control
                        print(f"    {cname:<14}{u:>6.2f}{pk:>12,.0f}"
                              f"{('YES' if ok else 'no'):>12}{sp/1e6:>12.1f}")
                        rows.append(dict(config=name, channel=cname, mode="horizon", u=u, weeks=H,
                                         peak=round(pk, 1),
                                         cum_infections=round(float(np.mean([r[1] for r in res])), 1),
                                         spend=round(sp, 1), spend_x_budget=0.0,
                                         controlled=int(ok), n_wards=cfg["n_wards"],
                                         script_version=SCRIPT_VERSION, run_utc=RUN_UTC))

    if rows:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved {len(rows)} rows -> {args.csv}")
        print("\nREADING IT")
        print("  u*        the coverage that must be sustained to control. Multiply by")
        print("            population x unit cost x 20 to get the budget Dhaka needs.")
        print("  k*        weeks of full coverage that achieve the same, front-loaded.")
        print("  If burst spends LESS than sustained, front-loading is genuinely better")
        print("  and the threshold effect is real. If it spends MORE, the front-loading")
        print("  advantage we saw was an artifact of how the budget was paced.")


if __name__ == "__main__":
    main()
