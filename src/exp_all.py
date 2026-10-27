#!/usr/bin/env python3
"""Policy-free experiments on one instance: delay ratios, the pre-specified
timing test and the channel comparison.

    python src/exp_all.py delay    --dncc
    python src/exp_all.py delay    --dncc --seasonal
    python src/exp_all.py prereg   --dncc
    python src/exp_all.py channels --dncc

Use --dncc10 for the 10-ward instance.

Options:
    mode              Experiment to run. One of: selfcheck, delay, prereg,
                      channels, efficacy.
    --dncc            54-ward instance only.
    --dncc10          10-ward instance only.
    --archetype_file  Parameter file.
    --archetype       Parameter set. Default Dengue_Dhaka_Advanced.
    --lambda_sigma    Risk-field dispersion. Default 0.8.
    --horizon         Season length in weeks. Default 20.
    --trials          Epidemic seeds per point. Default 40.
    --burst           Campaign length in weeks. Default 6.
    --coverage        Campaign coverage. Default 1.0.
    --max-start       Latest start week tried. Default 12.
    --seasonal        operational seasonal forcing; otherwise forcing is held
                      constant
    --season          Constant seasonal forcing level. Default 0.6.
    --fit-start       First week of the growth-rate fit. Default 2.
    --fit-end         Last week of the growth-rate fit. Default 6.
    --eta-grid        Adulticide efficacies to sweep. Default
                      0.10,0.20,0.30,0.40,0.50,0.62,0.75,0.90.
    --max-k           longest campaign considered when locating the point
                      beyond which extra weeks buy nothing Default 12.
    --csv             Output CSV.
"""
import argparse, csv, datetime as _dt, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage_threshold import run_fixed, make_env, GRAPH_CONFIGS
from src.env.physics import run_dynamics_step

SCRIPT_VERSION = "1.0"
RUN_UTC = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
CHANNELS = ("vector_control", "case_management", "larviciding")
GENERATION_INTERVAL_WEEKS = 16.0 / 7.0


def run_traj(env, seed, horizon, schedule, channels, params=None):
    env.reset(seed=seed)
    true_H = {k: v.copy() for k, v in env.true_H.items()}
    true_M = {k: v.copy() for k, v in env.true_M.items()}
    rng = np.random.default_rng(seed)
    n = env.num_sites
    p = env.ep_params if params is None else params
    cw = int(env.calendar_week)
    peak = 0.0
    I_series, M_series = [], []
    for w in range(horizon):
        u = float(schedule[w])
        acts = {ch: np.full(n, u if ch in channels else 0.0) for ch in CHANNELS}
        true_H, true_M = run_dynamics_step(true_H, true_M, acts, cw, rng, p, env.neighbors)
        tot = float(true_H["I"].sum())
        I_series.append(tot)
        M_series.append(float(true_M["S"].sum() + true_M["E"].sum() + true_M["I"].sum()))
        peak = max(peak, tot)
        cw += 1
    return dict(peak=peak, I=np.array(I_series), M=np.array(M_series),
                start_calendar_week=int(env.calendar_week) - horizon)


def flat_params(env, level):
    import copy
    p = copy.deepcopy(env.ep_params)
    p["seasonality"] = {"forcing_type": "sinusoidal", "peak_week": 28,
                        "amplitude": 0.0, "baseline": level}
    return p


def gmean(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(np.maximum(x, 1e-12)).mean()))


def boot_ci(x, n=10000, alpha=0.05, seed=0, geometric=False):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    if geometric:
        lx = np.log(np.maximum(x, 1e-12))
        return tuple(np.exp(np.quantile(
            r.choice(lx, size=(n, len(lx)), replace=True).mean(1), [alpha/2, 1-alpha/2])))
    return tuple(np.quantile(
        r.choice(x, size=(n, len(x)), replace=True).mean(1), [alpha/2, 1-alpha/2]))


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {path}")


def mode_selfcheck(env, args, name, cfg):
    print("=" * 84)
    print("SELF-CHECK: does run_traj reproduce coverage_threshold.run_fixed?")
    print("=" * 84)
    H = args.horizon
    cases = [("no intervention", [0.0] * H, ("vector_control",)),
             ("sustained u=0.5", [0.5] * H, ("vector_control",)),
             ("6-week burst",   [1.0] * 6 + [0.0] * (H - 6), ("vector_control",)),
             ("larviciding u=1", [1.0] * H, ("larviciding",))]
    ok = True
    print(f"  {'case':<20}{'run_fixed':>14}{'run_traj':>14}{'match':>8}")
    for label, sched, chans in cases:
        for s in range(args.trials):
            a = run_fixed(env, s, H, sched, chans)[0]
            b = run_traj(env, s, H, sched, chans)["peak"]
            if abs(a - b) > 1e-6:
                ok = False
                print(f"  {label:<20}{a:>14,.1f}{b:>14,.1f}{'NO':>8}  seed {s}")
                break
        else:
            a = run_fixed(env, 0, H, sched, chans)[0]
            b = run_traj(env, 0, H, sched, chans)["peak"]
            print(f"  {label:<20}{a:>14,.1f}{b:>14,.1f}{'yes':>8}")
    print()
    if ok:
        print("  PASS. Trajectory stepping matches the project's own. Experiments below")
        print("  are measuring the same model coverage_threshold.py measures.")
    else:
        print("  FAIL. Do not trust any result from this script until fixed.")
    return ok


def mode_delay(env, args, name, cfg):
    H, T = args.horizon, args.burst
    params = None if args.seasonal else flat_params(env, args.season)
    starts = list(range(1, min(args.max_start, H - T) + 1))

    print("=" * 84)
    print(f"COST OF DELAY   {name}   {T}-week campaign at coverage {args.coverage}")
    print(f"  forcing: {'seasonal (operational)' if args.seasonal else f'constant s={args.season}'}")
    print(f"  {args.trials} seeds, shared across start weeks so ratios can be paired")
    print("  No control threshold enters this calculation.")
    print("=" * 84)

    peaks = {}
    for st in starts:
        sched = [0.0] * (st - 1) + [1.0] * T
        sched = (sched + [0.0] * H)[:H]
        peaks[st] = np.array([run_traj(env, s, H, sched, ("vector_control",), params)["peak"]
                              for s in range(args.trials)])
    base = peaks[1]

    print(f"\n  {'start':>6}{'mean peak':>13}{'ratio':>9}{'95% CI (paired)':>20}")
    rows = []
    for st in starts:
        r = peaks[st] / np.maximum(base, 1e-9)
        g = gmean(r); lo, hi = boot_ci(r, geometric=True)
        print(f"  {st:>6}{peaks[st].mean():>13,.0f}{g:>8.2f}x{f'[{lo:.2f}, {hi:.2f}]':>20}")
        rows.append(dict(config=name, start_week=st, mean_peak=round(peaks[st].mean(), 1),
                         sd=round(peaks[st].std(), 1), ratio=round(g, 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         forcing="seasonal" if args.seasonal else f"constant_{args.season}",
                         burst=T, trials=args.trials, n_wards=cfg["n_wards"],
                         script_version=SCRIPT_VERSION, run_utc=RUN_UTC))

    print("\n  THE OPERATIONAL CLAIM")
    for k, lab in ((2, "one week late"), (3, "two weeks late")):
        if k in peaks:
            r = peaks[k] / np.maximum(base, 1e-9)
            lo, hi = boot_ci(r, geometric=True)
            print(f"    {lab:<16} x{gmean(r):.2f}   95% CI [{lo:.2f}, {hi:.2f}]"
                  f"   {'supported' if lo > 1.0 else 'NOT SUPPORTED (CI includes 1.0)'}")
    ftag = "seasonal" if args.seasonal else f"const{args.season}"
    write_csv(args.csv or os.path.join("evaluation_dengue", f"exp_delay_{name}_{ftag}.csv"), rows)


def mode_prereg(env, args, name, cfg):
    bio = env.ep_params
    mm, hm = bio["mosquito_model"], bio["human_model"]
    etaF = bio["action_semantics"]["vector_control"]["efficacy"]
    m = mm["carrying_capacity_multiplier"] * (1 - mm["mosquito_mortality_rate"] / mm["mosquito_birth_rate"])
    R0 = (args.season * mm["biting_rate_per_week"]**2
          * mm["transmission_prob_human_to_mosquito"] * mm["transmission_prob_mosquito_to_human"]
          * m / (mm["mosquito_mortality_rate"] * hm["recovery_rate"])
          * mm["extrinsic_incubation_rate"] / (mm["extrinsic_incubation_rate"] + mm["mosquito_mortality_rate"]))
    r0 = np.log(R0) / GENERATION_INTERVAL_WEEKS
    r1 = np.log(max(R0 * (1 - etaF), 1e-12)) / GENERATION_INTERVAL_WEEKS
    H, T = args.horizon, args.burst
    tau = (H - T) + (r1 / r0) * T

    print("=" * 84)
    print("PRE-REGISTERED PREDICTION  (computed before the model is stepped)")
    print("=" * 84)
    print(f"  constant forcing s = {args.season}, campaign {T} wk, horizon {H}")
    print(f"  R0 = {R0:.4f}   r0 = {r0:+.4f}/wk   r1 = {r1:+.4f}/wk")
    print(f"  fitting window: start weeks {args.fit_start}-{args.fit_end}, FIXED IN ADVANCE")
    print(f"  P1 slope = r0 = {r0:.4f}/wk     P2 two regimes beat one     P3 break at {tau:.1f} wk")
    print("=" * 84)

    params = flat_params(env, args.season)
    starts = list(range(1, H - T + 1))
    rows = []
    print(f"\n  {'start':>6}{'mean peak':>14}{'log peak':>11}")
    for st in starts:
        sched = [0.0] * (st - 1) + [1.0] * T
        sched = (sched + [0.0] * H)[:H]
        pk = np.array([run_traj(env, s, H, sched, ("vector_control",), params)["peak"]
                       for s in range(args.trials)])
        lp = float(np.log(max(pk.mean(), 1e-9)))
        print(f"  {st:>6}{pk.mean():>14,.1f}{lp:>11.4f}")
        rows.append(dict(config=name, start_week=st, mean_peak=round(pk.mean(), 1),
                         log_peak=round(lp, 4), pred_r0=round(r0, 4),
                         pred_tau_star=round(tau, 2), n_wards=cfg["n_wards"],
                         script_version=SCRIPT_VERSION, run_utc=RUN_UTC))

    x = np.array([r["start_week"] for r in rows], float)
    y = np.array([r["log_peak"] for r in rows], float)
    win = (x >= args.fit_start) & (x <= args.fit_end)
    slope = float(np.polyfit(x[win], y[win], 1)[0])
    xf, yf = x[x >= args.fit_start], y[x >= args.fit_start]
    best = None
    for k in range(1, len(xf) - 1):
        sl, ic = np.polyfit(xf[:k+1], yf[:k+1], 1)
        lvl = yf[k+1:].mean()
        res = np.sum((yf[:k+1] - (sl*xf[:k+1] + ic))**2) + np.sum((yf[k+1:] - lvl)**2)
        if best is None or res < best[0]:
            best = (res, xf[k])
    r2_lin = np.corrcoef(xf, yf)[0, 1]**2
    r2_two = 1 - best[0] / np.sum((yf - yf.mean())**2)

    print("\n  VERDICT")
    print(f"    P1 slope {slope:+.4f} vs {r0:+.4f}   "
          f"{'MATCH' if abs(slope-r0) < 0.35*abs(r0) else 'MISS'}")
    print(f"    P2 R2 {r2_lin:.3f} -> {r2_two:.3f}   "
          f"{'MATCH' if r2_two > r2_lin + 0.02 else 'MISS'}")
    print(f"    P3 break {best[1]:.1f} vs {tau:.1f} wk   "
          f"{'MATCH' if abs(best[1]-tau) <= 2.0 else 'MISS'}")
    write_csv(args.csv or os.path.join("evaluation_dengue", f"exp_prereg_{name}.csv"), rows)


def mode_channels(env, args, name, cfg):
    H = args.horizon
    arms = {"none": ((), 0.0),
            "larviciding": (("larviciding",), args.coverage),
            "adulticide": (("vector_control",), args.coverage)}
    print("=" * 84)
    print(f"CHANNEL TRAJECTORIES   {name}   coverage {args.coverage}   "
          f"{args.trials} seeds   seasonal forcing")
    print("=" * 84)
    res = {}
    for nm, (chans, cov) in arms.items():
        sched = [cov] * H
        runs = [run_traj(env, s, H, sched, chans) for s in range(args.trials)]
        res[nm] = dict(I=np.mean([r["I"] for r in runs], axis=0),
                       M=np.mean([r["M"] for r in runs], axis=0))

    print("\n  INFECTIONS")
    print("   wk " + "".join(f"{n:>14}" for n in arms))
    for w in range(H):
        print(f"  {w+1:>3} " + "".join(f"{res[n]['I'][w]:>14,.0f}" for n in arms))
    print("\n  ADULT VECTOR ABUNDANCE (millions)")
    print("   wk " + "".join(f"{n:>14}" for n in arms))
    for w in range(H):
        print(f"  {w+1:>3} " + "".join(f"{res[n]['M'][w]/1e6:>14.3f}" for n in arms))

    pk = {n: int(np.argmax(res[n]["I"])) + 1 for n in arms}
    print("\n  PEAK WEEK / PEAK VALUE")
    for n in arms:
        print(f"    {n:<14} week {pk[n]:>2}   {res[n]['I'].max():>12,.0f}")
    print("\n  week-1 infections, larviciding vs none: "
          f"{res['larviciding']['I'][0]:,.0f} vs {res['none']['I'][0]:,.0f}")
    print(f"  week-1 abundance,   larviciding vs none: "
          f"{res['larviciding']['M'][0]/1e6:.3f}M vs {res['none']['M'][0]/1e6:.3f}M")

    rows = [dict(config=name, week=w+1,
                 **{f"I_{n}": round(float(res[n]["I"][w]), 1) for n in arms},
                 **{f"M_{n}": round(float(res[n]["M"][w]), 1) for n in arms},
                 n_wards=cfg["n_wards"], script_version=SCRIPT_VERSION, run_utc=RUN_UTC)
            for w in range(H)]
    write_csv(args.csv or os.path.join("evaluation_dengue", f"exp_channels_{name}.csv"), rows)


def mode_efficacy(env, args, name, cfg):
    import copy
    H, T = args.horizon, args.burst
    grid = [float(x) for x in args.eta_grid.split(",")]
    base_eta = env.ep_params["action_semantics"]["vector_control"]["efficacy"]

    print("=" * 96)
    print(f"ADULTICIDE EFFICACY SWEEP   {name}")
    print(f"  calibrated value {base_eta}; sweeping {grid}")
    print(f"  Dhaka bottle assays: permethrin 0.024-0.238, deltamethrin 0.489-0.943")
    print(f"  {args.trials} seeds, seasonal forcing, {T}-week campaign")
    print("  Policy-free: coverage is written into the dynamics, no policy involved.")
    print("=" * 96)

    def params_at(eta):
        p = copy.deepcopy(env.ep_params)
        p["action_semantics"]["vector_control"]["efficacy"] = eta
        return p

    def peaks(sched, p):
        return np.array([run_traj(env, s, H, sched, ("vector_control",), p)["peak"]
                         for s in range(args.trials)])

    none_sched = [0.0] * H
    rows = []
    print(f"\n  {'eta_F':>6}{'untreated':>11}{'sustained':>11}{'burst k*':>10}"
          f"{'burst peak':>12}{'front/sust':>12}{'1wk delay':>11}{'95% CI':>16}")
    for eta in grid:
        p = params_at(eta)
        untreated = peaks(none_sched, p).mean()

        u_sust = T / H
        sust = peaks([u_sust] * H, p).mean()

        floor, kstar = None, None
        for k in range(1, min(args.max_k, H) + 1):
            pk = peaks([1.0] * k + [0.0] * (H - k), p).mean()
            if floor is not None and pk > floor * 0.99:
                kstar = k - 1
                break
            floor = pk
        if kstar is None:
            kstar = min(args.max_k, H)
        burst = peaks([1.0] * T + [0.0] * (H - T), p).mean()

        b1 = peaks([1.0] * T + [0.0] * (H - T), p)
        b2 = peaks([0.0] + [1.0] * T + [0.0] * (H - T - 1), p)
        r = b2 / np.maximum(b1, 1e-9)
        g = gmean(r); lo, hi = boot_ci(r, geometric=True)

        adv = sust / max(burst, 1e-9)
        print(f"  {eta:>6.2f}{untreated:>11,.0f}{sust:>11,.0f}{kstar:>10}"
              f"{burst:>12,.0f}{adv:>11.2f}x{g:>10.2f}x{f'[{lo:.2f}, {hi:.2f}]':>16}")
        rows.append(dict(config=name, eta_F=eta, untreated=round(untreated, 1),
                         sustained=round(sust, 1), burst=round(burst, 1),
                         k_star=kstar, front_vs_sustained=round(adv, 4),
                         delay_1wk=round(g, 4), ci_lo=round(lo, 4),
                         ci_hi=round(hi, 4), trials=args.trials,
                         n_wards=cfg["n_wards"], script_version=SCRIPT_VERSION,
                         run_utc=RUN_UTC))

    print("\n" + "=" * 96)
    print("DO THE STRUCTURAL CONCLUSIONS SURVIVE?")
    print("=" * 96)
    s1 = [r for r in rows if r["front_vs_sustained"] > 1.0]
    s2 = [r for r in rows if r["ci_lo"] > 1.0]
    s3 = [r for r in rows if r["k_star"] < min(args.max_k, H)]
    print(f"  S1 front-loading beats sustained at equal spend : "
          f"{len(s1)}/{len(rows)} efficacies"
          + ("" if len(s1) == len(rows)
             else f"   fails at eta_F = "
                  f"{[r['eta_F'] for r in rows if r['front_vs_sustained'] <= 1.0]}"))
    print(f"  S2 one-week delay raises the peak (CI excludes 1) : "
          f"{len(s2)}/{len(rows)} efficacies"
          + ("" if len(s2) == len(rows)
             else f"   fails at eta_F = "
                  f"{[r['eta_F'] for r in rows if r['ci_lo'] <= 1.0]}"))
    print(f"  S3 a finite campaign reaches the peak floor      : "
          f"{len(s3)}/{len(rows)} efficacies")
    print()
    print("  The delay ratio is the quantity the paper leads with. Report its")
    print("  range across the sweep, not only its value at the calibrated point.")

    write_csv(args.csv or os.path.join("evaluation_dengue",
                                       f"exp_efficacy_{name}.csv"), rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["selfcheck", "delay", "prereg",
                                    "channels", "efficacy"])
    p.add_argument("--dncc", action="store_true")
    p.add_argument("--dncc10", action="store_true")
    p.add_argument("--archetype_file", default=os.path.join("configs", "archetypes.json"))
    p.add_argument("--archetype", default="Dengue_Dhaka_Advanced")
    p.add_argument("--lambda_sigma", type=float, default=0.8)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--burst", type=int, default=6)
    p.add_argument("--coverage", type=float, default=1.0)
    p.add_argument("--max-start", dest="max_start", type=int, default=12)
    p.add_argument("--seasonal", action="store_true",
                   help="operational seasonal forcing; otherwise forcing is held constant")
    p.add_argument("--season", type=float, default=0.60)
    p.add_argument("--fit-start", dest="fit_start", type=int, default=2)
    p.add_argument("--fit-end", dest="fit_end", type=int, default=6)
    p.add_argument("--eta-grid", dest="eta_grid",
                   default="0.10,0.20,0.30,0.40,0.50,0.62,0.75,0.90",
                   help='Adulticide efficacies to sweep.')
    p.add_argument("--max-k", dest="max_k", type=int, default=12,
                   help="longest campaign considered when locating the "
                        "point beyond which extra weeks buy nothing")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    name = "dncc" if args.dncc else "dncc10" if args.dncc10 else None
    if name is None:
        print("pass --dncc or --dncc10"); sys.exit(1)
    cfg = GRAPH_CONFIGS[name]
    env = make_env(args, cfg)
    env.reset(seed=0)

    if args.mode != "selfcheck":
        print("running self-check first\n")
        if not mode_selfcheck(env, args, name, cfg):
            print("\nself-check failed -- refusing to run the experiment")
            sys.exit(1)
        print()

    {"selfcheck": mode_selfcheck, "delay": mode_delay,
     "prereg": mode_prereg, "channels": mode_channels,
     "efficacy": mode_efficacy}[args.mode](env, args, name, cfg)


if __name__ == "__main__":
    main()
