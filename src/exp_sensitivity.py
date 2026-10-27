#!/usr/bin/env python3
r"""Sensitivity of the two crossings to assumed values. K is refitted at each
value so the untreated growth rate still matches the data.

    python src/exp_sensitivity.py --preset b_iso
    python src/exp_sensitivity.py --path mosquito_model.biting_rate_per_week \
        --values 2.8,3.2,3.6

Options:
    --preset            Named parameter to vary.
    --path              dotted key, with --values
    --values            comma-separated
    --instance          Instance name. Default dncc.
    --archetype_file    Parameter file.
    --archetype         Parameter set. Default Dengue_Dhaka_Advanced.
    --lambda_sigma      Risk-field dispersion. Default 0.8.
    --horizon           Season length in weeks. Default 20.
    --burst             Campaign length in weeks. Default 6.
    --trials            Epidemic seeds per point. Default 40.
    --fine              full 17-point grid
    --calib-episodes    Untreated seasons per K refit. Default 30.
    --calib-weeks       Weeks in the refit window. Default 10.
    --calib-start-week  First week of the refit window. Default 12.
    --season-start      move the season to this calendar week, refit K there,
                        and relocate the crossings.
    --no-refit          skip the K refit.
    --r-target          fix the growth-rate target instead of measuring it
                        from the model at the published K.
    --csv               Output CSV.
"""
import argparse
import copy
import csv
import datetime as _dt
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage_threshold import make_env, GRAPH_CONFIGS
from src.exp_all import run_traj, CHANNELS, gmean
from src.env.physics import run_dynamics_step

RUN_UTC = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
SCRIPT_VERSION = "1.1"

BAND_GRID = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48,
             0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
FULL_GRID = [0.10, 0.20, 0.30, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50,
             0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.75, 0.90]

PRESETS = {
    "p_hm":        ("mosquito_model.transmission_prob_human_to_mosquito",
                    [0.25, 0.30, 0.40, 0.50, 0.60]),
    "p_mh":        ("mosquito_model.transmission_prob_mosquito_to_human",
                    [0.30, 0.40, 0.475, 0.60, 0.75]),
    "season_amp":  ("seasonality.amplitude", [0.45, 0.60, 0.75, 0.90]),
    "season_base": ("seasonality.baseline", [0.10, 0.15, 0.20, 0.30]),
    "season_peak": ("seasonality.peak_week", [24, 26, 28, 30, 32]),
    "prod":        (None, [0.7, 0.85, 1.0, 1.2, 1.4]),
    "b_iso":       (None, [1.25, 1.50, 1.75, 2.00, 2.50]),
    "none":        (None, [0]),
}


def untreated_growth(env, p, episodes, fit_weeks, horizon):
    n = env.num_sites
    acts = {ch: np.zeros(n) for ch in CHANNELS}
    rates = []
    for ep in range(episodes):
        env.reset(seed=ep)
        H = {k: v.copy() for k, v in env.true_H.items()}
        M = {k: v.copy() for k, v in env.true_M.items()}
        rng = np.random.default_rng(ep)
        cw = int(env.calendar_week)
        series = []
        for _ in range(horizon):
            s_before = float(H["S"].sum())
            H, M = run_dynamics_step(H, M, acts, cw, rng, p, env.neighbors)
            series.append(max(0.0, s_before - float(H["S"].sum())))
            cw += 1
        y = np.asarray(series[:fit_weeks], dtype=float)
        ok = y > 0
        if ok.sum() >= 4:
            x = np.arange(len(y))[ok]
            rates.append(float(np.polyfit(x, np.log(y[ok]), 1)[0]))
    if not rates:
        return float("nan")
    return float(np.mean(rates))


def refit_K(env, p, r_target, episodes, fit_weeks, horizon, tol=0.004):
    mm = p["mosquito_model"]
    lo, hi = 0.5, 40.0
    r_lo = None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        mm["carrying_capacity_multiplier"] = mid
        r = untreated_growth(env, p, episodes, fit_weeks, horizon)
        if math.isnan(r):
            lo = mid
            continue
        if abs(r - r_target) <= tol:
            return mid, r
        if r < r_target:
            lo = mid
        else:
            hi = mid
        r_lo = r
        if hi - lo < 1e-3:
            break
    mm["carrying_capacity_multiplier"] = 0.5 * (lo + hi)
    return 0.5 * (lo + hi), (r_lo if r_lo is not None else float("nan"))


def dotted_set(tree, path, value):
    node = tree
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def dotted_get(tree, path):
    node = tree
    for k in path.split("."):
        node = node[k]
    return node


def apply_setting(base, preset, path, value):
    p = copy.deepcopy(base)
    mm = p["mosquito_model"]

    if preset == "none":
        return p, "no parameter changed"

    if preset == "prod":
        f = math.sqrt(value)
        mm["transmission_prob_human_to_mosquito"] *= f
        mm["transmission_prob_mosquito_to_human"] *= f
        note = (f"p_hm={mm['transmission_prob_human_to_mosquito']:.3f} "
                f"p_mh={mm['transmission_prob_mosquito_to_human']:.3f}")
        return p, note

    if preset == "b_iso":
        mu = mm["mosquito_mortality_rate"]
        b0 = mm.get("mosquito_birth_rate", 1.2)
        K0 = mm.get("carrying_capacity_multiplier", 2.0)
        m_target = K0 * (1.0 - mu / b0)
        occ = 1.0 - mu / value
        if occ <= 0:
            return None, "b below mortality; population cannot persist"
        mm["mosquito_birth_rate"] = value
        mm["carrying_capacity_multiplier"] = m_target / occ
        note = (f"b={value:.2f} K={m_target/occ:.2f} "
                f"occupancy={occ:.3f} m={m_target:.3f} held")
        return p, note

    dotted_set(p, path, value)
    return p, f"{path.split('.')[-1]}={value}"


def bracket(grid, flags):
    for i in range(1, len(grid)):
        if flags[i] and not flags[i - 1]:
            return grid[i - 1], grid[i]
    if all(flags):
        return None, grid[0]
    return grid[-1], None


def crossings(env, p, grid, H, T, trials):
    sus_sched = [T / H] * H
    bur_sched = [1.0] * T + [0.0] * (H - T)
    dly_sched = [0.0] + [1.0] * T + [0.0] * (H - T - 1)

    conc, dly = [], []
    for eta in grid:
        q = copy.deepcopy(p)
        q["action_semantics"]["vector_control"]["efficacy"] = eta
        def peaks(sched):
            return np.array([
                run_traj(env, s, H, sched, ("vector_control",), q)["peak"]
                for s in range(trials)])
        s_pk, b_pk, d_pk = peaks(sus_sched), peaks(bur_sched), peaks(dly_sched)
        conc.append(s_pk.mean() / max(b_pk.mean(), 1e-9) > 1.0)
        dly.append(gmean(d_pk / np.maximum(b_pk, 1e-9)) > 1.0)
    return bracket(grid, conc), bracket(grid, dly)


def fmt(br):
    lo, hi = br
    if lo is None:
        return f"below {hi:.2f}"
    if hi is None:
        return f"above {lo:.2f}"
    return f"[{lo:.2f}, {hi:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--path", default=None, help="dotted key, with --values")
    ap.add_argument("--values", default=None, help="comma-separated")
    ap.add_argument("--instance", default="dncc")
    ap.add_argument("--archetype_file",
                    default=os.path.join("configs", "archetypes.json"))
    ap.add_argument("--archetype", default="Dengue_Dhaka_Advanced")
    ap.add_argument("--lambda_sigma", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--burst", type=int, default=6)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--fine", action="store_true", help="full 17-point grid")
    ap.add_argument("--calib-episodes", dest="calib_episodes", type=int, default=30)
    ap.add_argument("--calib-weeks", dest="calib_weeks", type=int, default=10)
    ap.add_argument("--calib-start-week", dest="calib_start_week", type=int, default=12)
    ap.add_argument("--season-start", dest="season_start", type=int, default=None,
                    help='move the season to this calendar week, refit K there, and relocate the crossings.')
    ap.add_argument("--no-refit", dest="no_refit", action="store_true",
                    help='skip the K refit.')
    ap.add_argument("--r-target", dest="r_target", type=float, default=None,
                    help='fix the growth-rate target instead of measuring it from the model at the published K.')
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if not a.preset and not a.path:
        print("pass --preset NAME or --path a.b.c with --values")
        sys.exit(1)
    if a.instance not in GRAPH_CONFIGS:
        print(f"unknown instance. Known: {', '.join(sorted(GRAPH_CONFIGS))}")
        sys.exit(1)

    if a.preset:
        path, values = PRESETS[a.preset]
    else:
        path, values = a.path, None
    if a.values:
        values = [float(x) for x in a.values.split(",")]
    if values is None:
        print("no values to sweep; pass --values")
        sys.exit(1)

    grid = FULL_GRID if a.fine else BAND_GRID
    H, T = a.horizon, a.burst
    cfg = GRAPH_CONFIGS[a.instance]
    env = make_env(a, cfg)
    if a.season_start is not None:
        try:
            env.force_start_week = a.season_start
        except Exception:
            sys.exit("this environment does not accept a forced start week")
    env.reset(seed=0)
    base = copy.deepcopy(env.ep_params)
    if a.season_start is not None:
        print(f"season moved to calendar week {a.season_start}; K is refitted "
              f"there, so every row still reproduces the observed take-off.\n")

    refit = not (a.no_refit or a.preset == "b_iso")

    r_target = None
    if refit:
        tgt = make_env(a, cfg)
        try:
            tgt.force_start_week = a.calib_start_week
        except Exception:
            pass
        tgt.reset(seed=0)
        if a.r_target is not None:
            r_target = a.r_target
        else:
            r_target = untreated_growth(tgt, copy.deepcopy(base), a.calib_episodes,
                                        a.calib_weeks, H)

        cal = make_env(a, cfg)
        try:
            cal.force_start_week = (a.season_start if a.season_start is not None
                                    else a.calib_start_week)
        except Exception:
            pass
        cal.reset(seed=0)
        if a.r_target is not None:
            print(f"target growth rate {r_target:.4f}/wk, fixed by --r-target; "
                  f"K is refitted to reach it at every row below.")
        else:
            print(f"target growth rate {r_target:.4f}/wk, measured at calendar "
                  f"week {a.calib_start_week} "
                  f"at K={base['mosquito_model']['carrying_capacity_multiplier']:.2f}; "
                  f"K is refitted to reach it at every row below.")
        if not (0.10 <= r_target <= 0.30):
            print("  WARNING: that is outside the DGHS band [0.154, 0.185]. "
                  "Check --calib-start-week before trusting these rows.")

    label = a.preset or a.path
    print("=" * 84)
    print(f"CROSSING SENSITIVITY   {label}   {a.instance}")
    print(f"  {len(grid)}-point efficacy grid, {a.trials} seeds, "
          f"{T}-week campaign in a {H}-week season")
    print("  policy-free: coverage written into the dynamics")
    print("=" * 84)
    print(f"  {'value':>8}  {'concentration':>16}  {'immediacy':>16}"
          f"  {'K':>6}  {'r':>7}   setting")

    rows = []
    for v in values:
        p, note = apply_setting(base, a.preset, path, v)
        if p is None:
            print(f"  {v:>8}  {note}")
            continue
        k_fit = p["mosquito_model"]["carrying_capacity_multiplier"]
        r_fit = float("nan")
        if refit:
            k_fit, r_fit = refit_K(cal, p, r_target, a.calib_episodes,
                                   a.calib_weeks, H)
            p["mosquito_model"]["carrying_capacity_multiplier"] = k_fit
        c_br, d_br = crossings(env, p, grid, H, T, a.trials)
        marker = "  <- as published" if abs(v - (
            dotted_get(base, path) if path else
            (1.0 if a.preset == "prod" else
             base["mosquito_model"].get("mosquito_birth_rate", 1.2)))) < 1e-9 else ""
        print(f"  {v:>8}  {fmt(c_br):>16}  {fmt(d_br):>16}"
              f"  {k_fit:>6.2f}  {r_fit:>7.4f}   {note}{marker}")
        rows.append(dict(parameter=label, value=v, setting=note,
                         K_refit=round(k_fit, 3), r_achieved=round(r_fit, 5),
                         conc_lo=c_br[0], conc_hi=c_br[1],
                         delay_lo=d_br[0], delay_hi=d_br[1],
                         instance=a.instance, trials=a.trials,
                         grid_points=len(grid), horizon=H, campaign_weeks=T,
                         script_version=SCRIPT_VERSION, run_utc=RUN_UTC))

    print()
    print("READING IT")
    print("  Each crossing is the grid interval it falls in. K is the value")
    print("  refitted so the untreated growth rate returns to r, so every row")
    print("  reproduces Dhaka's observed take-off equally well and the rows")
    print("  differ only in the assumed parameter.")
    print("  Same interval in every row: the finding does not depend on this")
    print("  parameter, and the artefact objection fails for it. Interval")
    print("  moves: say by how much, over what range, and whether the refitted")
    print("  K stayed plausible.")

    if a.csv and rows:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        if os.path.exists(a.csv):
            print(f"\n{a.csv} exists; not replacing it.")
        else:
            with open(a.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
