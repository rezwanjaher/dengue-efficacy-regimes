#!/usr/bin/env python3
"""Checks run before training: the untreated epidemic rises to a single peak,
the initial infection survives the first step, R0 lies in the literature band,
and intervention changes the outcome. Called by train.py, or on its own:

    python src/preflight_epidemic.py --graph data/dncc_graph.json         --econ DNCC_Seasonal_Budget

Options:
    --graph           Graph file. Default data/dncc_graph.json.
    --econ            Economic scenario. Default DNCC_Seasonal_Budget.
    --archetype_file  Parameter file.
    --archetype       Parameter set. Default Dengue_Dhaka_Advanced.
    --lambda_sigma    Risk-field dispersion. Default 0.8.
    --rho             Persistence of the ward risk field. Default 0.2.
    --regime          Season-start prior. Default estimated.
    --history         Past seasons behind the prior. Default 5.
    --horizon         Season length in weeks. Default 20.
    --substeps        fallback if the config does not declare
                      dynamics_substeps Default 1.
    --seeds           Epidemic seeds checked.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GENERATION_INTERVAL_DAYS = 16.0
MIN_STEPS_PER_GENERATION = 4.0
MAX_STEP_TRANSITION      = 0.50
R0_BAND                  = (1.5, 8.0)
DGHS_GROWTH_BAND         = (0.154, 0.185)
ATTACK_RATE_BAND         = (0.01, 0.80)

MIN_LOG_R2               = 0.75
MAX_LOCAL_MAXIMA         = 3
MAX_GROWTH_DROPS         = 3
MIN_SUCCESSIVE_RATIO     = 0.45

GROUP = "Urban Public Health"
_results = []


def _check(name, critical=True):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                msg = fn(*a, **k)
                print(f"  PASS  {name}"); print(f"          {msg}")
                _results.append((name, True, msg, critical))
                return True
            except AssertionError as e:
                tag = "FAIL" if critical else "WARN"
                print(f"  {tag}  {name}"); print(f"          {e}")
                _results.append((name, False, str(e), critical))
                return False
        return wrapped
    return deco


def _bio(cfg, archetype):
    return cfg["archetype_groups"][GROUP][archetype]["biological_facts"]


def closed_form_r0(bio, s=None):
    mm, hm = bio["mosquito_model"], bio["human_model"]
    a = float(mm["biting_rate_per_week"]);   mu = float(mm["mosquito_mortality_rate"])
    b = float(mm["mosquito_birth_rate"]);    K  = float(mm["carrying_capacity_multiplier"])
    sig = float(mm["extrinsic_incubation_rate"]); gam = float(hm["recovery_rate"])
    phm = float(mm["transmission_prob_human_to_mosquito"])
    pmh = float(mm["transmission_prob_mosquito_to_human"])
    if s is None:
        seas = bio.get("seasonality", {})
        s = float(seas.get("baseline", 0.2)) + float(seas.get("amplitude", 0.75))
    occ = 1.0 - mu / b
    m = K * max(occ, 0.0)
    return s * a**2 * phm * pmh * m / (mu * gam) * (sig / (sig + mu)), m, occ


def simulate_untreated(env, horizon, eval_seed):
    zero = np.full((env.num_sites, env.action_dim_per_node), -1.0, dtype=np.float32)
    zero[:, 0] = -20.0
    env.reset(seed=eval_seed)
    S0 = float(env.true_H["S"].sum())
    series = []
    for _ in range(horizon):
        env.step(zero.reshape(-1))
        series.append(float(env.true_H["I"].sum()))
    S1 = float(env.true_H["S"].sum())
    return np.array(series), (S0 - S1) / S0 if S0 > 0 else 0.0


def trajectory_diagnostics(y):
    y = np.asarray(y, float)
    pk = int(np.argmax(y))
    g = np.maximum(y[:pk + 1], 1e-9)
    if len(g) >= 3:
        x = np.arange(len(g))
        r2 = float(np.corrcoef(x, np.log(g))[0, 1] ** 2)
        slope = float(np.polyfit(x, np.log(g), 1)[0])
    else:
        r2, slope = 0.0, 0.0
    ratios = g[1:] / g[:-1] if len(g) > 1 else np.array([1.0])
    loc = [i for i in range(1, len(y) - 1)
           if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > 0.10 * y.max()]
    return dict(peak_week=pk + 1, peak=y.max(), r2=r2, slope=slope,
                n_maxima=len(loc), maxima_weeks=[i + 1 for i in loc],
                n_drops=int((ratios < 0.70).sum()),
                worst_ratio=float(ratios.min()) if len(ratios) else 1.0)


@_check("E1 resolution: enough integration steps per generation interval")
def e1_resolution(bio, substeps):
    dt_days = 7.0 / substeps
    spg = GENERATION_INTERVAL_DAYS / dt_days
    mm, hm = bio["mosquito_model"], bio["human_model"]
    rates = {"recovery": float(hm["recovery_rate"]),
             "mosquito_mortality": float(mm["mosquito_mortality_rate"]),
             "incubation": float(mm["extrinsic_incubation_rate"])}
    per_step = {k: v / substeps for k, v in rates.items()}
    worst = max(per_step, key=per_step.get)
    assert spg >= MIN_STEPS_PER_GENERATION, (
        f"only {spg:.1f} integration steps per {GENERATION_INTERVAL_DAYS:.0f}-day dengue "
        f"generation interval (need >= {MIN_STEPS_PER_GENERATION:.0f}). Timestep is "
        f"{dt_days:.2f} d with substeps={substeps}. Too few steps per generation makes the "
        f"compartments move in lockstep and the epidemic oscillates. FIX: integrate the "
        f"dynamics at daily resolution inside the weekly decision epoch (substeps=7). "
        f"Do NOT lower the transition rates to compensate -- that trades a numerical error "
        f"for a biological one.")
    assert per_step[worst] <= MAX_STEP_TRANSITION, (
        f"per-step transition '{worst}' = {per_step[worst]:.3f} exceeds "
        f"{MAX_STEP_TRANSITION}. At this size the compartment empties each step. "
        f"Increase substeps rather than lowering the rate.")
    return (f"{spg:.1f} steps/generation, dt={dt_days:.2f} d, "
            f"largest per-step transition {per_step[worst]:.3f} ({worst})")


@_check("E2 seeding: the initial infected cohort survives step 1")
def e2_seeding(series, bio):
    gam = float(bio["human_model"]["recovery_rate"])
    assert series[0] > 1.0, (
        f"only {series[0]:.3f} infected after week 1. The initial cohort was cleared "
        f"almost entirely in one step (recovery_rate={gam}). The epidemic then restarts "
        f"from noise, so week-1 decisions are made with no signal -- an artefact, not a "
        f"finding. Usually the same cause as E1.")
    return f"{series[0]:,.1f} infected after week 1"


@_check("E3 smoothness: the untreated epidemic does not oscillate")
def e3_smoothness(d):
    problems = []
    if d["n_maxima"] > MAX_LOCAL_MAXIMA:
        problems.append(f"{d['n_maxima']} local maxima above 10% of peak, at weeks "
                        f"{d['maxima_weeks']} (need <= {MAX_LOCAL_MAXIMA})")
    if d["n_drops"] > MAX_GROWTH_DROPS:
        problems.append(f"{d['n_drops']} weeks where infections fell >30% DURING the "
                        f"growth phase (need <= {MAX_GROWTH_DROPS})")
    if d["worst_ratio"] < MIN_SUCCESSIVE_RATIO:
        problems.append(f"worst week-on-week ratio during growth = {d['worst_ratio']:.3f} "
                        f"(need >= {MIN_SUCCESSIVE_RATIO})")
    assert not problems, (
        "the untreated trajectory oscillates instead of rising to a single peak: "
        + "; ".join(problems) +
        ". This is generation-cycle ringing from too coarse a timestep -- see E1. "
        "Nothing measured on an oscillating epidemic is reportable: 'peak' becomes "
        "whichever wave happens to be largest, and any policy that suppresses one wave "
        "scores as if it solved the season.")
    return (f"peak week {d['peak_week']}, {d['n_maxima']} local maxima, "
            f"{d['n_drops']} growth-phase drops, worst ratio {d['worst_ratio']:.3f}")


@_check("E3b growth phase is approximately exponential", critical=False)
def e3b_loglinearity(d):
    assert d["r2"] >= MIN_LOG_R2, (
        f"log-linear R2 over the growth phase = {d['r2']:.3f} (would prefer "
        f">= {MIN_LOG_R2}). If E3 PASSED, this is not ringing -- the epidemic is "
        f"smooth but not exponential, usually because R0 sits close to 1 or counts "
        f"are low enough that the early weeks are noisy. Re-fit "
        f"carrying_capacity_multiplier with src/calibrate_k.py and re-check. "
        f"If E3 also failed, treat this as confirmation of ringing.")
    return f"log-linear R2 = {d['r2']:.3f}, fitted growth {d['slope']:.3f}/wk"


@_check("E4 R0 is inside the literature band for a tropical zone")
def e4_r0(bio):
    r0, m, occ = closed_form_r0(bio)
    assert occ > 0, (f"vector occupancy 1-mu/b = {occ:.3f} <= 0: mortality exceeds birth "
                     f"rate, the vector population collapses")
    lo, hi = R0_BAND
    assert lo <= r0 <= hi, (
        f"closed-form R0 = {r0:.3f} at peak seasonality, outside the literature band "
        f"[{lo}, {hi}]. Liu et al. 2020 (Environ Res 182:109114) report a tropical-zone "
        f"mean of 3.44 across 65 studies. Adjust carrying_capacity_multiplier -- it is "
        f"the free parameter.")
    return f"R0 = {r0:.3f} at peak season, {m:.2f} vectors/person, occupancy {occ:.3f}"


@_check("E5 emergent growth rate matches the DGHS take-off band", critical=False)
def e5_growth(d):
    lo, hi = DGHS_GROWTH_BAND
    r = d["slope"]
    assert lo <= r <= hi, (
        f"emergent growth rate {r:.3f}/wk is outside the DGHS band [{lo}, {hi}] "
        f"(2024-2026 take-off). Recalibrate carrying_capacity_multiplier with "
        f"src/calibrate_k.py. WARNING only -- a single-episode fit is noisy; "
        f"calibrate_k.py averages over 30.")
    return f"emergent growth {r:.3f}/wk, inside [{lo}, {hi}]"


@_check("E6 attack rate is epidemiologically plausible")
def e6_attack(attack):
    lo, hi = ATTACK_RATE_BAND
    assert lo <= attack <= hi, (
        f"untreated attack rate {attack*100:.1f}% of the susceptible pool is outside "
        f"[{lo*100:.0f}%, {hi*100:.0f}%]. Below the band there is no epidemic to control; "
        f"above it the whole city is infected and every policy looks alike.")
    return f"{attack*100:.1f}% of susceptibles infected over the season"


@_check("E7 intervention changes the outcome")
def e7_response(env, horizon, eval_seed, untreated_peak):
    full = np.full((env.num_sites, env.action_dim_per_node), 1.0, dtype=np.float32)
    full[:, 0] = -20.0
    env.reset(seed=eval_seed)
    env.total_episode_budget *= 1e6
    env.budget_remaining_for_episode = env.total_episode_budget
    series = []
    for _ in range(horizon):
        env.step(full.reshape(-1))
        series.append(float(env.true_H["I"].sum()))
    treated = max(series)
    ratio = treated / untreated_peak if untreated_peak > 0 else 1.0
    assert ratio < 0.5, (
        f"full coverage with unlimited budget only reduces the peak to {ratio*100:.1f}% "
        f"of untreated ({treated:,.0f} vs {untreated_peak:,.0f}). Either the control "
        f"channels are not reaching the dynamics, or the instance is uncontrollable and "
        f"every policy comparison will be degenerate.")
    return f"full coverage cuts the peak to {ratio*100:.1f}% of untreated"


def run_epidemic_preflight(args):
    from src.env.smip_env import SMIPEnv

    cfg = json.load(open(args.archetype_file))
    bio = _bio(cfg, args.archetype)
    substeps = int(bio.get("dynamics_substeps",
                           cfg.get("dynamics_substeps", args.substeps)))

    print("=" * 92)
    print("EPIDEMIC PREFLIGHT")
    print(f"  graph {args.graph} | econ {args.econ} | horizon {args.horizon} "
          f"| lambda_sigma {args.lambda_sigma}")
    print(f"  schema {cfg.get('schema_version')} | dynamics substeps {substeps} "
          f"(dt = {7.0/substeps:.2f} days)")
    print("=" * 92)

    ok = e1_resolution(bio, substeps)

    env = SMIPEnv(archetype_filepath=args.archetype_file, species_group=GROUP,
                  species_name=args.archetype, economic_scenarios=[args.econ],
                  graph_filepath_override=args.graph, force_T_HORIZON=args.horizon,
                  lambda_sigma=args.lambda_sigma, rho=args.rho,
                  prior_regime=args.regime, prior_history=args.history)

    peaks = []
    for sd in args.seeds:
        series, attack = simulate_untreated(env, args.horizon, sd)
        d = trajectory_diagnostics(series)
        peaks.append((sd, series, attack, d))

    sd, series, attack, d = sorted(peaks, key=lambda t: t[3]["peak"])[len(peaks) // 2]
    print(f"\n  untreated trajectory, eval seed {sd} (median of {len(peaks)}):")
    print("   wk " + " ".join(f"{i+1:>8}" for i in range(len(series))))
    print("    I " + " ".join(f"{v:>8,.0f}" for v in series))
    print()

    ok &= e2_seeding(series, bio)
    all_smooth = all(e3_smoothness(t[3]) for t in peaks)
    ok &= all_smooth
    e3b_loglinearity(d)
    ok &= e4_r0(bio)
    e5_growth(d)
    ok &= e6_attack(attack)
    ok &= e7_response(env, args.horizon, sd, d["peak"])

    crit = [r for r in _results if not r[1] and r[3]]
    warn = [r for r in _results if not r[1] and not r[3]]
    print("\n" + "=" * 92)
    if crit:
        print(f"EPIDEMIC PREFLIGHT FAILED -- {len(crit)} critical check(s). DO NOT TRAIN.")
        for n, _, m, _ in crit:
            print(f"  - {n}")
        print("\nNothing measured on a badly-posed epidemic is reportable. Fix the model,")
        print("then re-run this, then calibrate K, then train.")
    else:
        print(f"EPIDEMIC PREFLIGHT PASSED"
              + (f" with {len(warn)} warning(s)" if warn else ""))
    print("=" * 92)
    return not crit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="data/dncc_graph.json")
    p.add_argument("--econ", default="DNCC_Seasonal_Budget")
    p.add_argument("--archetype_file", default=os.path.join("configs", "archetypes.json"))
    p.add_argument("--archetype", default="Dengue_Dhaka_Advanced")
    p.add_argument("--lambda_sigma", type=float, default=0.8)
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--regime", default="estimated")
    p.add_argument("--history", type=int, default=5)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--substeps", type=int, default=1,
                   help="fallback if the config does not declare dynamics_substeps")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 3, 7, 11, 14])
    a = p.parse_args()
    sys.exit(0 if run_epidemic_preflight(a) else 1)


if __name__ == "__main__":
    main()
