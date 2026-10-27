#!/usr/bin/env python3
r"""Calibrate the carrying-capacity multiplier K to the observed growth rate of
dengue admissions. For each candidate K, runs untreated seasons and fits the
weekly growth rate over the take-off window.

    python src/calibrate_k.py --graph data/dncc_graph.json \
        --econ DNCC_Seasonal_Budget --k-grid 3.5,4.0,4.5,5.0,5.5,6.0 --episodes 30 \
        --out evaluation_dengue/calibration_k_full.csv

Options:
    --graph       Graph file. Default data/dncc_graph.json.
    --econ        Economic scenario. Default DNCC_Seasonal_Budget.
    --k-grid      Candidate values of K. Default 3.5,4.0,4.5,5.0,5.5,6.0.
    --episodes    Untreated seasons per K. Default 30.
    --horizon     Season length in weeks. Default 20.
    --fit-weeks   Weeks of the take-off used for the log-linear fit (before
                  saturation). Default 10.
    --start-week  Calendar week the episode starts. Default 12.
    --seed        Random seed.
    --set-k       Skip the sweep; just write this K into
                  configs/archetypes.json.
    --out         Output path. Default calibration_k.csv.
"""
import argparse, json, os, sys, tempfile, shutil
import numpy as np

PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

ARCH_PATH = os.path.join(PROJECT_ROOT, "configs", "archetypes.json")
GROUP = "Urban Public Health"
ARCHETYPES = ("Dengue_Dhaka_Base", "Dengue_Dhaka_Advanced")
DGHS_BAND = (0.154, 0.185)


def set_k_in_file(path, k):
    with open(path) as f:
        cfg = json.load(f)
    g = cfg["archetype_groups"][GROUP]
    for a in ARCHETYPES:
        g[a]["biological_facts"]["mosquito_model"]["carrying_capacity_multiplier"] = float(k)
    b1 = g[ARCHETYPES[0]]["biological_facts"]
    b2 = g[ARCHETYPES[1]]["biological_facts"]
    assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True), \
        "biological_facts differ between Base and Advanced -- refusing to write"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def closed_form_r0(cfg, s=0.95):
    m_ = cfg["archetype_groups"][GROUP]["Dengue_Dhaka_Advanced"]["biological_facts"]
    mm, hm = m_["mosquito_model"], m_["human_model"]
    a, mu, b = mm["biting_rate_per_week"], mm["mosquito_mortality_rate"], mm["mosquito_birth_rate"]
    K, sig = mm["carrying_capacity_multiplier"], mm["extrinsic_incubation_rate"]
    phm, pmh = mm["transmission_prob_human_to_mosquito"], mm["transmission_prob_mosquito_to_human"]
    gam = hm["recovery_rate"]
    m = K * max(0.0, 1 - mu / b)
    return s * a**2 * phm * pmh * m / (mu * gam) * (sig / (sig + mu)), m


def measure_growth(arch_path, graph, econ, horizon, episodes, seed, fit_weeks, start_week):
    from src.env.smip_env import SMIPEnv

    env = SMIPEnv(
        archetype_filepath=arch_path, species_group=GROUP,
        species_name="Dengue_Dhaka_Advanced", economic_scenarios=[econ],
        graph_filepath_override=graph, force_T_HORIZON=horizon,
        force_start_week=start_week,
        force_perfect_surveys=True, force_free_info=True,
        rho=0.2, prior_regime="estimated", prior_history=5, lambda_sigma=0.8,
    )
    zero = np.full((env.num_sites, env.action_dim_per_node), -1.0, dtype=np.float32)
    zero[:, 0] = -20.0
    rates = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        series = []
        for _ in range(horizon):
            _, _, term, _, info = env.step(zero.reshape(-1))
            series.append(float(info["new_infections"]))
            if term:
                break
        y = np.asarray(series[:fit_weeks], dtype=float)
        ok = y > 0
        if ok.sum() >= 4:
            x = np.arange(len(y))[ok]
            rates.append(float(np.polyfit(x, np.log(y[ok]), 1)[0]))
    if not rates:
        return float("nan"), float("nan"), 0
    return float(np.mean(rates)), float(np.std(rates)), len(rates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="data/dncc_graph.json")
    p.add_argument("--econ", default="DNCC_Seasonal_Budget")
    p.add_argument("--k-grid", default="3.5,4.0,4.5,5.0,5.5,6.0")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--fit-weeks", type=int, default=10,
                   help="Weeks of the take-off used for the log-linear fit (before saturation).")
    p.add_argument("--start-week", type=int, default=12,
                   help='Calendar week the episode starts.')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set-k", type=float, default=None,
                   help="Skip the sweep; just write this K into configs/archetypes.json.")
    p.add_argument("--out", default="calibration_k.csv")
    a = p.parse_args()

    if a.set_k is not None:
        set_k_in_file(ARCH_PATH, a.set_k)
        with open(ARCH_PATH) as f:
            r0, m = closed_form_r0(json.load(f))
        print(f"Wrote K={a.set_k} to BOTH archetypes in {ARCH_PATH}")
        print(f"  closed-form R0(0.95) = {r0:.3f}, vectors/person m = {m:.3f}")
        return

    ks = [float(x) for x in a.k_grid.split(",")]
    tmpdir = tempfile.mkdtemp(prefix="calib_k_")
    tmp_arch = os.path.join(tmpdir, "archetypes.json")
    rows = []
    import math
    _s = 0.2 + 0.75 * 0.5 * (1 + math.cos(2 * math.pi * (a.start_week - 28) / 52.0))
    print(f"start week {a.start_week} -> seasonality multiplier {_s:.3f} "
          f"(peak is 0.95 at week 28)")
    print(f"R_eff at the start week is R0(.95) x {_s/0.95:.3f}; if that is below 1 the "
          f"epidemic decays and r_emergent will be negative.\n")
    print(f"{'K':>6} {'m':>7} {'R0(.95)':>9} {'r_emergent':>11} {'sd':>7} {'n':>4}  {'in DGHS band':>13}")
    try:
        for k in ks:
            shutil.copy(ARCH_PATH, tmp_arch)
            set_k_in_file(tmp_arch, k)
            with open(tmp_arch) as f:
                r0, m = closed_form_r0(json.load(f))
            r, sd, n = measure_growth(tmp_arch, a.graph, a.econ, a.horizon,
                                      a.episodes, a.seed, a.fit_weeks, a.start_week)
            hit = "YES" if DGHS_BAND[0] <= r <= DGHS_BAND[1] else ""
            print(f"{k:>6.2f} {m:>7.3f} {r0:>9.3f} {r:>11.4f} {sd:>7.4f} {n:>4}  {hit:>13}")
            rows.append(dict(K=k, m=m, R0_095=r0, r_emergent=r, r_sd=sd, n_episodes=n))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")
    print(f"DGHS target band: r in [{DGHS_BAND[0]}, {DGHS_BAND[1]}] /wk")
    print("Pick the K whose r_emergent lands in the band, then:")
    print("  python src/calibrate_k.py --set-k <value>")
    rs = [r["r_emergent"] for r in rows]
    if all(r < 0 for r in rs if r == r):
        print("\n*** ALL growth rates are NEGATIVE. This is almost certainly the START WEEK,")
        print("*** not K. The epidemic decays outside the transmission season by design.")
        print(f"*** Current --start-week {a.start_week} gives seasonality {_s:.3f}.")
        print("*** Re-run with --start-week 12 (or later) and check the header line.")
    else:
        print("\nIf NO K lands in the band, do not proceed to training -- the discrepancy")
        print("is not in K, and training on an uncalibrated instance wastes the run.")


if __name__ == "__main__":
    main()
