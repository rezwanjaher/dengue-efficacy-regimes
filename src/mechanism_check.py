#!/usr/bin/env python3
"""mechanism_check.py -- what produces the two efficacy thresholds?

For efficacies on both sides of each threshold, this runs three campaigns at
the same spend (seasonal forcing, 40 seeds, the published 20-week season):
  burst1  six weeks of full adulticide from week 1
  burst2  the same six weeks, one week later
  sust    coverage 0.3 every week (equal total spend)
and records, as means over seeds: the peak, the week it falls in, the share of
residents still susceptible when the campaign ends and at season end, and the
infections when the campaign ends. If susceptible depletion drives the
thresholds, the losing campaign should end with a high susceptible share and
a later, larger peak.

Nothing in src/ is edited; source MD5s are checked before and after. The
trajectory function below copies exp_all.run_traj and adds the susceptible
series; a self-check confirms it returns the same peaks.

Run from the repository root:
    python src/mechanism_check.py --dncc
    python src/mechanism_check.py --dncc10
Outputs go to evaluation_dengue/mechanism/.
The outputs shipped with the release are already in that folder, and the
script refuses to overwrite them: rename the folder first to rerun.
"""
import argparse, copy, csv, hashlib, os, sys, datetime as dt
import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
SRC = ["src/exp_all.py", "src/coverage_threshold.py",
       "src/env/smip_env.py", "src/env/physics.py", "configs/archetypes.json"]
ETAS = [0.30, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.62, 0.75]
OUT = os.path.join("evaluation_dengue", "mechanism")
H, T, TRIALS = 20, 6, 40


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8"); self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()


def md5s():
    return {p: hashlib.md5(open(p, "rb").read()).hexdigest() for p in SRC}


def traj(env, seed, schedule, params, run_dynamics_step):
    env.reset(seed=seed)
    tH = {k: v.copy() for k, v in env.true_H.items()}
    tM = {k: v.copy() for k, v in env.true_M.items()}
    rng = np.random.default_rng(seed)
    n = env.num_sites
    N = float(tH["N"].sum())
    cw = int(env.calendar_week)
    I, S = [], []
    for w in range(H):
        u = float(schedule[w])
        acts = {ch: np.full(n, u if ch == "vector_control" else 0.0)
                for ch in ("vector_control", "case_management", "larviciding")}
        tH, tM = run_dynamics_step(tH, tM, acts, cw, rng, params, env.neighbors)
        I.append(float(tH["I"].sum())); S.append(float(tH["S"].sum()) / N)
        cw += 1
    return np.array(I), np.array(S)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dncc", action="store_true")
    g.add_argument("--dncc10", action="store_true")
    a = ap.parse_args()
    name = "dncc" if a.dncc else "dncc10"
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f"mechanism_{name}.csv")
    log = os.path.join(OUT, f"mechanism_{name}.log")
    for p in (out_csv, log):
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")
    sys.stdout = Tee(log)
    print(f"mechanism_check  {name}  {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    before = md5s()
    for p, m in before.items():
        print(f"  md5 {m}  {p}")

    import src.exp_all as X
    from src.coverage_threshold import make_env, GRAPH_CONFIGS
    from src.env.physics import run_dynamics_step
    sys.argv = ["x"]
    args = argparse.Namespace(archetype_file=os.path.join("configs", "archetypes.json"),
                              archetype="Dengue_Dhaka_Advanced", lambda_sigma=0.8,
                              horizon=H)
    env = make_env(args, GRAPH_CONFIGS[name])
    env.reset(seed=0)

    def params_at(eta):
        p = copy.deepcopy(env.ep_params)
        p["action_semantics"]["vector_control"]["efficacy"] = eta
        return p

    sched = {"burst1": [1.0] * T + [0.0] * (H - T),
             "burst2": [0.0] + [1.0] * T + [0.0] * (H - T - 1),
             "sust": [T / H] * H}
    end = {"burst1": T, "burst2": T + 1, "sust": H}

    print("\nself-check against exp_all.run_traj (2 seeds, eta 0.54, burst1)")
    p = params_at(0.54)
    for s in (0, 1):
        mine = traj(env, s, sched["burst1"], p, run_dynamics_step)[0].max()
        ref = X.run_traj(env, s, H, sched["burst1"], ("vector_control",), p)["peak"]
        print(f"  seed {s}: {mine:.4f} vs {ref:.4f}")
        if abs(mine - ref) > 1e-6 * max(1.0, ref):
            sys.exit("  MISMATCH; stopping")
    print("  match")

    rows = []
    print(f"\n{'eta':>5} {'sched':>7} {'peak':>10} {'pk_wk':>6} {'S_end_campaign':>15}"
          f" {'S_season_end':>13} {'I_end_campaign':>15}")
    for eta in ETAS:
        p = params_at(eta)
        for k, sc in sched.items():
            pk, wk, s_end, s_fin, i_end = [], [], [], [], []
            for s in range(TRIALS):
                I, S = traj(env, s, sc, p, run_dynamics_step)
                pk.append(I.max()); wk.append(int(I.argmax()) + 1)
                s_end.append(S[end[k] - 1]); s_fin.append(S[-1]); i_end.append(I[end[k] - 1])
            r = dict(config=name, eta_F=eta, schedule=k, peak=round(np.mean(pk), 1),
                     peak_week_mean=round(np.mean(wk), 2),
                     peak_after_campaign_share=round(np.mean(np.array(wk) > end[k]), 3),
                     S_share_end_campaign=round(np.mean(s_end), 4),
                     S_share_season_end=round(np.mean(s_fin), 4),
                     I_end_campaign=round(np.mean(i_end), 1), trials=TRIALS)
            rows.append(r)
            print(f"{eta:>5.2f} {k:>7} {r['peak']:>10,.0f} {r['peak_week_mean']:>6.1f}"
                  f" {r['S_share_end_campaign']:>15.4f} {r['S_share_season_end']:>13.4f}"
                  f" {r['I_end_campaign']:>15,.0f}")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_csv}")
    print("source files unchanged:", before == md5s())


if __name__ == "__main__":
    main()
