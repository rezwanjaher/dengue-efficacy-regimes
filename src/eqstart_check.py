#!/usr/bin/env python3
"""eqstart_check.py -- does starting mosquitoes at equilibrium move the crossings?

The simulator starts every ward's mosquito population at the week's carrying
capacity. This script reruns the efficacy sweep (the 17-point grid, 40 seeds)
and the seasonal delay sweep with the population scaled to its equilibrium
share of capacity (default 0.534). Nothing in src/ is edited: SMIPEnv.reset is
wrapped in memory, and source MD5s are checked before and after.

Run from the repository root:
    python src/eqstart_check.py --dncc
    python src/eqstart_check.py --dncc10
Outputs go to evaluation_dengue/eqstart/. Step 1 checks that the wrapper, with
occupancy 1.0, reproduces the published row at efficacy 0.54 and stops if not.
The outputs shipped with the release are already in that folder, and the
script refuses to overwrite them: rename the folder first to rerun.
"""
import argparse, hashlib, os, sys, datetime as dt

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
SRC = ["src/exp_all.py", "src/coverage_threshold.py",
       "src/env/smip_env.py", "src/env/physics.py", "configs/archetypes.json"]
GRID = "0.10,0.20,0.30,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.75,0.90"
OUT = os.path.join("evaluation_dengue", "eqstart")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8"); self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()


def md5s():
    return {p: hashlib.md5(open(p, "rb").read()).hexdigest() for p in SRC}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dncc", action="store_true")
    g.add_argument("--dncc10", action="store_true")
    ap.add_argument("--occupancy", type=float, default=0.534)
    a = ap.parse_args()
    name = "dncc" if a.dncc else "dncc10"

    os.makedirs(OUT, exist_ok=True)
    eff_csv = os.path.join(OUT, f"exp_efficacy_{name}_eqstart.csv")
    del_csv = os.path.join(OUT, f"exp_delay_{name}_seasonal_eqstart.csv")
    ver_csv = os.path.join(OUT, f"verify_{name}_occ1.csv")
    log = os.path.join(OUT, f"eqstart_{name}.log")
    for p in (eff_csv, del_csv, ver_csv, log):
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")
    sys.stdout = Tee(log)
    print(f"eqstart_check  {name}  occupancy={a.occupancy}  "
          f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    before = md5s()
    for p, m in before.items():
        print(f"  md5 {m}  {p}")

    import src.exp_all as X
    from src.env.smip_env import SMIPEnv
    orig_reset = SMIPEnv.reset
    state = {"occ": 1.0}

    def reset(self, *args, **kw):
        out = orig_reset(self, *args, **kw)
        if state["occ"] != 1.0:
            for c in ("S", "E", "I"):
                self.true_M[c] = self.true_M[c] * state["occ"]
        return out
    SMIPEnv.reset = reset

    def run(argv):
        sys.argv = ["exp_all.py"] + argv
        X.main()

    flag = "--" + name
    print("\nSTEP 1  verify: with occupancy 1.0 the wrapper must reproduce the "
          "published sweep at eta_F = 0.54")
    state["occ"] = 1.0
    run(["efficacy", flag, "--eta-grid", "0.54", "--csv", ver_csv])
    import csv
    v = next(csv.DictReader(open(ver_csv)))
    pub = os.path.join("evaluation_dengue", f"exp_efficacy_{name}_full.csv")
    if os.path.exists(pub):
        p = [r for r in csv.DictReader(open(pub)) if abs(float(r["eta_F"]) - 0.54) < 1e-9][0]
        ok = all(abs(float(v[k]) - float(p[k])) < 1e-3
                 for k in ("front_vs_sustained", "delay_1wk", "k_star"))
        print(f"  wrapper  : fvs={v['front_vs_sustained']} delay={v['delay_1wk']} k*={v['k_star']}")
        print(f"  published: fvs={p['front_vs_sustained']} delay={p['delay_1wk']} k*={p['k_star']}")
        if not ok:
            sys.exit("  MISMATCH: the wrapper does not reproduce the published run; stopping")
        print("  match")
    else:
        print(f"  {pub} not found; cannot verify, stopping")
        sys.exit(1)

    print(f"\nSTEP 2  efficacy sweep with mosquitoes at {a.occupancy} of capacity")
    state["occ"] = a.occupancy
    run(["efficacy", flag, "--eta-grid", GRID, "--csv", eff_csv])

    print(f"\nSTEP 3  seasonal delay sweep at the reference efficacy, same start")
    run(["delay", flag, "--seasonal", "--csv", del_csv])

    after = md5s()
    print("\nsource files unchanged:", before == after)
    print("next: python src/find_crossings.py --glob "
          f"\"{OUT}/exp_efficacy_{name}_eqstart.csv\"")


if __name__ == "__main__":
    main()
