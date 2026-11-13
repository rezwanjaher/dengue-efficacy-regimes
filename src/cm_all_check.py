#!/usr/bin/env python3
"""cm_all_check.py -- the detection sweep with case management charged on all
estimated infections.

In src/evaluate_dengue.py, test-and-target pays for case management per
DETECTED infection (cost_to_cm = I_found[i] * c_cm), while the physics applies
case management to every infectious resident of the ward. Better detection
then raises the bill without raising the effect. This script charges per
ESTIMATED infection instead (I_hat_debiased[i] * c_cm), so the cost no longer
depends on the detection rate, and reruns the published detection sweep.

Nothing in src/ is edited. The one line is replaced in memory: the module's
source is read, the line swapped (exactly one match required), and the result
executed into the already-imported module before the harness imports it.
Source MD5s are checked before and after.

Uses the trained agents. Run from the repository root, one efficacy at a time:
    python src/cm_all_check.py --eta-f 0.40
    python src/cm_all_check.py --eta-f 0.45
    python src/cm_all_check.py --eta-f 0.54
Logs go to evaluation_dengue/sweeps_cm_all/, one per detection level, and are
read the same way as the logs in evaluation_dengue/sweeps_ls08/.
The outputs shipped with the release are already in that folder, and the
script refuses to overwrite them: rename the folder first to rerun.
"""
import argparse, hashlib, os, sys, datetime as dt

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
SRC = ["src/evaluate_dengue.py", "src/run_all_evals.py", "src/sweep_surveillance.py",
       "src/env/smip_env.py", "src/env/physics.py", "configs/archetypes.json"]
OLD = "cost_to_cm = I_found[i] * c_cm"
NEW = "cost_to_cm = I_hat_debiased[i] * c_cm"
LOGDIR = os.path.join("evaluation_dengue", "sweeps_cm_all")


def md5s():
    return {p: hashlib.md5(open(p, "rb").read()).hexdigest() for p in SRC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta-f", dest="eta_f", type=float, required=True)
    a = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)
    tag = f"etaF{a.eta_f}"
    if any(tag in f for f in os.listdir(LOGDIR)):
        sys.exit(f"logs for {tag} already exist in {LOGDIR}; refusing to overwrite")
    print(f"cm_all_check  eta_F={a.eta_f}  {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    before = md5s()
    for p, m in before.items():
        print(f"  md5 {m}  {p}")

    import src.evaluate_dengue as ed
    src = open(ed.__file__, encoding="utf-8").read()
    n = src.count(OLD)
    if n != 1:
        sys.exit(f"expected exactly one '{OLD}' in {ed.__file__}, found {n}; stopping")
    exec(compile(src.replace(OLD, NEW), ed.__file__, "exec"), ed.__dict__)
    print(f"  patched in memory: {OLD}  ->  {NEW}")

    import src.sweep_surveillance as sw
    sys.argv = ["sweep_surveillance.py", "--param", "eta_S",
                "--eta-f", str(a.eta_f), "--logdir", LOGDIR]
    sw.main()

    import src.run_all_evals as H
    if H.run_heuristic_week is not ed.run_heuristic_week:
        print("WARNING: the harness did not use the patched function; discard these logs")
    else:
        print("  harness used the patched function")
    print("source files unchanged:", before == md5s())


if __name__ == "__main__":
    main()
