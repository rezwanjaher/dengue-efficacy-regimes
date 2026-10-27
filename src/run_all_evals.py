#!/usr/bin/env python3
r"""Evaluation harness: every policy, budget multiplier and seed on one instance.

    python src/run_all_evals.py --all --version_tag V18 --lambda_sigma 0.8 \
        --gamma 0.99 --free-heuristics --survey-share 0.25 \
        --csv evaluation_dengue/v18_free_cap25.csv

Add --eta-F to set adulticide efficacy. Always pass --lambda_sigma 0.8 and
--version_tag V18 as above; the defaults differ. Each run writes a .meta.json
file beside its CSV with the full command.

Options:
    --rho               Persistence used at eval (match training). Default
                        0.2.
    --regime            Prior regime used at eval (match training). One of:
                        known, estimated, flat. Default estimated.
    --history           Prior history H (match training). Default 5.
    --lambda_sigma      Risk-field dispersion; the paper uses 0.8. Default
                        1.2.
    --graph             Graph file, e.g. data/dncc_graph.json.
    --archetype         Parameter set, e.g. Dengue_Dhaka_Advanced. Default
                        Dengue_Dhaka_Advanced.
    --econ              Economic scenario, e.g. DNCC_Seasonal_Budget.
    --budget            Season budget at x1.0, in BDT.
    --models_dir        Evaluate only the checkpoints in this folder.
    --all               Run both instances, using every seed folder under
                        --models_root.
    --dncc10            Batch mode: 10-ward graph only.
    --dncc              Batch mode: 54-ward DNCC graph only.
    --models_root       Root folder scanned for seed run directories in batch
                        mode.
    --V13               Only match run folders containing this version tag.
                        Default V13.
    --version_tag       Version tag of the run folders, e.g. V18. Default V13.
    --csv               CSV output path (default
                        <OUT_DIR>/master_results.csv).
    --mults             Comma-separated budget multipliers, e.g. 1.0,2.0.
    --survey-share      Cap on the share of the weekly budget surveillance may
                        claim, for the belief heuristics. Default 1.0.
    --eta-F             Override adulticide efficacy for the whole run.
    --gamma             Discount for the reported cumulative economic cost V.
                        Default 0.99.
    --no-strict-budget  Warn instead of aborting when a rollout overspends its
                        season budget.
    --free-heuristics   Give the heuristics the whole season budget from week
                        1, as the agent has.
    --checkpoint        Evaluate one checkpoint (.zip).
"""

import os, sys, glob, re, json, csv
import datetime as _dt
import subprocess as _sp
import numpy as np

sys.path.insert(0, os.getcwd())
from stable_baselines3 import PPO
from src.env.smip_env import SMIPEnv
from src.evaluate_dengue import run_heuristic_week
import src.evaluate_dengue as _ed

import argparse

BASE_BUDGET = 800_990_197

FREE_HEURISTICS = False

EVAL_GAMMA = 0.99
STRICT_BUDGET = True
BUDGET_TOL = 1e-6


SCRIPT_VERSION = "1.9"
SURVEY_SHARE_USED = 1.0
RUN_UTC = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
CHANNELS = ("vector_control", "case_management", "larviciding")
SPEND_KEYS = ("survey", "vc", "cm", "larv")


def _git_commit():
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        return None


def _unit_costs(env):
    out = {}
    for ch in CHANNELS:
        c = env.ep_params['action_semantics'][ch]['cost']
        out[ch] = float(np.mean(c)) if hasattr(c, '__len__') else float(c)
    return out


def _acc_coverage(acc, actions):
    if not actions:
        return
    for ch in CHANNELS:
        v = actions.get(ch)
        if v is not None:
            acc[ch] += float(np.mean(v))


def _audit_budget(tag, spent, budget):
    over = spent - budget
    if over > BUDGET_TOL * max(1.0, budget):
        msg = (f"BUDGET VIOLATION [{tag}]: spent {spent:,.2f} > budget "
               f"{budget:,.2f} (over by {over:,.2f}, {100*over/budget:.4f}%)")
        if STRICT_BUDGET:
            raise AssertionError(msg)
        print("  !! " + msg)
    return (spent / budget) if budget > 0 else 0.0


CONTROL_THRESHOLD = 5000
HORIZON = 20
ANNUALIZE = 52 / 20
BUDGET_MULTIPLES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
KAPPA_AT_1X = None
_POP_CACHE = None
REF_MULT = 2.0

EVAL_RHO = 0.2
EVAL_REGIME = "estimated"
EVAL_HISTORY = 5
EVAL_LAMBDA_SIGMA = 1.2
EVAL_MODELS_DIR = None
EVAL_CHECKPOINT = None
EVAL_GRAPH = None
EVAL_ARCHETYPE = "Dengue_Dhaka_Advanced"
EVAL_ECON = None
EVAL_ETA_F = None


def _apply_eta(env):
    if EVAL_ETA_F is None:
        return env
    env.ep_params['action_semantics']['vector_control']['efficacy'] = float(EVAL_ETA_F)
    return env


def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluation harness: every policy, budget multiplier and seed on one instance.')
    p.add_argument("--rho", type=float, default=0.2,
                   help='Persistence used at eval (match training).')
    p.add_argument("--regime", type=str, default="estimated",
                   choices=["known", "estimated", "flat"],
                   help='Prior regime used at eval (match training).')
    p.add_argument("--history", type=int, default=5, help="Prior history H (match training).")
    p.add_argument("--lambda_sigma", type=float, default=1.2, help='Risk-field dispersion; the paper uses 0.8.')
    p.add_argument("--graph", type=str, default=None,
                   help='Graph file, e.g. data/dncc_graph.json.')
    p.add_argument("--archetype", type=str, default="Dengue_Dhaka_Advanced",
                   help='Parameter set, e.g. Dengue_Dhaka_Advanced.')
    p.add_argument("--econ", type=str, default=None,
                   help='Economic scenario, e.g. DNCC_Seasonal_Budget.')
    p.add_argument("--budget", type=float, default=None,
                   help='Season budget at x1.0, in BDT.')
    p.add_argument("--models_dir", type=str, default=None,
                   help='Evaluate only the checkpoints in this folder.')
    p.add_argument("--all", action="store_true",
                   help='Run both instances, using every seed folder under --models_root.')
    p.add_argument("--dncc10", action="store_true", help="Batch mode: 10-ward graph only.")
    p.add_argument("--dncc", action="store_true", help="Batch mode: 54-ward DNCC graph only.")
    p.add_argument("--models_root", type=str, default=os.path.join("rl_models", "test"),
                   help="Root folder scanned for seed run directories in batch mode.")
    p.add_argument("--V13", dest="version_tag", action="store_const",
                   const="V13", default="V13",
                   help='Only match run folders containing this version tag.')
    p.add_argument("--version_tag", dest="version_tag", type=str, default="V13",
                   help='Version tag of the run folders, e.g. V18.')
    p.add_argument("--csv", type=str, default=None,
                   help="CSV output path (default <OUT_DIR>/master_results.csv).")
    p.add_argument("--mults", type=str, default=None,
                   help='Comma-separated budget multipliers, e.g. 1.0,2.0.')
    p.add_argument("--survey-share", type=float, default=1.0,
                   help='Cap on the share of the weekly budget surveillance may claim, for the belief heuristics.')
    p.add_argument("--eta-F", dest="eta_F", type=float, default=None,
                   help='Override adulticide efficacy for the whole run.')
    p.add_argument("--gamma", type=float, default=0.99,
                   help='Discount for the reported cumulative economic cost V.')
    p.add_argument("--no-strict-budget", action="store_true",
                   help="Warn instead of aborting when a rollout overspends its season budget.")
    p.add_argument("--free-heuristics", action="store_true",
                   help='Give the heuristics the whole season budget from week 1, as the agent has.')
    p.add_argument("--checkpoint", type=str, default=None,
                   help='Evaluate one checkpoint (.zip).')
    return p.parse_args()

HEURISTICS = [
    "H0_Do_Nothing",
    "H1_Blanket_Fogging",
    "H2_Test_and_Target",
    "H3_Random_Larviciding",
    "H4_Greedy_Belief",
    "H4_Greedy_Oracle",
    "H5_MyopicLP_Belief",
    "H5_MyopicLP_Oracle",
]
FAIR_POLICIES = ["H0_Do_Nothing", "H1_Blanket_Fogging", "H2_Test_and_Target",
                 "H3_Random_Larviciding", "H4_Greedy_Belief", "H5_MyopicLP_Belief",
                 "RL_Agent"]
ORACLE_POLICIES = ["H4_Greedy_Oracle", "H5_MyopicLP_Oracle"]
OUT_DIR = "evaluation_dengue"


GRAPH_CONFIGS = {
    "dncc10": dict(graph="data/dncc10_graph.json", econ="DNCC10_Seasonal_Budget",
                   budget=None, dir_tag="dncc10_graph", n_wards=10),
    "dncc":   dict(graph="data/dncc_graph.json",   econ="DNCC_Seasonal_Budget",
                   budget=None, dir_tag="dncc_graph",   n_wards=54),
}
LEGACY_BUDGETS = {"dncc10": 104_158_737, "dncc": 521_000_000}


def _resolve_budget(cfg, archetype_file, archetype, n_wards):
    with open(archetype_file) as f:
        j = json.load(f)
    sc = j["archetype_groups"]["Urban Public Health"][archetype]["economic_scenarios"][cfg["econ"]]
    per = sc["total_episode_budget_per_site"]
    if isinstance(per, dict):
        raise ValueError(
            f"{cfg['econ']}.total_episode_budget_per_site is a distribution, not a scalar. "
            f"Schema 18.0 requires point values; re-check archetypes.json.")
    return float(per) * int(n_wards)
MODELS_ROOT = os.path.join("rl_models", "test")
EVAL_ARCHETYPE_FILE = os.path.join("configs", "archetypes.json")
VERSION_TAG = "V13"
CSV_ROWS = []
EPISODE_ROWS = []


def _ckpt_steps(path):
    b = os.path.basename(path)
    m = re.search(r'_(\d+)_steps', b)
    if m:
        return int(m.group(1))
    m = re.search(r'final_model_(\d+)k', b, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 1000
    m = re.search(r'final_model_(\d+)', b, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def latest_ckpt(folder):
    z = glob.glob(os.path.join(folder, "*.zip"))
    return max(z, key=_ckpt_steps) if z else None


def discover_seed_folders(dir_tag):
    tag = f"_{EVAL_REGIME}_rho{int(round(EVAL_RHO*100)):03d}_"
    out = {}
    for d in sorted(glob.glob(os.path.join(MODELS_ROOT, "*"))):
        if not os.path.isdir(d):
            continue
        b = os.path.basename(d)
        if VERSION_TAG not in b or dir_tag not in b or tag not in b:
            continue
        m = re.search(r'_S(\d+)(?:_|$)', b)
        key = f"S{m.group(1)}" if m else b
        p = latest_ckpt(d)
        if p:
            out[key] = p
    return dict(sorted(out.items()))


def make_env():
    kw = dict(archetype_filepath=EVAL_ARCHETYPE_FILE,
              species_group="Urban Public Health", species_name=EVAL_ARCHETYPE,
              economic_scenarios=[EVAL_ECON] if EVAL_ECON else ["Balanced_Medium_Budget"],
              force_T_HORIZON=HORIZON,
              rho=EVAL_RHO, prior_regime=EVAL_REGIME,
              prior_history=EVAL_HISTORY, lambda_sigma=EVAL_LAMBDA_SIGMA)
    if EVAL_GRAPH:
        kw["graph_filepath_override"] = EVAL_GRAPH
    return SMIPEnv(**kw)


def find_seed_checkpoints():
    if EVAL_CHECKPOINT:
        if not os.path.isfile(EVAL_CHECKPOINT):
            print(f"ERROR: --checkpoint '{EVAL_CHECKPOINT}' is not a file.")
            return {}
        m = re.search(r'_(\d+)_steps', EVAL_CHECKPOINT)
        parent = os.path.basename(os.path.dirname(EVAL_CHECKPOINT))
        label = f"{parent}@{int(m.group(1)):,}steps" if m else os.path.basename(EVAL_CHECKPOINT)
        return {label: EVAL_CHECKPOINT}

    if EVAL_MODELS_DIR:
        base = EVAL_MODELS_DIR
        if not os.path.isdir(base):
            print(f"ERROR: --models_dir '{base}' is not a directory.")
            return {}
        seeds = {}

        latest_zip = latest_ckpt

        top = latest_zip(base)
        if top:
            seeds[os.path.basename(base.rstrip("/\\")) or "test"] = top
        for d in sorted(glob.glob(os.path.join(base, "*"))):
            if os.path.isdir(d):
                p = latest_zip(d)
                if p:
                    seeds[os.path.basename(d)] = p
        return seeds

    tag = f"_{EVAL_REGIME}_rho{int(round(EVAL_RHO*100)):03d}_"
    seeds = {}
    patterns = [os.path.join("rl_models", f"*V13*{tag}*"),
                os.path.join("rl_models", f"*{tag}*"),
                os.path.join("rl_models", "S*")]
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            if not os.path.isdir(d):
                continue
            label = os.path.basename(d)
            m_seed = re.search(r'S(\d+)$', label)
            key = f"S{m_seed.group(1)}" if m_seed else label
            cks = []
            for f in glob.glob(os.path.join(d, "*.zip")):
                m = re.search(r'_(\d+)_steps', f)
                if m:
                    cks.append((int(m.group(1)), f))
            if cks and key not in seeds:
                seeds[key] = max(cks)[1]
        if seeds:
            break
    return seeds


def outbreak_seeds(n=15, thresh=50000):
    env = make_env(); out = []
    for s in range(n):
        env.reset(seed=s); dn = 0
        for _ in range(HORIZON):
            env.step(np.full((env.num_sites, 4), -1.0).flatten())
            dn = max(dn, env.true_H['I'].sum())
        if dn > thresh:
            out.append(s)
    return out


def heuristic_peak(policy, mult, eval_seed):
    env = make_env(); env.reset(seed=eval_seed); _apply_eta(env)
    true_H = {k: v.copy() for k, v in env.true_H.items()}
    true_M = {k: v.copy() for k, v in env.true_M.items()}
    if FREE_HEURISTICS:
        weekly, pool = 0.0, BASE_BUDGET * mult
    else:
        weekly, pool = (BASE_BUDGET * mult) / HORIZON, 0.0
    rng = np.random.default_rng(eval_seed)
    peak = 0
    cw = env.calendar_week
    belief = None
    delta = env.ep_params.get('delta_penalty_human_I', 20000.0)
    budget_total = BASE_BUDGET * mult
    V, spent, disc = 0.0, 0.0, 1.0
    cov = {ch: 0.0 for ch in CHANNELS}
    costs_ep = _unit_costs(env)
    surv = 0.0
    spend = {k: 0.0 for k in SPEND_KEYS}
    for _ in range(HORIZON):
        pool += weekly
        pre = pool
        true_H, true_M, costs, new_inf, pool, belief = run_heuristic_week(
            cw, true_H, true_M, env.neighbors, policy, pool, env.ep_params, rng,
            belief=belief)
        step_cost = float(pre - pool)
        step_inf = float(np.sum(new_inf))
        spent += step_cost
        V += disc * (step_cost + delta * step_inf)
        disc *= EVAL_GAMMA
        _acc_coverage(cov, getattr(_ed, "LAST_ACTIONS", None))
        surv += float(getattr(_ed, "LAST_SURVEY_FRAC", 0.0) or 0.0)
        _sp_wk = getattr(_ed, "LAST_SPEND", None) or {}
        for k in SPEND_KEYS:
            spend[k] += float(_sp_wk.get(k, 0.0))
        peak = max(peak, true_H['I'].sum())
        cw += 1
    util = _audit_budget(f"{policy} x{mult}, seed {eval_seed}", spent, budget_total)
    cov = {k: v / HORIZON for k, v in cov.items()}
    return peak, V, util, cov, costs_ep, surv / HORIZON, spend


def agent_peak(model, mult, eval_seed, collect_actions=None):
    env = make_env()
    obs, _ = env.reset(seed=eval_seed)
    env.total_episode_budget = BASE_BUDGET * mult
    env.budget_remaining_for_episode = BASE_BUDGET * mult
    _apply_eta(env)
    peak = 0
    delta = env.ep_params.get('delta_penalty_human_I', 20000.0)
    budget_total = BASE_BUDGET * mult
    V, spent, disc = 0.0, 0.0, 1.0
    cov = {ch: 0.0 for ch in CHANNELS}
    costs_ep = _unit_costs(env)
    surv = 0.0
    spend = {k: 0.0 for k in SPEND_KEYS}
    steps = 0
    for _ in range(HORIZON):
        a, _ = model.predict(obs, deterministic=True)
        if collect_actions is not None:
            arr = np.asarray(a, dtype=np.float64).reshape(env.num_sites, -1)
            collect_actions.append(arr)
        obs, _, term, trunc, info = env.step(a)
        step_cost = float(info.get("cost", 0.0))
        step_inf = float(info.get("new_infections", 0.0))
        spent += step_cost
        V += disc * (step_cost + delta * step_inf)
        disc *= EVAL_GAMMA
        _acc_coverage(cov, getattr(env, "_last_final_actions", None))
        _ac = info.get("costs_by_type") or {}
        for k in SPEND_KEYS:
            spend[k] += float(_ac.get(k, 0.0))
        _cs = env.ep_params['action_semantics']['survey']['cost']
        _cs = float(np.mean(_cs)) if hasattr(_cs, '__len__') else float(_cs)
        _denom = _cs * float(np.sum(env.dhaka_populations))
        if _denom > 0:
            surv += min(1.0, float(_ac.get("survey", 0.0)) / _denom)
        steps += 1
        peak = max(peak, env.true_H['I'].sum())
        if term or trunc:
            break
    util = _audit_budget(f"RL x{mult}, seed {eval_seed}", spent, budget_total)
    cov = {k: v / max(1, steps) for k, v in cov.items()}
    return peak, V, util, cov, costs_ep, surv / max(1, steps), spend


def observation_signal(models, ob_seeds, mult=2.5):
    print("\n" + "=" * 88)
    print(f"OBSERVATION-SIGNAL DIAGNOSTIC  (at x{mult} budget)")
    print("  Can the agent SEE infection? (ceiling on any targeting it could learn)")
    print("=" * 88)

    chans = {1: "obs[1] believed_I", 3: "obs[3] mosq_abund", 14: "obs[14] lambda_hat"}
    stats = {c: {"std": [], "corr": []} for c in chans}
    belief_err = []

    model = list(models.values())[0]
    for s in ob_seeds[:5]:
        e = make_env()
        obs, _ = e.reset(seed=s)
        e.total_episode_budget = BASE_BUDGET * mult
        e.budget_remaining_for_episode = BASE_BUDGET * mult
        _apply_eta(e)
        for _ in range(HORIZON):
            trueI = e.true_H['I'].copy()
            for c in chans:
                if c < obs.shape[1]:
                    v = np.asarray(obs[:, c], dtype=np.float64)
                    stats[c]["std"].append(float(np.std(v)))
                    if np.std(v) > 1e-12 and np.std(trueI) > 1e-12:
                        rv = np.argsort(np.argsort(v))
                        ri = np.argsort(np.argsort(trueI))
                        stats[c]["corr"].append(float(np.corrcoef(rv, ri)[0, 1]))
                    else:
                        stats[c]["corr"].append(0.0)
            bi = e.believed_H['I']
            denom = max(1.0, float(trueI.sum()))
            belief_err.append(float(np.abs(bi - trueI).sum() / denom))
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = e.step(a)
            if term or trunc:
                break

    print(f"  {'signal':<24}{'cross_ward_std':>16}{'corr_with_true_I':>20}")
    print("  " + "-" * 60)
    corr1 = None
    for c, name in chans.items():
        if not stats[c]["std"]:
            continue
        sd, cr = float(np.mean(stats[c]["std"])), float(np.mean(stats[c]["corr"]))
        if c == 1:
            corr1 = cr
        print(f"  {name:<24}{sd:>16.4f}{cr:>20.3f}")
    print("  " + "-" * 60)
    print(f"  mean |belief_I - true_I| / total_true_I : {np.mean(belief_err):.3f}")

    print("\n  INTERPRETATION:")
    if corr1 is not None:
        if abs(corr1) < 0.15:
            print(f"   ! OBSERVABILITY FAILURE: obs[1] tracks true infection at corr={corr1:.3f}.")
            print("     The infection signal the policy reads is essentially UNRELATED to")
            print("     where infection actually is. No curriculum or extra training can")
            print("     produce targeted actions from an untargeted observation -- fix the")
            print("     survey/belief pipeline first. This is UPSTREAM of the RL entirely.")
        elif abs(corr1) < 0.4:
            print(f"   ~ WEAK signal: obs[1] corr={corr1:.3f}. Information is present but noisy.")
            print("     Targeting is possible but hard; belief quality AND training regime")
            print("     are both plausible contributors. Don't blame training alone.")
        else:
            print(f"   + SIGNAL PRESENT: obs[1] corr={corr1:.3f}. The agent CAN see infection.")
            print("     If its ACTIONS still don't correlate with infection (see next table),")
            print("     that is a LEARNING failure -> curriculum / reward / exploration.")
    return stats


def action_concentration(models, ob_seeds, mult=2.5):
    print("\n" + "=" * 88)
    print(f"RL ACTION-CONCENTRATION DIAGNOSTIC  (at x{mult} budget)")
    print("  Is the agent TARGETING wards, or spraying near-uniformly?")
    print("  uniform-spray reference: top10_share ~ 0.077, CoV ~ 0, corr_with_I ~ 0")
    print("=" * 88)

    chan_names = ["survey", "case_mgmt", "fogging", "larviciding"]
    per_channel = {}

    for lbl, model in models.items():
        for s in ob_seeds[:5]:
            acts = []
            env_probe = make_env()
            env_probe.reset(seed=s)
            e = make_env()
            obs, _ = e.reset(seed=s)
            e.total_episode_budget = BASE_BUDGET * mult
            e.budget_remaining_for_episode = BASE_BUDGET * mult
            _apply_eta(e)
            for _ in range(HORIZON):
                a, _ = model.predict(obs, deterministic=True)
                arr = np.asarray(a, dtype=np.float64).reshape(e.num_sites, -1)
                trueI = e.true_H['I'].copy()
                acts.append((arr, trueI))
                obs, _, term, trunc, _ = e.step(a)
                if term or trunc:
                    break
            for arr, trueI in acts:
                n_ch = arr.shape[1]
                for c in range(n_ch):
                    v = arr[:, c]
                    mean = float(np.mean(v))
                    std = float(np.std(v))
                    cov = std / (abs(mean) + 1e-9)
                    pos = np.maximum(v - v.min(), 0.0)
                    tot = pos.sum() + 1e-9
                    top10 = float(np.sort(pos)[::-1][:10].sum() / tot)
                    if np.std(v) > 1e-12 and np.std(trueI) > 1e-12:
                        rv = np.argsort(np.argsort(v))
                        ri = np.argsort(np.argsort(trueI))
                        corr = float(np.corrcoef(rv, ri)[0, 1])
                    else:
                        corr = 0.0
                    per_channel.setdefault(c, {"std": [], "cov": [], "top10": [], "corr": []})
                    per_channel[c]["std"].append(std)
                    per_channel[c]["cov"].append(cov)
                    per_channel[c]["top10"].append(top10)
                    per_channel[c]["corr"].append(corr)

    print(f"  {'channel':<14}{'cross_ward_std':>16}{'CoV':>10}{'top10_share':>14}{'corr_with_I':>14}")
    print("  " + "-" * 66)
    verdicts = []
    for c in sorted(per_channel):
        name = chan_names[c] if c < len(chan_names) else f"ch{c}"
        d = per_channel[c]
        std, cov = np.mean(d["std"]), np.mean(d["cov"])
        t10, corr = np.mean(d["top10"]), np.mean(d["corr"])
        print(f"  {name:<14}{std:>16.4f}{cov:>10.3f}{t10:>14.3f}{corr:>14.3f}")
        verdicts.append((name, std, cov, t10, corr))
    print("  " + "-" * 66)

    print("\n  INTERPRETATION:")
    collapsed = [n for n, s, cv, t, cr in verdicts if s < 0.01 or (t < 0.12 and abs(cr) < 0.1)]
    targeting = [n for n, s, cv, t, cr in verdicts if t >= 0.15 and abs(cr) >= 0.2]
    if collapsed:
        print(f"   ! NEAR-UNIFORM on: {', '.join(collapsed)}")
        print("     These channels are effectively blanket-applied. Any RL advantage")
        print("     on them comes from TIMING / BUDGET SPLIT, not spatial targeting.")
    if targeting:
        print(f"   + GENUINE TARGETING on: {', '.join(targeting)}")
        print("     These channels concentrate budget on high-infection wards.")
    if not collapsed and not targeting:
        print("     Mixed/intermediate -- report the raw numbers, don't over-claim.")
    print("\n  (Compare against policy.py's documented collapse: per-ward std ~0.0001,")
    print("   intent CoV ~0.001. If you see that here, the actor collapsed again.)")
    return verdicts


def convergence(models, ob_seeds):
    print("\n" + "=" * 72)
    print(f"1. CONVERGENCE  (final mean peak at {REF_MULT}x budget, per seed)")
    print("=" * 72)
    vals = []
    for lbl, model in models.items():
        mp = float(np.mean([agent_peak(model, REF_MULT, s)[0] for s in ob_seeds]))
        vals.append(mp)
        print(f"  {lbl:<6} final mean peak = {mp:>12,.0f}")
    if len(vals) >= 2:
        cov = np.std(vals) / (np.mean(vals) + 1e-9)
        print(f"\n  cross-seed mean = {np.mean(vals):,.0f}   std = {np.std(vals):,.0f}"
              f"   CoV = {cov*100:.1f}%")
        print("  (pair with the per-seed plateau from convergence_analysis.py)")
    return vals


def _ep_row(policy, train_seed, eval_seed, mult, t):
    peak, V, util, cov, costs_ep, surv, spend = t
    tot_spend = sum(spend.values()) or 1.0
    c_vc = costs_ep.get("vector_control", float("nan"))
    kb = (BASE_BUDGET * mult) / (_POP_CACHE * c_vc * HORIZON) if _POP_CACHE and c_vc else ""
    k_meas = cov.get("vector_control", 0.0)
    row = dict(policy=policy, train_seed=train_seed, eval_seed=eval_seed,
               budget_mult=mult, peak=round(peak, 1), cum_disc_cost=round(V, 1),
               budget_util=round(util, 4),
               kappa_measured=round(k_meas, 4),
               m_measured=round(min(k_meas, 1 - k_meas), 4),
               kappa_budget=(round(kb, 4) if kb != "" else ""))
    row["survey_frac"] = round(surv, 4)
    for ch in CHANNELS:
        row[f"cov_{ch}"] = round(cov.get(ch, 0.0), 4)
        row[f"cost_{ch}"] = round(costs_ep.get(ch, 0.0), 3)
    for k in SPEND_KEYS:
        row[f"spend_{k}"] = round(spend.get(k, 0.0), 1)
        row[f"spendshare_{k}"] = round(spend.get(k, 0.0) / tot_spend, 4)
    row["eta_F"] = ("" if EVAL_ETA_F is None else round(float(EVAL_ETA_F), 4))
    row["script_version"] = SCRIPT_VERSION
    row["run_utc"] = RUN_UTC
    return row


def budget_sweep(models, ob_seeds):
    print("\n" + "=" * 100)
    print("2. BUDGET SWEEP  --  mean peak infection by policy x budget")
    print("   heuristics deterministic given seeds; RL row = mean +/- std across seeds")
    print(f"   (* = controlled, peak < {CONTROL_THRESHOLD:,})")
    print("=" * 100)
    print(f"{'policy':<24}" + "".join(f"{('x%.1f' % m):>10}" for m in BUDGET_MULTIPLES))
    print("-" * 100)

    results, stds = {}, {}
    vals, utils = {}, {}
    for pol in HEURISTICS:
        results[pol] = {}; vals[pol] = {}; utils[pol] = {}
        row = f"{pol:<24}"
        for m in BUDGET_MULTIPLES:
            trio = [heuristic_peak(pol, m, s) for s in ob_seeds]
            for es, t in zip(ob_seeds, trio):
                EPISODE_ROWS.append(_ep_row(pol, "n/a", es, m, t))
            mp = float(np.mean([t[0] for t in trio]))
            results[pol][m] = mp
            vals[pol][m] = float(np.mean([t[1] for t in trio]))
            utils[pol][m] = float(np.mean([t[2] for t in trio]))
            row += f"{mp:>9,.0f}" + ("*" if mp < CONTROL_THRESHOLD else " ")
        print(row)

    per_seed_rl = {lbl: {} for lbl in models}
    per_seed_V = {lbl: {} for lbl in models}
    if models:
        results["RL_Agent"] = {}; stds["RL_Agent"] = {}
        vals["RL_Agent"] = {}; utils["RL_Agent"] = {}
        row = f"{'RL_Agent (mean)':<24}"
        for m in BUDGET_MULTIPLES:
            per_seed, per_seed_v, per_seed_u = [], [], []
            for lbl_m, mdl in models.items():
                trio = [agent_peak(mdl, m, s) for s in ob_seeds]
                for es, t in zip(ob_seeds, trio):
                    EPISODE_ROWS.append(_ep_row("RL_Agent", lbl_m, es, m, t))
                per_seed.append(float(np.mean([t[0] for t in trio])))
                per_seed_v.append(float(np.mean([t[1] for t in trio])))
                per_seed_u.append(float(np.mean([t[2] for t in trio])))
            for lbl, v, vv in zip(models.keys(), per_seed, per_seed_v):
                per_seed_rl[lbl][m] = v; per_seed_V[lbl][m] = vv
            mean, std = float(np.mean(per_seed)), float(np.std(per_seed))
            results["RL_Agent"][m] = mean; stds["RL_Agent"][m] = std
            vals["RL_Agent"][m] = float(np.mean(per_seed_v))
            utils["RL_Agent"][m] = float(np.mean(per_seed_u))
            row += f"{mean:>9,.0f}" + ("*" if mean < CONTROL_THRESHOLD else " ")
        print(row)
        row = f"{'RL_Agent (std)':<24}"
        for m in BUDGET_MULTIPLES:
            row += f"{stds['RL_Agent'][m]:>10,.0f}"
        print(row)
    print("-" * 100)

    if KAPPA_AT_1X:
        print("\n2a. COVERAGE AXIS  (kappa = fraction of network reachable per week at full strength)")
        print(f"{'budget mult':<24}" + "".join(f"{('x%g' % m):>10}" for m in BUDGET_MULTIPLES))
        print(f"{'kappa':<24}" + "".join(f"{min(1.0, KAPPA_AT_1X*m):>10.3f}" for m in BUDGET_MULTIPLES))
        print(f"{'m = min(kappa, 1-kappa)':<24}"
              + "".join(f"{min(min(1.0,KAPPA_AT_1X*m), 1-min(1.0,KAPPA_AT_1X*m)):>10.3f}"
                        for m in BUDGET_MULTIPLES))
        print("   The bound is ~proportional to m, so VO should track this row and vanish")
        print("   where m = 0. A peak elsewhere points to the rival C_V-near-criticality story.")

    print("\n2b. CUMULATIVE DISCOUNTED ECONOMIC COST  V = sum_t g^t (cost + delta*new_inf)")
    print(f"    gamma = {EVAL_GAMMA}   (billions BDT; lower is better)")
    print(f"{'policy':<24}" + "".join(f"{('x%.1f' % m):>10}" for m in BUDGET_MULTIPLES))
    print("-" * 100)
    for pol in vals:
        print(f"{pol:<24}" + "".join(f"{vals[pol][m]/1e9:>10.3f}" for m in BUDGET_MULTIPLES))
    print("-" * 100)
    print("\n2b-CI. V WITH 95% BOOTSTRAP CI  (billions BDT; n = eval seeds x train seeds)")
    print(f"{'policy':<24}" + "".join(f"{('x%.1f' % m):>18}" for m in BUDGET_MULTIPLES))
    print("-" * 150)
    for pol in vals:
        row = f"{pol:<24}"
        for m in BUDGET_MULTIPLES:
            xs = [r["cum_disc_cost"] for r in EPISODE_ROWS
                  if "config" not in r and r["policy"] == pol and r["budget_mult"] == m]
            if xs:
                mu, lo, hi = _bootstrap_ci(xs)
                row += f"{mu/1e9:>8.2f}[{lo/1e9:.2f},{hi/1e9:.2f}]"
            else:
                row += f"{'-':>18}"
        print(row)
    print("-" * 150)
    print("   Overlapping intervals between two policies at the same budget mean the")
    print("   ranking at that budget is NOT supported by this sample.")

    print("\n2d. MEASURED MEAN COVERAGE PER CHANNEL  (fraction of network per week)")
    for ch in CHANNELS:
        print(f"  -- {ch} --")
        print(f"{'policy':<24}" + "".join(f"{('x%g' % m):>10}" for m in BUDGET_MULTIPLES))
        for pol in vals:
            row = f"{pol:<24}"
            for m in BUDGET_MULTIPLES:
                xs = [r[f"cov_{ch}"] for r in EPISODE_ROWS
                      if "config" not in r and r["policy"] == pol and r["budget_mult"] == m]
                row += f"{np.mean(xs):>10.3f}" if xs else f"{'-':>10}"
            print(row)
    print("\n  A channel pinned at 1.000 is saturated and cannot absorb more budget;")
    print("  one well below 1.000 at high budget is where the spare money is going.")

    print("\n2e. SPEND SHARE BY CHANNEL + SURVEYED FRACTION")
    for pol in vals:
        rows = [r for r in EPISODE_ROWS if "config" not in r and r["policy"] == pol]
        if not rows:
            continue
        print(f"  {pol}")
        print(f"{'':<8}" + "".join(f"{('x%g' % m):>9}" for m in BUDGET_MULTIPLES))
        for key, lbl in [("spendshare_survey", "survey$"), ("spendshare_vc", "fog$"),
                         ("spendshare_cm", "casemgt$"), ("spendshare_larv", "larv$"),
                         ("survey_frac", "surveyd")]:
            line = f"{lbl:<8}"
            for m in BUDGET_MULTIPLES:
                xs = [r[key] for r in rows if r["budget_mult"] == m]
                line += f"{np.mean(xs):>9.3f}" if xs else f"{'-':>9}"
            print(line)

    print("\n2c. BUDGET UTILISATION  (spent / season budget; >1.0 would abort)")
    print(f"{'policy':<24}" + "".join(f"{('x%.1f' % m):>10}" for m in BUDGET_MULTIPLES))
    print("-" * 100)
    for pol in utils:
        print(f"{pol:<24}" + "".join(f"{utils[pol][m]:>10.3f}" for m in BUDGET_MULTIPLES))
    print("-" * 100)
    return results, stds, per_seed_rl, vals, utils, per_seed_V


def collect_csv(cfg_name, cfg, results, stds, per_seed_rl, thr, n_seeds,
                vals=None, utils=None, per_seed_V=None):
    proto = "free_heuristics" if FREE_HEURISTICS else "paced_heuristics"

    vals = vals or {}; utils = utils or {}; per_seed_V = per_seed_V or {}
    for r in EPISODE_ROWS:
        if "config" not in r:
            r["config"] = cfg_name; r["n_wards"] = cfg["n_wards"]
            r["protocol"] = proto; r["gamma"] = EVAL_GAMMA
            r["base_budget"] = cfg["budget"]
            r["survey_share"] = SURVEY_SHARE_USED
            r["kappa"] = (round(min(1.0, KAPPA_AT_1X*r["budget_mult"]), 4) if KAPPA_AT_1X else "")

    def add(policy, seed, m, peak, V=None, util=None):
        CSV_ROWS.append(dict(
            config=cfg_name, n_wards=cfg["n_wards"], graph=cfg["graph"],
            econ=cfg["econ"], base_budget=cfg["budget"], protocol=proto,
            gamma=EVAL_GAMMA, survey_share=SURVEY_SHARE_USED,
            policy=policy, seed=seed, budget_mult=m,
            kappa=(round(min(1.0, KAPPA_AT_1X*m), 4) if KAPPA_AT_1X else ""),
            m_min_k=(round(min(min(1.0,KAPPA_AT_1X*m), 1-min(1.0,KAPPA_AT_1X*m)), 4)
                     if KAPPA_AT_1X else ""),
            mean_peak=round(peak, 1),
            cum_disc_cost=("" if V is None else round(V, 1)),
            budget_util=("" if util is None else round(util, 4)),
            controlled=int(peak < CONTROL_THRESHOLD),
            controls_at=(thr.get(policy) if thr.get(policy) is not None else ""),
            n_rl_seeds=n_seeds,
            eta_F=("" if EVAL_ETA_F is None else round(float(EVAL_ETA_F), 4)),
            script_version=SCRIPT_VERSION, run_utc=RUN_UTC))

    for pol in results:
        for m in BUDGET_MULTIPLES:
            add(pol, "mean" if pol == "RL_Agent" else "n/a", m, results[pol][m],
                vals.get(pol, {}).get(m), utils.get(pol, {}).get(m))
    for lbl, d in per_seed_rl.items():
        for m, v in d.items():
            add("RL_Agent", lbl, m, v, per_seed_V.get(lbl, {}).get(m), None)


def _bootstrap_ci(x, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return (float(x.mean()) if x.size else 0.0,) * 3
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2)))


def write_meta(path, args, configs_run):
    meta = {
        "script": os.path.basename(__file__),
        "script_version": SCRIPT_VERSION,
        "run_utc": RUN_UTC,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "resolved": {
            "survey_share": SURVEY_SHARE_USED,
            "rho": EVAL_RHO, "regime": EVAL_REGIME, "history": EVAL_HISTORY,
            "lambda_sigma": EVAL_LAMBDA_SIGMA, "gamma": EVAL_GAMMA,
            "free_heuristics": FREE_HEURISTICS, "strict_budget": STRICT_BUDGET,
            "horizon": HORIZON, "budget_multiples": BUDGET_MULTIPLES,
            "archetype": EVAL_ARCHETYPE, "econ": EVAL_ECON,
            "control_threshold": CONTROL_THRESHOLD,
            "eta_F_override": EVAL_ETA_F,
        },
        "configs": configs_run,
        "csv": os.path.basename(path),
        "episodes_csv": os.path.basename(path)[:-4] + "_episodes.csv",
    }
    mp = path[:-4] + ".meta.json" if path.endswith(".csv") else path + ".meta.json"
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved provenance -> {mp}")


def write_csv(path):
    if CSV_ROWS:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(CSV_ROWS[0].keys()))
            w.writeheader(); w.writerows(CSV_ROWS)
        print(f"\nSaved aggregate CSV ({len(CSV_ROWS)} rows) -> {path}")
    if EPISODE_ROWS:
        ep = path[:-4] + "_episodes.csv" if path.endswith(".csv") else path + "_episodes.csv"
        with open(ep, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(EPISODE_ROWS[0].keys()))
            w.writeheader(); w.writerows(EPISODE_ROWS)
        print(f"Saved per-episode CSV ({len(EPISODE_ROWS)} rows) -> {ep}")
        print("   -> bootstrap CIs for every table are computed from THIS file.")


def thresholds(results):
    print("\n" + "=" * 72)
    print("3. BUDGET THRESHOLD FOR CONTROL, BY POLICY")
    print("=" * 72)
    thr = {}
    for pol in results:
        t = next((m for m in BUDGET_MULTIPLES if results[pol][m] < CONTROL_THRESHOLD), None)
        thr[pol] = t
        tag = "  [oracle: upper bound, not a fair competitor]" if pol in ORACLE_POLICIES else ""
        if t is None:
            print(f"  {pol:<24} not controlled within x{BUDGET_MULTIPLES[-1]}{tag}")
        else:
            print(f"  {pol:<24} controlled at x{t:.1f}"
                  f"  (~{t*BASE_BUDGET*ANNUALIZE/1e9:.1f}B BDT/yr){tag}")
    return thr


def value_of_information(thr):
    print("\n" + "=" * 72)
    print("4. VALUE OF INFORMATION  (oracle vs belief, same allocation logic)")
    print("=" * 72)
    for name, oracle, belief in [("Greedy", "H4_Greedy_Oracle", "H4_Greedy_Belief"),
                                 ("Myopic LP", "H5_MyopicLP_Oracle", "H5_MyopicLP_Belief")]:
        o, b = thr.get(oracle), thr.get(belief)
        if o and b:
            gap = b - o
            print(f"  {name:<10} oracle x{o:.1f} -> belief x{b:.1f}"
                  f"   value of information = {gap:.1f}x budget"
                  f"  (~{gap*BASE_BUDGET*ANNUALIZE/1e9:.1f}B BDT/yr)")
        else:
            print(f"  {name:<10} oracle x{o} / belief x{b} (one side uncontrolled)")
    print("\n  Knowing WHERE infection is is worth a sizeable multiple of the entire")
    print("  intervention budget -- surveillance capacity is a first-order lever.")


def lp_vs_greedy(results, thr):
    print("\n" + "=" * 72)
    print("5. MYOPIC LP vs GREEDY  (positioning vs Long et al. 2018)")
    print("=" * 72)
    print("  Long et al. (Ebola): Myopic LP averted 86-96% vs greedy's 72-85%")
    print("  -- one-period lookahead won decisively there. On dengue:")
    print()
    print(f"  {'budget':>7} {'Greedy_Or':>11} {'LP_Or':>11} {'Greedy_Bel':>12} {'LP_Bel':>12}")
    for m in BUDGET_MULTIPLES:
        print(f"  {m:>7.1f} {results['H4_Greedy_Oracle'][m]:>11,.0f}"
              f" {results['H5_MyopicLP_Oracle'][m]:>11,.0f}"
              f" {results['H4_Greedy_Belief'][m]:>12,.0f}"
              f" {results['H5_MyopicLP_Belief'][m]:>12,.0f}")
    print()
    print(f"  thresholds:  greedy oracle x{thr.get('H4_Greedy_Oracle')} vs LP oracle x{thr.get('H5_MyopicLP_Oracle')}"
          f"  |  greedy belief x{thr.get('H4_Greedy_Belief')} vs LP belief x{thr.get('H5_MyopicLP_Belief')}")
    print("  If LP ~= greedy, one-period optimization adds little on dengue -- the")
    print("  reversal our structural analysis predicts (Thm 1 + weak coupling).")


def fair_ranking(results, thr):
    print("\n" + "=" * 72)
    print("6. FAIR RANKING  (information-limited policies only)")
    print("=" * 72)
    rows = [(p, thr.get(p), results[p][2.5]) for p in FAIR_POLICIES if p in results]
    rows.sort(key=lambda r: (r[1] is None, r[1] if r[1] else 99, r[2]))
    print(f"  {'policy':<24}{'controls at':>13}{'peak @2.5x':>13}")
    for p, t, pk in rows:
        print(f"  {p:<24}{('x%.1f' % t) if t else '>x4.0':>13}{pk:>13,.0f}")
    print("\n  Oracle variants are excluded here: they observe the true state for free")
    print("  and are reported only as upper bounds / value-of-information references.")


def relative_gain(results, thr):
    print("\n" + "=" * 96)
    print("6b. RELATIVE GAIN OF THE BEST POLICY  (% reduction in peak infection vs each rival)")
    print("    positive = best policy is better by that %;  computed per budget")
    print("=" * 96)

    avail = [p for p in FAIR_POLICIES if p in results]
    if not avail:
        return

    header = f"{'vs policy':<24}" + "".join(f"{('x%.1f' % m):>11}" for m in BUDGET_MULTIPLES)
    print(header)
    print("-" * 96)

    best_at = {m: min(avail, key=lambda p: results[p][m]) for m in BUDGET_MULTIPLES}

    for p in avail:
        row = f"{p:<24}"
        for m in BUDGET_MULTIPLES:
            b = best_at[m]
            if p == b:
                row += f"{'BEST':>11}"
            else:
                bp, pp = results[b][m], results[p][m]
                gain = 100.0 * (pp - bp) / (pp + 1e-9)
                row += f"{gain:>10.1f}%"
        print(row)
    print("-" * 96)
    print("  best policy by budget: " + ", ".join(f"x{m}={best_at[m]}" for m in BUDGET_MULTIPLES))
    print("\n  NOTE: a policy can be 'BEST' at one budget and mid-pack at another.")
    print("  Report the budget alongside any headline gain -- an unqualified")
    print("  '<policy> is X% better' is not supportable from this table.")


def run_analysis(cfg_name=None, cfg=None):
    _e = make_env()
    _e.reset(seed=0)
    _apply_eta(_e)
    _pop = float(np.sum(_e.dhaka_populations))
    _c = _e.ep_params['action_semantics']['vector_control']['cost']
    _c = float(np.mean(_c)) if hasattr(_c, '__len__') else float(_c)
    globals()['_POP_CACHE'] = _pop
    globals()['KAPPA_AT_1X'] = BASE_BUDGET / (_pop * _c * HORIZON)
    print("\n" + "*" * 78)
    print(f"*  COVERAGE AXIS   kappa at x1.0 = {KAPPA_AT_1X:.4f}")
    print(f"*    population {_pop:,.0f} | fogging unit cost {_c:.2f} BDT | horizon {HORIZON}w")
    print(f"*    theory: VO peaks at kappa=0.5  -> x{0.5/KAPPA_AT_1X:.3f}")
    print(f"*    theory: VO -> 0 at kappa=1     -> x{1.0/KAPPA_AT_1X:.3f}")
    print("*" * 78 + "\n")

    seed_ckpts = globals().get("_BATCH_SEEDS") or find_seed_checkpoints()
    if seed_ckpts:
        print(f"Discovered {len(seed_ckpts)} seed(s): {', '.join(seed_ckpts.keys())}")
    else:
        print("No matching checkpoints -- running heuristics + LP only.")
    models = {lbl: PPO.load(p, device="cpu") for lbl, p in seed_ckpts.items()}

    ob_seeds = outbreak_seeds()
    print(f"Outbreak evaluation seeds (n={len(ob_seeds)}): {ob_seeds}")
    os.makedirs(OUT_DIR, exist_ok=True)

    if models:
        convergence(models, ob_seeds)
    results, stds, per_seed_rl, vals, utils, per_seed_V = budget_sweep(models, ob_seeds)
    thr = thresholds(results)
    value_of_information(thr)
    lp_vs_greedy(results, thr)
    if models:
        fair_ranking(results, thr)
    relative_gain(results, thr)
    if models:
        observation_signal(models, ob_seeds, mult=2.5)
        action_concentration(models, ob_seeds, mult=2.5)

    dump = {
        "mean_peak": {p: {str(m): results[p][m] for m in BUDGET_MULTIPLES} for p in results},
        "rl_std": {str(m): stds.get("RL_Agent", {}).get(m) for m in BUDGET_MULTIPLES} if stds else {},
        "thresholds": {p: thr[p] for p in thr},
        "eval_seeds": ob_seeds,
        "n_rl_seeds": len(models),
    }
    path = os.path.join(OUT_DIR, "master_results.json")
    with open(path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved consolidated results -> {path}")

    print("\n" + "=" * 72)
    print("7. HEADLINE ECONOMIC TABLE")
    print("=" * 72)
    if seed_ckpts:
        rep = list(seed_ckpts.values())[0]
        print(f"  python src/evaluate_dengue.py --trials 20 --model {rep}")
    else:
        print("  python src/evaluate_dengue.py --trials 20")
    print("  (Monte-Carlo lives/cases/economic loss by scenario; kept separate as it")
    print("   is not a budget sweep.)")

    if cfg is None:
        cfg = dict(graph=EVAL_GRAPH or "default", econ=EVAL_ECON or "default",
                   budget=BASE_BUDGET, n_wards="")
        cfg_name = cfg_name or "single"
    collect_csv(cfg_name, cfg, results, stds, per_seed_rl, thr, len(models),
                vals=vals, utils=utils, per_seed_V=per_seed_V)


def main():
    global EVAL_RHO, EVAL_REGIME, EVAL_HISTORY, EVAL_LAMBDA_SIGMA, EVAL_MODELS_DIR, EVAL_CHECKPOINT
    global EVAL_GRAPH, EVAL_ARCHETYPE, EVAL_ECON, BASE_BUDGET, FREE_HEURISTICS
    global EVAL_ETA_F
    args = parse_args()
    FREE_HEURISTICS = args.free_heuristics
    globals()['EVAL_GAMMA'] = args.gamma
    EVAL_ETA_F = args.eta_F
    if EVAL_ETA_F is not None:
        print(f'=== ETA_F OVERRIDE ===  adulticide efficacy forced to {EVAL_ETA_F:.3f} '
              f'(calibrated value ignored). Results are NOT comparable to an '
              f'unflagged run; write this to its own --csv.')
    _ed.SURVEY_SHARE = float(args.survey_share)
    globals()['SURVEY_SHARE_USED'] = float(args.survey_share)
    if args.mults:
        globals()['BUDGET_MULTIPLES'] = [float(x) for x in args.mults.split(",") if x.strip()]
        globals()['REF_MULT'] = min(BUDGET_MULTIPLES, key=lambda m: abs(m - 2.0))
        print(f"=== BUDGET GRID  ===  {len(BUDGET_MULTIPLES)} points: "
              f"{', '.join('x%g' % m for m in BUDGET_MULTIPLES)}")
    globals()['STRICT_BUDGET'] = not args.no_strict_budget
    EVAL_RHO, EVAL_REGIME = args.rho, args.regime
    EVAL_HISTORY, EVAL_LAMBDA_SIGMA = args.history, args.lambda_sigma
    EVAL_MODELS_DIR = args.models_dir
    EVAL_CHECKPOINT = args.checkpoint
    EVAL_GRAPH = args.graph
    EVAL_ARCHETYPE = args.archetype
    EVAL_ECON = args.econ
    if args.budget is not None:
        BASE_BUDGET = args.budget
    print(f"=== EVAL REGIME ===  rho={EVAL_RHO}  regime={EVAL_REGIME}  "
          f"history={EVAL_HISTORY}  lambda_sigma={EVAL_LAMBDA_SIGMA}")
    print(f"=== SURVEY CAP  ===  share={args.survey_share:.2f}"
          + ("  (uncapped: surveillance may claim the whole weekly pool)"
             if args.survey_share >= 1.0 else "  (capped)"))
    print(f"=== V / AUDIT   ===  gamma={EVAL_GAMMA}  "
          f"strict_budget={'ON (abort on overspend)' if STRICT_BUDGET else 'off (warn only)'}")
    print(f"=== BUDGET PROTO ===  RL=free (whole season up front)  "
          f"heuristics={'FREE (matched)' if FREE_HEURISTICS else 'paced 1/HORIZON (default)'}")
    if EVAL_GRAPH or args.budget is not None or EVAL_ARCHETYPE != "Dengue_Dhaka_Advanced":
        print(f"=== DNCC/SUBGRAPH ===  graph={EVAL_GRAPH or 'default'}  "
              f"archetype={EVAL_ARCHETYPE}  econ={EVAL_ECON or 'default'}  "
              f"base_budget={BASE_BUDGET:,.0f}")
    if EVAL_CHECKPOINT:
        print(f"=== CHECKPOINT  ===  {EVAL_CHECKPOINT}  (discovery BYPASSED)")
        print("    Make sure the rho/regime above match how this model was trained.")
    elif EVAL_MODELS_DIR:
        print(f"=== MODELS DIR  ===  {EVAL_MODELS_DIR}  (regime-tag discovery BYPASSED)")
        print("    Make sure the rho/regime above match how this model was trained.")
    else:
        print("(episodes and checkpoint discovery both use these; match them to training)")
    print()


    global MODELS_ROOT, VERSION_TAG
    MODELS_ROOT = args.models_root
    VERSION_TAG = args.version_tag
    wanted = []
    if args.all:
        wanted = ["dncc10", "dncc"]
    else:
        if args.dncc10: wanted.append("dncc10")
        if args.dncc:   wanted.append("dncc")

    csv_path = args.csv or os.path.join(OUT_DIR, "master_results.csv")
    _configs_run = []
    os.makedirs(OUT_DIR, exist_ok=True)

    if not wanted:
        run_analysis()
        _configs_run.append({"name": "single", "graph": EVAL_GRAPH, "econ": EVAL_ECON,
                             "base_budget": BASE_BUDGET,
                             "checkpoints": {k: v for k, v in
                                             (globals().get("_BATCH_SEEDS") or {}).items()}})
        write_csv(csv_path)
        write_meta(csv_path, args, _configs_run)
        return

    if args.checkpoint:
        print("NOTE: --checkpoint is ignored in batch mode; seeds are auto-discovered.")
    for i, name in enumerate(wanted):
        cfg = GRAPH_CONFIGS[name]
        cfg["budget"] = _resolve_budget(cfg, EVAL_ARCHETYPE_FILE, EVAL_ARCHETYPE, cfg["n_wards"])
        _legacy = LEGACY_BUDGETS.get(name)
        if _legacy and abs(cfg["budget"] - _legacy) > 1:
            print(f"\n  NOTE: season budget resolved from archetypes.json = "
                  f"{cfg['budget']:,.0f} BDT; the hardcoded value used before v2.0 was "
                  f"{_legacy:,.0f} ({100*(_legacy-cfg['budget'])/cfg['budget']:+.1f}%). "
                  f"Results are NOT comparable to pre-v2.0 runs in absolute BDT.")
        found = discover_seed_folders(cfg["dir_tag"])
        print("\n" + "#" * 100)
        print(f"# BATCH {i+1}/{len(wanted)}:  {name}  ({cfg['n_wards']} wards)")
        print(f"#   graph  = {cfg['graph']}")
        print(f"#   econ   = {cfg['econ']}   base_budget = {cfg['budget']:,}")
        print(f"#   seeds  = {len(found)} discovered under {MODELS_ROOT}")
        for k, v in found.items():
            print(f"#            {k}: {os.path.basename(os.path.dirname(v))}/{os.path.basename(v)}")
        print("#" * 100)
        if not found:
            print(f"  SKIPPED: no {VERSION_TAG} run folders matching '{cfg['dir_tag']}' "
                  f"with tag _{EVAL_REGIME}_rho{int(round(EVAL_RHO*100)):03d}_")
            continue
        globals()["EVAL_GRAPH"] = cfg["graph"]
        globals()["EVAL_ECON"] = cfg["econ"]
        globals()["BASE_BUDGET"] = cfg["budget"]
        globals()["EVAL_CHECKPOINT"] = None
        globals()["EVAL_MODELS_DIR"] = None
        globals()["_BATCH_SEEDS"] = found
        _configs_run.append({"name": name, "graph": cfg["graph"], "econ": cfg["econ"],
                             "base_budget": cfg["budget"], "n_wards": cfg["n_wards"],
                             "kappa_at_1x": KAPPA_AT_1X, "checkpoints": dict(found)})
        run_analysis(name, cfg)
    write_csv(csv_path)
    write_meta(csv_path, args, _configs_run)


if __name__ == "__main__":
    main()
