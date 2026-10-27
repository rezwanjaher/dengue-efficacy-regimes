"""The heuristic policies and the weekly step they share with the harness.
Used by run_all_evals.py, which produces every reported result.

Options:
    --trials  Number of Monte Carlo trials. Default 5.
    --model   Explicit path to an RL checkpoint (.zip).
"""

import os
import sys
import argparse
try:
    from scipy.optimize import linprog as _linprog
    _HAVE_LINPROG = True
except Exception:          
    _HAVE_LINPROG = False
    _linprog = None

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
import json
import re

PACE_HEURISTICS = True  

try:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _project_root not in sys.path: sys.path.insert(0, _project_root)
except NameError:
    _project_root = os.path.abspath(os.path.join(os.getcwd()));
    if 'src' in _project_root: _project_root = os.path.dirname(_project_root)
    if _project_root not in sys.path: sys.path.insert(0, _project_root)

from src.env.smip_env import SMIPEnv
from src.env.archetype_parser import ArchetypeParser
from src.env.physics import run_dynamics_step, calculate_seasonality_multiplier

plt.rcParams['font.family'] = 'Arial'; plt.rcParams['svg.fonttype'] = 'none'

EVAL_SEED = 123

def find_latest_checkpoint(folder_path: str) -> str or None:
    if not os.path.isdir(folder_path): return None
    zip_files = [f for f in os.listdir(folder_path) if f.endswith(".zip")]
    if not zip_files: return None

    def step_count(f):
        m = re.search(r'_(\d+)k', f)          
        if m: return int(m.group(1)) * 1000
        m = re.search(r'_(\d+)_steps', f)      
        if m: return int(m.group(1))
        return -1

    def sort_key(f):
        return (step_count(f), os.path.getmtime(os.path.join(folder_path, f)))

    best = max(zip_files, key=sort_key)
    print(f"[eval] selected model: {best} (step_count={step_count(best)})")
    return os.path.join(folder_path, best)

def load_dhaka_graph():
    graph_path = os.path.join(_project_root, "dhaka_graph_populated.json")
    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    N = np.array(data["N_initial"], dtype=np.float32)
    neighbors = {int(k): v for k, v in data["neighbors"].items()}
    ward_names = data["ward_names"]
    return N, neighbors, ward_names

def generate_dengue_scenario(level, params, rng):
    N, neighbors, _ = load_dhaka_graph()
    
    h_I = np.zeros_like(N)
    top_wards = np.argsort(N)[-5:]
    for idx in top_wards:
        h_I[idx] = N[idx] * 0.01 
        
    h_S = np.maximum(0, N - h_I)
    h_R = np.zeros_like(N)
    
    true_H = {'N': N, 'S': h_S, 'I': h_I, 'R': h_R}
    
    season_mult = calculate_seasonality_multiplier(18, params['seasonality'])
    carrying_cap = params['mosquito_model'].get('carrying_capacity_multiplier', 2.0)
    m_total = N * carrying_cap * season_mult
    
    m_I = np.zeros_like(N)
    m_E = np.zeros_like(N)
    m_E[h_I > 0] = m_total[h_I > 0] * 0.02
    m_S = np.maximum(0, m_total - m_E - m_I)
    
    true_M = {'S': m_S, 'E': m_E, 'I': m_I}
    
    weeks_to_spread = {"Low": 1, "Medium": 4, "High": 8}[level]
    calendar_week = 18
    
    for _ in range(weeks_to_spread):
        no_actions = {'vector_control': np.zeros_like(N), 'case_management': np.zeros_like(N)}
        true_H, true_M = run_dynamics_step(true_H, true_M, no_actions, calendar_week, rng, params, neighbors)
        calendar_week += 1
        
    return true_H, true_M, neighbors, calendar_week

LAST_ACTIONS = None
LAST_SURVEY_FRAC = None
LAST_SPEND = None
SURVEY_SHARE = 1.0


def run_heuristic_week(week, true_H, true_M, neighbors, policy_name, budget, params, rng,
                       belief=None):
    N = true_H['N']
    num_sites = len(N)
    vc_coverage = np.zeros(num_sites)
    cm_coverage = np.zeros(num_sites)
    larv_coverage = np.zeros(num_sites)

    BELIEF_DECAY = 0.6
    if belief is None:
        belief = {
            'I_hat': np.zeros(num_sites),
            'age': np.zeros(num_sites),
            'ever_seen': np.zeros(num_sites),
        }
    
    survey_cost = 0.0; cm_cost = 0.0; vc_cost = 0.0
    larv_cost = 0.0
    globals()['LAST_SURVEY_FRAC'] = 0.0
    budget_left = float(budget)
    
    action_params = params['action_semantics']
    c_surv = action_params['survey']['cost']
    c_cm = action_params['case_management']['cost']
    c_vc = action_params['vector_control']['cost']

    def survey_and_update_belief(budget_left, survey_cost):
        eff = action_params['survey'].get("efficacy", 0.75)
        belief['age'] += 1.0
        belief['I_hat'] = belief['I_hat'] * BELIEF_DECAY
        surveyed = np.zeros(num_sites)
        share = float(globals().get('SURVEY_SHARE', 1.0))
        survey_pool = budget_left * share
        for i in np.argsort(-N):
            cost = N[i] * c_surv
            if survey_pool >= cost:
                surveyed[i] = 1
                survey_pool -= cost
                budget_left -= cost
                survey_cost += cost
        for i in range(num_sites):
            if surveyed[i] == 1:
                reading = rng.binomial(np.round(true_H['I'][i]).astype(int), eff) / max(eff, 1e-6)
                belief['I_hat'][i] = reading
                belief['age'][i] = 0.0
                belief['ever_seen'][i] = 1.0
        globals()['LAST_SURVEY_FRAC'] = float(np.mean(surveyed))
        return belief['I_hat'].copy(), budget_left, survey_cost

    if policy_name == "H0_Do_Nothing":
        pass

    elif policy_name == "H1_Blanket_Fogging":
        for i in np.argsort(-N):
            if budget_left <= 0: break
            cost_to_fog_ward = N[i] * c_vc
            if cost_to_fog_ward > 0:
                frac = max(0.0, min(1.0, budget_left / cost_to_fog_ward))
                vc_coverage[i] = frac
                spent = frac * cost_to_fog_ward
                budget_left -= spent
                vc_cost += spent

    elif policy_name == "H2_Test_and_Target":
        eff = action_params['survey'].get("efficacy", 0.75)
        I_hat_debiased, budget_left, survey_cost = survey_and_update_belief(budget_left, survey_cost)
        I_found = np.maximum(0.0, I_hat_debiased * eff)

        for i in np.argsort(-I_found):
            if budget_left <= 0 or I_found[i] <= 0: break
            cost_to_cm = I_found[i] * c_cm
            cm_frac = max(0.0, min(1.0, budget_left / (cost_to_cm + 1e-6)))
            cm_coverage[i] = cm_frac
            spent_cm = cm_frac * cost_to_cm
            budget_left -= spent_cm
            cm_cost += spent_cm
            
            cost_to_vc = N[i] * c_vc
            vc_frac = max(0.0, min(1.0, budget_left / (cost_to_vc + 1e-6)))
            vc_coverage[i] = vc_frac
            spent_vc = vc_frac * cost_to_vc
            budget_left -= spent_vc
            vc_cost += spent_vc

    elif policy_name == "H3_Random_Larviciding":
        c_larv = action_params.get('larviciding', {}).get('cost', c_vc)
        random_order = np.arange(num_sites)
        rng.shuffle(random_order)
        for i in random_order:
            if budget_left <= 0: break
            cost_to_larv = N[i] * c_larv
            if cost_to_larv > 0:
                frac = max(0.0, min(1.0, budget_left / cost_to_larv))
                larv_coverage[i] = frac
                spent = frac * cost_to_larv
                budget_left -= spent
                larv_cost += spent

    elif policy_name in ("H5_MyopicLP_Belief", "H5_MyopicLP_Oracle"):
        c_larv = action_params.get('larviciding', {}).get('cost', c_vc)
        eta_F = action_params['vector_control'].get('efficacy', 0.5)

        if policy_name == "H5_MyopicLP_Oracle":
            I_signal = true_H['I']                     
        else:
            I_signal, budget_left, survey_cost = survey_and_update_belief(budget_left, survey_cost)

        mm = params['mosquito_model']
        trans_m2h = mm.get('transmission_prob_mosquito_to_human',
                           mm.get('transmission_prob_human_to_mosquito', 0.5))
        K = (mm['biting_rate_per_week'] * trans_m2h * true_H['S']
             * (np.maximum(0.0, I_signal) / np.maximum(1.0, N)))

        cost_full = c_vc * N
        if _HAVE_LINPROG and K.sum() > 0 and budget_left > 0:
            res = _linprog(c=-(K * eta_F), A_ub=cost_full.reshape(1, -1), b_ub=[budget_left],
                           bounds=[(0.0, 1.0)] * num_sites, method="highs")
            u = np.clip(res.x, 0.0, 1.0) if res.success else np.zeros(num_sites)
        else:
            u = np.zeros(num_sites)
            ratio = np.where(cost_full > 0, (K * eta_F) / cost_full, 0.0)
            bl_tmp = budget_left
            for i in np.argsort(-ratio):
                if bl_tmp <= 0 or ratio[i] <= 0: break
                spend = min(cost_full[i], bl_tmp); u[i] = spend / cost_full[i]; bl_tmp -= spend
        
        vc_coverage = u
        spent_fog = float((u * cost_full).sum())
        budget_left -= spent_fog; vc_cost += spent_fog

        #             l = max(0.0, min(1.0, budget_left / cost_larv))
        #             larv_coverage[i] = l
        #             budget_left -= l * cost_larv

    elif policy_name in ("H6_GreedyCoupled_Belief", "H6_GreedyCoupled_Oracle"):
        c_larv = action_params.get('larviciding', {}).get('cost', c_vc)
        eps_H = params['spatial_spread']['human_mobility_network_rate']
        w = np.ones(num_sites)
        for j in range(num_sites):
            nbrs = neighbors.get(j, [])
            deg = len(nbrs)
            if deg == 0: continue
            for i in nbrs:
                if 0 <= i < num_sites:
                    w[i] += eps_H / deg

        if policy_name == "H6_GreedyCoupled_Oracle":
            rank_I = true_H['I'] * w
        else:
            I_hat_now, budget_left, survey_cost = survey_and_update_belief(budget_left, survey_cost)
            rank_I = I_hat_now * w

        for i in np.argsort(-rank_I):
            if budget_left <= 0: break
            if policy_name == "H6_GreedyCoupled_Belief" and rank_I[i] <= 0:
                break
            cost_fog = N[i] * c_vc
            if cost_fog > 0:
                f = max(0.0, min(1.0, budget_left / cost_fog))
                vc_coverage[i] = f
                spent = f * cost_fog; budget_left -= spent; vc_cost += spent
            if budget_left <= 0: break
            cost_larv = N[i] * c_larv
            if cost_larv > 0:
                l = max(0.0, min(1.0, budget_left / cost_larv))
                larv_coverage[i] = l
                budget_left -= l * cost_larv

    elif policy_name in ("H7_GreedyDensity_Belief", "H7_GreedyDensity_Oracle"):
        c_larv = action_params.get('larviciding', {}).get('cost', c_vc)
        density = true_M['S'] / np.maximum(1.0, N)

        if policy_name == "H7_GreedyDensity_Oracle":
            rank_I = true_H['I'] * density
        else:
            I_hat_now, budget_left, survey_cost = survey_and_update_belief(budget_left, survey_cost)
            rank_I = I_hat_now * density

        for i in np.argsort(-rank_I):
            if budget_left <= 0: break
            if policy_name == "H7_GreedyDensity_Belief" and rank_I[i] <= 0:
                break
            cost_fog = N[i] * c_vc
            if cost_fog > 0:
                f = max(0.0, min(1.0, budget_left / cost_fog))
                vc_coverage[i] = f
                spent = f * cost_fog; budget_left -= spent; vc_cost += spent
            if budget_left <= 0: break
            cost_larv = N[i] * c_larv
            if cost_larv > 0:
                l = max(0.0, min(1.0, budget_left / cost_larv))
                larv_coverage[i] = l
                budget_left -= l * cost_larv

    elif policy_name in ("H4_Greedy_Belief", "H4_Greedy_Oracle"):
        c_larv = action_params.get('larviciding', {}).get('cost', c_vc)

        if policy_name == "H4_Greedy_Oracle":
            rank_I = true_H['I']                       
        else:
            rank_I, budget_left, survey_cost = survey_and_update_belief(budget_left, survey_cost)

        for i in np.argsort(-rank_I):
            if budget_left <= 0: break
            if policy_name == "H4_Greedy_Belief" and rank_I[i] <= 0:
                break  
            cost_fog = N[i] * c_vc
            if cost_fog > 0:
                f = max(0.0, min(1.0, budget_left / cost_fog))
                vc_coverage[i] = f
                spent = f * cost_fog; budget_left -= spent; vc_cost += spent
            if budget_left <= 0: break
            cost_larv = N[i] * c_larv
            if cost_larv > 0:
                l = max(0.0, min(1.0, budget_left / cost_larv))
                larv_coverage[i] = l
                budget_left -= l * cost_larv

    actions = {'vector_control': vc_coverage, 'case_management': cm_coverage,
               'larviciding': larv_coverage}
    globals()['LAST_ACTIONS'] = {k: np.asarray(v).copy() for k, v in actions.items()}
    old_S = true_H['S'].copy()
    globals()['LAST_SPEND'] = {'survey': float(survey_cost), 'vc': float(vc_cost),
                               'cm': float(cm_cost), 'larv': float(larv_cost)}
    
    next_true_H, next_true_M = run_dynamics_step(true_H, true_M, actions, week, rng, params, neighbors)
    new_infections = np.sum(np.maximum(0, old_S - next_true_H['S']))
    
    costs = {'survey': survey_cost, 'cm': cm_cost, 'vc': vc_cost, 'larv': larv_cost}
    return next_true_H, next_true_M, costs, new_infections, budget_left, belief

def plot_epidemic_charts(history_df, scenario_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    weeks = history_df['week'].unique()
    
    def format_ax(ax, title, ylabel):
        ax.set_title(title, fontsize=10, weight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('Week', fontsize=9)
        sns.despine()
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    ax.plot(weeks, history_df['susceptible'], color='#0d5a94', label='Susceptible (S)', linewidth=2)
    ax.plot(weeks, history_df['infected'], color='#d62728', label='Infected (I)', linewidth=2)
    ax.plot(weeks, history_df['recovered'], color='#2ca02c', label='Recovered (R)', linewidth=2)
    format_ax(ax, 'Dengue S-I-R Epidemic Curve', 'Population')
    ax.legend(frameon=False)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y/1e6:.1f}M'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_epidemic_curve.svg'))
    plt.close()

def main():
    EVALUATION_HORIZON = 20 
    DENGUE_CFR = 0.005 
    
    print("--- Loading BIOLOGICAL parameters from archetypes.json ---")
    archetype_filepath = os.path.join(_project_root, 'configs', 'archetypes.json')
    parser = ArchetypeParser(archetype_filepath)
    EVAL_SPECIES = "Dengue_Dhaka_Advanced"  
    archetype_data = parser.get_archetype_data("Urban Public Health", EVAL_SPECIES)
    
    params_rng = np.random.default_rng(seed=EVAL_SEED)
    params = parser.sample_parameters(archetype_data, "Balanced_Medium_Budget", params_rng)
    
    arg_parser = argparse.ArgumentParser(description='The heuristic policies and the weekly step they share with the harness.')
    arg_parser.add_argument("--trials", type=int, default=5, help="Number of Monte Carlo trials.")
    arg_parser.add_argument("--model", type=str, default=None, help='Explicit path to an RL checkpoint (.zip).')
    args = arg_parser.parse_args()
    
    N_arr, _, _ = load_dhaka_graph()
    num_sites = len(N_arr)

    per_site_budget = params.get("total_episode_budget_per_site", 7_500_000.0)
    episode_budget_total = float(per_site_budget * num_sites)
    total_budget_weekly = episode_budget_total / EVALUATION_HORIZON
    
    base_models_dir = os.path.join(_project_root, "rl_models")
    if args.model:
        evaluation_run_dir = args.model
    else:
        evaluation_run_dir = find_latest_checkpoint(base_models_dir)
        if not evaluation_run_dir:
            folders = [os.path.join(base_models_dir, d) for d in os.listdir(base_models_dir) if os.path.isdir(os.path.join(base_models_dir, d))]
            if folders: evaluation_run_dir = find_latest_checkpoint(max(folders, key=os.path.getmtime))
        
    print(f"Loading RL Agent from: {evaluation_run_dir}")
    rl_model = PPO.load(evaluation_run_dir, device='cpu') if evaluation_run_dir else None
        
    policies_to_test = ["H0_Do_Nothing", "H1_Blanket_Fogging", "H2_Test_and_Target", "H3_Random_Larviciding", "H4_Greedy_Belief", "H4_Greedy_Oracle", "H5_MyopicLP_Belief", "H5_MyopicLP_Oracle", "RL_Agent"]
    if not rl_model: policies_to_test.remove("RL_Agent")
    
    scenarios_to_run = ["Low", "Medium", "High"]
    all_results = []
    
    print("\n" + "#"*80)
    print("### RUNNING 20-WEEK DHAKA DENGUE EVALUATION ###")
    print(f"### GLOBAL EVAL SEED: {EVAL_SEED} | TRIALS: {args.trials} ###")
    print("#"*80)
    
    for scenario_level in scenarios_to_run:
        print(f"\n--- Generating '{scenario_level}' Severity Scenario ---")
        scenario_rng = np.random.default_rng(seed=EVAL_SEED)
        true_H_init, true_M_init, neighbors_init, start_week = generate_dengue_scenario(scenario_level, params, scenario_rng)
        _, _, ward_names = load_dhaka_graph()
        
        for policy in policies_to_test:
            trial_stats = []
            spatial_actions = np.zeros((len(ward_names), 3)) 
            
            total_incidence_all_trials = 0
            total_spent_all_trials = 0
            
            for trial in tqdm(range(args.trials), desc=f"Testing {policy} ({scenario_level})", leave=False):
                true_H = {k: v.copy() for k,v in true_H_init.items()}
                true_M = {k: v.copy() for k,v in true_M_init.items()}
                
                trial_rng = np.random.default_rng(seed=EVAL_SEED + trial)
                cal_week = start_week

                episode_budget = float(total_budget_weekly * EVALUATION_HORIZON)
                heuristic_budget_remaining = episode_budget
                heuristic_pool = episode_budget

                if policy == "RL_Agent":
                    env = SMIPEnv(archetype_filepath=archetype_filepath, species_group="Urban Public Health", species_name=EVAL_SPECIES, economic_scenarios=["Balanced_Medium_Budget"], force_T_HORIZON=EVALUATION_HORIZON)
                    obs, _ = env.reset(seed=EVAL_SEED + trial)
                    env.ep_params = params
                    env.true_H = {k: v.copy() for k,v in true_H.items()}
                    env.true_M = {k: v.copy() for k,v in true_M.items()}
                    env.believed_H = {k: v.copy() for k,v in true_H.items()}
                    env.believed_M = {k: v.copy() for k,v in true_M.items()}
                    env.calendar_week = cal_week
                    env.total_episode_budget = episode_budget
                    env.budget_remaining_for_episode = episode_budget
                    obs = env._get_observation()
                
                trial_incidence = 0
                trial_spent = 0
                heuristic_belief = None
                
                for week in range(1, EVALUATION_HORIZON + 1):
                    if policy == "RL_Agent":
                        action, _ = rl_model.predict(obs, deterministic=True)
                        obs, _, _, _, info = env.step(action)
                        new_cases = info.get('new_infections', 0)
                        costs = info.get("costs_by_type", {})
                        true_H = env.true_H
                        
                        if week == 1 and trial == 0:
                            active_action = action[:env.num_sites * env.action_dim_per_node].reshape((env.num_sites, env.action_dim_per_node))
                            survey_intent = 1 / (1 + np.exp(-active_action[:, 0]))
                            cm_intent = (active_action[:, 1] + 1) / 2.0
                            vc_intent = (active_action[:, 2] + 1) / 2.0
                            
                            spatial_actions[:, 0] = survey_intent
                            spatial_actions[:, 1] = cm_intent
                            spatial_actions[:, 2] = vc_intent
                    else:
                        weeks_left = max(1, EVALUATION_HORIZON - week + 1)
                        weekly_cap = (heuristic_pool / weeks_left) if PACE_HEURISTICS else heuristic_pool
                        
                        true_H, true_M, costs, new_cases, cap_left, heuristic_belief = run_heuristic_week(
                            cal_week, true_H, true_M, neighbors_init, policy,
                            weekly_cap, params, trial_rng,
                            belief=heuristic_belief)
                            
                        spent_this_week = weekly_cap - cap_left
                        heuristic_pool -= spent_this_week
                        heuristic_budget_remaining = heuristic_pool
                        cal_week += 1
                    
                    trial_incidence += new_cases
                    trial_spent += sum(costs.values())
                    
                    if trial == 0: 
                        trial_stats.append({
                            'week': week, 'susceptible': np.sum(true_H['S']), 'infected': np.sum(true_H['I']), 'recovered': np.sum(true_H['R']),
                            'cost_survey': costs.get('survey', 0), 'cost_treat': costs.get('cm', 0), 'cost_remove': costs.get('vc', 0)
                        })
                
                total_incidence_all_trials += trial_incidence
                total_spent_all_trials += trial_spent
            
            avg_incidence = total_incidence_all_trials / args.trials
            avg_spent = total_spent_all_trials / args.trials
            
            human_life_loss = avg_incidence * DENGUE_CFR
            penalty = avg_incidence * params.get('delta_penalty_human_I', 20000.0) 
            
            all_results.append({
                "Policy": policy, "Scenario": scenario_level,
                "Estimated Lives Lost": int(human_life_loss),
                "Total New Cases": int(avg_incidence),
                "Total BDT Spent": avg_spent,
                "Total Economic Loss": avg_spent + penalty
            })
            
            output_dir = os.path.join(_project_root, "evaluation_dengue", f"Scenario_{scenario_level}")
            plot_epidemic_charts(pd.DataFrame(trial_stats), scenario_level, os.path.join(output_dir, policy))
            
            if policy == "RL_Agent":
                df_spatial = pd.DataFrame({
                    "Ward_Name": ward_names, "Population": N_arr, 
                    "Week1_Survey_Intent": spatial_actions[:, 0], 
                    "Week1_CaseMgmt_Intent": spatial_actions[:, 1], 
                    "Week1_VectorControl_Intent": spatial_actions[:, 2]
                })
                df_spatial.to_csv(os.path.join(output_dir, "RL_Agent_Spatial_Strategy_Week1.csv"), index=False)

    df_summary = pd.DataFrame(all_results)
    print("\n" + "="*110)
    print("DENGUE INCIDENCE & POLICY COMPARISON (20 WEEKS)")
    print("="*110)
    
    format_dict = {
        'Estimated Lives Lost': '{:,}',
        'Total New Cases': '{:,}',
        'Total BDT Spent': lambda x: f'৳{x/1e6:.1f}M' if x > 100000 else f'{x:,.0f}',
        'Total Economic Loss': lambda x: f'৳{x/1e6:.1f}M' if x > 100000 else f'{x:,.0f}'
    }
    
    summary_grouped = df_summary.groupby(['Scenario', 'Policy']).mean()
    for col, fmt in format_dict.items():
        if isinstance(fmt, str):
            summary_grouped[col] = summary_grouped[col].apply(lambda x: fmt.format(int(x)))
        else:
            summary_grouped[col] = summary_grouped[col].apply(fmt)
            
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary_grouped)
    print("="*110)
    
    summary_dir = os.path.join(_project_root, "evaluation_dengue", "Summary")
    os.makedirs(summary_dir, exist_ok=True)
    df_summary.to_csv(os.path.join(summary_dir, "overall_dengue_results.csv"), index=False)

if __name__ == '__main__':
    main()