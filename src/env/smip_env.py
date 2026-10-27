"""The environment: ward states, belief, budget and the weekly decision.
"""

import os
import json
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from src.env.archetype_parser import ArchetypeParser
from src.env.physics import run_dynamics_step, calculate_seasonality_multiplier

class SMIPEnv(gym.Env):
    def __init__(self, archetype_filepath: str, species_group: str, species_name: str, economic_scenarios: list,
                 force_T_HORIZON=None, force_perfect_surveys=False, force_start_week=None,
                 force_free_info=False, greedy_residual=False,
                 rho=0.2, prior_regime="estimated", prior_history=5, lambda_sigma=1.2,
                 graph_filepath_override=None,
                 reward_mode="weighted", overspend_penalty=10.0):
        super().__init__()

        #                 cumulative spend exceeds the season budget.
        if reward_mode not in ("weighted", "constrained"):
            raise ValueError(f"reward_mode must be 'weighted' or 'constrained', "
                             f"got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.overspend_penalty = float(overspend_penalty)

        self.graph_filepath_override = graph_filepath_override

        #                  persistence); ~0.7-1.0 = novel-serotype/emergence.
        #                  start historical prior lambda_hat is.
        #   prior_history: # past seasons behind an 'estimated' prior (error ~1/sqrt(H)).
        self.rho = float(rho)
        self.prior_regime = str(prior_regime)
        self.prior_history = int(prior_history)
        self.lambda_sigma = float(lambda_sigma)

        self.force_T_HORIZON = force_T_HORIZON
        self.force_perfect_surveys = force_perfect_surveys
        self.force_start_week = force_start_week
        self.greedy_residual = greedy_residual
        self.force_free_info = force_free_info
        
        self.parser = ArchetypeParser(archetype_filepath)
        self.archetype_data = self.parser.get_archetype_data(species_group, species_name)
        self.economic_scenarios_to_sample = economic_scenarios
        
        self._load_dhaka_graph()
        
        self.action_dim_per_node = 4
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.num_sites * self.action_dim_per_node,), dtype=np.float32)
        
        self.node_feature_dim = 15
        
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_sites, self.node_feature_dim), dtype=np.float32)
        
        self.MIN_T_HORIZON, self.MAX_T_HORIZON = 15, 25

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is None: options = {}
        self.belief_seeder_rng = self.np_random.spawn(1)[0]
        
        passed_horizon = options.get("T_HORIZON") if options else None
        if passed_horizon is not None:
            self.T_HORIZON = passed_horizon
        elif self.force_T_HORIZON is not None: 
            self.T_HORIZON = self.force_T_HORIZON
        else: 
            self.T_HORIZON = self.np_random.integers(self.MIN_T_HORIZON, self.MAX_T_HORIZON + 1)
            
        if self.force_start_week is not None:
            self.start_calendar_week = self.force_start_week
            self.start_week_needs_seasonal_placement = False
        else:
            self.start_week_needs_seasonal_placement = True

        self.current_step = 1

        self.chosen_economic_scenario = self.np_random.choice(self.economic_scenarios_to_sample)
        self.ep_params = self.parser.sample_parameters(self.archetype_data, self.chosen_economic_scenario, self.np_random)

        if self.start_week_needs_seasonal_placement:
            peak_week = int(round(self.ep_params.get('seasonality', {}).get('peak_week', 36)))
            horizon = self.T_HORIZON if self.T_HORIZON else 20
            earliest = max(1, peak_week - horizon)
            latest = max(earliest + 1, peak_week)
            self.start_calendar_week = int(self.np_random.integers(earliest, latest + 1))

        self.calendar_week = self.start_calendar_week
        if self.force_perfect_surveys:
            try:
                self.ep_params['action_semantics']['survey']['efficacy'] = 1.0
            except KeyError:
                pass
                
        self._initialize_epidemic_states()
        
        total_budget_per_site = self.ep_params.get("total_episode_budget_per_site", 500000.0)
        self.total_episode_budget = float(total_budget_per_site * self.num_sites)
        self.budget_remaining_for_episode = self.total_episode_budget
        
        return self._get_observation(), {}

    def _load_dhaka_graph(self):
        override = getattr(self, "graph_filepath_override", None)
        if override:
            graph_path = override
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"graph_filepath_override not found: {graph_path}")
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            graph_path = os.path.join(project_root, "dhaka_graph_populated.json")

            if not os.path.exists(graph_path):
                graph_path = "dhaka_graph_populated.json"
                if not os.path.exists(graph_path):
                    raise FileNotFoundError("CRITICAL: dhaka_graph_populated.json not found.")

        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
            
        self.num_sites = graph_data["num_sites"]
        self.dhaka_populations = np.array(graph_data["N_initial"], dtype=np.float32)
        raw_neighbors = graph_data["neighbors"]
        self.neighbors = {int(k): v for k, v in raw_neighbors.items()}

        self.LAMBDA_SIGMA = float(getattr(self, "lambda_sigma", 1.2))
        self.RHO = float(getattr(self, "rho", 0.2))
        self.PRIOR_REGIME = getattr(self, "prior_regime", "estimated")
        self.PRIOR_HISTORY = int(getattr(self, "prior_history", 5))
        _field_rng = np.random.default_rng(20240723)
        _z = _field_rng.standard_normal(self.num_sites)
        lam = np.exp(self.LAMBDA_SIGMA * _z)
        self.lambda_field = (lam / lam.sum()).astype(np.float32)

        if self.PRIOR_REGIME == "known":
            self.lambda_hat = self.lambda_field.copy()
        elif self.PRIOR_REGIME == "flat":
            self.lambda_hat = np.full(self.num_sites, 1.0 / self.num_sites, dtype=np.float32)
        else:
            _est_rng = np.random.default_rng(20240721)
            err_sigma = 0.75 / np.sqrt(max(1, self.PRIOR_HISTORY))
            noisy = self.lambda_field * _est_rng.lognormal(0.0, err_sigma, size=self.num_sites)
            self.lambda_hat = (noisy / noisy.sum()).astype(np.float32)

    def _initialize_epidemic_states(self):
        h_N = self.dhaka_populations.copy()
        h_I = np.zeros_like(h_N, dtype=np.float32)
        h_R = np.zeros_like(h_N, dtype=np.float32)
        
        #            (endemic dengue; strong persistence).
        #            (novel-serotype / emergence; the 2023 paradigm-shift end).
        uniform_p = np.full(self.num_sites, 1.0 / self.num_sites, dtype=np.float32)
        p_ignite = (1.0 - self.RHO) * self.lambda_field + self.RHO * uniform_p
        p_ignite = p_ignite / p_ignite.sum()
        num_epicenters = self.np_random.integers(2, 6)
        epicenters = self.np_random.choice(self.num_sites, size=num_epicenters,
                                           replace=False, p=p_ignite)
        for idx in epicenters:
            scale = 0.005 + 0.015 * (self.lambda_field[idx] * self.num_sites) / \
                    (1.0 + self.lambda_field[idx] * self.num_sites)
            h_I[idx] = h_N[idx] * self.np_random.uniform(0.005, 0.005 + scale)
            
        h_S = np.maximum(0, h_N - h_I - h_R)
        self.true_H = {'N': h_N, 'S': h_S, 'I': h_I, 'R': h_R}
        
        season_mult = calculate_seasonality_multiplier(self.calendar_week, self.ep_params['seasonality'])
        carrying_cap = self.ep_params['mosquito_model'].get('carrying_capacity_multiplier', 2.0)
        m_total = h_N * carrying_cap * season_mult
        
        m_I = np.zeros_like(h_N, dtype=np.float32)
        m_E = np.zeros_like(h_N, dtype=np.float32)
        
        m_E[h_I > 0] = m_total[h_I > 0] * 0.02
        m_S = np.maximum(0, m_total - m_E - m_I)
        
        self.true_M = {'S': m_S, 'E': m_E, 'I': m_I}

        expected_total_I = float(h_I.sum()) if h_I.sum() > 0 else float(h_N.sum() * 0.001)
        prior_I = (self.lambda_hat * expected_total_I).astype(np.float32)
        prior_I = np.minimum(prior_I, h_N)
        bel_S = np.maximum(0.0, h_N - prior_I).astype(np.float32)
        self.believed_H = {'N': h_N.copy(), 'S': bel_S,
                           'I': prior_I, 'R': np.zeros_like(h_N, dtype=np.float32)}
        bel_mE = (m_total * 0.02 * (self.lambda_hat * self.num_sites)).astype(np.float32)
        bel_mE = np.minimum(bel_mE, m_total)
        self.believed_M = {'S': np.maximum(0.0, m_total - bel_mE).astype(np.float32),
                           'E': bel_mE,
                           'I': np.zeros_like(h_N, dtype=np.float32)}

    def _calculate_action_allocations(self, actions_raw, budget):
        survey_intent = 1 / (1 + np.exp(-actions_raw[:, 0])) 
        cm_intent = (actions_raw[:, 1] + 1) / 2.0            
        vc_intent = (actions_raw[:, 2] + 1) / 2.0            
        if actions_raw.shape[1] > 3:
            larv_intent = (actions_raw[:, 3] + 1) / 2.0
        else:
            larv_intent = np.zeros(self.num_sites)
        
        action_params = self.ep_params['action_semantics']
        cost_per_survey = action_params['survey']['cost']
        if getattr(self, 'force_free_info', False):
            cost_per_survey = 0.0
        cost_per_cm = action_params['case_management']['cost']
        cost_per_vc = action_params['vector_control']['cost']
        cost_per_larv = action_params.get('larviciding', {}).get('cost', cost_per_vc)
        
        CHANNEL_TAU = 0.5
        _ch_logits = np.array([
            float(np.mean(survey_intent)), float(np.mean(vc_intent)),
            float(np.mean(cm_intent)),     float(np.mean(larv_intent)),
        ], dtype=np.float64) / CHANNEL_TAU
        _ch_logits -= _ch_logits.max()
        _w = np.exp(_ch_logits); _w /= max(_w.sum(), 1e-12)
        _total_budget = float(budget)
        _chan_budget = {'survey': _w[0] * _total_budget, 'vc': _w[1] * _total_budget,
                        'cm': _w[2] * _total_budget, 'larv': _w[3] * _total_budget}
        self._last_channel_shares = {'survey': float(_w[0]), 'vector_control': float(_w[1]),
                                     'case_management': float(_w[2]), 'larviciding': float(_w[3])}

        actual_costs = {'survey': 0.0, 'cm': 0.0, 'vc': 0.0, 'larv': 0.0}

        budget_remaining = _chan_budget['survey']
        survey_decisions = self.np_random.binomial(1, survey_intent)
        surveys_performed = np.zeros(self.num_sites)
        
        for i in np.argsort(-survey_intent):
            if budget_remaining <= 0: break
            if survey_decisions[i] == 1:
                cost = self.true_H['N'][i] * cost_per_survey
                if budget_remaining >= cost:
                    surveys_performed[i] = 1
                    budget_remaining -= cost
                    actual_costs['survey'] += cost
                    
        vc_coverage = np.zeros(self.num_sites)
        _carry = budget_remaining
        budget_remaining = _chan_budget['vc'] + _carry

        MIN_USEFUL_FRAC = 0.0
        for i in np.argsort(-vc_intent):
            if budget_remaining <= 0: break
            if vc_intent[i] <= 0: continue
            full_cost = self.true_H['N'][i] * cost_per_vc
            if full_cost <= 0: continue
            affordable_frac = max(0.0, min(1.0, budget_remaining / (full_cost + 1e-6)))
            actual_frac = max(0.0, min(vc_intent[i], affordable_frac))
            if actual_frac < MIN_USEFUL_FRAC:
                continue
            vc_coverage[i] = actual_frac
            cost = actual_frac * full_cost
            budget_remaining -= cost
            actual_costs['vc'] += cost

        cm_coverage = np.zeros(self.num_sites)
        _carry = budget_remaining
        budget_remaining = _chan_budget['cm'] + _carry
        for i in np.argsort(-cm_intent):
            if budget_remaining <= 0: break
            known_infected = self.believed_H['I'][i]
            desired_cost = cm_intent[i] * known_infected * cost_per_cm
            if desired_cost > 0:
                affordable_frac = max(0.0, min(1.0, budget_remaining / (known_infected * cost_per_cm + 1e-6)))
                actual_frac = max(0.0, min(cm_intent[i], affordable_frac))
                cm_coverage[i] = actual_frac
                cost = actual_frac * known_infected * cost_per_cm
                budget_remaining -= cost
                actual_costs['cm'] += cost

        larv_coverage = np.zeros(self.num_sites)
        _carry = budget_remaining
        budget_remaining = _chan_budget['larv'] + _carry
        for i in np.argsort(-larv_intent):
            if budget_remaining <= 0: break
            if larv_intent[i] <= 0: continue
            full_cost = self.true_H['N'][i] * cost_per_larv
            if full_cost <= 0: continue
            affordable_frac = max(0.0, min(1.0, budget_remaining / (full_cost + 1e-6)))
            actual_frac = max(0.0, min(larv_intent[i], affordable_frac))
            if actual_frac < MIN_USEFUL_FRAC:
                continue
            larv_coverage[i] = actual_frac
            cost = actual_frac * full_cost
            budget_remaining -= cost
            actual_costs['larv'] += cost

        final_actions = {'vector_control': vc_coverage, 'case_management': cm_coverage,
                         'larviciding': larv_coverage}
        self._last_final_actions = final_actions
        self._last_raw_intents = {
            'survey': survey_intent.copy(),
            'case_management': cm_intent.copy(),
            'vector_control': vc_intent.copy(),
            'larviciding': larv_intent.copy(),
        }
        self._last_eff_intents = {
            'vector_control': vc_intent.copy(),
            'larviciding': larv_intent.copy(),
        }
        self._last_inf_norm = np.zeros(self.num_sites)
        return surveys_performed, final_actions, actual_costs

    def _compute_greedy_action(self):
        N = np.maximum(1.0, self.believed_H['N'])
        I = self.believed_H['I']
        ns = self.num_sites
        fog = np.zeros(ns); larv = np.zeros(ns)
        aps = self.ep_params['action_semantics']
        c_vc = aps['vector_control']['cost']
        c_larv = aps.get('larviciding', {}).get('cost', c_vc)
        bl = float(self.budget_remaining_for_episode)
        weeks_left = max(1, self.T_HORIZON - (self.current_step - 1))
        weekly_cap = self.budget_remaining_for_episode / weeks_left
        bl = min(bl, weekly_cap)
        for i in np.argsort(-I):
            if bl <= 0: break
            cf = N[i] * c_vc
            if cf > 0:
                f = max(0.0, min(1.0, bl / cf)); fog[i] = f; bl -= f * cf
            if bl <= 0: break
            cl = N[i] * c_larv
            if cl > 0:
                l = max(0.0, min(1.0, bl / cl)); larv[i] = l; bl -= l * cl
        return fog, larv

    def step(self, action):
        if self.num_sites == 0:
            return self._get_observation(), 0.0, True, False, {}

        active_action = action.reshape((self.num_sites, self.action_dim_per_node))

        if getattr(self, 'greedy_residual', False):
            g_fog, g_larv = self._compute_greedy_action()
            g_fog_raw = g_fog * 2 - 1
            g_larv_raw = g_larv * 2 - 1
            residual_scale = 1.0
            active_action = active_action.copy()
            active_action[:, 2] = np.clip(g_fog_raw + residual_scale * active_action[:, 2], -1.0, 1.0)
            active_action[:, 3] = np.clip(g_larv_raw + residual_scale * active_action[:, 3], -1.0, 1.0)

        surveys_performed, final_actions, actual_costs = self._calculate_action_allocations(active_action, self.budget_remaining_for_episode)
        
        total_cost_this_step = sum(actual_costs.values())
        self.budget_remaining_for_episode = max(0.0, float(self.budget_remaining_for_episode - total_cost_this_step))
        
        old_S = self.true_H['S'].copy()
        next_true_H, next_true_M = run_dynamics_step(
            self.true_H, self.true_M, final_actions, 
            self.calendar_week, self.np_random, self.ep_params, self.neighbors
        )
        new_infections = np.maximum(0, old_S - next_true_H['S'])
        
        belief_rng = np.random.default_rng(self.belief_seeder_rng.integers(2**31))
        next_bel_H, next_bel_M = run_dynamics_step(
            self.believed_H, self.believed_M, final_actions,
            self.calendar_week, belief_rng, self.ep_params, self.neighbors
        )
        
        eff_survey = self.ep_params['action_semantics']['survey'].get('efficacy', 0.8)
        
        for i in range(self.num_sites):
            if surveys_performed[i] == 1:
                found_I = self.np_random.binomial(np.round(next_true_H['I'][i]).astype(int), eff_survey)
                next_bel_H['I'][i] = found_I
                projected_R = next_bel_H['R'][i]
                derived_S = next_bel_H['N'][i] - found_I - projected_R
                if derived_S < 0:
                    derived_S = 0.0
                    next_bel_H['R'][i] = max(0.0, next_bel_H['N'][i] - found_I)
                next_bel_H['S'][i] = derived_S
                
                true_m_S = next_true_M['S'][i]
                true_m_E = next_true_M['E'][i]
                true_m_I = next_true_M['I'][i]
                true_m_total = true_m_S + true_m_E + true_m_I + 1e-6
                p_S, p_E, p_I = true_m_S/true_m_total, true_m_E/true_m_total, true_m_I/true_m_total
                found_m = true_m_total * self.np_random.uniform(eff_survey, 1.0)
                
                next_bel_M['S'][i] = found_m * p_S
                next_bel_M['E'][i] = found_m * p_E
                next_bel_M['I'][i] = found_m * p_I
                
        self.true_H = next_true_H
        self.true_M = next_true_M
        self.believed_H = next_bel_H
        self.believed_M = next_bel_M

        if self.force_free_info:
            self.believed_H = {k: v.copy() for k, v in self.true_H.items()}
            self.believed_M = {k: v.copy() for k, v in self.true_M.items()}
        
        penalty_per_infection = self.ep_params.get('delta_penalty_human_I', 20000.0)
        total_infection_penalty = np.sum(new_infections) * penalty_per_infection
        total_economic_loss_bdt = total_cost_this_step + total_infection_penalty

        max_pop = np.sum(self.dhaka_populations)
        max_penalty = (max_pop * 0.05) * penalty_per_infection
        reward_scaling = max_penalty + self.total_episode_budget

        if self.reward_mode == "constrained":
            base_reward = float(-(total_infection_penalty / (reward_scaling + 1e-6)))
            _spent = float(self.total_episode_budget) - float(self.budget_remaining_for_episode)
            _over = _spent - float(self.total_episode_budget)
            if _over > 0:
                base_reward -= self.overspend_penalty * (_over / (self.total_episode_budget + 1e-6))
        else:
            base_reward = float(-(total_economic_loss_bdt / (reward_scaling + 1e-6)))

        reward = base_reward
        
        self.current_step += 1
        self.calendar_week += 1
        terminated = self.current_step > self.T_HORIZON
        
        info = {"cost": total_cost_this_step, "new_infections": np.sum(new_infections), "costs_by_type": actual_costs}
        return self._get_observation(), float(reward), terminated, False, info
    
    def _get_observation(self):
        obs = np.zeros((self.num_sites, self.node_feature_dim), dtype=np.float32)
        if self.num_sites > 0:
            N = np.maximum(1.0, self.believed_H['N'])
            obs[:, 0] = self.believed_H['S'] / N
            infection_frac = self.believed_H['I'] / N
            obs[:, 1] = np.log1p(infection_frac * 1000.0)
            obs[:, 2] = self.believed_H['R'] / N
            
            m_total_belief = self.believed_M['S'] + self.believed_M['E'] + self.believed_M['I']
            obs[:, 3] = m_total_belief / (N * 2.0)
            
            action_params = self.ep_params['action_semantics']
            obs[:, 4] = self.ep_params.get('alpha_benefit', 30000) / 30000.0
            obs[:, 5] = self.ep_params.get('delta_penalty_human_I', 20000) / 20000.0
            obs[:, 6] = action_params['case_management']['cost'] / 1000.0
            obs[:, 7] = action_params['vector_control']['cost'] / 2000.0
            obs[:, 8] = action_params['survey']['cost'] / 50.0
            obs[:, 9] = self.budget_remaining_for_episode / (self.total_episode_budget + 1e-6)
            
            week_frac = (self.calendar_week % 52) / 52.0
            obs[:, 10] = np.sin(2 * np.pi * week_frac)
            obs[:, 11] = np.cos(2 * np.pi * week_frac)
            obs[:, 12] = (self.T_HORIZON - (self.current_step - 1)) / self.MAX_T_HORIZON
            obs[:, 13] = action_params.get('larviciding', {}).get('cost', action_params['vector_control']['cost']) / 2000.0
            obs[:, 14] = np.log1p(self.lambda_hat * 1000.0)
            
        return obs