"""Ward-level vector-host dynamics: one weekly step, integrated in daily
sub-steps.
"""

import numpy as np

_ADJ_CACHE = {}


def _normalised_adjacency(neighbors, num_sites):
    key = (id(neighbors), num_sites)
    A = _ADJ_CACHE.get(key)
    if A is None:
        A = np.zeros((num_sites, num_sites), dtype=np.float64)
        for i in range(num_sites):
            nb = neighbors.get(i, [])
            if nb:
                A[i, list(nb)] = 1.0 / len(nb)
        _ADJ_CACHE[key] = A
    return A


def calculate_seasonality_multiplier(current_week, season_params):
    baseline = season_params.get("baseline", 0.2)
    amplitude = season_params.get("amplitude", 0.8)
    peak_week = season_params.get("peak_week", 28)
    wave = 0.5 * (1.0 + np.cos(2.0 * np.pi * (current_week - peak_week) / 52.0))
    return baseline + (amplitude * wave)


def run_dynamics_step(human_state, mosq_state, actions, current_week, rng, params, neighbors):
    num_sites = len(human_state['N'])
    substeps = int(params.get('dynamics_substeps', 7))
    substeps = max(1, substeps)
    dt = 1.0 / substeps

    h_N = human_state['N']
    h_S = np.asarray(human_state['S'], dtype=np.float64).copy()
    h_I = np.asarray(human_state['I'], dtype=np.float64).copy()
    h_R = np.asarray(human_state['R'], dtype=np.float64).copy()

    m_S = np.asarray(mosq_state['S'], dtype=np.float64).copy()
    m_E = np.asarray(mosq_state['E'], dtype=np.float64).copy()
    m_I = np.asarray(mosq_state['I'], dtype=np.float64).copy()

    mosq_params = params['mosquito_model']
    human_params = params['human_model']
    spread_params = params['spatial_spread']
    action_params = params['action_semantics']

    vc_coverage = actions.get('vector_control', np.zeros(num_sites))
    vc_efficacy = action_params['vector_control']['efficacy']
    survival_from_fogging = np.maximum(0.0, 1.0 - (vc_coverage * vc_efficacy))
    m_S *= survival_from_fogging
    m_E *= survival_from_fogging
    m_I *= survival_from_fogging

    cm_coverage = actions.get('case_management', np.zeros(num_sites))
    cm_efficacy = action_params['case_management']['efficacy']
    isolation_factor = np.maximum(0.0, 1.0 - (cm_coverage * cm_efficacy))

    season_mult = calculate_seasonality_multiplier(current_week, params['seasonality'])
    carrying_capacity_multiplier = mosq_params.get('carrying_capacity_multiplier', 2.0)
    m_K = h_N * carrying_capacity_multiplier * season_mult

    larv_coverage = actions.get('larviciding', np.zeros(num_sites))
    larv_efficacy = action_params.get('larviciding', {}).get('efficacy', 0.7)
    capacity_retained = np.maximum(0.0, 1.0 - (larv_coverage * larv_efficacy))
    m_K = m_K * capacity_retained

    birth_rate      = mosq_params.get('mosquito_birth_rate', 1.2)
    biting_rate     = mosq_params['biting_rate_per_week']
    trans_h2m       = mosq_params['transmission_prob_human_to_mosquito']
    trans_m2h       = mosq_params['transmission_prob_mosquito_to_human']
    incubation_rate = mosq_params['extrinsic_incubation_rate']
    mortality_rate  = mosq_params['mosquito_mortality_rate']
    recovery_rate   = human_params['recovery_rate']

    p_incub  = min(1.0, incubation_rate * dt)
    p_mort   = min(1.0, mortality_rate * dt)
    p_recov  = min(1.0, recovery_rate * dt)
    bite_sub = biting_rate * dt
    birth_sub = birth_rate * dt

    rho_m = spread_params['mosquito_dispersal_rate']
    rho_h = spread_params['human_mobility_network_rate']
    A = _normalised_adjacency(neighbors, num_sites)
    hN_safe = np.maximum(1.0, h_N)

    for _ in range(substeps):
        h_I_effective = h_I * isolation_factor

        m_I_pressure = (1.0 - rho_m) * m_I + rho_m * (A @ m_I)
        h_I_pressure = (1.0 - rho_h) * h_I_effective + rho_h * (A @ h_I_effective)

        total_m = m_S + m_E + m_I
        m_births = np.maximum(0.0, birth_sub * total_m *
                              (1.0 - (total_m / np.maximum(1.0, m_K))))

        prob_bite_infectious = np.minimum(1.0, h_I_pressure / hN_safe)
        foi_m = 1.0 - np.exp(-bite_sub * trans_h2m * prob_bite_infectious)
        new_exposed_m = rng.binomial(np.round(m_S).astype(int),
                                     np.minimum(1.0, foi_m))
        new_infectious_m = rng.binomial(np.round(m_E).astype(int), p_incub)

        m_S = m_S + m_births - new_exposed_m - (m_S * p_mort)
        m_E = m_E + new_exposed_m - new_infectious_m - (m_E * p_mort)
        m_I = m_I + new_infectious_m - (m_I * p_mort)
        m_S = np.maximum(0.0, m_S); m_E = np.maximum(0.0, m_E); m_I = np.maximum(0.0, m_I)

        bites_per_human = (bite_sub * m_I_pressure) / hN_safe
        foi_h = 1.0 - np.exp(-trans_m2h * bites_per_human)
        new_infected_h = rng.binomial(np.round(h_S).astype(int),
                                      np.minimum(1.0, foi_h))
        new_recovered_h = rng.binomial(np.round(h_I).astype(int), p_recov)

        h_S = np.maximum(0.0, h_S - new_infected_h)
        h_I = np.maximum(0.0, h_I + new_infected_h - new_recovered_h)
        h_R = np.maximum(0.0, h_R + new_recovered_h)

    total_h_after = h_S + h_I + h_R
    discrepancy = np.round(h_N) - total_h_after
    h_S = np.maximum(0.0, h_S + discrepancy)

    human_state_next = {'N': np.asarray(h_N, dtype=np.float32),
                        'S': h_S.astype(np.float32),
                        'I': h_I.astype(np.float32),
                        'R': h_R.astype(np.float32)}
    mosq_state_next = {'S': np.maximum(0.0, m_S).astype(np.float32),
                       'E': np.maximum(0.0, m_E).astype(np.float32),
                       'I': np.maximum(0.0, m_I).astype(np.float32)}
    return human_state_next, mosq_state_next
