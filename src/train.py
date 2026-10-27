r"""Train the PPO agent with the three-stage curriculum.

    python src/train.py --seed 0 --graph data/dncc_graph.json \
        --archetype Dengue_Dhaka_Advanced --econ DNCC_Seasonal_Budget \
        --lambda_sigma 0.8 --rho 0.2 --regime estimated --history 5 \
        --horizon 20 --steps 4000000 --reward_mode weighted

For the 10-ward instance use --graph data/dncc10_graph.json and
--econ DNCC10_Seasonal_Budget. The published agents use seeds 0 to 3.

Options:
    --seed               Training seed.
    --rho                Persistence of the ward risk field, in [0, 1].
                         Default 0.2.
    --regime             Season-start prior. One of: known, estimated, flat.
                         Default estimated.
    --history            Past seasons behind an estimated prior. Default 5.
    --lambda_sigma       Risk-field dispersion. Default 0.8.
    --steps              Total timesteps. Default 2500000.
    --graph              Graph file, e.g. data/dncc_graph.json.
    --archetype          Parameter set for stage 3. Default
                         Dengue_Dhaka_Advanced.
    --econ               Economic scenario for stage 3, e.g.
                         DNCC_Seasonal_Budget.
    --reward_mode        Training objective. One of: weighted, constrained.
                         Default weighted.
    --overspend_penalty  Overspend penalty under --reward_mode constrained.
                         Default 10.0.
    --horizon            Episode length in weeks. Default 20.
"""

import os
import sys
import torch
import re
import time
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


def parse_args():
    p = argparse.ArgumentParser(
        description='Train the PPO agent with the three-stage curriculum.')
    p.add_argument("--seed", type=int, required=True,
                   help='Training seed.')
    p.add_argument("--rho", type=float, default=0.2,
                   help='Persistence of the ward risk field, in [0, 1].')
    p.add_argument("--regime", type=str, default="estimated",
                   choices=["known", "estimated", "flat"],
                   help='Season-start prior.')
    p.add_argument("--history", type=int, default=5,
                   help='Past seasons behind an estimated prior.')
    p.add_argument("--lambda_sigma", type=float, default=0.8,
                   help='Risk-field dispersion.')
    p.add_argument("--steps", type=int, default=2_500_000,
                   help='Total timesteps.')
    p.add_argument("--graph", type=str, default=None,
                   help='Graph file, e.g. data/dncc_graph.json.')
    p.add_argument("--archetype", type=str, default="Dengue_Dhaka_Advanced",
                   help='Parameter set for stage 3.')
    p.add_argument("--econ", type=str, default=None,
                   help='Economic scenario for stage 3, e.g. DNCC_Seasonal_Budget.')
    p.add_argument("--reward_mode", type=str, default="weighted",
                   choices=["weighted", "constrained"],
                   help='Training objective.')
    p.add_argument("--overspend_penalty", type=float, default=10.0,
                   help='Overspend penalty under --reward_mode constrained.')
    p.add_argument("--horizon", type=int, default=20,
                   help='Episode length in weeks.')
    return p.parse_args()

PROJECT_ROOT = os.getcwd() 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.env.smip_env import SMIPEnv
from src.common.policy import GNNPolicy, GNNFeaturesExtractor

def find_latest_checkpoint_and_steps(folder_path: str) -> (str or None, int):
    if not os.path.isdir(folder_path): return None, 0
    checkpoints = [f for f in os.listdir(folder_path) if f.startswith("checkpoint_") and f.endswith(".zip")]
    if not checkpoints: return None, 0
    def get_steps(filename):
        match = re.search(r'_(\d+)_steps.zip', filename)
        return int(match.group(1)) if match else 0
    latest_ckpt_file = max(checkpoints, key=get_steps)
    return os.path.join(folder_path, latest_ckpt_file), get_steps(latest_ckpt_file)

def find_latest_run_folder(base_models_dir: str, run_prefix: str) -> str or None:
    if not os.path.isdir(base_models_dir): return None
    run_folders = [d for d in os.listdir(base_models_dir) if d.startswith(run_prefix) and os.path.isdir(os.path.join(base_models_dir, d))]
    if not run_folders: return None
    return max([os.path.join(base_models_dir, d) for d in run_folders], key=os.path.getmtime)

def main():
    args = parse_args()
    SPECIES_GROUP = "Urban Public Health"
    RUN_PREFIX_BASE = ("DENGUE_DHAKA_V19C" if args.reward_mode == "constrained"
                       else "DENGUE_DHAKA_V19")
    RESUME_FROM_BC = False

    TRAIN_SEED = args.seed
    TOTAL_TIMESTEPS = args.steps
    RHO = args.rho
    PRIOR_REGIME = args.regime
    PRIOR_HISTORY = args.history
    LAMBDA_SIGMA = args.lambda_sigma
    STAGE3_ARCHETYPE = args.archetype
    GRAPH_OVERRIDE = args.graph
    ECON_OVERRIDE = args.econ
    HORIZON_WEEKS = args.horizon
    REWARD_MODE = args.reward_mode
    OVERSPEND_PENALTY = args.overspend_penalty

    if not os.environ.get("SKIP_PREFLIGHT"):
        import subprocess
        _pf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflight.py")
        if os.path.exists(_pf):
            _cmd = [sys.executable, _pf, "--graph", GRAPH_OVERRIDE or "data/dncc_graph.json",
                    "--archetype", STAGE3_ARCHETYPE, "--lambda_sigma", str(LAMBDA_SIGMA), "--horizon", str(HORIZON_WEEKS)]
            if ECON_OVERRIDE:
                _cmd += ["--econ", ECON_OVERRIDE]
            print("Running preflight gate:", " ".join(_cmd))
            if subprocess.call(_cmd) != 0:
                print("\nPREFLIGHT FAILED -- training aborted. "
                      "Fix the failures, or set SKIP_PREFLIGHT=1 to override.")
                sys.exit(1)

    _graph_tag = ""
    if GRAPH_OVERRIDE:
        _stem = os.path.splitext(os.path.basename(GRAPH_OVERRIDE))[0]
        _graph_tag = f"_{_stem}"
    RUN_PREFIX = f"{RUN_PREFIX_BASE}{_graph_tag}_{PRIOR_REGIME}_rho{int(round(RHO*100)):03d}_S{TRAIN_SEED}"
    print(f"\n=== RUN CONFIG ===\n  seed={TRAIN_SEED}  rho={RHO}  regime={PRIOR_REGIME}"
          f"  history={PRIOR_HISTORY}  lambda_sigma={LAMBDA_SIGMA}"
          f"\n  archetype={STAGE3_ARCHETYPE}  graph={GRAPH_OVERRIDE or 'default 130-ward'}"
          f"  econ={ECON_OVERRIDE or 'archetype default'}"
          f"\n  run_prefix={RUN_PREFIX}\n")
    STAGE_1_STEPS = 150_000
    STAGE_2_STEPS = 300_000
    
    N_ENVS = 8
    SAVE_FREQ = 100_000
    
    TUTORIAL_SCENARIOS = ["Balanced_Medium_Budget"]
    STANDARD_SCENARIOS = None
    if ECON_OVERRIDE is None:
        print("ERROR: --econ is required. Use DNCC_Seasonal_Budget (54 wards) "
              "or DNCC10_Seasonal_Budget (10 wards).")
        sys.exit(1)

    if os.environ.get("SKIP_PREFLIGHT"):
        print("WARNING: SKIP_PREFLIGHT set -- epidemic preflight NOT run.")
    else:
        from src.preflight_epidemic import run_epidemic_preflight

        class _EpiArgs:
            pass
        _ea = _EpiArgs()
        _ea.graph = args.graph
        _ea.econ = ECON_OVERRIDE
        _ea.archetype_file = getattr(args, "archetype_file",
                                     os.path.join("configs", "archetypes.json"))
        _ea.archetype = args.archetype
        _ea.lambda_sigma = args.lambda_sigma
        _ea.rho = getattr(args, "rho", 0.2)
        _ea.regime = getattr(args, "regime", "estimated")
        _ea.history = getattr(args, "history", 5)
        _ea.horizon = getattr(args, "horizon", 20)
        _ea.substeps = 1
        _ea.seeds = [0, 3, 7, 11, 14]
        if not run_epidemic_preflight(_ea):
            print("\nEpidemic preflight failed. Not training.")
            sys.exit(1)
    
    archetype_filepath = os.path.join(PROJECT_ROOT, 'configs', 'archetypes.json')
    if GRAPH_OVERRIDE:
        graph_filepath = GRAPH_OVERRIDE if os.path.isabs(GRAPH_OVERRIDE) \
            else os.path.join(PROJECT_ROOT, GRAPH_OVERRIDE)
        if not os.path.exists(graph_filepath):
            print(f"ERROR: --graph '{graph_filepath}' not found."); sys.exit(1)
    else:
        graph_filepath = os.path.join(PROJECT_ROOT, "dhaka_graph_populated.json")

    is_colab = 'COLAB_GPU' in os.environ
    base_dir = '/content/drive/MyDrive/DRL_POMDP_DENGUE' if is_colab else PROJECT_ROOT
    base_models_dir = os.path.join(base_dir, "rl_models")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(base_models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    latest_run_path = find_latest_run_folder(base_models_dir, RUN_PREFIX)
    steps_already_trained = 0
    if latest_run_path:
        model_dir = latest_run_path
        run_name = os.path.basename(model_dir)
        latest_checkpoint_path, steps_already_trained = find_latest_checkpoint_and_steps(model_dir)
    else:
        run_name = f"{RUN_PREFIX}_{int(time.time())}"
        model_dir = os.path.join(base_models_dir, run_name)
        os.makedirs(model_dir, exist_ok=True)
        latest_checkpoint_path = None

    base_env_kwargs = {
        'archetype_filepath': archetype_filepath, 
        'species_group': SPECIES_GROUP, 
        'rho': RHO,
        'prior_regime': PRIOR_REGIME,
        'prior_history': PRIOR_HISTORY,
        'lambda_sigma': LAMBDA_SIGMA,
    }
    if GRAPH_OVERRIDE:
        base_env_kwargs['graph_filepath_override'] = graph_filepath

    #            It learns the physical control levers (fog + larvicide the
    #            infected wards) without the information confound.
    #            must learn to balance paying for info vs spending on control.
    if steps_already_trained < STAGE_1_STEPS:
        print("\n--- CURRICULUM STAGE 1: CONTROL TUTORIAL (perfect free info) ---")
        print("Agent sees true state for free; learns to fog + larvicide hotspots.")
        env_kwargs = base_env_kwargs.copy()
        env_kwargs.update({
            'species_name': 'Dengue_Dhaka_Base',
            'economic_scenarios': TUTORIAL_SCENARIOS,
            'force_T_HORIZON': HORIZON_WEEKS,
            'reward_mode': REWARD_MODE, 'overspend_penalty': OVERSPEND_PENALTY,
            'force_start_week': None,
            'force_perfect_surveys': True,
            'force_free_info': True
        })
        current_stage_target = STAGE_1_STEPS

    elif steps_already_trained < STAGE_2_STEPS:
        print("\n--- CURRICULUM STAGE 2: INFORMATION HAS A COST ---")
        print("Free info OFF, surveys now cost budget. Agent learns info/action balance.")
        env_kwargs = base_env_kwargs.copy()
        env_kwargs.update({
            'species_name': 'Dengue_Dhaka_Base',
            'economic_scenarios': TUTORIAL_SCENARIOS,
            'force_T_HORIZON': HORIZON_WEEKS,
            'reward_mode': REWARD_MODE, 'overspend_penalty': OVERSPEND_PENALTY,
            'force_start_week': None,
            'force_perfect_surveys': True,
            'force_free_info': False
        })
        current_stage_target = STAGE_2_STEPS

    else:
        print(f"\n--- CURRICULUM STAGE 3: STANDARD POMDP (full stochastic biology) ---")
        print(f"Activating '{STAGE3_ARCHETYPE}' profile with imperfect surveys!")
        env_kwargs = base_env_kwargs.copy()
        stage3_scenarios = [ECON_OVERRIDE] if ECON_OVERRIDE else STANDARD_SCENARIOS
        env_kwargs.update({
            'species_name': STAGE3_ARCHETYPE,
            'economic_scenarios': stage3_scenarios,
            'force_T_HORIZON': HORIZON_WEEKS,
            'reward_mode': REWARD_MODE, 'overspend_penalty': OVERSPEND_PENALTY,
            'force_start_week': None,
            'force_perfect_surveys': False,
            'force_free_info': False
        })
        current_stage_target = TOTAL_TIMESTEPS

    env = make_vec_env(SMIPEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs, seed=TRAIN_SEED)
    
    policy_kwargs = {
        'features_extractor_class': GNNFeaturesExtractor, 
        'features_extractor_kwargs': dict(features_dim=64, graph_filepath=graph_filepath)
    }

    if latest_checkpoint_path:
        print(f"\nResuming training from {latest_checkpoint_path}")
        model = PPO.load(latest_checkpoint_path, env=env, tensorboard_log=logs_dir)
    else:
        bc_ckpt = os.path.join(base_models_dir, "bc_warmstart.zip")
        if RESUME_FROM_BC and os.path.exists(bc_ckpt):
            print(f"\nWARM-START: loading BC greedy-imitation policy from {bc_ckpt}")
            print("Fine-tuning with FREE RL from the greedy basin (agent may exceed greedy).")
            model = PPO.load(bc_ckpt, env=env, tensorboard_log=logs_dir)
            model.num_timesteps = 0
        else:
            print("\nStarting new training run with STABILIZED PPO.")
            model = PPO(GNNPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logs_dir, seed=TRAIN_SEED,
                    n_steps=2048, batch_size=128, n_epochs=5, gamma=0.99, learning_rate=5e-5, 
                    ent_coef=0.01, target_kl=0.07, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_callback = CheckpointCallback(save_freq=max(SAVE_FREQ // N_ENVS, 1), save_path=model_dir, name_prefix="checkpoint")
    
    try:
        while model.num_timesteps < TOTAL_TIMESTEPS:
            steps_to_train = current_stage_target - model.num_timesteps
            
            model.learn(total_timesteps=steps_to_train, callback=checkpoint_callback, tb_log_name=run_name, reset_num_timesteps=False, progress_bar=True)
            
            if model.get_env() is not None:
                model.get_env().close()
            
            if model.num_timesteps >= STAGE_2_STEPS:
                print("\n--- TRANSITIONING TO STAGE 3 (FULL POMDP & ADVANCED BIOLOGY) ---")
                env_kwargs_next = base_env_kwargs.copy()
                env_kwargs_next.update({'species_name': STAGE3_ARCHETYPE, 'economic_scenarios': ([ECON_OVERRIDE] if ECON_OVERRIDE else STANDARD_SCENARIOS), 'force_T_HORIZON': HORIZON_WEEKS,
            'reward_mode': REWARD_MODE, 'overspend_penalty': OVERSPEND_PENALTY, 'force_start_week': None, 'force_perfect_surveys': False})
                current_stage_target = TOTAL_TIMESTEPS
                
            elif model.num_timesteps >= STAGE_1_STEPS:
                print("\n--- TRANSITIONING TO STAGE 2 (STRATEGIC TUTORIAL) ---")
                env_kwargs_next = base_env_kwargs.copy()
                env_kwargs_next.update({'species_name': 'Dengue_Dhaka_Base', 'economic_scenarios': TUTORIAL_SCENARIOS, 'force_T_HORIZON': HORIZON_WEEKS,
            'reward_mode': REWARD_MODE, 'overspend_penalty': OVERSPEND_PENALTY, 'force_start_week': None, 'force_perfect_surveys': True, 'force_free_info': False})
                current_stage_target = STAGE_2_STEPS
                
            stage_seed = TRAIN_SEED * 1_000_000 + int(model.num_timesteps)
            next_env = make_vec_env(SMIPEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv,
                                    env_kwargs=env_kwargs_next, seed=stage_seed)
            model.set_env(next_env)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("\n--- Training Complete ---")
        final_model_path = os.path.join(model_dir, f"final_model_{model.num_timesteps // 1000}k.zip")
        print(f"Saving final model to {final_model_path}")
        model.save(final_model_path)
        
        if model.get_env() is not None:
            model.get_env().close()

if __name__ == '__main__':
    main()