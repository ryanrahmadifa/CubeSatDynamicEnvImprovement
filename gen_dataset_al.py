import pandas as pd
import numpy as np
from tqdm import tqdm
from cubesat_mission import MissionConfig, SatelliteConfig
from cubesat_simulator import SatelliteSimulator
from compute_score import compute_score
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Benchmark-AL-Mat', 'src'))
from strategies.treebased_representativity import TreeBasedRegressor_Representativity

os.makedirs("outputs", exist_ok=True)

# ==== SETTINGS ====
INIT_SIZE    = 35    # random seed set for first iteration
BATCH_SIZE   = 35    # configs selected per AL iteration
TOTAL_BUDGET = 3500  # total simulations budget
RANDOM_SEED  = 42
AL_POOL_SIZE = 5000  # subsample of unlabeled pool passed to AL strategy (speed)
# ==================

ORBIT_PERIOD_SEC  = 5400
SECONDS_4ORBITS   = ORBIT_PERIOD_SEC * 4
SECONDS_8ORBITS   = ORBIT_PERIOD_SEC * 8


def setup_mission():
    mission = MissionConfig(altitude=500e3, inclination=98.0, time_resolution=1.0, n_orbits=15)
    for lat, lon in [(60.0, 0.0), (60.0, 72.0), (60.0, 144.0), (-60.0, -144.0), (-60.0, -72.0)]:
        mission.addGS(lat, lon)
    for lat, lon in [(30.0, 10.0), (-30.0, 100.0), (10.0, -50.0)]:
        mission.addPOI(lat, lon)
    return mission


def simulate_config(idx, row, mission, pos, vel, sun_flag, gs_flag, poi_flag, time_vector):
    config = SatelliteConfig(row)
    sim = SatelliteSimulator(config, mission)
    sim.logger.setLevel(logging.ERROR)

    results = sim.simulate(pos, sun_flag, gs_flag, poi_flag)

    idx_4orb  = np.searchsorted(time_vector, SECONDS_4ORBITS)
    idx_8orb  = np.searchsorted(time_vector, SECONDS_8ORBITS)

    results_4  = {k: v[:idx_4orb] for k, v in results.items() if isinstance(v, np.ndarray)}
    results_8  = {k: v[:idx_8orb] for k, v in results.items() if isinstance(v, np.ndarray)}
    results_15 = results

    score_4,  subscores_4  = compute_score(pos[:idx_4orb], results_4,  config, sun_flag[:idx_4orb], gs_flag[:idx_4orb], poi_flag[:idx_4orb], mission)
    score_8,  subscores_8  = compute_score(pos[:idx_8orb], results_8,  config, sun_flag[:idx_8orb], gs_flag[:idx_8orb], poi_flag[:idx_8orb], mission)
    score_15, subscores_15 = compute_score(pos,            results_15, config, sun_flag,            gs_flag,            poi_flag,            mission)

    output_row = row.copy()
    output_row['score_4orbits']  = score_4
    output_row['score_8orbits']  = score_8
    output_row['score_15orbits'] = score_15

    for key, val in subscores_4.items():
        output_row[f'subscore_4_{key}']  = val
    for key, val in subscores_8.items():
        output_row[f'subscore_8_{key}']  = val
    for key, val in subscores_15.items():
        output_row[f'subscore_15_{key}'] = val

    return output_row


def get_feature_columns(df):
    exclude = [c for c in df.columns if c.startswith('score_') or c.startswith('subscore_')]
    feature_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def main():
    print("Loading product line...")
    df_pool = pd.read_csv('cubesat_productline.csv', index_col=0)
    df_pool.columns = df_pool.columns.str.strip()
    print(f"Total configurations: {len(df_pool)}")

    print("Setting up mission and precomputing orbital geometry...")
    mission    = setup_mission()
    pos, vel, sun_flag, gs_flag, poi_flag = mission.compute()
    time_vector = np.arange(len(pos)) * mission.dt

    feature_cols = df_pool.select_dtypes(include=[np.number]).columns.tolist()

    # ==== STEP 1: Random seed set ====
    np.random.seed(RANDOM_SEED)
    init_indices = np.random.choice(df_pool.index, size=INIT_SIZE, replace=False).tolist()

    all_results   = []
    labeled_indices = []
    labeled_scores  = []

    print(f"\n--- Initial random simulation ({INIT_SIZE} configs) ---")
    for idx in tqdm(init_indices):
        row = df_pool.loc[idx]
        try:
            output_row = simulate_config(idx, row, mission, pos, vel, sun_flag, gs_flag, poi_flag, time_vector)
            all_results.append(output_row)
            labeled_indices.append(idx)
            labeled_scores.append(output_row['score_8orbits'])
        except Exception as e:
            print(f"[ERROR] Config {idx}: {e}")

    simulated_budget = len(labeled_indices)

    # ==== STEP 2: Active learning iterations ====
    strategy = TreeBasedRegressor_Representativity(random_state=RANDOM_SEED, min_samples_leaf=5)

    iteration = 0
    while simulated_budget < TOTAL_BUDGET:
        iteration += 1
        remaining = TOTAL_BUDGET - simulated_budget
        batch = min(BATCH_SIZE, remaining)

        print(f"\n--- AL Iteration {iteration} | Simulated so far: {simulated_budget} | Querying: {batch} ---")

        # Build labeled/unlabeled DataFrames for the strategy
        unlabeled_indices = [i for i in df_pool.index if i not in set(labeled_indices)]

        # Subsample unlabeled pool for AL to keep KMeans/distance loops tractable
        rng = np.random.default_rng(RANDOM_SEED + iteration)
        if len(unlabeled_indices) > AL_POOL_SIZE:
            pool_indices = rng.choice(unlabeled_indices, size=AL_POOL_SIZE, replace=False).tolist()
        else:
            pool_indices = unlabeled_indices

        X_labeled   = df_pool.loc[labeled_indices, feature_cols].reset_index(drop=True)
        X_unlabeled = df_pool.loc[pool_indices,    feature_cols].reset_index(drop=True)
        X_unlabeled.index = pool_indices  # preserve original indices for selection

        y_labeled   = pd.DataFrame({'score': labeled_scores})
        y_unlabeled = None

        try:
            selected = strategy.query(
                X_labeled, X_unlabeled,
                y_labeled, y_unlabeled,
                n_act=batch
            )
        except Exception as e:
            print(f"[AL ERROR] Strategy failed: {e}. Falling back to random selection.")
            selected = np.random.choice(unlabeled_indices, size=batch, replace=False).tolist()

        # Simulate selected configs
        for idx in tqdm(selected):
            row = df_pool.loc[idx]
            try:
                output_row = simulate_config(idx, row, mission, pos, vel, sun_flag, gs_flag, poi_flag, time_vector)
                all_results.append(output_row)
                labeled_indices.append(idx)
                labeled_scores.append(output_row['score_8orbits'])
                simulated_budget += 1
            except Exception as e:
                print(f"[ERROR] Config {idx}: {e}")

        # Checkpoint save every iteration
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_csv("outputs/satellite_config_scores_al_8orbits_checkpoint.csv", index=False)
        print(f"Checkpoint saved. Total simulated: {simulated_budget}")

    # ==== FINAL SAVE ====
    final_df = pd.DataFrame(all_results)
    final_df['al_order'] = range(len(final_df))  # row 0..34 = random seed, rest = AL-selected
    final_df.to_csv("outputs/satellite_config_scores_al_8orbit.csv", index=False)
    print(f"\nDone. {simulated_budget} configs simulated. Saved to outputs/satellite_config_scores_al_8orbits.csv")


if __name__ == "__main__":
    main()
