import pandas as pd
import numpy as np
from tqdm import tqdm
from cubesat_mission import *
from cubesat_simulator import SatelliteSimulator
from compute_score import compute_score
import logging
import os

# Output directory
os.makedirs("outputs", exist_ok=True)

# Constants
ORBIT_PERIOD_SEC = 5400  # Approximate period of LEO orbit in seconds (~90min)
SECONDS_4ORBITS = ORBIT_PERIOD_SEC * 4
SECONDS_8ORBITS = ORBIT_PERIOD_SEC * 8
SECONDS_15ORBITS = ORBIT_PERIOD_SEC * 15



def simulate_all_configs_partial():
    """
    Simulate CubeSat configurations and compute scores at 4, 8, and 15 orbits.
    """
    # Initialize mission (15 orbits)
    mission = MissionConfig(
        altitude=500e3,
        inclination=98.0,
        time_resolution=1.0,
        n_orbits=15
    )

    # Add multiple ground stations
    GS_coords = [(60.0, 0.0), (60.0, 72.0), (60.0, 144.0), (-60.0, -144.0), (-60.0, -72.0)]
    for lat, lon in GS_coords:
        mission.addGS(lat, lon)

    # Add points of interest
    POI_coords = [(30.0, 10.0), (-30.0, 100.0), (10.0, -50.0)]
    for lat, lon in POI_coords:
        mission.addPOI(lat, lon)

    # Precompute orbital data
    pos, vel, sun_flag, gs_flag, poi_flag = mission.compute()
    n = len(pos)
    time_vector = np.arange(n) * mission.dt

    # Load configurations
    df = pd.read_csv('cubesat_productline.csv')

    # Initialize outputs
    all_results = []

    # Loop over configurations
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx % 100 != 0:
            continue

        config = SatelliteConfig(row)
        sim = SatelliteSimulator(config, mission)
        sim.logger.setLevel(logging.ERROR)

        try:
            results = sim.simulate(pos, sun_flag, gs_flag, poi_flag)
        except Exception as e:
            print(f"Simulation failed for config {idx}: {e}")
            continue

        # Find indices corresponding to 4, 8, and 15 orbits
        idx_4orb = np.searchsorted(time_vector, SECONDS_4ORBITS)
        idx_8orb = np.searchsorted(time_vector, SECONDS_8ORBITS)
        idx_15orb = n  # full length

        # Partial datasets
        results_4 = {k: v[:idx_4orb] for k, v in results.items() if isinstance(v, np.ndarray)}
        results_8 = {k: v[:idx_8orb] for k, v in results.items() if isinstance(v, np.ndarray)}
        results_15 = results

        # Compute scores
        try:
            score_4, _ = compute_score(pos[:idx_4orb], results_4, config, sun_flag[:idx_4orb], gs_flag[:idx_4orb], poi_flag[:idx_4orb], mission)
            score_8, _ = compute_score(pos[:idx_8orb], results_8, config, sun_flag[:idx_8orb], gs_flag[:idx_8orb], poi_flag[:idx_8orb], mission)
            score_15, _ = compute_score(pos, results_15, config, sun_flag, gs_flag, poi_flag, mission)
        except Exception as e:
            print(f"Scoring failed for config {idx}: {e}")
            continue

        # Store
        output_row = row.copy()
        output_row['score_4orbits'] = score_4
        output_row['score_8orbits'] = score_8
        output_row['score_15orbits'] = score_15
        all_results.append(output_row)

    # Save final dataset
    final_df = pd.DataFrame(all_results)
    final_df.to_csv("outputs/satellite_config_scores_multi.csv", index=False)
    print("✅ Simulation complete. Results saved to outputs/satellite_config_scores_multi.csv")


if __name__ == "__main__":
    simulate_all_configs_partial()
