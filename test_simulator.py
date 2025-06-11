import numpy as np
import pandas as pd
import pytest

from cubesat_simulator import SatelliteConfig, SatelliteSimulator

class DummyMission:
    def __init__(self, dt):
        self.dt = dt

# Helper to create a default configuration row
@pytest.fixture(scope="module")
def default_row():
    # Minimal realistic values for all required config fields
    data = {
        # Communication
        'Communication_Communication band': 'X',
        'Communication_Frequency [GHz]': 8.0,
        'Communication_Beamwidth [deg]': 20.0,
        'Communication_D_r [m]': 500.0,
        'Communication_Required Eb/N0 [dB]': 10.0,
        'Communication_Efficiency': 0.5,
        'Communication_D_antenna [m]': 0.3,
        'Communication_Mass_communication [kg]': 2.0,
        'Communication_Data Rate [bps]': 1e6,
        # Solar Array
        'SolarArray_Solar Material': 'Si',
        'SolarArray_Efficiency': 0.3,
        'SolarArray_Specific Power [W/kg]': 150.0,
        'SolarArray_A_solar [m^2]': 0.5,
        'SolarArray_P_solar_max [W]': 200.0,
        'SolarArray_Mass_solar_array [kg]': 1.0,
        # Battery
        'Battery_Battery type': 'LiPo',
        'Battery_Number of cell': 4,
        'Battery_Diameter [m]': 0.05,
        'Battery_Length [m]': 0.1,
        'Battery_Volumetric Energy [Wh/m³]': 300.0,
        'Battery_Specific Energy [Wh/kg]': 200.0,
        'Battery_Capacity [Wh]': 50.0,
        'Battery_Mass_battery [kg]': 0.5,
        # Radiator
        'Radiator_Radiator material': 'Al',
        'Radiator_A_radiator [m2]': 0.2,
        'Radiator_rho [kg/m³]': 2700.0,
        'Radiator_emissivity': 0.85,
        'Radiator_thickness [m]': 0.005,
        'Radiator_Q_rad [W/K⁴]': 5.67e-8,
        'Radiator_Mass_radiator [kg]': 0.3,
        # OBC
        'OBC_cpu_DMIPS': 100.0,
        'OBC_cpu_power_idle': 1.0,
        'OBC_power max': 5.0,
        'OBC_Throughtput per W': 50.0,
        'OBC_cpu efficiency': 1e6,  # bytes per DMIPS per s
        'OBC_recovery rate': 5.0,
        # CubeSat
        'CubeSat mass': 5.0
    }
    return pd.Series(data)

@pytest.fixture(scope="module")
def simulator(default_row):
    config = SatelliteConfig(default_row)
    mission = DummyMission(dt=1)
    return SatelliteSimulator(config, mission)

# Basic output structure and lengths
def test_simulator_outputs_length(simulator):
    n = 20
    pos = np.zeros((n, 3))  # positions are unused internally
    sun_flag = np.ones(n)
    gs_flag = np.zeros(n)
    poi_flag = np.zeros(n)
    results = simulator.simulate(pos, sun_flag, gs_flag, poi_flag)
    # All expected keys
    expected_keys = {'state_of_charge', 'temperature', 'cpu_load',
                     'raw_buffer', 'processed_buffer', 'downlinked'}
    assert set(results.keys()) == expected_keys
    # Each array has length n
    for arr in results.values():
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == n

# SoC behavior: should drop during eclipse
def test_soc_decreases_in_eclipse(simulator):
    n = 10
    pos = [None] * n
    # first half in sunlight, second half eclipse
    sun_flag = np.array([1]*5 + [0]*5)
    gs_flag = np.zeros(n)
    poi_flag = np.zeros(n)
    results = simulator.simulate(pos, sun_flag, gs_flag, poi_flag)
    soc = results['state_of_charge']
    # SoC starts at 100, then decreases once eclipse begins
    assert soc[0] == pytest.approx(100.0)
    assert soc[6] < soc[4]

# Buffer accumulation and processing
def test_buffer_and_cpu_load(simulator):
    n = 15
    pos = [None] * n
    sun_flag = np.ones(n)
    gs_flag = np.zeros(n)
    poi_flag = np.zeros(n)
    # Trigger POI events at steps 1, 5, 9
    poi_flag[[1,5,9]] = 1
    results = simulator.simulate(pos, sun_flag, gs_flag, poi_flag)
    raw = results['raw_buffer']
    proc = results['processed_buffer']
    cpu = results['cpu_load']
    # Processed buffer should increase at POI steps
    assert proc[1] > proc[0], "Processed buffer did not increase at first POI"
    assert proc[5] > proc[4], "Processed buffer did not increase at second POI"
    assert proc[9] > proc[8], "Processed buffer did not increase at third POI"
    # Raw buffer should be drained by processing at POI steps
    assert raw[1] == pytest.approx(0.0), "Raw buffer not fully drained at first POI"
    assert raw[5] == pytest.approx(0.0), "Raw buffer not fully drained at second POI"
    assert raw[9] == pytest.approx(0.0), "Raw buffer not fully drained at third POI"
    # CPU load should rise after POI events
    assert cpu.max() > 0, "CPU load never increased"

# Downlink behavior: only when GS flag is 1
# Also ensure downlinked never exceeds downlink_rate * dt
def test_downlink_with_gs(simulator, default_row):
    n = 12
    pos = [None] * n
    sun_flag = np.ones(n)
    poi_flag = np.ones(n)
    # GS available only second half
    gs_flag = np.array([0]*6 + [1]*6)
    results = simulator.simulate(pos, sun_flag, gs_flag, poi_flag)
    down = results['downlinked']
    # No downlink in first half
    assert np.all(down[:6] == 0)
    # Some downlink in second half
    assert np.any(down[6:] > 0)
    # Check downlink rate limit
    config = SatelliteConfig(default_row)
    dt = simulator.mission.dt
    max_rate = config.downlink_rate * dt
    assert np.all(down <= max_rate), "Downlink exceeds configured max rate"
    
# Temperature evolution: non-constant under internal heat
def test_temperature_evolution(simulator):
    n = 30
    pos = [None] * n
    # allow some power generation & draws
    sun_flag = np.ones(n)
    gs_flag = np.zeros(n)
    poi_flag = np.zeros(n)
    results = simulator.simulate(pos, sun_flag, gs_flag, poi_flag)
    temp = results['temperature']
    # Temperature should deviate from initial
    assert not np.allclose(temp, temp[0])
    # Should remain finite and reasonable
    assert np.all(temp > 0)


# Invariant checks: bounds and monotonic properties
# (existing tests...)

# Transient thermal response: eclipse-induced temperature dip and recovery
def test_thermal_transient(simulator):
    n = 20
    pos = [None] * n
    # Sun for 8 steps, eclipse 4 steps, then sun for 8 steps
    sun_flag = np.array([1]*8 + [0]*4 + [1]*8)
    gs_flag = np.zeros(n)
    poi_flag = np.zeros(n)
    results = simulator.simulate(pos, sun_flag, gs_flag, poi_flag)
    temp = results['temperature']
    # Expect temperature decreasing during eclipse
    assert temp[7] > temp[9], "Temperature did not drop during eclipse"
    # Expect temperature recovering after eclipse
    assert temp[11] < temp[13], "Temperature did not recover after eclipse"

# State-of-charge safety: no complete depletion under nominal conditions
def test_soc_no_depletion(simulator):
    n = 50
    pos = [None] * n
    # Alternate sunlight and eclipse
    sun_flag = np.tile([1,0], n//2)
    gs_flag = np.zeros(n)
    poi_flag = np.zeros(n)
    results = simulator.simulate(pos, sun_flag, gs_flag, poi_flag)
    soc = results['state_of_charge']
    # Ensure SOC stays above zero
    assert np.min(soc) > 0.0, "State-of-charge depleted to zero or below"

# CPU load smoothing: monotonic ramps
def test_cpu_smoothing(simulator):
    n = 30
    pos = [None] * n
    # Trigger POI for first 10 steps, then none
    sun_flag = np.ones(n)
    gs_flag = np.zeros(n)
    poi_flag = np.array([1]*10 + [0]*(n-10))
    results = simulator.simulate(pos, sun_flag, gs_flag, poi_flag)
    cpu = results['cpu_load']
    # During POI, CPU should non-decreasing
    assert np.all(np.diff(cpu[:10]) >= 0), "CPU load not monotonically increasing during data burst"
    # After POI, CPU should non-increasing
    assert np.all(np.diff(cpu[10:]) <= 0), "CPU load not monotonically decreasing after data burst"

def test_buffer_caps_at_capacity(default_config, mission, pos, sun_flag, gs_flag):
    cfg = default_config()
    cfg.buffer_capacity = 1e6  # 1 MB to force cap quickly
    sim = SatelliteSimulator(cfg, mission)
    results = sim.simulate(pos, sun_flag, gs_flag, poi_flag=np.ones(len(pos)))
    raw = results['raw_buffer']
    # raw_buffer should never exceed capacity
    assert np.all(raw <= cfg.buffer_capacity + 1e-6), \
        "raw_buffer exceeded buffer_capacity"
    # And once at capacity, it should hold until processing drains it
    cap_idx = np.argmax(raw >= cfg.buffer_capacity - 1e-6)
    assert raw[cap_idx] == pytest.approx(cfg.buffer_capacity, rel=1e-6)
    # After some processing steps, raw_buffer should fall below capacity again
    assert np.any(raw[cap_idx+1:] < cfg.buffer_capacity)

if __name__ == "__main__":
    pytest.main()
