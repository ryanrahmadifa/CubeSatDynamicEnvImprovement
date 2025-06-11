import pytest
import numpy as np
from cubesat_simulator import SatelliteSimulator

from compute_score import (
    analyze_power_margin,
    analyze_thermal_compliance,
    analyze_comm_availability,
    analyze_science_yield,
    analyze_cpu_efficiency,
    compute_score
)

class DummyMission:
    def __init__(self, dt):
        self.dt = dt

class DummyConfig:
    def __init__(self, data_rate):
        self.data_rate = data_rate
        self.buffer_capacity = np.inf  # default unless overridden

# Power margin tests
def test_analyze_power_margin_full_and_zero():
    assert analyze_power_margin(np.array([100, 100])) == pytest.approx(1.0, rel=1e-3)
    assert analyze_power_margin(np.array([0, 50, 100])) == pytest.approx(0.0, rel=1e-3)
    assert analyze_power_margin(np.array([-10, 20]))   == pytest.approx(0.0, rel=1e-3)

# Thermal compliance tests
@pytest.mark.parametrize("temps, dt, safe_min, safe_max, expected", [
    (np.full(8, 290), 1, 260, 330, 1.0),
    (np.concatenate((np.full(5, 270), np.full(5, 350))), 2, 260, 330,
     pytest.approx(1 - (5*2)/(10*2), rel=1e-3))
])

def test_analyze_thermal_compliance(temps, dt, safe_min, safe_max, expected):
    assert analyze_thermal_compliance(temps, dt, safe_min, safe_max) == expected

# Communication availability tests
def test_analyze_comm_availability_no_passes():
    gs_flag, downlinked = np.zeros(10), np.zeros(10)
    score = analyze_comm_availability(
        gs_flag, downlinked,
        dt=1, target_passes=3,
        required_volume=100, target_duration=5
    )
    assert score == pytest.approx(0.0, rel=1e-6)

def test_analyze_comm_availability_long_pass_insufficient_volume():
    gs_flag, downlinked = np.ones(20), np.zeros(20)
    score = analyze_comm_availability(
        gs_flag, downlinked,
        dt=1, target_passes=2,
        required_volume=50, target_duration=10
    )
    expected = 0.4*(1/2) + 0.4*1 + 0.2*0
    assert score == pytest.approx(expected, rel=1e-6)

def test_analyze_comm_availability_partial():
    gs_flag    = np.array([1,0,1,0,1,0])
    downlinked = np.array([0,5,0,5,0,5])
    score = analyze_comm_availability(
        gs_flag, downlinked,
        dt=1, target_passes=3,
        required_volume=20, target_duration=2
    )
    pass_frac     = min(1, 3/3)
    duration_frac = min(1, 1/2)
    volume_frac   = min(1, 15/20)
    expected = 0.4*pass_frac + 0.4*duration_frac + 0.2*volume_frac
    assert score == pytest.approx(expected, rel=1e-6)

def test_science_yield_processing_downlink_revisit():
    # Create a POI pattern with three events
    poi_flag = np.zeros(10, dtype=int)
    poi_flag[[1, 4, 7]] = 1

    dt         = 1.0
    data_rate  = 1e6     # 1 MB/s
    data_budget = 3e6    # 3 MB total budget

    # Build raw_buf so that at each POI you get +1e6 bytes, then holds constant
    raw_buf  = np.zeros_like(poi_flag, dtype=float)
    for i in range(1, len(raw_buf)):
        raw_buf[i] = raw_buf[i-1] + poi_flag[i] * data_rate * dt

    # Case A: nothing processed, nothing downlinked
    proc_buf = np.zeros_like(raw_buf)
    down     = np.zeros_like(raw_buf)
    s0 = analyze_science_yield(raw_buf, proc_buf, down, poi_flag, dt, data_rate, data_budget)

    # Compute revisit penalty
    idxs = np.where(poi_flag == 1)[0]
    full_idxs = np.concatenate(([-1], idxs, [len(poi_flag)]))
    gaps = np.diff(full_idxs) * dt
    revisit = 1.0 / (1.0 + np.log(np.max(gaps) + 1e-6))

    # Expected: 0.4*0 (proc) + 0.4*0 (down) + 0.2*revisit
    expected_s0 = 0.2 * revisit
    assert s0 == pytest.approx(expected_s0, rel=1e-6)

    # Case B: full processing, no downlink
    proc_buf = raw_buf.copy()
    s1 = analyze_science_yield(raw_buf, proc_buf, down, poi_flag, dt, data_rate, data_budget)
    # Expected: 0.4*1 + 0.4*0 + 0.2*revisit
    expected_s1 = 0.4 * 1.0 + 0.2 * revisit
    assert s1 == pytest.approx(expected_s1, rel=1e-6)
    assert s1 > s0

    # Case C: full processing and full downlink
    down = proc_buf.copy()
    s2 = analyze_science_yield(raw_buf, proc_buf, down, poi_flag, dt, data_rate, data_budget)
    # Expected: 0.4*1 + 0.4*1 + 0.2*revisit
    expected_s2 = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * revisit
    assert s2 == pytest.approx(expected_s2, rel=1e-6)
    assert s2 > s1
    # Final sanity
    assert 0.0 <= s2 <= 1.0

# CPU efficiency tests (now only activity & smoothness)
def test_analyze_cpu_efficiency_regular():
    cpu_load = np.array([0,1,1,0])
    raw_buf  = np.zeros(4)
    activity, smooth = analyze_cpu_efficiency(cpu_load, raw_buf, buffer_capacity=10)
    assert activity == pytest.approx(0.5, rel=1e-6)
    assert smooth   == pytest.approx(0.75, rel=1e-6)

def test_analyze_cpu_efficiency_constant_load():
    cpu_load = np.full(5, 2)
    raw_buf  = np.zeros(5)
    activity, smooth = analyze_cpu_efficiency(cpu_load, raw_buf, buffer_capacity=5)
    assert activity == pytest.approx(1.0, rel=1e-6)
    assert smooth   == pytest.approx(1.0, rel=1e-6)

# Processing and transmission efficiency tests
def test_processing_and_transmission_efficiency():
    steps = 6
    dt    = 1
    mission = DummyMission(dt)
    data_rate = 2
    config = DummyConfig(data_rate)
    soc    = np.full(steps, 100)
    cpu_l  = np.zeros(steps)
    raw_b  = np.zeros(steps)
    poi    = np.ones(steps)
    # Prepare processed_buffer and downlinked
    processed = np.full(steps, data_rate)
    down      = np.full(steps, data_rate/2)
    results = {
        'state_of_charge': soc,
        'cpu_load': cpu_l,
        'raw_buffer': raw_b,
        'processed_buffer': processed,
        'downlinked': down,
        'temperature': np.full(steps, 290)
    }
    final, subs = compute_score(
        None, results, config,
        sun_flag=np.ones(steps),
        gs_flag =np.ones(steps),
        poi_flag=poi,
        mission =mission
    )
    raw_gen   = np.sum(poi * data_rate * dt)
    exp_proc  = processed[-1] / (raw_gen + 1e-6)
    exp_trans = np.sum(down) / (processed[-1] + np.sum(down) + 1e-6)
    assert subs['Processing']   == pytest.approx(exp_proc,  rel=1e-6)
    assert subs['Transmission']== pytest.approx(exp_trans, rel=1e-6)

# Overall compute_score monotonicity
def test_compute_score_monotonic():
    steps = 5
    dt    = 1
    mission = DummyMission(dt)
    config  = DummyConfig(data_rate=1)
    base_results = {
        'state_of_charge': np.full(steps, 100),
        'cpu_load':         np.zeros(steps),
        'raw_buffer':       np.zeros(steps),
        'processed_buffer': np.zeros(steps),
        'downlinked':       np.zeros(steps),
        'temperature':      np.full(steps, 290)
    }
    sun = np.ones(steps)
    gs  = np.ones(steps)
    poi = np.zeros(steps)

    score_base, _ = compute_score(None, base_results, config, sun, gs, poi, mission)

    # Lower SoC should not increase the final score
    low_soc_results = base_results.copy()
    low_soc_results['state_of_charge'] = np.linspace(50, 50, steps)
    score_low, _ = compute_score(None, low_soc_results, config, sun, gs, poi, mission)

    assert score_low <= score_base

if __name__ == "__main__":
    pytest.main()
