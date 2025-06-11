import numpy as np

def compute_score(pos, results, cfg,
                  sun_flag, gs_flag, poi_flag, mission,
                  safe_min=270, safe_max=330,
                  target_passes=4, target_duration=600,
                  required_volume=1e9, data_budget=1e9,
                  mass_ref=None,
                  mass_exponent=2.0):
    """
    Composite (0–10) across six domains: Power, Thermal, OBC, Comm, Payload, Mass.
    
    mass_ref: reference mass (same units as cfg.total_mass).
    mass_exponent: exponent for non-linear penalty: (mass_ref/mass_actual)**mass_exponent.
    """

    # Unpack
    soc       = results['state_of_charge']
    cpu_load  = results['cpu_load']
    raw_buf   = results['raw_buffer']
    proc_buf  = results['processed_buffer']
    down      = results['downlinked']
    temp      = results['temperature']
    dt        = mission.dt






    # --- Power domain (fully enhanced) ---
    energy_score     = (np.mean(soc) + np.min(soc)) / 2.0 / 100.0
    power_margin     = np.clip(np.min(soc) / 100.0, 0.0, 1.0)

    # 1) Low‐battery penalty
    low_thr          = getattr(cfg, 'soc_limit', 20.0)
    low_frac         = np.mean(soc < low_thr)
    low_batt_penalty = np.clip(1.0 - low_frac, 0.0, 1.0)

    # 2) Full‐battery penalty
    full_thr         = 0.99 * 100.0
    full_frac        = np.mean(soc > full_thr)
    full_batt_penalty= np.clip(1.0 - full_frac, 0.0, 1.0)

    # 3) Abrupt‐variation penalty
    d_soc            = np.abs(np.diff(soc))
    max_jump         = np.max(d_soc) if d_soc.size else 0.0
    variation_penalty= np.clip(1.0 - max_jump/10.0, 0.0, 1.0)

    # 4) SoC‐stability penalty (tight band is good)
    std_soc          = np.std(soc)
    # assume a “bad” band is ~50% swing
    stability_penalty= np.clip(1.0 - std_soc/50.0, 0.0, 1.0)

    # 5) Wasted‐energy penalty
    total_time       = len(soc) * dt
    # find timesteps when soc>=full_thr and gen>draw
    wasted = np.maximum(results['power_generated'] - 
                        (results['power_idle']+results['power_cpu']
                         +results['power_comms']+results['power_payload']), 0)
    wasted_when_full = wasted[(soc >= full_thr)]
    wasted_penalty   = 1.0 - (np.sum(wasted_when_full)*dt) / (np.sum(wasted)*dt + 1e-12)
    wasted_penalty   = np.clip(wasted_penalty, 0.0, 1.0)

    # 6) Harvest‐efficiency penalty
    potential = cfg.solar_area * cfg.solar_efficiency * cfg.solar_constant * np.sum(sun_flag) * dt
    actual    = np.sum(results['power_generated']) * dt
    harvest_frac = np.clip(actual / (potential + 1e-12), 0.0, 1.0)

    # 7) Cycle‐count penalty
    # count downward crossings through low_thr
    crossings = np.sum((soc[:-1] >= low_thr) & (soc[1:] < low_thr))
    max_cycles= 10  # for example, more than 10 deep discharges is “bad”
    cycle_penalty = np.clip(1.0 - crossings/ max_cycles, 0.0, 1.0)

    # 8) Combine all seven with custom weights
    power_dom = (
        0.40*energy_score   + 
        0.10*power_margin   +
        0.25*low_batt_penalty +
        0.05*full_batt_penalty+
        0.05*variation_penalty+
        0.05*stability_penalty+
        0.05*wasted_penalty  +
        0.05*harvest_frac    # treat harvest_frac as a “bonus” rather than a penalty
        # cycle_penalty could replace or share weight with one of the above
    )

    power_dom = np.clip(power_dom, 0.0, 1.0)


    # --- Thermal domain (enhanced) ---
    # 1) Base compliance (in-band fraction)
    tot_time       = len(temp) * dt
    oob_mask       = (temp < safe_min) | (temp > safe_max)
    oob_time       = np.sum(oob_mask) * dt
    thermal_base   = np.clip(1 - oob_time / tot_time, 0.0, 1.0)

    # 2) Rapid temperature variation penalty
    dT             = np.abs(np.diff(temp))
    max_dT         = np.max(dT) if dT.size else 0.0
    max_allowed_dT = 5.0   # e.g. 5 K per timestep acceptable
    rapid_penalty  = np.clip(1 - max_dT / max_allowed_dT, 0.0, 1.0)

    # 3) Comfort margin (time well within safe bounds)
    delta          = 5.0   # margin in K inside safe band
    comfort_mask   = (temp >= safe_min + delta) & (temp <= safe_max - delta)
    comfort_frac   = np.mean(comfort_mask)
    comfort_score  = np.clip(comfort_frac, 0.0, 1.0)

    # 4) Excursion count penalty (number of OOB entries)
    excursions     = np.sum((oob_mask[:-1] == False) & (oob_mask[1:] == True))
    max_excursions = 10    # threshold for “too many” excursions
    excursion_penalty = np.clip(1 - excursions / max_excursions, 0.0, 1.0)

    # 5) Weighted combination
    #    e.g. 40% base compliance, 20% rapid-change, 20% comfort, 20% excursions
    thermal_dom = (
        0.50 * thermal_base +
        0.10 * rapid_penalty +
        0.10 * comfort_score +
        0.30 * excursion_penalty
    )
    thermal_dom = np.clip(thermal_dom, 0.0, 1.0)




    # --- OBC domain (with full-buffer penalty) ---

    # 1) Throughput: fraction of raw data processed
    raw_gen         = np.sum(poi_flag * cfg.data_rate * dt) + 1e-12
    proc_score      = np.clip(proc_buf[-1] / raw_gen, 0.0, 1.0)

    # 2) Blocked penalty: CPU off when there’s data waiting
    demand_mask     = raw_buf > 0
    blocked_frac    = np.sum(demand_mask & (cpu_load == 0)) / (np.sum(demand_mask) + 1e-12)
    blocked_penalty = 1.0 - np.clip(blocked_frac, 0.0, 1.0)

    # 3) Activation penalty: penalize frequent on‐events
    rising_edges    = np.sum((cpu_load[1:] > 0) & (cpu_load[:-1] == 0))
    activation_frac = rising_edges / (len(cpu_load) - 1)
    activation_score= 1.0 - np.clip(activation_frac, 0.0, 1.0)

    # 4) Efficiency: bytes processed per Joule of CPU energy
    energy_cpu_j    = np.sum(results['power_cpu']) * dt
    bytes_proc      = proc_buf[-1]
    efficiency      = bytes_proc / (energy_cpu_j + 1e-12)
    max_bpj         = cfg.cpu_throughput_per_watt * cfg.cpu_processing_efficiency
    efficiency_score= np.clip(efficiency / (max_bpj + 1e-12), 0.0, 1.0)

    # 5) Full‐buffer penalty: fraction of time raw_buffer at capacity
    full_frac       = np.mean(raw_buf >= cfg.buffer_capacity)
    full_penalty    = 1.0 - np.clip(full_frac, 0.0, 1.0)

    # 6) Combine all sub‐scores (weights sum to 1):
    #    50% throughput, 20% blocked,  5% activation,
    #    15% efficiency, 10% full‐buffer penalty
    obc_dom = (
        0.60 * proc_score +
        0.05 * blocked_penalty +
        0.05 * activation_score +
        0.10 * efficiency_score +
        0.20 * full_penalty
    )
    obc_dom = np.clip(obc_dom, 0.0, 1.0)





    # --- Comm domain (with full‐processed‐buffer penalty) ---

    # 1) Count passes
    passes, in_pass = 0, False
    for f in gs_flag:
        if f and not in_pass:
            in_pass, passes = True, passes + 1
        elif not f and in_pass:
            in_pass = False
    pass_frac = min(1.0, passes / target_passes)

    # 2) Volume fraction
    vol_frac = min(1.0, np.sum(down) / required_volume)

    # 3) Availability score
    avail = 0.5 * pass_frac + 0.5 * vol_frac

    # 4) Utilization
    gs_time    = np.sum(gs_flag) * dt + 1e-12
    max_bytes  = cfg.downlink_rate * gs_time
    util_comm  = np.clip(np.sum(down) / max_bytes, 0.0, 1.0)

    # 5) Full‐processed‐buffer penalty: fraction of time processed_buffer is at capacity
    full_proc_frac    = np.mean(proc_buf >= cfg.buffer_capacity)
    full_proc_penalty = 1.0 - np.clip(full_proc_frac, 0.0, 1.0)

    # 6) Combine into final Comm score
    #    40% availability, 40% utilization, 20% full‐buffer penalty
    comm_dom = (
        0.20 * avail +
        0.60 * util_comm +
        0.20 * full_proc_penalty
    )
    comm_dom = np.clip(comm_dom, 0.0, 1.0)





    # --- Payload domain (availability & storage efficiency) ---
    # 1) Identify all POI timesteps
    poi_idxs = np.where(poi_flag == 1)[0]

    # 2) Fraction of POIs actually observed (raw data added)
    if poi_idxs.size:
        # at each POI, did raw_buffer jump by at least one data_rate chunk?
        obs_events = [
            (raw_buf[i] - raw_buf[i-1]) >= (cfg.data_rate * dt * 0.9)
            for i in poi_idxs
        ]
        avail_frac = sum(obs_events) / len(obs_events)
    else:
        avail_frac = 1.0  # no POIs ⇒ trivially available

    # 3) Storage efficiency: how much of the potential raw data actually got into the buffer
    potential_raw = len(poi_idxs) * cfg.data_rate * dt
    # sum of all positive raw_buffer increments
    added_bytes   = sum(
        max(raw_buf[i] - raw_buf[i-1], 0.0)
        for i in range(1, len(raw_buf))
    )
    storage_frac  = np.clip(added_bytes / (potential_raw + 1e-12), 0.0, 1.0)

    # 4) Processing & downlink efficiency: fraction of collected bytes that finish end-to-end
    completed_bytes = proc_buf[-1] + np.sum(down)
    proc_frac       = np.clip(completed_bytes / (added_bytes + 1e-12), 0.0, 1.0)

    # 5) Combine with simple weights
    #    50% availability, 25% storage, 25% completion
    payload_dom = (
        0.40 * avail_frac +
        0.30 * storage_frac +
        0.30 * proc_frac
    )
    payload_dom = np.clip(payload_dom, 0.0, 1.0)


    # --- Mass domain (non-linear penalty) ---
    if mass_ref is None or cfg.total_mass <= 0:
        mass_dom = 1.0
    else:
        ratio   = mass_ref / cfg.total_mass
        mass_dom= np.clip(ratio**mass_exponent, 0.0, 1.0)

    # --- Final assembly ---
    subscores = {
        'Power':   power_dom,
        'Thermal': thermal_dom,
        'OBC':     obc_dom,
        'Comm':    comm_dom,
        'Payload': payload_dom,
    }
    #final = np.mean(list(subscores.values())) * 10.0
    subscores_list = list(subscores.values())
    final = float(np.prod(subscores_list) ** (1.0/len(subscores_list))) * 10.0
    # 2) Mass penalty (clamped between 0.1 and 1.0 so you never go below 10% of base)
    #ratio        = mass_ref / cfg.total_mass
    #mass_penalty = np.clip(ratio**mass_exponent, 0.1, 1.0)

    # 3) Final score
    #final = final * mass_penalty

    return final, subscores


#def compute_score(pos, results, cfg,
#                  sun_flag, gs_flag, poi_flag, mission,
#                  safe_min=270, safe_max=330,
#                  target_passes=4, target_duration=600,
#                  required_volume=1e9, data_budget=1e9,
#                  mass_ref=None,
#                  mass_exponent=2.0):
#    """
#    Composite (0–10) across six domains: Power, Thermal, OBC, Comm, Payload, Mass.
#    
#    mass_ref: reference mass (same units as cfg.total_mass).
#    mass_exponent: exponent for non-linear penalty: (mass_ref/mass_actual)**mass_exponent.
#    """
#
#    # Unpack
#    soc       = results['state_of_charge']
#    cpu_load  = results['cpu_load']
#    raw_buf   = results['raw_buffer']
#    proc_buf  = results['processed_buffer']
#    down      = results['downlinked']
#    temp      = results['temperature']
#    dt        = mission.dt
#
#    # --- Power domain ---
#    energy_score = (np.mean(soc) + np.min(soc)) / 2.0 / 100.0
#    power_margin = np.clip(np.min(soc) / 100.0, 0.0, 1.0)
#    power_dom    = np.mean([energy_score, power_margin])
#
#    #need oob for low energy too
#
#    # --- Thermal domain ---
#    tot_time    = len(temp) * dt
#    oob_time    = np.sum((temp < safe_min) | (temp > safe_max)) * dt
#    thermal_dom = np.clip(1 - oob_time/tot_time, 0.0, 1.0)
#
#    # --- OBC domain ---
#    mean_load  = np.mean(cpu_load)
#    util       = mean_load / (cfg.cpu_dmips + 1e-12)
#    raw_gen    = np.sum(poi_flag * cfg.data_rate * dt) + 1e-12
#    total_proc = proc_buf[-1] + np.sum(down)
#    proc_eff   = np.clip(total_proc/raw_gen, 0.0, 1.0)
#    obc_dom    = np.mean([np.clip(util,0,1), proc_eff])
#
#    # --- Comm domain ---
#    # Availability
#    passes, durs, in_pass = [], [], False
#    for i,f in enumerate(gs_flag):
#        if f and not in_pass:
#            in_pass, start = True, i
#        elif not f and in_pass:
#            in_pass = False
#            passes.append((start, i-1))
#            durs.append((i-1-start+1)*dt)
#    if in_pass:
#        passes.append((start, len(gs_flag)-1))
#        durs.append((len(gs_flag)-1-start+1)*dt)
#    pass_frac = min(1.0, len(passes)/target_passes)
#    avg_dur   = np.mean(durs) if durs else 0.0
#    dur_frac  = min(1.0, avg_dur/target_duration)
#    vol_frac  = min(1.0, np.sum(down)/required_volume)
#    avail     = 0.4*pass_frac + 0.4*dur_frac + 0.2*vol_frac
#    # Utilization
#    gs_time   = np.sum(gs_flag)*dt + 1e-12
#    cap_bytes = cfg.downlink_rate * gs_time
#    util_comm = np.clip(np.sum(down)/cap_bytes, 0.0, 1.0)
#    comm_dom  = np.mean([avail, util_comm])
#
#    # --- Payload domain ---
#    proc_frac = proc_buf[-1] / raw_gen
#    down_frac = np.sum(down) / (proc_buf[-1] + 1e-12)
#    idxs      = np.where(poi_flag==1)[0]
#    if idxs.size:
#        full   = np.concatenate(([-1], idxs, [len(poi_flag)]))
#        gaps   = np.diff(full) * dt
#        revisit= 1.0/(1.0 + np.log(np.max(gaps)+1e-6))
#    else:
#        revisit= 0.0
#    payload_dom = np.clip(0.4*proc_frac + 0.4*down_frac + 0.2*revisit, 0.0, 1.0)
#
#    # --- Mass domain (non-linear penalty) ---
#    if mass_ref is None or cfg.total_mass <= 0:
#        mass_dom = 1.0
#    else:
#        ratio   = mass_ref / cfg.total_mass
#        mass_dom= np.clip(ratio**mass_exponent, 0.0, 1.0)
#
#    # --- Final assembly ---
#    subscores = {
#        'Power':   power_dom,
#        'Thermal': thermal_dom,
#        'OBC':     obc_dom,
#        'Comm':    comm_dom,
#        'Payload': payload_dom,
#    }
#    final = np.mean(list(subscores.values())) * 10.0
#    # 2) Mass penalty (clamped between 0.1 and 1.0 so you never go below 10% of base)
#    #ratio        = mass_ref / cfg.total_mass
#    #mass_penalty = np.clip(ratio**mass_exponent, 0.1, 1.0)
#
#    # 3) Final score
#    #final = final * mass_penalty
#
#    return final, subscores
#