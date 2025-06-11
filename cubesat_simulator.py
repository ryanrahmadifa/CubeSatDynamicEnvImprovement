import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from compute_score import compute_score
from cubesat_mission import *

Re = 6371e3  # Earth radius [m]
mu = 3.986004418e14  # Gravitational constant [m^3/s^2]
sun_vector = np.array([1, 0, 0])  # Arbitrary sun direction for sunlight flag

import logging

import logging
import numpy as np

class SatelliteSimulator:
    def __init__(self, config, mission):
        self.config  = config
        self.mission = mission

        # set up logger (once)
        self.logger = logging.getLogger('SatSim')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = ('%(asctime)s %(levelname)-5s '
                   '%(message)s')
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)   # change to DEBUG to see every print

    def simulate(self, pos, sun_flag, gs_flag, poi_flag, debug=False):
        """
        Run the satellite power/thermal/data-handling simulation.

        Parameters
        ----------
        pos : array_like
            Satellite position array (unused except for length)
        sun_flag : array_like
            0/1 whether in sunlight
        gs_flag : array_like
            0/1 whether in view of ground station
        poi_flag : array_like
            0/1 whether over a point of interest
        debug : bool
            If True, emit detailed debug logs every 100 steps

        Returns
        -------
        dict of np.ndarray
            All time series: state_of_charge, temperature, buffers, loads,
            power terms, heat terms, and state machines.
        """
        import numpy as np

        cfg, dt = self.config, self.mission.dt
        n = len(pos)

        # — Preconditions —
        assert dt > 0, "time step dt must be positive"
        assert len(sun_flag) == len(gs_flag) == len(poi_flag) == n, \
            "all input arrays must have same length"

        # time vector for logs
        time_arr = np.arange(n) * dt

        # --- Preallocate all series ---
        soc             = np.zeros(n); soc[0]            = 100.0
        temperature     = np.zeros(n); temperature[0]    = cfg.initial_temp
        raw_buffer      = np.zeros(n)
        processed_buffer= np.zeros(n)
        observed        = np.zeros(n)
        downlinked      = np.zeros(n)
        processed_amt   = np.zeros(n)
        cpu_load        = np.zeros(n)

        power_generated = np.zeros(n)
        power_idle      = np.zeros(n)
        power_payload   = np.zeros(n)
        power_cpu       = np.zeros(n)
        power_comms     = np.zeros(n)

        heat_solar      = np.zeros(n)
        heat_idle       = np.zeros(n)
        heat_payload    = np.zeros(n)
        heat_cpu        = np.zeros(n)
        heat_comms      = np.zeros(n)
        heat_imbalance  = np.zeros(n)
        heat_generated  = np.zeros(n)
        heat_radiated   = np.zeros(n)

        # States
        s_energy      = np.empty(n, dtype='<U12')
        s_heat        = np.empty(n, dtype='<U10')
        s_buffer_raw  = np.empty(n, dtype='<U10')
        s_buffer_proc = np.empty(n, dtype='<U10')
        s_payload     = np.empty(n, dtype='<U10')
        s_comm        = np.empty(n, dtype='<U12')
        s_cpu         = np.empty(n, dtype='<U12')
        s_safety_T    = np.empty(n, dtype='<U16')
        s_safety_E    = np.empty(n, dtype='<U16')
        s_electronics = np.empty(n, dtype='<U16')

        # Initial states
        s_energy[0]      = 'charging'
        s_heat[0]        = 'cooling'
        s_buffer_raw[0]  = 'empty'
        s_buffer_proc[0] = 'empty'
        s_payload[0]     = 'off'
        s_comm[0]        = 'off'
        s_cpu[0]         = 'off'
        s_safety_T[0]    = 'nominal'
        s_safety_E[0]    = 'nominal'
        s_electronics[0] = 'nominal'

        for i in range(1, n):
            # Inherit & reset
            raw_buffer[i]       = raw_buffer[i-1]
            processed_buffer[i] = processed_buffer[i-1]

            # Default power draws
            idle_draw    = cfg.idle_power
            payload_draw = cfg.payload_power_idle
            cpu_draw     = cfg.cpu_power_idle
            comms_draw   = cfg.comms_power_idle

            # — Electronics safety from previous step —
            if s_safety_T[i-1] != 'nominal':
                s_electronics[i] = 'safety_shutdown'
            elif s_safety_E[i-1] == 'low_battery':
                s_electronics[i] = 'forced_shutdown'
            else:
                s_electronics[i] = 'nominal'

            # Only if electronics are nominal do we run subsystems:
            if s_electronics[i] == 'nominal':
                # Decide payload state
                if poi_flag[i] > 0:
                    s_payload[i] = (
                        'blocked'
                        if raw_buffer[i-1] >= cfg.buffer_capacity
                        else 'observing'
                    )
                else:
                    s_payload[i] = 'off'

                # Decide CPU state
                if raw_buffer[i] > 0:
                    s_cpu[i] = (
                        'blocked'
                        if processed_buffer[i-1] >= cfg.buffer_capacity
                        else 'on'
                    )
                else:
                    s_cpu[i] = 'off'

                # Decide comm state
                if gs_flag[i] > 0:
                    s_comm[i] = (
                        'downloading'
                        if processed_buffer[i-1] > 0
                        else 'stalled'
                    )
                else:
                    s_comm[i] = 'off'

                # Data operations
                if s_payload[i] == 'observing':
                    obs = cfg.data_rate * dt
                    raw_buffer[i]   += obs
                    observed[i]      = obs

                if s_cpu[i] == 'on':
                    cpu_load[i] = cfg.cpu_dmips
                    proc_cap = min(
                        raw_buffer[i],
                        cpu_load[i] * cfg.cpu_processing_efficiency * dt
                    )
                    raw_buffer[i]        -= proc_cap
                    processed_buffer[i]  += proc_cap
                    processed_amt[i]      = proc_cap

                if s_comm[i] == 'downloading':
                    dl = min(cfg.downlink_rate * dt, processed_buffer[i])
                    processed_buffer[i] -= dl
                    downlinked[i]        = dl

                # Compute actual draws
                payload_draw = cfg.payload_power_idle + poi_flag[i] * cfg.payload_power_max
                cpu_draw     = (
                    cfg.cpu_power_idle
                    + cpu_load[i] * (cfg.cpu_power_max - cfg.cpu_power_idle) / cfg.cpu_dmips
                )
                comms_draw = (cfg.comms_power_idle 
                    + gs_flag[i] * (cfg.comms_power_max - cfg.comms_power_idle)) * 10

            # Store draws
            power_idle[i]    = idle_draw
            power_payload[i] = payload_draw
            power_cpu[i]     = cpu_draw
            power_comms[i]   = comms_draw
            total_draw       = idle_draw + payload_draw + cpu_draw + comms_draw

            # — Power generation & SoC update —
            pg = (
                cfg.solar_area
                * cfg.solar_efficiency
                * cfg.solar_constant
                * sun_flag[i]
            )
            power_generated[i] = pg

            delta_soc = (pg - total_draw) * dt / (cfg.battery_capacity * 3600.0) * 100.0
            soc[i] = np.clip(soc[i-1] + delta_soc, 0.0, 100.0)

            # — Heat & temperature update —
            heat_solar[i]     = max(0.0, pg * (1 - cfg.solar_efficiency))
            heat_idle[i]      = idle_draw * 0.6
            heat_payload[i]   = payload_draw * 0.7
            heat_cpu[i]       = cpu_draw * 0.9
            heat_comms[i]     = comms_draw * 0.8
            heat_imbalance[i] = abs(pg - total_draw) * 0.2

            heat_generated[i] = (
                cfg.absorptivity * pg
                + heat_idle[i]
                + heat_payload[i]
                + heat_cpu[i]
                + heat_comms[i]
                + heat_imbalance[i]
            )
            prev_temp = temperature[i-1]
            rad        = (
                cfg.emissivity
                * 5.67e-8
                * cfg.radiative_area
                * prev_temp**4
            )
            heat_radiated[i] = rad

            net_heat       = heat_generated[i] - rad
            temperature[i] = prev_temp + net_heat * dt / cfg.thermal_mass

            # — State machines —
            s_energy[i]     = 'charging'  if delta_soc > 0 else 'discharging'
            s_heat[i]       = 'heating'   if temperature[i] > prev_temp else 'cooling'

            # Raw buffer
            dr = raw_buffer[i] - raw_buffer[i-1]
            if raw_buffer[i] <= 0:
                s_buffer_raw[i] = 'empty'
            elif raw_buffer[i] >= cfg.buffer_capacity:
                s_buffer_raw[i] = 'full'
            elif dr > 0:
                s_buffer_raw[i] = 'filling'
            elif dr < 0:
                s_buffer_raw[i] = 'draining'
            else:
                s_buffer_raw[i] = 'stable'

            # Processed buffer
            dp = processed_buffer[i] - processed_buffer[i-1]
            if processed_buffer[i] <= 0:
                s_buffer_proc[i] = 'empty'
            elif processed_buffer[i] >= cfg.buffer_capacity:
                s_buffer_proc[i] = 'full'
            elif dp > 0:
                s_buffer_proc[i] = 'filling'
            elif dp < 0:
                s_buffer_proc[i] = 'draining'
            else:
                s_buffer_proc[i] = 'stable'

            # Safety
            s_safety_T[i] = (
                'high_temperature' if temperature[i] >= cfg.max_temp
                else 'low_temperature'  if temperature[i] <= cfg.min_temps
                else 'nominal'
            )
            s_safety_E[i] = (
                'low_battery' if soc[i] <= cfg.soc_limit
                else 'full'       if soc[i] >= 100
                else 'nominal'
            )

            # — Post-conditions (sanity checks) —
            assert 0 <= raw_buffer[i] <= cfg.buffer_capacity
            assert 0 <= processed_buffer[i] <= cfg.buffer_capacity
            assert payload_draw >= cfg.payload_power_idle
            assert cpu_draw     >= cfg.cpu_power_idle
            assert comms_draw   >= cfg.comms_power_idle
            assert total_draw   >= 0
            assert 0 <= soc[i] <= 100

            # Optional detailed logging
            if debug and (i % 100) == 0:
                self.logger.debug(
                    f"i={i:4d} t={time_arr[i]:6.1f}s SoC={soc[i]:6.2f}% "
                    f"P[idl={idle_draw:.1f} pay={payload_draw:.1f} cpu={cpu_draw:.1f} comm={comms_draw:.1f}] "
                    f"pg={pg:.1f} td={total_draw:.1f} "
                    f"Δraw={dr:+.1f} Δproc={dp:+.1f} dl={downlinked[i]:.1f} "
                    f"ΔSoC={delta_soc:+.2f}% "
                    f"H[sol={heat_solar[i]:.1f} idl={heat_idle[i]:.1f} pay={heat_payload[i]:.1f} "
                    f"cpu={heat_cpu[i]:.1f} comm={heat_comms[i]:.1f}] "
                    f"Hgen={heat_generated[i]:.1f} Hrad={heat_radiated[i]:.1f} "
                    f"ΔHimb={heat_imbalance[i]:.1f}"
                )

        return {
            'state_of_charge':     soc,
            'temperature':         temperature,
            'raw_buffer':          raw_buffer,
            'processed_buffer':    processed_buffer,
            'downlinked':          downlinked,
            'processed_amt':       processed_amt,
            'observed':            observed,
            'cpu_load':            cpu_load,
            'power_generated':     power_generated,
            'power_idle':          power_idle,
            'power_payload':       power_payload,
            'power_cpu':           power_cpu,
            'power_comms':         power_comms,
            'heat_solar':          heat_solar,
            'heat_idle':           heat_idle,
            'heat_payload':        heat_payload,
            'heat_cpu':            heat_cpu,
            'heat_comms':          heat_comms,
            'heat_generated':      heat_generated,
            'heat_radiated':       heat_radiated,
            'heat_diff':           heat_imbalance,
            's_energy':            s_energy,
            's_heat':              s_heat,
            's_buffer_raw':        s_buffer_raw,
            's_buffer_proc':       s_buffer_proc,
            's_payload':           s_payload,
            's_comm':              s_comm,
            's_cpu':               s_cpu,
            's_safety_T':          s_safety_T,
            's_safety_E':          s_safety_E,
            's_electronics':       s_electronics,
            'sun_flag':            sun_flag,
            'gs_flag':             gs_flag,
            'poi_flag':            poi_flag,
            'buffer_capacity':     cfg.buffer_capacity
        }


            # — LOGGING every 1000 steps —
            #f i % 100 == 0:
            #   clipped    = '*' if soc[i] in (0.0, 100.0) else ''
            #   msg = (
            #       f"i={i:5d} t={time_arr[i]:7.1f}s "
            #       f"SoC={soc[i]:6.2f}{clipped}% "
            #       f"T={temperature[i]:6.2f}ΔT={temperature[i]-prev_temp:+.2f} "
            #       f"pg={pg:6.1f} td={total_draw:6.1f} "
            #       f"ΔSoC={delta_soc:+6.2f} "
            #       f"flags={sun_flag[i]},{gs_flag[i]},{poi_flag[i]} "
            #       f"states=[E:{s_energy[i]} H:{s_heat[i]} "
            #       f"P:{s_payload[i]} C:{s_comm[i]} CPU:{s_cpu[i]}] "
            #       f"safety[T:{s_safety_T[i]},E:{s_safety_E[i]}]"
            #   )
            #   self.logger.debug(msg)
                
            

def apply_plot_style(ax, title, xlabel, ylabel, ylim=None):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)

def apply_plot_style(ax, title, xlabel, ylabel, ylim=None):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)

def plot_simulation_dashboard(results, return_fig=False):
    """
    3x4 Dashboard with integrated buffer margin in the Buffer & Downlink subplot.
    Panels:
      0: SoC, 1: Temp, 2: Buffer & Downlink + margin, 3: CPU Load,
      4: Payload vs Downlink, 5: Power Gen vs Draw,
      6: Power Breakdown, 7: Heat Breakdown,
      8: Heat Gen vs Radiated, 9: GS Timeline, 10: POI Timeline, 11: Eclipse/Sunlight
    """
    dt = 1
    n = len(results['state_of_charge'])
    t = np.arange(n)
    
    # Auxiliary series
    raw_buf = results['raw_buffer']
    proc_buf = results['processed_buffer']
    cum_down = np.cumsum(results['downlinked'])
    capacity = results['buffer_capacity']
    margin = raw_buf / capacity
    
    # Flags
    sun = results['sun_flag']
    gs  = results['gs_flag']
    poi = results['poi_flag']
    
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    axs = axs.flatten()
    fig.suptitle("Satellite System Simulation Dashboard", fontsize=18, fontweight='bold')
    
    # Panel 0: SoC
    axs[0].plot(results['state_of_charge'], color='tab:blue')
    apply_plot_style(axs[0], "Battery SoC", "Time Step", "SoC (%)")
    
    # Panel 1: Temperature
    axs[1].plot(results['temperature'], color='tab:red')
    apply_plot_style(axs[1], "Temperature (K)", "Time Step", "Temp (K)")
    
    # Panel 2: Buffer & Downlink with Margin
    ax2 = axs[2]
    ax2.plot(raw_buf, label="Raw", color='tab:orange')
    ax2.plot(proc_buf, label="Processed", color='tab:purple')
    ax2.plot(cum_down, label="Cum. Downlinked", color='tab:green')
    # Capacity line
    ax2.axhline(capacity, color='tab:red', linestyle='--', label="Capacity")
    # Shade region above margin threshold (e.g. margin > 0.8)
    ax2.fill_between(t, raw_buf, capacity, where=(margin >= 0.8), color='red', alpha=0.1,
                     label="Near Full")
    ax2.legend(fontsize=8)
    apply_plot_style(ax2, "Buffer & Downlink (Bytes)", "Time Step", "Bytes")
    
    # Panel 3: CPU Load
    axs[3].plot(results['cpu_load'], color='tab:green')
    apply_plot_style(axs[3], "CPU Load (DMIPS)", "Time Step", "DMIPS")
    
    # Panel 4: Payload vs Downlink
    payload_rate = np.diff(raw_buf, prepend=raw_buf[0]) / dt
    downlink_rate = results['downlinked'] / dt
    axs[4].plot(payload_rate, label="Payload Rate", color='tab:orange')
    axs[4].plot(downlink_rate, label="Downlink Rate", color='tab:purple')
    axs[4].legend(fontsize=8)
    apply_plot_style(axs[4], "Payload vs Downlink Rate", "Time Step", "Bytes/s")
    
    # Panel 5: Power Gen vs Draw
    total_draw = results['power_idle'] + results['power_cpu'] + results['power_comms'] + results['power_payload']
    axs[5].plot(results['power_generated'], label="Generated", color='tab:cyan')
    axs[5].plot(total_draw, label="Total Draw", color='tab:gray')
    axs[5].legend(fontsize=8)
    apply_plot_style(axs[5], "Power Gen vs Draw", "Time Step", "Power (W)")
    
    # Panel 6: Power Breakdown (stacked)
    comp_p = np.vstack([results['power_idle'], results['power_cpu'], results['power_comms'], results['power_payload']])
    axs[6].stackplot(t, comp_p, labels=['Idle','CPU','Comms','Payload'], alpha=0.8)
    axs[6].legend(loc='upper left', fontsize=8)
    apply_plot_style(axs[6], "Power Draw Breakdown", "Time Step", "Power (W)")
    
    # Panel 7: Heat Breakdown (stacked)
    comp_h = np.vstack([results['heat_solar'], results['heat_idle'], results['heat_cpu'], results['heat_comms'], results['heat_diff']])
    axs[7].stackplot(t, comp_h, labels=['Solar','Idle','CPU','Comms','Imbalance'], alpha=0.8)
    axs[7].legend(loc='upper left', fontsize=8)
    apply_plot_style(axs[7], "Heat Breakdown", "Time Step", "Heat Rate (W)")
    
    # Panel 8: Heat Gen vs Radiated
    axs[8].plot(t, results['heat_generated'], label="Heat Gen", color='tab:orange')
    axs[8].plot(t, results['heat_radiated'], label="Radiated", color='tab:blue')
    axs[8].legend(fontsize=8)
    apply_plot_style(axs[8], "Heat Gen vs Radiated", "Time Step", "Heat Rate (W)")
    
    # Panel 9: GS Contact Timeline
    axs[9].imshow(gs[np.newaxis,:], aspect='auto', cmap='Greys', vmin=0, vmax=1)
    axs[9].set_yticks([])
    axs[9].set_xlabel("Time Step")
    axs[9].set_title("GS Contact Timeline")
    
    # Panel 10: POI Contact Timeline
    axs[10].imshow(poi[np.newaxis,:], aspect='auto', cmap='Oranges', vmin=0, vmax=1)
    axs[10].set_yticks([])
    axs[10].set_xlabel("Time Step")
    axs[10].set_title("POI Contact Timeline")
    
    # Panel 11: Eclipse/Sunlight
    axs[11].imshow(sun[np.newaxis,:], aspect='auto', cmap='YlOrBr', vmin=0, vmax=1)
    axs[11].set_yticks([])
    axs[11].set_xlabel("Time Step")
    axs[11].set_title("Eclipse (gray) / Sunlight (yellow)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if return_fig:
        return fig
    else:
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def apply_plot_style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def apply_plot_style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_simulation_dashboard_with_states(results, return_fig=False):
    """
    3×4 Dashboard with integrated state overlays and legends.
    Panels:
      0: SoC + Battery State
      1: Temperature + Thermal Safety State
      2: Buffer & Downlink + Margin
      3: CPU Load + CPU State ('off','on','forced_shutdown','safety_shutdown')
      4: Payload vs Downlink Rate + Payload & Comm States
      5: Power Gen vs Draw + Energy State
      6: Power Breakdown
      7: Heat Breakdown
      8: Heat Gen vs Radiated + Heat State
      9: POI & GS Timeline
      10: Eclipse/Sunlight
      11: Reserved
    """
    n = len(results['state_of_charge'])
    t = np.arange(n)

    # Core series
    soc = results['state_of_charge']
    temp = results['temperature']
    raw_buf = results['raw_buffer']
    proc_buf = results['processed_buffer']
    down = results['downlinked']
    cum_down = np.cumsum(down)
    cap = results['buffer_capacity']
    margin = raw_buf / cap
    sun = results['sun_flag']
    gs = results['gs_flag']
    poi = results['poi_flag']

    # State arrays
    batt_state   = results['s_safety_E']
    therm_state  = results['s_safety_T']
    cpu_state    = results['s_cpu']
    payload_state= results['s_payload']
    comm_state   = results['s_comm']
    energy_state = results['s_energy']
    heat_state   = results['s_heat']

    # Colormap & maps
    batt_cmap   = ListedColormap(['white', 'red', 'lightblue'])
    batt_map    = {'nominal':0, 'low_battery':1, 'full':2}
    therm_cmap  = ListedColormap(['white', 'red', 'gray'])
    therm_map   = {'nominal':0, 'high_temperature':1, 'low_temperature':2}
    cpu_cmap    = ListedColormap(['white', 'green', 'gray', 'red'])
    cpu_map     = {'off':0, 'on':1, 'forced_shutdown':2, 'safety_shutdown':3}
    payload_cmap= ListedColormap(['lightgray', 'green', 'orange'])
    payload_map = {'off':0, 'observing':1, 'blocked':2}
    comm_cmap   = ListedColormap(['lightgray', 'blue', 'orange'])
    comm_map    = {'off':0, 'downloading':1, 'stalled':2}
    energy_cmap = ListedColormap(['lightcoral', 'lightgreen'])
    energy_map  = {'discharging':0, 'charging':1}
    heat_cmap   = ListedColormap(['lightblue', 'orange'])
    heat_map    = {'cooling':0, 'heating':1}

    # Encode states
    batt_codes   = [batt_map.get(s,0)   for s in batt_state]
    therm_codes  = [therm_map.get(s,0)  for s in therm_state]
    cpu_codes    = [cpu_map.get(s,0)    for s in cpu_state]
    pay_codes    = [payload_map.get(s,0)for s in payload_state]
    comm_codes   = [comm_map.get(s,0)   for s in comm_state]
    energy_codes = [energy_map.get(s,0) for s in energy_state]
    heat_codes   = [heat_map.get(s,0)   for s in heat_state]

    # Set up figure
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    axs = axs.flatten()
    fig.suptitle("Satellite System Simulation Dashboard", fontsize=18, fontweight='bold')

    # Panel 0: SoC + Battery
    ax0 = axs[0]
    ax0.plot(soc, color='tab:blue')
    ymin, ymax = ax0.get_ylim()
    ax0.imshow([batt_codes], aspect='auto', cmap=batt_cmap,
               extent=[0, n, ymax, ymax + (ymax-ymin)*0.05])
    ax0.set_ylim(ymin, ymax + (ymax-ymin)*0.05)
    apply_plot_style(ax0, "Battery SoC", "Time Step", "SoC (%)")
    batt_handles = [
        Patch(color='white',     label='Nominal'),
        Patch(color='red',       label='Low Battery'),
        Patch(color='lightblue', label='Full')
    ]
    ax0.legend(handles=batt_handles, loc='upper right', fontsize=8)

    # Panel 1: Temperature + Thermal
    ax1 = axs[1]
    ax1.plot(temp, color='tab:red')
    ymin, ymax = ax1.get_ylim()
    ax1.imshow([therm_codes], aspect='auto', cmap=therm_cmap,
               extent=[0, n, ymax, ymax + (ymax-ymin)*0.05])
    ax1.set_ylim(ymin, ymax + (ymax-ymin)*0.05)
    apply_plot_style(ax1, "Temperature (K)", "Time Step", "Temp")
    therm_handles = [
        Patch(color='white', label='Nominal'),
        Patch(color='red',   label='High Temp'),
        Patch(color='gray',  label='Low Temp')
    ]
    ax1.legend(handles=therm_handles, loc='upper right', fontsize=8)

    # Panel 2: Buffer & Downlink
    ax2 = axs[2]
    ax2.plot(raw_buf, label="Raw",       color='tab:orange')
    ax2.plot(proc_buf, label="Processed", color='tab:purple')
    ax2.plot(cum_down, label="Cum Down",  color='tab:green')
    ax2.axhline(cap, linestyle='--', color='tab:red', label="Capacity")
    ax2.fill_between(t, raw_buf, cap, where=(margin>=0.8),
                     color='red', alpha=0.1, label="Near Full")
    ax2.legend(fontsize=8)
    apply_plot_style(ax2, "Buffer & Downlink (Bytes)", "Time Step", "Bytes")

    # Panel 3: CPU Load + State
    ax3 = axs[3]
    ax3.plot(results['cpu_load'], color='tab:green')
    ymin, ymax = ax3.get_ylim()
    ax3.imshow([cpu_codes], aspect='auto', cmap=cpu_cmap,
               extent=[0, n, ymax, ymax + (ymax-ymin)*0.05])
    ax3.set_ylim(ymin, ymax + (ymax-ymin)*0.05)
    apply_plot_style(ax3, "CPU Load (DMIPS)", "Time Step", "DMIPS")
    cpu_handles = [
        Patch(color='white', label='Off'),
        Patch(color='green', label='On'),
        Patch(color='gray',  label='Forced SD'),
        Patch(color='red',   label='Safety SD')
    ]
    ax3.legend(handles=cpu_handles, loc='upper right', fontsize=8)

    # Panel 4: Payload vs Downlink + States
    ax4 = axs[4]
    payload_rate = np.diff(raw_buf, prepend=raw_buf[0])
    downlink_rate = down
    ax4.plot(payload_rate,    label="Payload Rate",  color='tab:orange')
    ax4.plot(downlink_rate,   label="Downlink Rate", color='tab:purple')
    ax4.legend(loc='upper left', fontsize=8)
    ymin, ymax = ax4.get_ylim()
    ax4.imshow([pay_codes], aspect='auto', cmap=payload_cmap,
               extent=[0, n, ymax, ymax + (ymax-ymin)*0.05])
    ax4.imshow([comm_codes], aspect='auto', cmap=comm_cmap,
               extent=[0, n, ymax + (ymax-ymin)*0.05, ymax + 2*(ymax-ymin)*0.05])
    ax4.set_ylim(ymin, ymax + 2*(ymax-ymin)*0.05)
    apply_plot_style(ax4, "Payload vs Downlink Rate", "Time Step", "Bytes/s")
    payload_handles = [
        Patch(color='lightgray', label='Off'),
        Patch(color='green',     label='Observing'),
        Patch(color='orange',    label='Blocked')
    ]
    comm_handles = [
        Patch(color='lightgray', label='Comm Off'),
        Patch(color='blue',      label='Downloading'),
        Patch(color='orange',    label='Stalled')
    ]
    ax4.legend(handles=payload_handles + comm_handles, loc='upper right', fontsize=8)

    # Panel 5: Power Gen vs Draw + Energy
    ax5 = axs[5]
    total_draw = (results['power_idle'] + results['power_cpu'] +
                  results['power_comms'] + results['power_payload'])
    ax5.plot(results['power_generated'], label="Generated", color='tab:cyan')
    ax5.plot(total_draw,                label="Total Draw", color='tab:gray')
    ymin, ymax = ax5.get_ylim()
    ax5.imshow([energy_codes], aspect='auto', cmap=energy_cmap,
               extent=[0, n, ymax, ymax + (ymax-ymin)*0.05])
    ax5.set_ylim(ymin, ymax + (ymax-ymin)*0.05)
    apply_plot_style(ax5, "Power Gen vs Draw", "Time Step", "Power (W)")
    gen_handles, gen_labels = ax5.get_legend_handles_labels()
    energy_handles = [
        Patch(color='lightcoral', label='Discharging'),
        Patch(color='lightgreen', label='Charging')
    ]
    ax5.legend(handles=gen_handles + energy_handles, fontsize=8)

    # Panel 6: Power Breakdown
    ax6 = axs[6]
    comp_p = np.vstack([results['power_idle'], results['power_cpu'],
                        results['power_comms'], results['power_payload']])
    ax6.stackplot(t, comp_p, labels=['Idle','CPU','Comms','Payload'], alpha=0.8)
    ax6.legend(loc='upper left', fontsize=8)
    apply_plot_style(ax6, "Power Draw Breakdown", "Time Step", "Power (W)")

    # Panel 7: Heat Breakdown
    ax7 = axs[7]
    comp_h = np.vstack([results['heat_solar'], results['heat_idle'],
                        results['heat_cpu'], results['heat_comms'],
                        results['heat_diff']])
    ax7.stackplot(t, comp_h, labels=['Solar','Idle','CPU','Comms','Imbalance'], alpha=0.8)
    ax7.legend(loc='upper left', fontsize=8)
    apply_plot_style(ax7, "Heat Breakdown", "Time Step", "Heat Rate (W)")

    # Panel 8: Heat Gen vs Radiated + Heat State
    ax8 = axs[8]
    ax8.plot(t, results['heat_generated'], label="Heat Gen", color='tab:orange')
    ax8.plot(t, results['heat_radiated'], label="Radiated", color='tab:blue')
    ymin, ymax = ax8.get_ylim()
    ax8.imshow([heat_codes], aspect='auto', cmap=heat_cmap,
               extent=[0, n, ymax, ymax + (ymax-ymin)*0.05])
    ax8.set_ylim(ymin, ymax + (ymax-ymin)*0.05)
    apply_plot_style(ax8, "Heat Gen vs Radiated", "Time Step", "Heat Rate (W)")
    hgen_handles, h_labels = ax8.get_legend_handles_labels()
    heat_handles = [
        Patch(color='lightblue', label='Cooling'),
        Patch(color='orange',     label='Heating')
    ]
    ax8.legend(handles=hgen_handles + heat_handles, fontsize=8)

    # Panel 9: POI & GS Timeline
    ax9 = axs[9]
    ax9.imshow(poi[np.newaxis,:], aspect='auto', cmap='Oranges',
               extent=[0, n, 0, 1], vmin=0, vmax=1)
    ax9.imshow(gs[np.newaxis,:], aspect='auto', cmap='Greys',
               extent=[0, n, 1, 2], vmin=0, vmax=1)
    ax9.set_yticks([0.5, 1.5]); ax9.set_yticklabels(['POI','GS'])
    ax9.set_xlabel("Time Step"); ax9.set_title("POI & GS Timeline")

    # Panel 10: Eclipse/Sunlight
    ax10 = axs[10]
    ax10.imshow(sun[np.newaxis,:], aspect='auto', cmap='YlOrBr', vmin=0, vmax=1)
    ax10.set_yticks([])
    ax10.set_xlabel("Time Step"); ax10.set_title("Eclipse (gray) / Sunlight (yellow)")

    # Panel 11: Reserved
    axs[11].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if return_fig:
        return fig
    else:
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def plot_simulation_dashboard_with_states_v2(results, return_fig=False):
    """
    3×4 Dashboard with integrated state overlays and legends.
    Panels:
      0: SoC + Battery State
      1: Temperature + Thermal Safety State
      2: Buffer (raw & processed) + Buffer States
      3: Data-Rates (payload, CPU, comms) + Their States
      4: Power Gen, Draw & Breakdown + Energy State
      5: Heat Gen, Radiated & Breakdown + Heat State
      6: POI & GS Timeline
      7: Eclipse/Sunlight
      8-11: Reserved
    """
    n = len(results['state_of_charge'])
    t = np.arange(n)

    # Core series
    soc      = results['state_of_charge']
    temp     = results['temperature']
    raw_buf  = results['raw_buffer']
    proc_buf = results['processed_buffer']
    payload_rate   = results['observed']
    processed_rate = results['processed_amt']
    downlink_rate  = results['downlinked']
    sun    = results['sun_flag']
    gs     = results['gs_flag']
    poi    = results['poi_flag']

    # States
    batt_state    = results['s_safety_E']
    therm_state   = results['s_safety_T']
    buffer_state  = results['s_buffer_raw']
    cpu_state     = results['s_cpu']
    payload_state = results['s_payload']
    comm_state    = results['s_comm']
    energy_state  = results['s_energy']
    heat_state    = results['s_heat']

    # Colormaps + maps
    batt_cmap = ListedColormap(['white','red','lightblue'])
    batt_map  = {'nominal':0,'low_battery':1,'full':2}

    therm_cmap = ListedColormap(['white','red','gray'])
    therm_map  = {'nominal':0,'high_temperature':1,'low_temperature':2}

    buf_cmap = ListedColormap(['white','orange','purple','lightgreen'])
    buf_map  = {'empty':0,'filling':1,'draining':2,'full':3}

    data_maps = {
        'cpu':     ({'off':0,'on':1,'forced_shutdown':2,'safety_shutdown':3}, ['white','green','gray','red']),
        'payload': ({'off':0,'observing':1,'blocked':2},            ['lightgray','green','orange']),
        'comm':    ({'off':0,'downloading':1,'stalled':2},           ['lightgray','blue','orange']),
    }

    energy_cmap = ListedColormap(['lightcoral','lightgreen'])
    energy_map  = {'discharging':0,'charging':1}

    heat_cmap = ListedColormap(['lightblue','orange'])
    heat_map  = {'cooling':0,'heating':1}

    # encode codes
    batt_codes   = [batt_map.get(s, 0) for s in batt_state]
    therm_codes  = [therm_map.get(s, 0) for s in therm_state]
    buf_codes    = [buf_map.get(s, 0)  for s in buffer_state]
    cpu_codes    = [data_maps['cpu'][0].get(s,0)     for s in cpu_state]
    pay_codes    = [data_maps['payload'][0].get(s,0) for s in payload_state]
    comm_codes   = [data_maps['comm'][0].get(s,0)    for s in comm_state]
    energy_codes = [energy_map[s] for s in energy_state]
    heat_codes   = [heat_map[s]   for s in heat_state]

    # figure
    fig, axs = plt.subplots(3,4,figsize=(20,12))
    axs = axs.flatten()
    fig.suptitle("Satellite System Simulation Dashboard", fontsize=18, fontweight='bold')

    def apply_style(ax,title,xlabel,ylabel):
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    # Panel 0: SoC + Battery
    ax0 = axs[0]
    ax0.plot(soc, color='tab:blue')
    ymin,ymax = ax0.get_ylim()
    ax0.imshow([batt_codes], aspect='auto', cmap=batt_cmap,
               extent=[0,n,ymax,ymax+(ymax-ymin)*0.05])
    ax0.set_ylim(ymin,ymax+(ymax-ymin)*0.05)
    apply_style(ax0, "Battery SoC", "Step","SoC (%)")
    batt_legs = [Patch(color=c,label=l) for c,l in zip(['white','red','lightblue'],['Nominal','Low Batt','Full'])]
    ax0.legend(handles=batt_legs,loc='upper right',fontsize=8)

    # Panel 1: Temp + Thermal
    ax1 = axs[1]
    ax1.plot(temp, color='tab:red')
    ymin,ymax = ax1.get_ylim()
    ax1.imshow([therm_codes], aspect='auto', cmap=therm_cmap,
               extent=[0,n,ymax,ymax+(ymax-ymin)*0.05])
    ax1.set_ylim(ymin,ymax+(ymax-ymin)*0.05)
    apply_style(ax1, "Temperature (K)", "Step","Temp")
    therm_legs = [Patch(color=c,label=l) for c,l in zip(['white','red','gray'],['Nominal','High','Low'])]
    ax1.legend(handles=therm_legs,loc='upper right',fontsize=8)

    # Panel 2: Buffers + States
    ax2 = axs[2]
    ax2.plot(raw_buf,  label="Raw Buffer",  color='tab:orange')
    ax2.plot(proc_buf, label="Proc Buffer", color='tab:purple')
    ymin,ymax = ax2.get_ylim()
    ax2.imshow([buf_codes], aspect='auto', cmap=buf_cmap,
               extent=[0,n,ymax,ymax+(ymax-ymin)*0.05])
    ax2.set_ylim(ymin,ymax+(ymax-ymin)*0.05)
    apply_style(ax2, "Buffers (Bytes)", "Step","Bytes")
    buf_legs = [Patch(color=c,label=l) for c,l in zip(['white','orange','purple','lightgreen'],['Empty','Filling','Draining','Full'])]
    ax2.legend(handles=buf_legs,loc='upper left',fontsize=8)

    # Panel 3: Data-Rates + States
    ax3 = axs[3]
    ax3.plot(payload_rate,   label="Obs Rate",     color='tab:orange')
    ax3.plot(processed_rate, label="Proc Rate",    color='tab:purple')
    ax3.plot(downlink_rate,  label="Downlink Rate",color='tab:cyan')
    ymin,ymax = ax3.get_ylim()
    ax3.imshow([pay_codes], aspect='auto', cmap=ListedColormap(data_maps['payload'][1]),
               extent=[0,n,ymax,ymax+(ymax-ymin)*0.03])
    ax3.imshow([cpu_codes], aspect='auto', cmap=ListedColormap(data_maps['cpu'][1]),
               extent=[0,n,ymax+(ymax-ymin)*0.03,ymax+2*(ymax-ymin)*0.03])
    ax3.imshow([comm_codes], aspect='auto', cmap=ListedColormap(data_maps['comm'][1]),
               extent=[0,n,ymax+2*(ymax-ymin)*0.03,ymax+3*(ymax-ymin)*0.03])
    ax3.set_ylim(ymin,ymax+3*(ymax-ymin)*0.03)
    apply_style(ax3, "Data Rates (B/s)", "Step","Bytes/s")
    pay_leg = [Patch(color=c,label=l) for c,l in zip(data_maps['payload'][1],data_maps['payload'][0].keys())]
    cpu_leg = [Patch(color=c,label=l) for c,l in zip(data_maps['cpu'][1],data_maps['cpu'][0].keys())]
    comm_leg= [Patch(color=c,label=l) for c,l in zip(data_maps['comm'][1],data_maps['comm'][0].keys())]
    ax3.legend(handles=pay_leg+cpu_leg+comm_leg, loc='upper right',fontsize=6)

    # Panel 4: Power Gen, Draw & Breakdown + Energy
    ax4 = axs[4]
    total_draw = results['power_idle']+results['power_cpu']+results['power_comms']+results['power_payload']
    ax4.plot(results['power_generated'], label="Gen",  color='tab:green')
    ax4.plot(total_draw,                label="Draw", color='tab:gray')
    comp_p = [results['power_idle'], results['power_cpu'], results['power_comms'], results['power_payload']]
    ax4.stackplot(t, *comp_p, labels=['Idle','CPU','Comms','Payload'], alpha=0.6)
    ymin,ymax = ax4.get_ylim()
    ax4.imshow([energy_codes], aspect='auto', cmap=energy_cmap,
               extent=[0,n,ymax,ymax+(ymax-ymin)*0.05])
    ax4.set_ylim(ymin,ymax+(ymax-ymin)*0.05)
    apply_style(ax4, "Power Gen, Draw & Breakdown", "Step","W")
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles=handles + [Patch(color='lightcoral',label='Discharging'), Patch(color='lightgreen',label='Charging')], loc='upper left', fontsize=8)

    # Panel 5: Heat Gen, Rad & Breakdown + Heat
    ax5 = axs[5]
    ax5.plot(results['heat_generated'], label="Gen",      color='tab:orange')
    ax5.plot(results['heat_radiated'],  label="Radiated", color='tab:blue')
    comp_h = [results['heat_solar'], results['heat_idle'], results['heat_cpu'], results['heat_comms'], results['heat_diff']]
    ax5.stackplot(t, *comp_h, labels=['Solar','Idle','CPU','Comms','Imbalance'], alpha=0.6)
    ymin,ymax = ax5.get_ylim()
    ax5.imshow([heat_codes], aspect='auto', cmap=heat_cmap,
               extent=[0,n,ymax,ymax+(ymax-ymin)*0.05])
    ax5.set_ylim(ymin,ymax+(ymax-ymin)*0.05)
    apply_style(ax5, "Heat Gen, Rad & Breakdown", "Step","W")
    h_handles, h_labels = ax5.get_legend_handles_labels()
    ax5.legend(handles=h_handles + [Patch(color='lightblue',label='Cooling'),Patch(color='orange',label='Heating')], loc='upper left', fontsize=8)

    # Panel 6: POI & GS Timeline
    ax6 = axs[6]
    ax6.imshow(poi[np.newaxis,:], aspect='auto', cmap='Oranges', extent=[0,n,0,1])
    ax6.imshow(gs[np.newaxis,:],  aspect='auto', cmap='Greys',   extent=[0,n,1,2])
    ax6.set_yticks([0.5,1.5]); ax6.set_yticklabels(['POI','GS'])
    apply_style(ax6, "POI & GS Timeline", "Step","")

    # Panel 7: Eclipse/Sunlight
    ax7 = axs[7]
    ax7.imshow(sun[np.newaxis,:], aspect='auto', cmap='YlOrBr', vmin=0, vmax=1)
    ax7.set_yticks([])
    apply_style(ax7, "Sunlight (yellow) / Eclipse", "Step","")

    for ax in axs[8:]:
        ax.axis('off')

    plt.tight_layout(rect=[0,0,1,0.96])
    if return_fig:
        return fig
    else:
        plt.show()



# === 3D Orbit + Animated State Dashboard ===
def animate_orbit_with_metrics(pos, results, Re, gs_ecef=None, poi_ecef=None):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 4, figure=fig)

    ax3d = fig.add_subplot(gs[:, :2], projection='3d')
    ax_soc = fig.add_subplot(gs[0, 2:])
    ax_temp = fig.add_subplot(gs[1, 2:])
    ax_cpu = fig.add_subplot(gs[2, 2:])
    ax_buf = fig.add_subplot(gs[3, 2:])

    ax3d.set_title("Animated 3D Orbit with Satellite Metrics", fontsize=14, fontweight='bold')
    ax3d.set_xlim([-2*Re, 2*Re])
    ax3d.set_ylim([-2*Re, 2*Re])
    ax3d.set_zlim([-2*Re, 2*Re])
    ax3d.set_xlabel('X [km]'); ax3d.set_ylabel('Y [km]'); ax3d.set_zlabel('Z [km]')
    ax3d.set_box_aspect([1, 1, 1])

    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x = Re * np.cos(u) * np.sin(v)
    y = Re * np.sin(u) * np.sin(v)
    z = Re * np.cos(v)
    ax3d.plot_surface(x, y, z, color='lightblue', alpha=0.3)

    # Add Ground Station and Point of Interest if provided
    if gs_ecef is not None:
        ax3d.scatter(*gs_ecef, color='green', s=50, label='Ground Station')
    if poi_ecef is not None:
        ax3d.scatter(*poi_ecef, color='red', s=50, label='Point of Interest')

    satellite_dot, = ax3d.plot([], [], [], 'ko', label='Satellite')
    orbit_line, = ax3d.plot([], [], [], 'k-', linewidth=1.5, alpha=0.7)

    soc_line, = ax_soc.plot([], [], color='tab:blue', linewidth=1.5)
    temp_line, = ax_temp.plot([], [], color='tab:red', linewidth=1.5)
    cpu_line, = ax_cpu.plot([], [], color='tab:green', linewidth=1.5)
    buf_line, = ax_buf.plot([], [], color='tab:orange', linewidth=1.5)

    time_array = np.arange(len(pos))
    ax_soc.set_xlim(time_array[0], time_array[-1])
    ax_temp.set_xlim(time_array[0], time_array[-1])
    ax_cpu.set_xlim(time_array[0], time_array[-1])
    ax_buf.set_xlim(time_array[0], time_array[-1])

    apply_plot_style(ax_soc, "Battery SoC", "%", ylim=(-5, 105))
    apply_plot_style(ax_temp, "Temperature", "K", ylim=(53, 443))
    apply_plot_style(ax_cpu, "CPU Load", "DMIPS", ylim=(0, max(1, results['cpu_load'].max())*1.1))
    apply_plot_style(ax_buf, "Data Buffer", "Bytes", ylim=(0, max(1, results['buffer'].max())*1.1))

    def init():
        satellite_dot.set_data([], [])
        satellite_dot.set_3d_properties([])
        orbit_line.set_data([], [])
        orbit_line.set_3d_properties([])
        soc_line.set_data([], [])
        temp_line.set_data([], [])
        cpu_line.set_data([], [])
        buf_line.set_data([], [])
        return satellite_dot, orbit_line, soc_line, temp_line, cpu_line, buf_line

    def update(frame):
        satellite_dot.set_data([pos[frame, 0]], [pos[frame, 1]])
        satellite_dot.set_3d_properties([pos[frame, 2]])
        orbit_line.set_data(pos[:frame+1, 0], pos[:frame+1, 1])
        orbit_line.set_3d_properties(pos[:frame+1, 2])

        soc_line.set_data(time_array[:frame+1], results['state_of_charge'][:frame+1])
        temp_line.set_data(time_array[:frame+1], results['temperature'][:frame+1])
        cpu_line.set_data(time_array[:frame+1], results['cpu_load'][:frame+1])
        buf_line.set_data(time_array[:frame+1], results['buffer'][:frame+1])

        return satellite_dot, orbit_line, soc_line, temp_line, cpu_line, buf_line

    ani = animation.FuncAnimation(fig, update, frames=len(pos), init_func=init, interval=20, blit=True)
    ax3d.legend()
    plt.tight_layout()
    plt.show()


def plot_orbit_flags(pos, vel, sun_flag, gs_flag, poi_flag, Re):
    X = pos[:, 0]
    Y = pos[:, 1]
    Vx = vel[:, 0]
    Vy = vel[:, 1]
    speed = np.linalg.norm(vel, axis=1)
    skip = max(1, len(X) // 20)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Orbital Analysis Dashboard", fontsize=14, fontweight='bold')

    # Orbit + Vectors
    axs[0, 0].plot(X, Y, color='black')
    axs[0, 0].quiver(X[::skip], Y[::skip], Vx[::skip], Vy[::skip], angles='xy', scale_units='xy', scale=1, color='blue')
    axs[0, 0].add_patch(plt.Circle((0, 0), Re, fill=False, color='gray'))
    axs[0, 0].set_aspect('equal')
    apply_plot_style(axs[0, 0], "Orbit + Velocity Vectors")

    # Speed profile
    axs[0, 1].plot(speed / 1000, color='purple')  # convert to km/s
    apply_plot_style(axs[0, 1], "Speed Profile", "Speed (km/s)")

    # Sun exposure
    axs[0, 2].scatter(X, Y, c=sun_flag, cmap='YlOrRd', s=10)
    axs[0, 2].add_patch(plt.Circle((0, 0), Re, fill=False, color='gray'))
    axs[0, 2].set_aspect('equal')
    apply_plot_style(axs[0, 2], "Sun Exposure Map")

    # Ground Station visibility
    axs[1, 0].scatter(X, Y, c=gs_flag, cmap='viridis', s=10)
    axs[1, 0].add_patch(plt.Circle((0, 0), Re, fill=False, color='gray'))
    axs[1, 0].set_aspect('equal')
    apply_plot_style(axs[1, 0], "Ground Station Visibility")

    # POI visibility
    axs[1, 1].scatter(X, Y, c=poi_flag, cmap='coolwarm', s=10)
    axs[1, 1].add_patch(plt.Circle((0, 0), Re, fill=False, color='gray'))
    axs[1, 1].set_aspect('equal')
    apply_plot_style(axs[1, 1], "Point of Interest Visibility")

    # Visibility flags over time
    axs[1, 2].plot(sun_flag, label='Sun', color='gold')
    axs[1, 2].plot(gs_flag, label='GS', color='green')
    axs[1, 2].plot(poi_flag, label='PoI', color='red')
    apply_plot_style(axs[1, 2], "Visibility Flags Over Time")
    axs[1, 2].legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_radar_chart(subscores, labels=None, title='CubeSat Performance Radar', return_fig=False):
    """
    Plot a radar chart of normalized subscores.
    
    Args:
      - subscores: dict of metric name → score in [0,1]
      - labels:    optional list of display labels; must have len == number of metrics
      - title:     title of the plot
      - return_fig: if True, returns the matplotlib Figure instead of plt.show()
    """
    metrics = list(subscores.keys())
    N = len(metrics)
    # 1) compute the angles for each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # close the loop

    # 2) collect the values, closing the loop
    values = [subscores[m] for m in metrics]
    values = np.array(values + [values[0]])

    # 3) set up figure
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # 4) assign tick labels only if length matches
    if labels is not None and len(labels) == N:
        tick_labels = labels
    else:
        tick_labels = metrics
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tick_labels)

    ax.set_title(title, y=1.1)
    ax.set_ylim(0, 1)

    if return_fig:
        return fig
    else:
        plt.show()

  
import pandas as pd

import os
from tqdm import tqdm

import cProfile
import pickle
import pstats

# Create folder if it doesn't exist
output_dir = "dashboards"
os.makedirs(output_dir, exist_ok=True)
output_dir = "scores"
os.makedirs(output_dir, exist_ok=True)


def simulate_all_configs():
    """
    Your existing loop over DataFrame rows:
        for idx, row in df.iterrows():
            # build config, simulate, score...
    """
    # Initialize mission
    mission = MissionConfig(
        altitude=500e3,
        inclination=98.0,
        time_resolution=1.0,
        n_orbits=15
    )
    
    #mission.addGS(52.0, 5.0)
    #mission.addGS(0.0, -75.0)
    #mission.addPOI(30.0, 90.0)
    #mission.addPOI(-10.0, 100.0)
    
    GS_coords = [
    ( 60.0,   0.0),   # 60N,   0E
    ( 60.0,  72.0),   # 60N,  72E
    ( 60.0, 144.0),   # 60N, 144E
    (  -60.0, -144.0),# 60S, 144W
    ( -60.0,  -72.0), # 60S,  72W
    ]

    for lat, lon in GS_coords:
        mission.addGS(lat, lon)
        #print(f"Added GS at  lat={lat:6.2f}, lon={lon:6.2f}")


    # 3 POIs: again within 98 latitude, spaced to sample different continents
    POI_coords = [
        ( 30.0,  10.0),   # 30N,  10E (e.g. Mediterranean)
        ( -30.0, 100.0),  # 30S, 100E (e.g. Australia)
        (  10.0, -50.0),  # 10N,  50W (e.g. Atlantic)
    ]

    for lat, lon in POI_coords:
        mission.addPOI(lat, lon)
        #print(f"Added POI at lat={lat:6.2f}, lon={lon:6.2f}")

    # Pre-compute geometry and flags
    pos, vel, sun_flag, gs_flag, poi_flag = mission.compute()
    n = len(pos)

    # Load configurations
    df = pd.read_csv('cubesat_productline.csv')  # or your dataframe source

    #plot_orbit_flags(pos, vel, sun_flag, gs_flag, poi_flag, Re)

    scores = []
    configs_used = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Only test every Nth or specific subset if large
        #if idx % 30 != 0:
        if idx % 100 != 0:
            continue
        
        config = SatelliteConfig(row)
        sim = SatelliteSimulator(config, mission)
        sim.logger.setLevel(logging.DEBUG)     # only anomalies


        # Run simulation with invariant checks inside simulate()
        try:
            results = sim.simulate(pos, sun_flag, gs_flag, poi_flag, debug=False)
        except AssertionError as ae:
            print(f"[INVARIANT FAILED] Config {idx}: {ae}")
            continue
        except Exception as e:
            print(f"[ERROR] Simulation failed for config {idx}: {e}")
            continue

        # Post-simulation assertions
        soc = results['state_of_charge']
        temp = results['temperature']
        cpu = results['cpu_load']
        raw = results['raw_buffer']
        proc = results['processed_buffer']
        down = results['downlinked']

        assert len(soc) == n, "Length mismatch: soc"
        assert np.all((soc >= 0) & (soc <= 100)), "SoC out of [0,100]"
        assert np.all(np.isfinite(temp)) and np.all(temp > 0), "Invalid temperature"
        assert np.all((cpu >= 0) & (cpu <= config.cpu_dmips)), "CPU load out of bounds"
        assert np.all(raw >= 0), "Raw buffer negative"
        assert np.all(proc >= 0), "Processed buffer negative"
        assert np.all(down >= 0), "Downlinked negative"

        # Compute scoring
        try:
            final_score, subscores = compute_score(
                pos, results, config,
                sun_flag, gs_flag, poi_flag, mission,
                safe_min=260, safe_max=330,
                target_passes=4, target_duration=600,
                required_volume=1e9, data_budget=2e9,
                mass_ref=1.0,        # target mass = 0.5kg
                mass_exponent=2.0    # square-law penalty
            )
            #print(final_score)
            #print(subscores)
        except AssertionError as ae:
            print(f"[SCORE INVARIANT FAILED] Config {idx}: {ae}")
            continue
        except Exception as e:
            print(f"[ERROR] Scoring failed for config {idx}: {e}")
            continue

        # Post-scoring asserts
        assert 0.0 <= final_score <= 100.0, f"Final score out of bounds: {final_score}"
        for key, val in subscores.items():
            assert 0.0 <= val <= 1.0, f"Subscore {key} out of [0,1]: {val}"

        scores.append(final_score)
        configs_used.append(idx)
        
        #print(f"[INFO] Config {idx} score : {final_score} saving dashboard...")
        #fig = plot_simulation_dashboard_with_states_v2(results, return_fig=True)
        #fig.savefig(f"dashboards/simulation_dashboard_{idx}.png", dpi=300)
        #plt.close(fig)
        #
        #fig = plot_radar_chart(
        #    subscores,
        #    labels=[
        #      "Power", "Thermal", "OBC", "Comm", "Payload"
        #    ],
        #    title="CubeSat Configuration Performance", return_fig=True
        #)
        #fig.savefig(f"scores/simulation_scores_{idx}.png", dpi=300)
        #plt.close(fig)
        #
    results_df = df.loc[configs_used].reset_index(drop=True)
    results_df['score'] = scores
    results_df.to_csv('satellite_config_scores.csv', index=False)
    print("Results saved to satellite_config_scores.csv")

if __name__ == '__main__':
     # 1) Profile the simulation sweep and save stats to 'prof.out'
    simulate_all_configs()
    #cProfile.run("simulate_all_configs()", "prof.out")