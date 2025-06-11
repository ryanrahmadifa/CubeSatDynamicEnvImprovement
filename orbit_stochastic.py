import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from pprint import pprint

# SatelliteConfig adapted for real satellite parameters
class SatelliteConfig:
    def __init__(self, row):
        # === Communication ===
        self.comm_band = row['Communication_Communication band']
        self.frequency = row['Communication_Frequency [GHz]']
        self.beamwidth = row['Communication_Beamwidth [deg]']
        self.comm_d_r = row['Communication_D_r [m]']
        self.required_ebn0 = row['Communication_Required Eb/N0 [dB]']
        self.comm_efficiency = row['Communication_Efficiency']
        self.comm_d_antenna = row['Communication_D_antenna [m]']
        self.comm_mass = row['Communication_Mass_communication [kg]']
        self.downlink_rate = row['Communication_Data Rate [bps]']
        
        k_boltzmann = 1.38064852e-23  # J/K
        temperature = 290  # K
        bandwidth = self.downlink_rate  # Hz, simplifié
        noise_power = k_boltzmann * temperature * bandwidth
        eb_n0_linear = 10 ** (self.required_ebn0 / 10)
        self.comms_power_max = (noise_power * eb_n0_linear) / self.comm_efficiency

        # === Solar Array ===
        self.solar_material = row['SolarArray_Solar Material']
        self.solar_efficiency = row['SolarArray_Efficiency']
        self.specific_power = row['SolarArray_Specific Power [W/kg]']
        self.solar_area = row['SolarArray_A_solar [m^2]']
        self.solar_power_max = row['SolarArray_P_solar_max [W]']
        self.solar_mass = row['SolarArray_Mass_solar_array [kg]']
        self.solar_constant = 1361  # [W/m^2], constant

        # === Battery ===
        self.battery_type = row['Battery_Battery type']
        self.n_cell = row['Battery_Number of cell']
        self.battery_diameter = row['Battery_Diameter [m]']
        self.battery_length = row['Battery_Length [m]']
        self.volumetric_energy = row['Battery_Volumetric Energy [Wh/m³]']
        self.specific_energy = row['Battery_Specific Energy [Wh/kg]']
        self.battery_capacity = row['Battery_Capacity [Wh]']
        self.battery_mass = row['Battery_Mass_battery [kg]']
        self.battery_voltage = 3.7 * self.n_cell  # simple model

        # === Radiator ===
        self.radiator_material = row['Radiator_Radiator material']
        self.radiative_area = row['Radiator_A_radiator [m2]']
        self.rho_radiator = row['Radiator_rho [kg/m³]']
        self.emissivity = row['Radiator_emissivity']
        self.radiator_thickness = row['Radiator_thickness [m]']
        self.radiator_q_rad = row['Radiator_Q_rad [W/K⁴]']
        self.radiator_mass = row['Radiator_Mass_radiator [kg]']

        # === CPU Configuration (from OBC_data) ===
        self.cpu_dmips = row['OBC_cpu_DMIPS']                                # DMIPS max
        self.cpu_power_idle = row['OBC_cpu_power_idle']  
        self.cpu_power_max = row['OBC_power max'] 
        self.cpu_throughput_per_watt = row['OBC_Throughtput per W']          # DMIPS per W
        self.cpu_processing_efficiency = row['OBC_cpu efficiency']           # Bytes per DMIPS per s
        self.processing_load_per_sec = (self.cpu_power_max - self.cpu_power_idle) * self.cpu_throughput_per_watt * 0.8  # si tu ajoutes ce paramètre
        self.processing_output_rate = 0.5e6                                   # Can be static or computed if available
        self.cpu_recovery_rate = row['OBC_recovery rate']                    # DMIPS/s cooldown rate

        # === Payload ===
        self.payload_power = 2.0  # [W], typical

        # === Data ===
        self.data_rate = 2e6  # Bytes/sec during observation
        #self.buffer_capacity = 2e9  # Bytes (for simulation limit)
        
         # === CubeSat ===
        self.total_mass = row['CubeSat mass']
        self.absorptivity = 0.8  # default if not provided
        self.internal_heat = 5.0
        self.thermal_mass = self.total_mass * 800  # J/K
        self.initial_temp = 300.0  # [K]

        # === Idle Power ===
        self.idle_power = 0.1 + self.cpu_power_idle   # [W], assumed baseline
        
        #config_dict = self.__dict__
        #print("\n[SatelliteConfig] Full Configuration Parameters:")
        #pprint(config_dict, sort_dicts=False)
        #print("-" * 70)


# === Orbit Class ===
class Orbit:
    def __init__(self, mu, inclination_deg, x0, y0, z0, vx0, vy0, vz0):
        self.mu = mu
        self.inclination_deg = inclination_deg
        self.state = np.array([x0, y0, z0, vx0, vy0, vz0])

    def apply_inclination(self, vec):
        inc = np.deg2rad(self.inclination_deg)
        R = np.array([[1, 0, 0], [0, np.cos(inc), -np.sin(inc)], [0, np.sin(inc), np.cos(inc)]])
        return R @ vec

    def propagate(self, T, N):
        dt = T / N
        state = self.state.copy()
        pos = np.zeros((N, 3))
        vel = np.zeros((N, 3))

        def deriv(s):
            r = np.linalg.norm(s[:3])
            a = -self.mu * s[:3] / r**3
            return np.hstack((s[3:], a))

        for i in range(N):
            pos[i], vel[i] = self.apply_inclination(state[:3]), self.apply_inclination(state[3:])
            k1 = deriv(state)
            k2 = deriv(state + 0.5 * dt * k1)
            k3 = deriv(state + 0.5 * dt * k2)
            k4 = deriv(state + dt * k3)
            state += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return pos, vel

# === Ground Station Class ===
class GroundStation:
    def __init__(self, lat_deg, lon_deg, Re, elev_mask_deg=10.0):
        self.lat = lat_deg
        self.lon = lon_deg
        self.ecef = self.latlon_to_ecef(lat_deg, lon_deg, Re)
        self.elev_mask_deg = elev_mask_deg

    def latlon_to_ecef(self, lat_deg, lon_deg, Re):
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        x = Re * np.cos(lat) * np.cos(lon)
        y = Re * np.cos(lat) * np.sin(lon)
        z = Re * np.sin(lat)
        return np.array([x, y, z])

    def compute_visibility(self, pos):
        v2gs = pos - self.ecef
        norm_v2gs = np.linalg.norm(v2gs, axis=1)
        gs_hat = self.ecef / np.linalg.norm(self.ecef)
        dot_gs = v2gs @ gs_hat
        cos_el = dot_gs / norm_v2gs
        sin_el = np.sin(np.deg2rad(self.elev_mask_deg))
        gs_raw = (cos_el - sin_el) / (1 - sin_el)
        return np.clip(gs_raw, 0.0, 1.0)

# === Point of Interest Class ===
class PointOfInterest:
    def __init__(self, lat_deg, lon_deg, Re, elev_mask_deg=10.0):
        self.lat = lat_deg
        self.lon = lon_deg
        self.ecef = self.latlon_to_ecef(lat_deg, lon_deg, Re)
        self.elev_mask_deg = elev_mask_deg

    def latlon_to_ecef(self, lat_deg, lon_deg, Re):
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        x = Re * np.cos(lat) * np.cos(lon)
        y = Re * np.cos(lat) * np.sin(lon)
        z = Re * np.sin(lat)
        return np.array([x, y, z])

    def compute_visibility(self, pos):
        v2gs = pos - self.ecef
        norm_v2gs = np.linalg.norm(v2gs, axis=1)
        gs_hat = self.ecef / np.linalg.norm(self.ecef)
        dot_gs = v2gs @ gs_hat
        cos_el = dot_gs / norm_v2gs
        sin_el = np.sin(np.deg2rad(self.elev_mask_deg))
        gs_raw = (cos_el - sin_el) / (1 - sin_el)
        return np.clip(gs_raw, 0.0, 1.0)


# === Mission Configuration ===
class MissionConfig:
    def __init__(self, poi_lat, poi_lon, gs_lat, gs_lon, altitude_km=500.0):
        self.mu = 398600.4418
        self.Re = 6371.0

        self.hp = altitude_km
        self.ha = altitude_km
        self.inclination_deg = 0.0  # Equatorial orbit

        self.rp = self.Re + self.hp
        self.ra = self.Re + self.ha
        self.a = 0.5 * (self.rp + self.ra)
        self.e = (self.ra - self.rp) / (self.ra + self.rp)
        self.orbit_period = 2 * np.pi * np.sqrt(self.a ** 3 / self.mu)

        self.x0, self.y0, self.z0 = self.rp, 0.0, 0.0
        self.vx0 = 0.0
        self.vy0 = np.sqrt(self.mu / self.rp)
        self.vz0 = 0.0

        self.orbit = Orbit(self.mu, self.inclination_deg, self.x0, self.y0, self.z0,
                           self.vx0, self.vy0, self.vz0)

        self.gs = GroundStation(gs_lat, gs_lon, self.Re)
        self.poi = PointOfInterest(poi_lat, poi_lon, self.Re)

        self.dt = self.orbit_period / 5000
        self.sun_angle_deg = 180.0

    def compute_sun_flag(self, pos):
        phi_sun = np.deg2rad(self.sun_angle_deg)
        sun_vec = np.array([np.cos(phi_sun), np.sin(phi_sun), 0])
        pos_unit = pos / np.linalg.norm(pos, axis=1)[:, None]
        return np.clip(pos_unit @ sun_vec, 0.0, 1.0)

import numpy as np

class SatelliteSimulator:
    def __init__(self, config, mission):
        self.config = config
        self.mission = mission

    #def simulate(self, pos, sun_flag, gs_flag, poi_flag):
    #    n = len(pos)
    #    dt = self.mission.dt
    #
    #    soc = np.zeros(n)
    #    temperature = np.zeros(n)
    #    raw_buffer = np.zeros(n)
    #    processed_buffer = np.zeros(n)
    #    cpu_load = np.zeros(n)
    #
    #    soc[0] = 100.0
    #    temperature[0] = self.config.initial_temp
    #
    #    for i in range(1, n):
    #        # Power generation from solar panels
    #        power_generated = self.config.solar_area * self.config.solar_efficiency * self.config.solar_constant * sun_flag[i]
    #
    #        # === Power Draw Sources ===
    #        idle_power_draw = self.config.idle_power
    #        cpu_power_draw = 0.0
    #        comms_power_draw = 0.0
    #
    #        if soc[i - 1] > 20:
    #            # Observation: add raw data
    #            raw_buffer[i] = raw_buffer[i-1] + self.config.data_rate * poi_flag[i] * dt
    #
    #            # CPU processing only if there's raw data
    #            if raw_buffer[i] > 0 and poi_flag[i - 1] > 0:
    #                cpu_load[i] = max(0.0, cpu_load[i-1] + (self.config.processing_load_per_sec - self.config.cpu_recovery_rate) * dt)
    #                cpu_power_draw = self.config.processing_load_per_sec * 0.6
    #                cpu_load[i] = min(cpu_load[i], self.config.cpu_dmips)
    #            else:
    #                cpu_load[i] = max(0.0, cpu_load[i-1] - self.config.cpu_recovery_rate * dt)
    #                cpu_power_draw = self.config.processing_load_per_sec * 0.1
    #                
    #            # Process data from raw buffer into processed buffer
    #            processing_capacity = cpu_load[i] * self.config.cpu_processing_efficiency * dt
    #            processed = min(raw_buffer[i], processing_capacity)
    #            raw_buffer[i] -= processed
    #            processed_buffer[i] = processed_buffer[i-1] + processed
    #
    #            # Downlink processed data
    #            downlinked = min(self.config.downlink_rate * gs_flag[i] * dt, raw_buffer[i])
    #            raw_buffer[i] -= downlinked
    #        
    #        else:
    #            cpu_load[i] = max(0.0, cpu_load[i-1] - self.config.cpu_recovery_rate * dt)
    #            raw_buffer[i] = raw_buffer[i-1]
    #
    #        # Comms draw
    #        comms_power_draw = (gs_flag[i] + poi_flag[i]) * 2.0 * 0.6
    #        total_power_draw = idle_power_draw + cpu_power_draw + comms_power_draw
    #
    #        # Battery update
    #        battery_energy_joules = self.config.battery_capacity * 3600
    #        delta_soc = (power_generated - total_power_draw) * dt / battery_energy_joules * 100.0
    #        soc[i] = np.clip(soc[i-1] + delta_soc, 0.0, 100.0)
    #
    #        # Temperature evolution
    #        temp_prev = temperature[i-1]
    #        internal_heat = (
    #            power_generated * (1 - self.config.solar_efficiency) +
    #            idle_power_draw * 0.9 +
    #            cpu_power_draw * 0.9 +
    #            comms_power_draw * 0.8 +
    #            abs(power_generated - total_power_draw) * 0.1
    #        )
    #        net_heat = (self.config.absorptivity * power_generated +
    #                    internal_heat -
    #                    self.config.emissivity * 5.67e-8 * self.config.radiative_area * (temp_prev ** 4))
    #        temperature[i] = temp_prev + (net_heat * dt) / self.config.thermal_mass
    #
    #    return {
    #        'state_of_charge': soc,
    #        'temperature': temperature,
    #        'cpu_load': cpu_load,
    #        'raw_buffer': raw_buffer,
    #        'processed_buffer': processed_buffer
    #    }
        
    def simulate(self, pos, sun_flag, gs_flag, poi_flag):
        n = len(pos)
        dt = self.mission.dt

        soc = np.zeros(n)
        temperature = np.zeros(n)
        raw_buffer = np.zeros(n)
        processed_buffer = np.zeros(n)
        cpu_load = np.zeros(n)
        downlinked = np.zeros(n)

        soc[0] = 100.0
        temperature[0] = self.config.initial_temp

        for i in range(1, n):
            # Power generation from solar panels
            power_generated = self.config.solar_area * self.config.solar_efficiency * self.config.solar_constant * sun_flag[i]

            # === Power Draw Sources ===
            idle_power_draw = self.config.idle_power
            cpu_power_draw = 0.0
            comms_power_draw = 0.0

            if soc[i-1] > 20:
                # Observation: add raw data
                raw_buffer[i] = raw_buffer[i-1] + self.config.data_rate * poi_flag[i] * dt

                # CPU processing only if there's raw data
                if raw_buffer[i] > 0:
                    cpu_load[i] = np.clip(cpu_load[i-1] + self.config.processing_load_per_sec * dt, 0, self.config.cpu_dmips)            
         
                else:
                    cpu_load[i] = np.clip(cpu_load[i-1] - self.config.cpu_recovery_rate * dt, 0, self.config.cpu_dmips)

                # Process data from raw buffer into processed buffer
                processing_capacity = min(raw_buffer[i], cpu_load[i] * self.config.cpu_processing_efficiency * dt)
                raw_buffer[i] -= processing_capacity
                processed_buffer[i] = processed_buffer[i-1] + processing_capacity

                # Downlink processed data
                downlinked[i] = min(self.config.downlink_rate * gs_flag[i] * dt, processed_buffer[i])
                processed_buffer[i] -= downlinked[i]
            else:
                cpu_load[i] = np.clip(cpu_load[i-1] - self.config.cpu_recovery_rate * dt, 0, self.config.cpu_dmips)
                raw_buffer[i] = raw_buffer[i-1]
                
            cpu_power_draw = self.config.cpu_power_idle + (cpu_load[i] / self.config.cpu_dmips) * (self.config.cpu_power_max - self.config.cpu_power_idle)
            comms_power_draw = poi_flag[i] * self.config.payload_power + gs_flag[i] * self.config.comms_power_max * self.config.comm_efficiency
            total_power_draw = idle_power_draw + cpu_power_draw + comms_power_draw

            # Battery update
            battery_energy_joules = self.config.battery_capacity * 3600
            delta_soc = (power_generated - total_power_draw) * dt / battery_energy_joules * 100.0
            soc[i] = np.clip(soc[i-1] + delta_soc, 0.0, 100.0)

            # Temperature evolution
            temp_prev = temperature[i-1]
            internal_heat = (
                power_generated * (1 - self.config.solar_efficiency) +
                idle_power_draw * 0.9 +
                cpu_power_draw * 0.9 +
                comms_power_draw * 0.8 +
                abs(power_generated - total_power_draw) * 0.1
            )
            net_heat = (self.config.absorptivity * power_generated +
                        internal_heat -
                        self.config.emissivity * 5.67e-8 * self.config.radiative_area * (temp_prev ** 4))
            temperature[i] = temp_prev + (net_heat * dt) / self.config.thermal_mass
            
            # === DEBUG PRINT BLOCK ===
            #if i % 500 == 0:  # print every 100 steps
            #    print(f"[DEBUG] Step {i}")
            #    print(f"  Power Generated     : {power_generated:.2f} W")
            #    print(f"  CPU Power Draw      : {cpu_power_draw:.2f} W")
            #    print(f"  Comms Power Draw    : {comms_power_draw:.2f} W")
            #    print(f"  Total Power Draw    : {total_power_draw:.2f} W")
            #    print(f"  Raw Buffer          : {raw_buffer[i]:.2f} Bytes")
            #    print(f"  Processed Buffer    : {processed_buffer[i]:.2f} Bytes")
            #    print(f"  CPU Load            : {cpu_load[i]:.2f} DMIPS")
            #    print(f"  SoC                 : {soc[i]:.2f} %")
            #    print(f"  Temperature         : {temperature[i]:.2f} K")
            #    print("-" * 60)

        return {
            'state_of_charge': soc,
            'temperature': temperature,
            'cpu_load': cpu_load,
            'raw_buffer': raw_buffer,
            'processed_buffer': processed_buffer,
            'downlinked': downlinked
        }    

def compute_score(results, config):
    soc = results['state_of_charge']
    cpu_load = results['cpu_load']
    raw_buffer = results['raw_buffer']
    processed_buffer = results['processed_buffer']
    temperature = results['temperature']
    downlinked = results['downlinked']

    # Énergie
    avg_soc = np.mean(soc)
    min_soc = np.min(soc)
    energy_score = (avg_soc + min_soc) / 2 / 100

    # CPU
    avg_cpu_load = np.mean(cpu_load) / config.cpu_dmips

    # Traitement
    final_raw = raw_buffer[-1]
    final_processed = processed_buffer[-1]
    processing_efficiency = final_processed / (final_raw + final_processed + 1e-6)

    # Transmission buffer
    transmission_efficiency = 1.0 - (processed_buffer[-1] / (final_processed + 1e-6))

    # Données téléchargées normalisées (par exemple sur 1 Go max)
    total_downlinked = np.sum(downlinked)
    max_possible = 4e9  # 4 GB
    downlink_score = total_downlinked / max_possible
    downlink_score = np.clip(downlink_score, 0, 1)

    # Température
    overheat_penalty = np.sum(temperature > 340) / len(temperature)

    # Score final
    score = (
        0.20 * energy_score +
        0.20 * avg_cpu_load +
        0.15 * processing_efficiency +
        0.15 * transmission_efficiency +
        0.20 * downlink_score -
        0.10 * overheat_penalty  
    )

    final_score = score / (config.total_mass + 1e-6)  # pour éviter division par zéro

    return max(0, min(final_score, 10.0))  # éventuellement plafonner le score


# === Common Plotting Style ===
def apply_plot_style(ax, title, ylabel=None, ylim=None):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time Step", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if ylim is not None:
        if isinstance(ylim, tuple) and len(ylim) == 2:
            ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=9)


# === System Simulation Dashboard (Dual Buffers) ===
def plot_simulation_dashboard(results, return_fig=False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Satellite System Simulation Results", fontsize=14, fontweight='bold')

    axs[0, 0].plot(results['state_of_charge'], color="tab:blue")
    apply_plot_style(axs[0, 0], "Battery State of Charge", "SoC (%)", ylim=(-5, 105))

    axs[0, 1].plot(results['temperature'], color="tab:red")
    apply_plot_style(axs[0, 1], "Satellite Temperature", "Temperature (K)", ylim=(
            min(0, min(results['temperature'].min(), results['temperature'].min())) * 0.9,
            max(1, max(results['temperature'].max(), results['temperature'].max())) * 1.1
        ))

    axs[1, 0].plot(results['raw_buffer'], color="tab:orange", label="Raw Data")
    axs[1, 0].plot(results['processed_buffer'], color="tab:purple", label="Processed Data")
    axs[1, 0].legend(fontsize=9)
    apply_plot_style(
        axs[1, 0],
        "Buffer Usage",
        "Buffer (Bytes)",
        ylim=(
            min(0, min(results['raw_buffer'].min(), results['processed_buffer'].min())) * 0.9,
            max(1, max(results['raw_buffer'].max(), results['processed_buffer'].max())) * 1.1
        )
    )

    axs[1, 1].plot(results['cpu_load'], color="tab:green")
    apply_plot_style(axs[1, 1], "CPU Load", "Load (DMIPS)", ylim=(0, max(1, results['cpu_load'].max()) * 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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



# === Orbit + Flags Dashboard ===
def plot_orbit_flags(pos, vel, sun_flag, gs_flag, poi_flag, Re):
    X = pos[:, 0]; Y = pos[:, 1]
    Vx = vel[:, 0]; Vy = vel[:, 1]
    speed = np.linalg.norm(vel, axis=1)
    skip = max(1, len(X) // 20)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Orbital Analysis Dashboard", fontsize=14, fontweight='bold')

    axs[0,0].plot(X, Y, color='black')
    axs[0,0].quiver(X[::skip], Y[::skip], Vx[::skip], Vy[::skip], angles='xy', scale_units='xy', scale=1, color='blue')
    axs[0,0].add_patch(plt.Circle((0, 0), Re, fill=False, color='gray'))
    axs[0,0].set_aspect('equal')
    apply_plot_style(axs[0,0], "Orbit + Velocity Vectors")

    axs[0,1].plot(speed, color='purple')
    apply_plot_style(axs[0,1], "Speed Profile", "Speed (km/s)")

    axs[0,2].scatter(X, Y, c=sun_flag, cmap='YlOrRd', s=10)
    axs[0,2].add_patch(plt.Circle((0, 0), Re, fill=False, color='gray'))
    axs[0,2].set_aspect('equal')
    apply_plot_style(axs[0,2], "Sun Exposure Map")

    axs[1,0].scatter(X, Y, c=gs_flag, cmap='viridis', s=10)
    axs[1,0].add_patch(plt.Circle((0, 0), Re, fill=False, color='gray'))
    axs[1,0].set_aspect('equal')
    apply_plot_style(axs[1,0], "Ground Station Visibility")

    axs[1,1].scatter(X, Y, c=poi_flag, cmap='coolwarm', s=10)
    axs[1,1].add_patch(plt.Circle((0, 0), Re, fill=False, color='gray'))
    axs[1,1].set_aspect('equal')
    apply_plot_style(axs[1,1], "Point of Interest Visibility")

    axs[1,2].plot(sun_flag, label='Sun', color='gold')
    axs[1,2].plot(gs_flag, label='GS', color='green')
    axs[1,2].plot(poi_flag, label='PoI', color='red')
    apply_plot_style(axs[1,2], "Visibility Flags Over Time")
    axs[1,2].legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

import pandas as pd

# Load the updated CubeSat configuration file
file_path = "./cubesat_productline.csv"
df = pd.read_csv(file_path)

import os
from tqdm import tqdm

# Create folder if it doesn't exist
output_dir = "dashboards"
os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    # Define equatorial locations
    poi_lat, poi_lon = 0.0, 45.0     # Example: Gulf of Guinea
    gs_lat, gs_lon = 0.0, -75.0      # Example: Ecuador

    mission = MissionConfig(poi_lat, poi_lon, gs_lat, gs_lon, altitude_km=500.0)

    num_orbits = 1
    pos, vel = mission.orbit.propagate(mission.orbit_period * num_orbits, 5000)

    sun_flag = mission.compute_sun_flag(pos)
    gs_flag = mission.gs.compute_visibility(pos)
    poi_flag = mission.poi.compute_visibility(pos)

    #print(f"GS visibility: {np.mean(gs_flag > 0) * 100:.2f}%")
    #print(f"PoI visibility: {np.mean(poi_flag) * 100:.2f}%")
    #print(f"Sun visibility: {np.mean(sun_flag > 0) * 100:.2f}%")

    #plot_orbit_flags(pos, vel, sun_flag, gs_flag, poi_flag, mission.Re)


    scores = []
    configs_used = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if idx % 100 == 0:
                config = SatelliteConfig(row)
                sim = SatelliteSimulator(config, mission)
                results = sim.simulate(pos, sun_flag, gs_flag, poi_flag)
                
                score = compute_score(results, config)

                # Store the score and config index
                scores.append(score)
                configs_used.append(idx)

                # Optional: Save dashboard every N steps
                if score > 2.5:
                    print(f"[INFO] Config {idx} score : {score} saving dashboard...")
                    fig = plot_simulation_dashboard(results, return_fig=True)
                    fig.savefig(f"dashboards/simulation_dashboard_{idx}.png", dpi=300)
                    plt.close(fig)

        except Exception as e:
            print(f"[ERROR] Failed to simulate config {idx}: {e}")
            continue

    # Create the final DataFrame and export
    score_df = df.loc[configs_used].copy()
    score_df['Score'] = scores

    # Save to CSV
    score_df.to_csv("satellite_config_scores.csv", index=False)
    print("[INFO] Score CSV saved as 'satellite_config_scores.csv'")