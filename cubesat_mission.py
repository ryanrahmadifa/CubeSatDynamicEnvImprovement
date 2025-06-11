import numpy as np

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
        self.comms_power_max = (noise_power * eb_n0_linear) / self.comm_efficiency * 1e6
        self.comms_power_idle = self.comms_power_max / 10

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
        self.payload_power_max = 2.0  # [W], typical
        self.payload_power_idle = 0.3 # idle

        # === Data ===
        self.data_rate = 1e5  # Bytes/sec during observation
        self.buffer_capacity = 1e9  # Bytes (for simulation limit)
        
         # === CubeSat ===
        self.total_mass = row['CubeSat mass']
        self.absorptivity = 0.8  # default if not provided
        self.internal_heat = 5.0
        self.thermal_mass = self.total_mass * 800  # J/K
        self.initial_temp = 300.0  # [K]
        self.max_temp = 340
        self.min_temps = 260
        self.soc_limit = 0

        # === Idle Power ===
        self.idle_power = 0.1 + self.cpu_power_idle   # [W], assumed baseline
        
        #config_dict = self.__dict__
        #print("\n[SatelliteConfig] Full Configuration Parameters:")
        #pprint(config_dict, sort_dicts=False)
        #print("-" * 70)

# Constants
Re = 6371e3  # Earth radius [m]
mu = 3.986004418e14  # Gravitational constant [m^3/s^2]
sun_vector = np.array([1, 0, 0])  # Arbitrary sun direction for sunlight flag


class GroundStation:
    def __init__(self, lat, lon, Re):
        self.lat = np.radians(lat)
        self.lon = np.radians(lon)
        self.Re = Re
        self.pos = self.Re * np.array([
            np.cos(self.lat) * np.cos(self.lon),
            np.cos(self.lat) * np.sin(self.lon),
            np.sin(self.lat)
        ])

    def is_visible(self, sat_pos):
        vec = sat_pos - self.pos
        dist = np.linalg.norm(vec, axis=1)
        los_vec = vec / dist[:, None]
        dot = np.dot(los_vec, self.pos / self.Re)
        return dot > 0.0


class PointOfInterest:
    def __init__(self, lat, lon, Re):
        self.lat = np.radians(lat)
        self.lon = np.radians(lon)
        self.Re = Re
        self.pos = self.Re * np.array([
            np.cos(self.lat) * np.cos(self.lon),
            np.cos(self.lat) * np.sin(self.lon),
            np.sin(self.lat)
        ])

    def is_visible(self, sat_pos):
        vec = sat_pos - self.pos
        dist = np.linalg.norm(vec, axis=1)
        los_vec = vec / dist[:, None]
        dot = np.dot(los_vec, self.pos / self.Re)
        return dot > 0.0

class Orbit:
    def __init__(self, altitude, inclination_deg, mu=3.986e14, Re=6371e3):
        """
        Initialise une orbite circulaire inclinée autour de la Terre.

        Args:
            altitude (float): Altitude de l’orbite en mètres.
            inclination_deg (float): Inclinaison orbitale en degrés.
            mu (float): Constante gravitationnelle (par défaut Terre).
            Re (float): Rayon de la Terre en mètres.
        """
        self.altitude = altitude
        self.inclination = np.radians(inclination_deg)
        self.mu = mu
        self.Re = Re

        # Rayon orbital total (centre Terre à satellite)
        self.r0 = self.Re + self.altitude

        # Vitesse circulaire
        self.v0 = np.sqrt(self.mu / self.r0)

        # === Position initiale dans le plan orbital ===
        # On place le satellite au point (r0, 0, 0)
        pos0 = np.array([self.r0, 0, 0])

        # === Vitesse initiale (orthogonale à la position), avec inclinaison ===
        vx = 0.0
        vy = self.v0 * np.cos(self.inclination)
        vz = self.v0 * np.sin(self.inclination)
        vel0 = np.array([vx, vy, vz])

        # État initial du satellite : [x, y, z, vx, vy, vz]
        self.state = np.hstack((pos0, vel0))

    def propagate(self, T, N):
        """
        Propagation orbitale par intégration de Runge-Kutta d'ordre 4.

        Args:
            T (float): Durée totale de la propagation (en secondes).
            N (int): Nombre d'étapes (résolution).

        Returns:
            pos (ndarray): Positions [N, 3]
            vel (ndarray): Vitesses [N, 3]
        """
        dt = T / N
        state = self.state.copy()
        pos = np.zeros((N, 3))
        vel = np.zeros((N, 3))

        def deriv(s):
            r = np.linalg.norm(s[:3])
            a = -self.mu * s[:3] / r**3
            return np.hstack((s[3:], a))

        for i in range(N):
            pos[i] = state[:3]
            vel[i] = state[3:]

            k1 = deriv(state)
            k2 = deriv(state + 0.5 * dt * k1)
            k3 = deriv(state + 0.5 * dt * k2)
            k4 = deriv(state + dt * k3)
            state += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return pos, vel


class MissionConfig:
    def __init__(self, altitude=500e3, inclination=98.0, time_resolution=1.0, n_orbits=15):
        self.altitude = altitude
        self.inclination = np.deg2rad(inclination)
        self.dt = time_resolution
        self.n_orbits = n_orbits
        self.Re = 6371e3
        self.mu = 3.986004418e14
        self.omega_earth = 2 * np.pi / (24 * 3600)

        self.orbit = orbit = Orbit(altitude, inclination)
        self.gs_list = []
        self.poi_list = []

    def addGS(self, lat, lon):
        self.gs_list.append((np.deg2rad(lat), np.deg2rad(lon)))

    def addPOI(self, lat, lon):
        self.poi_list.append((np.deg2rad(lat), np.deg2rad(lon)))

    def compute(self):
        T = 2 * np.pi * np.sqrt((self.Re + self.altitude) ** 3 / self.mu)
        N = int(self.n_orbits * T / self.dt)

        pos, vel = self.orbit.propagate(T * self.n_orbits, N)
        time = np.arange(N) * self.dt
        
        sun_flag = self.compute_sun_flag(pos)
        gs_flag = self.compute_visibility_flags(pos, time, self.gs_list)
        poi_flag = self.compute_visibility_flags(pos, time, self.poi_list)
        
         # Debug summary
        summary = {
            "Total Steps": len(pos),
            "Sunlight Coverage (%)": np.mean(sun_flag) * 100,
            "Average GS visibility": np.mean(gs_flag),
            "Max GS visible at once": np.max(gs_flag),
            "Average POI visibility": np.mean(poi_flag),
            "Max POI visible at once": np.max(poi_flag),
            "Position Norm [km]": [np.linalg.norm(p) / 1000 for p in pos[:5]],
            "Velocity Norm [km/s]": [np.linalg.norm(v) / 1000 for v in vel[:5]]
        }

        return pos, vel, sun_flag, gs_flag, poi_flag

    def compute_sun_flag(self, pos):
        sun_vector = np.array([1.0, 0.0, 0.0])
        dot = np.dot(pos, sun_vector)
        return (dot > 0).astype(int)

    def compute_visibility_flags(self, pos, time, targets):
        flags = np.zeros(len(pos))
        for lat, lon0 in targets:
            for i, (x, y, z) in enumerate(pos):
                lon = lon0 + self.omega_earth * time[i]
                lon = (lon + np.pi) % (2 * np.pi) - np.pi
                r = np.array([
                    self.Re * np.cos(lat) * np.cos(lon),
                    self.Re * np.cos(lat) * np.sin(lon),
                    self.Re * np.sin(lat)
                ])
                dist = np.linalg.norm(r - np.array([x, y, z]))
                visible = dist < 2500e3
                flags[i] += visible
        return np.clip(flags, 0, 1)