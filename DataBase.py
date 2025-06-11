import pandas as pd
import math

# Define the input dictionary (as you provided)
Radiator = { # units: rho_radiator [kg/m^3], emissivity_radiator [None], absorptivity_radiator [None], t_radiator [m]
    'Quartzmirror': {'rho_radiator':1333, 'emissivity_radiator':0.8, 'absorptivity_radiator':0.08, 't_radiator':0.008},
    'Silveredteflon_2mil': {'rho_radiator':2100, 'emissivity_radiator':0.66, 'absorptivity_radiator':0.09, 't_radiator':0.002},
    'Silveredteflon_5mil': {'rho_radiator':2100, 'emissivity_radiator':0.78, 'absorptivity_radiator':0.09, 't_radiator':0.005},
    'Aluminizedteflon_2mil': {'rho_radiator':2200, 'emissivity_radiator':0.78, 'absorptivity_radiator':0.16, 't_radiator':0.002},
    'Aluminizedteflon_5mil': {'rho_radiator':2200, 'emissivity_radiator':0.66, 'absorptivity_radiator':0.16, 't_radiator':0.005},
    'Whitepaints_S13G-LO': {'rho_radiator':1300, 'emissivity_radiator':0.85, 'absorptivity_radiator':0.25, 't_radiator':0.003},
    'Whitepaints_Z93': {'rho_radiator':1300, 'emissivity_radiator':0.92, 'absorptivity_radiator':0.20, 't_radiator':0.001},
    'Whitepaints_ZOT': {'rho_radiator':1300, 'emissivity_radiator':0.91, 'absorptivity_radiator':0.20, 't_radiator':0.002},
    'Whitepaints_chemglazeA276': {'rho_radiator':1300, 'emissivity_radiator':0.88, 'absorptivity_radiator':0.28, 't_radiator':0.001},
    'Blackpaints_chemglazeZ306': {'rho_radiator':1200, 'emissivity_radiator':0.89, 'absorptivity_radiator':0.98, 't_radiator':0.0015},
    'Blackpaints_3Mblackvelvet': {'rho_radiator':1200, 'emissivity_radiator':0.84, 'absorptivity_radiator':0.97, 't_radiator':0.002},
    'Aluminizedkapton_0.5mil': {'rho_radiator':1420, 'emissivity_radiator':0.55, 'absorptivity_radiator':0.34, 't_radiator':0.0005},
    'Aluminizedkapton_1mil': {'rho_radiator':1420, 'emissivity_radiator':0.67, 'absorptivity_radiator':0.38, 't_radiator':0.001},
    'Aluminizedkapton_2mil': {'rho_radiator':1420, 'emissivity_radiator':0.75, 'absorptivity_radiator':0.41, 't_radiator':0.002},
    'Aluminizedkapton_5mil': {'rho_radiator':1420, 'emissivity_radiator':0.86, 'absorptivity_radiator':0.46, 't_radiator':0.005},
    'Metallic_VDA': {'rho_radiator':2700, 'emissivity_radiator':0.04, 'absorptivity_radiator':0.17, 't_radiator':0.002},
    'Metallic_bareAl': {'rho_radiator':2700, 'emissivity_radiator':0.10, 'absorptivity_radiator':0.17, 't_radiator':0.001},
    'Metallic_VDG': {'rho_radiator':8400, 'emissivity_radiator':0.03, 'absorptivity_radiator':0.30, 't_radiator':0.001},
    'Metallic_anodizedAl': {'rho_radiator':2750, 'emissivity_radiator':0.88, 'absorptivity_radiator':0.86, 't_radiator':0.002},
    'Aluminizedmylar': {'rho_radiator':1500, 'emissivity_radiator':0.34, 'absorptivity_radiator':0.30, 't_radiator':0.00025},
    'Betacloth': {'rho_radiator':1600, 'emissivity_radiator':0.86, 'absorptivity_radiator':0.32, 't_radiator':0.003},
    'Astroquartz': {'rho_radiator':2000, 'emissivity_radiator':0.80, 'absorptivity_radiator':0.22, 't_radiator':0.0015},
    'Maxorb': {'rho_radiator':2100, 'emissivity_radiator':0.1, 'absorptivity_radiator':0.9, 't_radiator':0.0015},
}

OBC = {
    'ISIS_Onboard_Computer': {
        'cpu_dmips': 100,
        'cpu_power_idle': 0.3,
        'cpu_power_max': 0.9,
        'cpu_processing_efficiency': 1e4,
        'cpu_recovery_rate': 2.0,
        'cpu_throughput_per_watt': 3e5,
        'mass_kg': 0.045
    },
    'Pumpkin_OBC': {
        'cpu_dmips': 16,
        'cpu_power_idle': 0.1,
        'cpu_power_max': 0.3,
        'cpu_processing_efficiency': 5e3,
        'cpu_recovery_rate': 1.0,
        'cpu_throughput_per_watt': 2e5,
        'mass_kg': 0.035
    },
    'ClydeSpace_OBC': {
        'cpu_dmips': 160,
        'cpu_power_idle': 0.4,
        'cpu_power_max': 1.2,
        'cpu_processing_efficiency': 1.2e4,
        'cpu_recovery_rate': 2.5,
        'cpu_throughput_per_watt': 4e5,
        'mass_kg': 0.050
    },
    'GomSpace_A3200': {
        'cpu_dmips': 140,
        'cpu_power_idle': 0.6,
        'cpu_power_max': 1.5,
        'cpu_processing_efficiency': 8e3,
        'cpu_recovery_rate': 2.0,
        'cpu_throughput_per_watt': 4e5,
        'mass_kg': 0.058
    },
    'CubeComputer_CC_A60': {
        'cpu_dmips': 225,
        'cpu_power_idle': 0.2,
        'cpu_power_max': 1.0,
        'cpu_processing_efficiency': 1.5e4,
        'cpu_recovery_rate': 1.5,
        'cpu_throughput_per_watt': 6e5,
        'mass_kg': 0.040
    },
    'BlueCanyon_XACT': {
        'cpu_dmips': 500,
        'cpu_power_idle': 1.5,
        'cpu_power_max': 3.0,
        'cpu_processing_efficiency': 2e4,
        'cpu_recovery_rate': 3.0,
        'cpu_throughput_per_watt': 5e5,
        'mass_kg': 0.075
    },
    'Hyperion_OBC': {
        'cpu_dmips': 60,
        'cpu_power_idle': 0.25,
        'cpu_power_max': 0.7,
        'cpu_processing_efficiency': 6e3,
        'cpu_recovery_rate': 1.2,
        'cpu_throughput_per_watt': 3e5,
        'mass_kg': 0.038
    },
    'Sirius_OBC': {
        'cpu_dmips': 250,
        'cpu_power_idle': 1.0,
        'cpu_power_max': 2.0,
        'cpu_processing_efficiency': 1.8e4,
        'cpu_recovery_rate': 2.0,
        'cpu_throughput_per_watt': 5e5,
        'mass_kg': 0.065
    },
    'Innovative_iOBC': {
        'cpu_dmips': 90,
        'cpu_power_idle': 0.4,
        'cpu_power_max': 1.1,
        'cpu_processing_efficiency': 7e3,
        'cpu_recovery_rate': 1.5,
        'cpu_throughput_per_watt': 3e5,
        'mass_kg': 0.042
    },
    'NASA_PhoneSat': {
        'cpu_dmips': 1000,
        'cpu_power_idle': 1.5,
        'cpu_power_max': 2.5,
        'cpu_processing_efficiency': 2.5e4,
        'cpu_recovery_rate': 4.0,
        'cpu_throughput_per_watt': 6e5,
        'mass_kg': 0.120
    }
}


Battery = {
    'NiCd': {'d_battery':0.018, 'l_battery':0.065, 'v_battery':54000, 'e_battery':30},
    'NiH2_individual': {'d_battery':0.021, 'l_battery':0.070, 'v_battery':64500, 'e_battery':43},
    'NiH2_common': {'d_battery':0.046, 'l_battery':0.080, 'v_battery':84000, 'e_battery':56},
    'NiH2_single': {'d_battery':0.018, 'l_battery':0.065, 'v_battery':85500, 'e_battery':57},
    'LiSO2': {'d_battery':0.021, 'l_battery':0.070, 'v_battery':144000, 'e_battery':90},
    'LiCF': {'d_battery':0.018, 'l_battery':0.065, 'v_battery':215600, 'e_battery':98},
    'LiSOCl2': {'d_battery':0.021, 'l_battery':0.070, 'v_battery':176000, 'e_battery':110},
    'NaS': {'d_battery':0.046, 'l_battery':0.080, 'v_battery':399000, 'e_battery':210},
    'LiIon_18650': {'d_battery':0.018, 'l_battery':0.065, 'v_battery':184500, 'e_battery':155.1},
    'LiIon_21700': {'d_battery':0.018, 'l_battery':0.065, 'v_battery':228500, 'e_battery':190.4},
    'LiFe_18650': {'d_battery':0.018, 'l_battery':0.065, 'v_battery':69730, 'e_battery':57.75},
    'LiFe_21700': {'d_battery':0.021, 'l_battery':0.070, 'v_battery':69730, 'e_battery':57.75},
    'LiFePO4': {'d_battery':0.021, 'l_battery':0.070, 'v_battery':103000, 'e_battery':89},
    'Li-polymer': {'d_battery':0.018, 'l_battery':0.065, 'v_battery':169500, 'e_battery':119}
}

Solar_array = { # units: e_solar [W/kg], efficiency_solar [None]
    'Polyimide1': {'e_solar':50, 'efficiency_solar':0.28},
    'Polyimide2': {'e_solar':75, 'efficiency_solar':0.148},
    'CFRP_substrate1': {'e_solar':69, 'efficiency_solar':0.15},
    'CFRP_substrate2': {'e_solar':84, 'efficiency_solar':0.185},
    'PCB1': {'e_solar':95, 'efficiency_solar':0.18},
    'PCB2': {'e_solar':100, 'efficiency_solar':0.174},
    'Additive_manufacture_substrate': {'e_solar':53.6, 'efficiency_solar':0.199},
    'FlexiblePV': {'e_solar':100, 'efficiency_solar':0.30},
    'Triple_junction_GaAs': {'e_solar':46, 'efficiency_solar':22},
    'Carbon_fiber': {'e_solar':84.5, 'efficiency_solar':0.21},
}

Communication = { # units: f [GHz], theta_t [deg], Eb_No_req [dB], eff [None]
    'VHF': {'f':0.3, 'theta_t':60, 'D_r':0.1, 'Eb_No_req':8, 'eff':0.59},
    'UHF1': {'f':0.5, 'theta_t':60, 'D_r':0.08, 'Eb_No_req':10, 'eff':0.51},
    'UHF2': {'f':0.8, 'theta_t':75, 'D_r':0.1, 'Eb_No_req':12, 'eff':0.57},
    'L_band1': {'f':1.35, 'theta_t':60, 'D_r':0.09, 'Eb_No_req':9, 'eff':0.62},
    'L_band2': {'f':1.7, 'theta_t':45, 'D_r':0.075, 'Eb_No_req':13, 'eff':0.55},
    'S_band1': {'f':2.4, 'theta_t':30, 'D_r':0.1, 'Eb_No_req':11, 'eff':0.71},
    'S_band2': {'f':2.0, 'theta_t':30, 'D_r':0.1, 'Eb_No_req':15, 'eff':0.68},
    'S_band3': {'f':3.6, 'theta_t':22.5, 'D_r':0.08, 'Eb_No_req':7.5, 'eff':0.63},
    'C_band': {'f':4.0, 'theta_t':22.5, 'D_r':0.075, 'Eb_No_req':8, 'eff':0.5},
    'X_band1': {'f':8.0, 'theta_t':10, 'D_r':0.05, 'Eb_No_req':12, 'eff':0.7},
    'X_band2': {'f':8.0, 'theta_t':15, 'D_r':0.1, 'Eb_No_req':14, 'eff':0.82},
}


#------------
# Radiator
#------------

def compute_radiator_sizing(A_radiator, t_radiator, emissivity_radiator, rho_radiator):

    sigma = 5.67e-8  # Stefan–Boltzmann constant [W/m²·K⁴]

    Q_rad = A_radiator * sigma * emissivity_radiator
    M_thermal_radiator = A_radiator * rho_radiator * t_radiator

    return Q_rad, M_thermal_radiator

A_radiator = 0.03  # m²

# Evaluate each radiator material
results_radiator = {}
radiator_results_list = []

for name, props in Radiator.items():
    rho = props['rho_radiator']
    emissivity = props['emissivity_radiator']
    thickness = props['t_radiator']

    Q_rad, M_rad = compute_radiator_sizing(A_radiator, thickness, emissivity, rho)

    radiator_data = {
        'Radiator material': name,
        'A_radiator [m2]': A_radiator,
        'rho [kg/m³]': rho,
        'emissivity': emissivity,
        'thickness [m]': thickness,
        'Q_rad [W/K⁴]': Q_rad,
        'Mass_radiator [kg]': M_rad,
    }

    radiator_results_list.append(radiator_data)

df_rad = pd.DataFrame(radiator_results_list)
df_rad.to_csv("radiator_technologies.csv", index=False)
print(df_rad)

#------------
# OBS
#------------

OBC_results_list = []

for name, props in OBC.items():
    DMIPS = props['cpu_dmips']
    cpu_power_idle = props['cpu_power_idle']
    cpu_power_max = props['cpu_power_max']
    cpu_processing_efficiency = props['cpu_processing_efficiency']
    cpu_recovery_rate = props['cpu_recovery_rate']
    cpu_throughput_per_watt = props['cpu_throughput_per_watt']
    mass_kg = props['mass_kg']

    OBC_data = {
        'cpu_DMIPS' : DMIPS,
        'cpu_power_idle' : cpu_power_idle,
        'power max' : cpu_power_max,
        'cpu efficiency' : cpu_processing_efficiency,
        'recovery rate' : cpu_recovery_rate,
        'Throughtput per W' : cpu_throughput_per_watt,
        'Mass_OBC' : mass_kg
    }
    
    OBC_results_list.append(OBC_data)

df_OBC = pd.DataFrame(OBC_results_list)
df_OBC.to_csv("OBC_technologies.csv", index=False)
print(df_OBC)

#------------
# Battery
#------------

def compute_battery_subsystem(n_cell, d_battery, l_battery, v_battery, e_battery):

    V_battery = 0.25 * math.pi * d_battery**2 * l_battery  # m³
    b_capacity = v_battery * V_battery * n_cell  # Wh
    M_batt = b_capacity / e_battery  # kg
    return b_capacity, M_batt

n_cell = 3
results = {}
battery_results_list = []

for name, props in Battery.items():
    d = props['d_battery']
    l = props['l_battery']
    v = props['v_battery']
    e = props['e_battery']

    capacity, mass = compute_battery_subsystem(n_cell, d, l, v, e)

    battery_data = {
        'Battery type': name,
        'Number of cell': n_cell,
        'Diameter [m]': d,
        'Length [m]': l,
        'Volumetric Energy [Wh/m³]': v,
        'Specific Energy [Wh/kg]': e,
        'Capacity [Wh]': capacity,
        'Mass_battery [kg]': mass
    }

    battery_results_list.append(battery_data)

# Convert to DataFrame
df_battery = pd.DataFrame(battery_results_list)
df_battery.to_csv("battery_technologies.csv", index=False)
print(df_battery)


#------------
# Solar array
#------------

def compute_solar_array(A_solar, I_solar, efficiency_solar, e_solar):

    P_solar_max = A_solar * I_solar * efficiency_solar
    M_solar = P_solar_max / e_solar
    return P_solar_max, M_solar

A_solar = 0.03  # m²
I_solar = 1367  # W/m² (standard solar constant)
solar_results_list = []

for name, props in Solar_array.items():
    eta = props['efficiency_solar']
    e_sp = props['e_solar']

    # Fix efficiency if it looks like a percentage
    if eta > 1:
        eta = eta / 100

    P_max, M_array = compute_solar_array(A_solar, I_solar, eta, e_sp)

    solar_data = {
        'Solar Material': name,
        'Efficiency': eta,
        'Specific Power [W/kg]': e_sp,
        'A_solar [m^2]': A_solar,
        'P_solar_max [W]': P_max,
        'Mass_solar_array [kg]': M_array,
    }

    solar_results_list.append(solar_data)

df_solar = pd.DataFrame(solar_results_list)
df_solar.to_csv("solar_array_technologies.csv", index=False)
print(df_solar)


#---------------
# Communication
#---------------

import numpy as np

def compute_communication_sizing(f, theta_t, D_r, Eb_No_req, eff):

    D_antenna = 21 / (f * theta_t)

    # Mass estimation
    if f < 1:
        R = f / 0.4
        M_comm = R ** 3 * 0.078
    elif 1 <= f < 2:
        R = f / 1.6
        M_comm = R ** 3 * 0.138
    elif 2 <= f <= 4:
        R = f / 2.2
        M_comm = R ** 3 * 0.195
    else:
        R = f / 8
        M_comm = R ** 3 * 0.27

    # Constants
    P_t = 5
    e_t = 27
    L_l = -2
    S = 1200
    e_r = 0.2
    L_a = -0.3
    T_s = 500

    P_t_dB = 10 * np.log10(P_t)
    G_pt = 44.3 - 10 * np.log10(theta_t ** 2)
    L_pt = -12 * (e_t / theta_t) ** 2
    G_t = G_pt + L_pt
    L_s = (20 * np.log10(3e8) - 20 * np.log10(4 * np.pi) -
           20 * np.log10(S * 1000) - 20 * np.log10(f) - 180.0)
    G_rp = (-159.59 + 20 * np.log10(D_r) + 20 * np.log10(f) +
            10 * np.log10(eff) + 180.0)
    theta_r = 21 / (f * D_r)
    L_pr = -12 * (e_r / theta_r) ** 2
    G_r = G_rp + L_pr

    exponent = (P_t_dB + L_l + G_t + L_pr + L_s + L_a + G_r + 228.6 -
                10 * np.log10(T_s) - Eb_No_req) / 10.0
    data_rate = 10 ** exponent

    return D_antenna, M_comm, data_rate

communication_results_list = []

for name, props in Communication.items():
    f = props['f']
    theta_t = props['theta_t']
    D_r = props['D_r']
    Eb_No_req = props['Eb_No_req']
    eff = props['eff']

    D_antenna, M_comm, data_rate = compute_communication_sizing(f, theta_t, D_r, Eb_No_req, eff)

    communication_data = {
        'Communication band': name,
        'Frequency [GHz]': f,
        'Beamwidth [deg]': theta_t,
        'D_r [m]': D_r,
        'Required Eb/N0 [dB]': Eb_No_req,
        'Efficiency': eff,
        'D_antenna [m]': D_antenna,
        'Mass_communication [kg]': M_comm,
        'Data Rate [bps]': data_rate
    }

    communication_results_list.append(communication_data)

df_comm = pd.DataFrame(communication_results_list)
df_comm.to_csv("communication_technologies.csv", index=False)
print(df_comm)


#-------------
# Generating complete static database
#-------------

from itertools import product

# Combine all pairs and sum their mass
combined_results = []

for item1, item2, item3, item4, item5 in product(communication_results_list, OBC_results_list, solar_results_list, battery_results_list, radiator_results_list):
    combined = {}  # new combined dictionary

    # Merge both dictionaries (with prefix to avoid conflicts)
    combined.update({f'Communication_{k}': v for k, v in item1.items()})
    combined.update({f'OBC_{k}': v for k, v in item2.items()})
    combined.update({f'SolarArray_{k}': v for k, v in item3.items()})
    combined.update({f'Battery_{k}': v for k, v in item4.items()})
    combined.update({f'Radiator_{k}': v for k, v in item5.items()})

    # Add combined mass
    combined['CubeSat mass'] = item1['Mass_communication [kg]'] + item2['Mass_OBC'] + item3['Mass_solar_array [kg]'] + item4['Mass_battery [kg]'] + item5['Mass_radiator [kg]']

    combined_results.append(combined)

# Convert to DataFrame
df = pd.DataFrame(combined_results)

# Save to CSV
df.to_csv("cubesat_productline.csv", index=True)