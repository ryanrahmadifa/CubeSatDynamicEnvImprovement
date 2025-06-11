import numpy as np
import matplotlib.pyplot as plt

# 2D RK4 orbit generator with adjustable sun direction and continuous GS visibility
def gen_orbit_xy_with_flags(num_nodes, x0, y0, vx0, vy0, 
                             mu=398600.4418, Re=6371.0,
                             apogee_angle_deg=180.0, elev_mask_deg=10.0,
                             sun_angle_deg=0.0):
    # Integrate in orbital plane
    r0 = np.hypot(x0, y0)
    v0 = np.hypot(vx0, vy0)
    energy = v0**2/2 - mu/r0
    a = -mu/(2*energy)
    T = 2*np.pi * np.sqrt(a**3/mu)
    dt = T/num_nodes

    X = np.zeros(num_nodes); Y = np.zeros(num_nodes)
    Vx = np.zeros(num_nodes); Vy = np.zeros(num_nodes)
    state = np.array([x0, y0, vx0, vy0])

    def deriv(s):
        x, y, vx, vy = s
        r = np.hypot(x, y)
        return np.array([vx, vy, -mu*x/r**3, -mu*y/r**3])

    for i in range(num_nodes):
        X[i], Y[i], Vx[i], Vy[i] = state
        k1 = deriv(state)
        k2 = deriv(state + 0.5*dt*k1)
        k3 = deriv(state + 0.5*dt*k2)
        k4 = deriv(state +   dt*k3)
        state += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Rotate so apogee at desired angle
    phi_apo = np.deg2rad(apogee_angle_deg)
    ca, sa = np.cos(phi_apo - np.pi), np.sin(phi_apo - np.pi)
    Xr = ca*X - sa*Y
    Yr = sa*X + ca*Y

    # Satellite position unit vectors
    r_norm = np.hypot(Xr, Yr)
    r_hat = np.vstack((Xr, Yr)).T / r_norm[:, None]

    # Compute sun exposure using adjustable sun direction
    phi_sun = np.deg2rad(sun_angle_deg)
    sun_vec = np.array([np.cos(phi_sun), np.sin(phi_sun)])
    dot_sun = np.dot(r_hat, sun_vec)
    sun_flag = np.clip(dot_sun, 0.0, 1.0)

    # Ground station at same longitude on equator
    gs_pos = np.array([Re*np.cos(phi_apo), Re*np.sin(phi_apo)])
    gs_hat = gs_pos / np.linalg.norm(gs_pos)
    sat_pos = np.vstack((Xr, Yr)).T
    v2gs = sat_pos - gs_pos
    v_norm = np.linalg.norm(v2gs, axis=1)
    dot_gs = v2gs.dot(gs_hat)

    # Continuous cosine elevation
    cos_el = dot_gs / v_norm
    sin_el = np.sin(np.deg2rad(elev_mask_deg))
    gs_raw = (cos_el - sin_el) / (1 - sin_el)
    gs_flag = np.clip(gs_raw, 0.0, 1.0)

    return Xr, Yr, Vx, Vy, sun_flag, gs_flag

# Example usage with modified sun direction
nn = 500
Re = 6371.0
hp, ha = 300.0, 500.0
rp, ra = Re+hp, Re+ha
a = 0.5*(rp+ra)
e = (ra-rp)/(ra+rp)
x0, y0 = rp, 0.0
vx0, vy0 = 0.0, np.sqrt(398600.4418*(1+e)/(a*(1-e)))

# Parameters
apogee_angle = 135.0  # degrees
elev_mask = 10.0      # GS elevation mask
sun_angle = 180.0      # degrees from +x axis

# Generate orbit and flags with custom sun angle
X, Y, Vx, Vy, sun_flag, gs_flag = gen_orbit_xy_with_flags(
    nn, x0, y0, vx0, vy0,
    apogee_angle_deg=apogee_angle,
    elev_mask_deg=elev_mask,
    sun_angle_deg=sun_angle)

# Compute speed and skip for vectors
speed = np.hypot(Vx, Vy)
skip = max(1, nn//20)

# Plot updated dashboard
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 1) Rotated orbit + velocity vectors
axs[0,0].plot(X, Y, linewidth=1)
axs[0,0].quiver(X[::skip], Y[::skip], Vx[::skip], Vy[::skip],
                angles='xy', scale_units='xy', scale=1)
axs[0,0].add_patch(plt.Circle((0,0), Re, fill=False))
axs[0,0].set_aspect('equal'); axs[0,0].set_title('Orbit + Velocity Vectors')

# 2) Speed vs node index
axs[0,1].plot(speed)
axs[0,1].set_title('Speed vs Node Index'); axs[0,1].set_xlabel('Node Index'); axs[0,1].set_ylabel('Speed (km/s)')

# 3) Orbit + Earth outline
axs[0,2].plot(X, Y, linewidth=1)
axs[0,2].add_patch(plt.Circle((0,0), Re, fill=False))
axs[0,2].set_aspect('equal'); axs[0,2].set_title('Orbit + Earth')

# 4) Sun exposure map with custom sun direction
sc0 = axs[1,0].scatter(X, Y, c=sun_flag, cmap='YlOrRd', s=10, vmin=0, vmax=1)
axs[1,0].add_patch(plt.Circle((0,0), Re, fill=False))
axs[1,0].set_aspect('equal')
axs[1,0].set_title(f'Sun Exposure (Sun @ {sun_angle:.1f}°)')
fig.colorbar(sc0, ax=axs[1,0], label='sun_flag')

# 5) Continuous GS visibility map
sc1 = axs[1,1].scatter(X, Y, c=gs_flag, cmap='viridis', s=10, vmin=0, vmax=1)
axs[1,1].add_patch(plt.Circle((0,0), Re, fill=False))
axs[1,1].set_aspect('equal')
axs[1,1].set_title(f'GS Visibility @ Apo {apogee_angle:.1f}°')
fig.colorbar(sc1, ax=axs[1,1], label='gs_flag')

# 6) Flags vs node index
axs[1,2].plot(sun_flag, label='sun_flag')
axs[1,2].plot(gs_flag,  label='gs_flag')
axs[1,2].set_title('Flags vs Node Index'); axs[1,2].set_xlabel('Node Index'); axs[1,2].set_ylabel('Flag Value')
axs[1,2].legend(); axs[1,2].grid(True)

plt.tight_layout()
plt.show()
