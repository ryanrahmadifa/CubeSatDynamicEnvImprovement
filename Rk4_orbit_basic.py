import numpy as np
import matplotlib.pyplot as plt

def gen_orbit(num_nodes, x0, y0, vx0, vy0,
              mu=398600.4418, Re=6371.0,
              rotate_angle=0.0, comm_half_angle=np.deg2rad(5),
              day_offset=0.2):
    """
    Generate orbit and flags with normalized, offset daylight factor and communication availability.
    """
    # RK4 integration
    r0 = np.hypot(x0, y0); v0 = np.hypot(vx0, vy0)
    energy = v0**2/2 - mu/r0; a = -mu/(2*energy)
    T = 2*np.pi * np.sqrt(a**3/mu); dt = T/num_nodes

    X = np.zeros(num_nodes); Y = np.zeros(num_nodes)
    Vx = np.zeros(num_nodes); Vy = np.zeros(num_nodes)
    state = np.array([x0, y0, vx0, vy0])

    def deriv(s):
        x, y, vx, vy = s
        r = np.hypot(x, y)
        ax = -mu * x / r**3; ay = -mu * y / r**3
        return np.array([vx, vy, ax, ay])

    for i in range(num_nodes):
        X[i], Y[i], Vx[i], Vy[i] = state
        k1 = deriv(state)
        k2 = deriv(state + 0.5*dt*k1)
        k3 = deriv(state + 0.5*dt*k2)
        k4 = deriv(state +   dt*k3)
        state += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Rotate
    ca, sa = np.cos(rotate_angle), np.sin(rotate_angle)
    Xr = ca*X - sa*Y; Yr = sa*X + ca*Y
    Vxr = ca*Vx - sa*Vy; Vyr = sa*Vx + ca*Vy

    # Dot‐product daylight with offset and normalization
    sun_vector = np.array([-1.0, 0.0])
    r_hat = np.vstack((Xr, Yr)).T
    r_hat /= np.linalg.norm(r_hat, axis=1)[:, None]
    dot_vals = -np.dot(r_hat, sun_vector)  # +1 sun, -1 eclipse

    # Normalize across [−day_offset, +1] → [0,1]
    day_norm = (dot_vals + day_offset) / (1 + day_offset)
    day_flag = np.clip(day_norm, 0.0, 1.0)

    # Communication flag
    theta = np.mod(np.arctan2(Yr, Xr), 2*np.pi)
    apo_angle = np.mod(rotate_angle + np.pi, 2*np.pi)
    diff = np.mod(theta - apo_angle + np.pi, 2*np.pi) - np.pi
    comm_flag = np.clip(1 - np.abs(diff)/comm_half_angle, 0.0, 1.0)

    return Xr, Yr, Vxr, Vyr, day_flag, comm_flag

# Example usage
Re = 6371.0; hp, ha = 300.0, 500.0
rp, ra = Re+hp, Re+ha; a = 0.5*(rp+ra)
e = (ra-rp)/(ra+rp); x0, y0 = rp, 0.0
vp = np.sqrt(398600.4418*(1+e)/(a*(1-e)))
vx0, vy0 = 0.0, vp
rotate_angle = 7*np.pi/4 - np.pi
nn = 200; day_offset = 0.4

X, Y, Vx, Vy, day, comm = gen_orbit(
    nn, x0, y0, vx0, vy0, rotate_angle=rotate_angle, day_offset=day_offset)

speed = np.hypot(Vx, Vy)
skip = max(1, nn//20)

# Plot updated dashboard
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 1) Orbit + velocity
axs[0, 0].plot(X, Y); axs[0, 0].quiver(X[::skip], Y[::skip], Vx[::skip], Vy[::skip],
                                       angles='xy', scale_units='xy', scale=1)
axs[0, 0].set_aspect('equal'); axs[0, 0].set_title('Orbit + Velocity Vectors')

# 2) Speed vs node
axs[0, 1].plot(speed)
axs[0, 1].set_title('Speed vs Node Index')
axs[0, 1].set_xlabel('Node Index'); axs[0, 1].set_ylabel('Speed (km/s)')

# 3) Orbit + Earth
axs[0, 2].plot(X, Y); axs[0, 2].add_patch(plt.Circle((0, 0), Re, fill=False))
axs[0, 2].set_aspect('equal'); axs[0, 2].set_title('Orbit + Earth')

# 4) Normalized, shifted daylight factor
sc = axs[1, 0].scatter(X, Y, c=day, vmin=0, vmax=1, cmap='plasma', s=20)
axs[1, 0].add_patch(plt.Circle((0, 0), Re, fill=False))
axs[1, 0].set_aspect('equal'); axs[1, 0].set_title('Normalized Shifted Daylight')
fig.colorbar(sc, ax=axs[1, 0], label='day_flag')

# 5) Communication availability
axs[1, 1].scatter(X[comm>0], Y[comm>0], marker='.', s=20, label='Comm Avail')
axs[1, 1].scatter(X[comm==0], Y[comm==0], marker='.', s=20, label='No Comm')
axs[1, 1].add_patch(plt.Circle((0, 0), Re, fill=False))
axs[1, 1].set_aspect('equal'); axs[1, 1].set_title('Comm Availability')
axs[1, 1].legend()

# 6) Flags vs node
axs[1, 2].plot(day, label='day_flag'); axs[1, 2].plot(comm, label='comm_flag')
axs[1, 2].set_title('Flags vs Node Index'); axs[1, 2].set_xlabel('Node Index')
axs[1, 2].legend()

plt.tight_layout()
plt.show()
