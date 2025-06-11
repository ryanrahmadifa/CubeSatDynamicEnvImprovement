import numpy as np

# Constants
earth_rotation_rate = 7.2921150e-5  # rad/s
time_step = 1.0  # seconds
num_steps = 1000

# Initial longitude of a point (e.g., ground station) in radians
initial_longitude = np.deg2rad(0.0)

# Compute the longitude of the point over time due to Earth's rotation
longitudes = initial_longitude + earth_rotation_rate * np.arange(num_steps) * time_step
longitudes_deg = np.rad2deg(longitudes % (2 * np.pi))  # Wrap around 360°

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(longitudes_deg)
plt.title("Apparent Longitude Shift Due to Earth's Rotation")
plt.xlabel("Time Step")
plt.ylabel("Longitude (degrees)")
plt.grid(True)
plt.tight_layout()
plt.show()
