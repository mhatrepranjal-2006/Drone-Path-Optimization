import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment size
# -----------------------------
ENV_MIN = 0
ENV_MAX = 100

# -----------------------------
# Start and destination
# -----------------------------
start = np.array([5, 10])
destination = np.array([90, 85])

# -----------------------------
# Rectangular No-Fly Zone
# -----------------------------
nfz_x_min = 35
nfz_x_max = 55

nfz_y_min = 30
nfz_y_max = 65

# -----------------------------
# Create visualization
# -----------------------------
plt.figure(figsize=(8, 8))

# Start point
plt.scatter(
    start[0],
    start[1],
    s=120,
    marker="o",
    label="Start"
)

# Destination point
plt.scatter(
    destination[0],
    destination[1],
    s=180,
    marker="*",
    label="Destination"
)

# No-Fly Zone
no_fly_zone = plt.Rectangle(
    (nfz_x_min, nfz_y_min),
    nfz_x_max - nfz_x_min,
    nfz_y_max - nfz_y_min,
    fill=False,
    hatch="//",
    linewidth=2,
    label="No-Fly Zone"
)

plt.gca().add_patch(no_fly_zone)

# Environment limits
plt.xlim(ENV_MIN, ENV_MAX)
plt.ylim(ENV_MIN, ENV_MAX)

# Labels and formatting
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("2D Drone Delivery Environment")

plt.grid(True)
plt.legend()
plt.axis("equal")

plt.show()