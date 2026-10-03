import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment
# -----------------------------
ENV_MIN = 0
ENV_MAX = 100

# Fixed points
start = np.array([5, 10])
destination = np.array([90, 85])

# -----------------------------
# No-Fly Zone
# -----------------------------
nfz_x_min = 35
nfz_x_max = 55
nfz_y_min = 30
nfz_y_max = 65

# -----------------------------
# Initial waypoints
# -----------------------------
P1 = np.array([25, 25])
P2 = np.array([35, 72])
P3 = np.array([60, 72])
P4 = np.array([75, 80])

# Complete initial path
path = np.array([
    start,
    P1,
    P2,
    P3,
    P4,
    destination
])

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(8, 8))

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

# Initial path
plt.plot(
    path[:, 0],
    path[:, 1],
    marker="o",
    linewidth=2,
    label="Initial Path"
)

# Start
plt.scatter(
    start[0],
    start[1],
    s=120,
    marker="o",
    label="Start"
)

# Destination
plt.scatter(
    destination[0],
    destination[1],
    s=180,
    marker="*",
    label="Destination"
)

# Label intermediate waypoints
for i, point in enumerate(path[1:-1], start=1):
    plt.text(
        point[0] + 1,
        point[1] + 1,
        f"P{i}"
    )

# Environment limits
plt.xlim(ENV_MIN, ENV_MAX)
plt.ylim(ENV_MIN, ENV_MAX)

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

plt.title("Initial 2D Drone Delivery Path")

plt.grid(True)
plt.legend()
plt.axis("equal")

plt.show()