import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------------
# Environment
# -----------------------------------
ENV_MIN = 0
ENV_MAX = 100

# Altitude limits
Z_MIN = 10
Z_MAX = 40

# -----------------------------------
# Fixed start and destination
# -----------------------------------
start = np.array([5.0, 10.0, 10.0])
destination = np.array([90.0, 85.0, 10.0])

# -----------------------------------
# Initial 3D waypoints
# -----------------------------------
P1 = np.array([20.0, 40.0, 20.0])
P2 = np.array([35.0, 70.0, 25.0])
P3 = np.array([55.0, 75.0, 25.0])
P4 = np.array([72.5, 80.0, 20.0])

# Complete 3D path
path = np.array([
    start,
    P1,
    P2,
    P3,
    P4,
    destination
])

# -----------------------------------
# No-Fly Zone
# x = 35 to 55
# y = 30 to 65
# -----------------------------------
x1, x2 = 35, 55
y1, y2 = 30, 65
z1, z2 = Z_MIN, Z_MAX

# Vertices of the rectangular prism
vertices = [
    [x1, y1, z1],
    [x2, y1, z1],
    [x2, y2, z1],
    [x1, y2, z1],

    [x1, y1, z2],
    [x2, y1, z2],
    [x2, y2, z2],
    [x1, y2, z2]
]

faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],
    [vertices[4], vertices[5], vertices[6], vertices[7]],
    [vertices[0], vertices[1], vertices[5], vertices[4]],
    [vertices[1], vertices[2], vertices[6], vertices[5]],
    [vertices[2], vertices[3], vertices[7], vertices[6]],
    [vertices[3], vertices[0], vertices[4], vertices[7]]
]

# -----------------------------------
# 3D Visualization
# -----------------------------------
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

# Drone path
ax.plot(
    path[:, 0],
    path[:, 1],
    path[:, 2],
    marker="o",
    linewidth=2,
    label="Initial 3D Path"
)

# Start
ax.scatter(
    start[0],
    start[1],
    start[2],
    s=100,
    marker="o",
    label="Start"
)

# Destination
ax.scatter(
    destination[0],
    destination[1],
    destination[2],
    s=160,
    marker="*",
    label="Destination"
)

# No-Fly Zone prism
nfz = Poly3DCollection(
    faces,
    alpha=0.2
)

ax.add_collection3d(nfz)

# Label waypoints
for i, point in enumerate(path[1:-1], start=1):
    ax.text(
        point[0],
        point[1],
        point[2] + 1,
        f"P{i}"
    )

# Axis limits
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 45)

# Labels
ax.set_xlabel("X - Field Length")
ax.set_ylabel("Y - Field Width")
ax.set_zlabel("Z - Altitude")

ax.set_title(
    "Basic 3D Drone Delivery Environment"
)

ax.legend()

plt.show()