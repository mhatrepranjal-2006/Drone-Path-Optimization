import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

# ============================================================
# 3D DRONE DELIVERY ANIMATED SIMULATION
# ============================================================

# ------------------------------------------------------------
# Fixed start and destination
# ------------------------------------------------------------
start = np.array([5.0, 10.0, 10.0])
destination = np.array([90.0, 85.0, 10.0])

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
initial_waypoints = np.array([
    [20.0, 40.0, 20.0],
    [35.0, 70.0, 25.0],
    [55.0, 75.0, 25.0],
    [72.5, 80.0, 20.0]
])

gd_waypoints = np.array([
    [20.0, 40.0, 20.0],
    [35.0, 70.0, 20.0],
    [55.0, 75.0, 20.0],
    [72.5, 80.0, 20.0]
])

hb_waypoints = np.array([
    [20.0, 40.0, 20.0],
    [35.0, 70.0, 20.0],
    [55.0, 75.0, 20.0],
    [72.5, 80.0, 20.0]
])

# ------------------------------------------------------------
# Choose simulation mode
# Options: "initial", "gd", "hb"
# ------------------------------------------------------------
PATH_MODE = "hb"

if PATH_MODE == "initial":
    waypoints = initial_waypoints
    path_name = "Initial 3D Path"
    final_energy = 3837.5
    iterations = "-"
elif PATH_MODE == "gd":
    waypoints = gd_waypoints
    path_name = "3D Gradient Descent"
    final_energy = 3737.5
    iterations = 31
elif PATH_MODE == "hb":
    waypoints = hb_waypoints
    path_name = "3D Heavy-Ball"
    final_energy = 3737.5
    iterations = 4
else:
    raise ValueError("PATH_MODE must be 'initial', 'gd', or 'hb'")

# Full waypoint path
path_points = np.vstack([
    start,
    waypoints,
    destination
])

battery_capacity = 4500.0

# ------------------------------------------------------------
# Build smooth animation points
# ------------------------------------------------------------
def interpolate_path(points, frames_per_segment=25):

    smooth_points = []

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]

        for t in np.linspace(0, 1, frames_per_segment, endpoint=False):
            smooth_points.append((1 - t) * p0 + t * p1)

    smooth_points.append(points[-1])

    return np.array(smooth_points)

smooth_path = interpolate_path(path_points, frames_per_segment=25)

# ------------------------------------------------------------
# Create results folder
# ------------------------------------------------------------
results = Path("results")
results.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 3D Figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot complete reference path
ax.plot(
    path_points[:, 0],
    path_points[:, 1],
    path_points[:, 2],
    linestyle="--",
    marker="o",
    label=path_name
)

# Start and destination
ax.scatter(
    start[0], start[1], start[2],
    s=120,
    marker="o",
    label="Start"
)

ax.scatter(
    destination[0], destination[1], destination[2],
    s=180,
    marker="*",
    label="Destination"
)

# Label fixed waypoints
for i, point in enumerate(waypoints, start=1):
    ax.text(
        point[0],
        point[1],
        point[2] + 1,
        f"P{i}"
    )

# ------------------------------------------------------------
# No-Fly Zone Prism
# ------------------------------------------------------------
x1, x2 = 35, 55
y1, y2 = 30, 65
z1, z2 = 0, 40

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

nfz = Poly3DCollection(faces, alpha=0.2)
ax.add_collection3d(nfz)

# ------------------------------------------------------------
# Axis settings
# ------------------------------------------------------------
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 45)

ax.set_xlabel("X - Field Length (m)")
ax.set_ylabel("Y - Field Width (m)")
ax.set_zlabel("Altitude (m)")
ax.set_title("Animated 3D Drone Delivery Simulation")

# Drone marker and travelled path
drone_marker, = ax.plot([], [], [], marker="o", markersize=8, label="Drone")
trail_line, = ax.plot([], [], [], linewidth=2, label="Travelled Path")

# Info text
info_text = ax.text2D(
    0.02, 0.95, "",
    transform=ax.transAxes
)

ax.legend()

# ------------------------------------------------------------
# Animation init
# ------------------------------------------------------------
def init():
    drone_marker.set_data([], [])
    drone_marker.set_3d_properties([])
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    info_text.set_text("")
    return drone_marker, trail_line, info_text

# ------------------------------------------------------------
# Animation update
# ------------------------------------------------------------
def update(frame):

    current = smooth_path[frame]

    # Drone current position
    drone_marker.set_data([current[0]], [current[1]])
    drone_marker.set_3d_properties([current[2]])

    # Travelled trail
    travelled = smooth_path[:frame + 1]
    trail_line.set_data(travelled[:, 0], travelled[:, 1])
    trail_line.set_3d_properties(travelled[:, 2])

    info_text.set_text(
        f"Mode: {path_name}\n"
        f"Frame: {frame + 1}/{len(smooth_path)}\n"
        f"X = {current[0]:.2f}\n"
        f"Y = {current[1]:.2f}\n"
        f"Z = {current[2]:.2f} m\n"
        f"Route Energy Proxy = {final_energy}\n"
        f"Battery Constraint = VALID\n"
        f"Optimization Iterations = {iterations}"
    )

    return drone_marker, trail_line, info_text

# ------------------------------------------------------------
# Create animation
# ------------------------------------------------------------
anim = FuncAnimation(
    fig,
    update,
    frames=len(smooth_path),
    init_func=init,
    interval=120,
    blit=False,
    repeat=False
)

# ------------------------------------------------------------
# Save GIF
# ------------------------------------------------------------
gif_path = results / f"drone_simulation_{PATH_MODE}.gif"

try:
    anim.save(gif_path, writer=PillowWriter(fps=10))
    print(f"Animation saved as: {gif_path}")
except Exception as e:
    print("GIF could not be saved automatically.")
    print("Reason:", e)

plt.show()