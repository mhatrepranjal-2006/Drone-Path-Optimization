import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# FINAL 3D DRONE PATH OPTIMIZATION
# ============================================================

# ------------------------------------------------------------
# Fixed start and destination
# ------------------------------------------------------------
start = np.array([5.0, 10.0, 10.0])
destination = np.array([90.0, 85.0, 10.0])

# Initial 3D waypoints
X_initial = np.array([
    20.0, 40.0, 20.0,
    35.0, 70.0, 25.0,
    55.0, 75.0, 25.0,
    72.5, 80.0, 20.0
])

gamma = 2.0
battery_capacity = 4500.0

# ------------------------------------------------------------
# Constraints
# ------------------------------------------------------------
lower = np.array([
     0.0,   0.0, 20.0,
     0.0,  70.0, 20.0,
    55.0,  70.0, 20.0,
    60.0,   0.0, 20.0
])

upper = np.array([
     30.0, 100.0, 40.0,
     35.0, 100.0, 40.0,
    100.0, 100.0, 40.0,
    100.0, 100.0, 40.0
])


# ------------------------------------------------------------
# 3D Energy
# ------------------------------------------------------------
def energy_3d(X):

    path = np.vstack([
        start,
        X.reshape(-1, 3),
        destination
    ])

    d = path[1:] - path[:-1]

    dx = d[:, 0]
    dy = d[:, 1]
    dz = d[:, 2]

    return np.sum(
        dx**2 +
        dy**2 +
        gamma * dz**2
    )


# ------------------------------------------------------------
# 3D Path Length
# ------------------------------------------------------------
def path_length_3d(X):

    path = np.vstack([
        start,
        X.reshape(-1, 3),
        destination
    ])

    d = path[1:] - path[:-1]

    return np.sum(
        np.linalg.norm(d, axis=1)
    )


# ------------------------------------------------------------
# 3D Gradient
# ------------------------------------------------------------
def gradient_3d(X):

    path = np.vstack([
        start,
        X.reshape(-1, 3),
        destination
    ])

    gradient = []

    for i in range(1, len(path) - 1):

        previous = path[i - 1]
        current = path[i]
        next_point = path[i + 1]

        grad_x = (
            4 * current[0]
            - 2 * previous[0]
            - 2 * next_point[0]
        )

        grad_y = (
            4 * current[1]
            - 2 * previous[1]
            - 2 * next_point[1]
        )

        grad_z = gamma * (
            4 * current[2]
            - 2 * previous[2]
            - 2 * next_point[2]
        )

        gradient.extend([
            grad_x,
            grad_y,
            grad_z
        ])

    return np.array(gradient)


# ------------------------------------------------------------
# Projection
# ------------------------------------------------------------
def project_constraints(X):

    return np.minimum(
        np.maximum(X, lower),
        upper
    )


# ------------------------------------------------------------
# Constraint Handling
# ------------------------------------------------------------
def enforce_constraints(X_current, X_candidate):

    X_feasible = project_constraints(
        X_candidate
    )

    while energy_3d(X_feasible) > battery_capacity:

        X_feasible = 0.5 * (
            X_current + X_feasible
        )

    return X_feasible


# ------------------------------------------------------------
# 3D Gradient Descent
# ------------------------------------------------------------
def gradient_descent_3d(
    X_initial,
    alpha=0.1,
    tolerance=1e-6,
    max_iterations=10000
):

    X = X_initial.copy()

    history = [
        energy_3d(X)
    ]

    for iteration in range(
        1,
        max_iterations + 1
    ):

        gradient = gradient_3d(X)

        X_candidate = (
            X - alpha * gradient
        )

        X_new = enforce_constraints(
            X,
            X_candidate
        )

        history.append(
            energy_3d(X_new)
        )

        if np.linalg.norm(
            X_new - X
        ) < tolerance:

            return X_new, history, iteration

        X = X_new

    return X, history, max_iterations


# ------------------------------------------------------------
# 3D Heavy-Ball
# ------------------------------------------------------------
def heavy_ball_3d(
    X_initial,
    alpha=0.1,
    beta=0.4,
    tolerance=1e-6,
    max_iterations=10000
):

    X_previous = X_initial.copy()
    X = X_initial.copy()

    history = [
        energy_3d(X)
    ]

    for iteration in range(
        1,
        max_iterations + 1
    ):

        gradient = gradient_3d(X)

        momentum = beta * (
            X - X_previous
        )

        X_candidate = (
            X
            - alpha * gradient
            + momentum
        )

        X_new = enforce_constraints(
            X,
            X_candidate
        )

        history.append(
            energy_3d(X_new)
        )

        if np.linalg.norm(
            X_new - X
        ) < tolerance:

            return X_new, history, iteration

        X_previous = X
        X = X_new

    return X, history, max_iterations


# ------------------------------------------------------------
# Run both algorithms
# ------------------------------------------------------------
X_gd, gd_history, gd_iterations = (
    gradient_descent_3d(X_initial)
)

X_hb, hb_history, hb_iterations = (
    heavy_ball_3d(X_initial)
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
initial_energy = energy_3d(X_initial)
gd_energy = energy_3d(X_gd)
hb_energy = energy_3d(X_hb)

initial_length = path_length_3d(X_initial)
gd_length = path_length_3d(X_gd)
hb_length = path_length_3d(X_hb)

energy_reduction = (
    (initial_energy - gd_energy)
    / initial_energy
) * 100


print("\n======================================")
print("FINAL 3D DRONE OPTIMIZATION RESULTS")
print("======================================")

print("\nInitial 3D Path")
print("Energy:", round(initial_energy, 4))
print("Path Length:", round(initial_length, 4))

print("\n3D Gradient Descent")
print("Energy:", round(gd_energy, 4))
print("Path Length:", round(gd_length, 4))
print("Iterations:", gd_iterations)
print("Waypoints:")
print(np.round(X_gd.reshape(-1, 3), 4))

print("\n3D Heavy-Ball")
print("Energy:", round(hb_energy, 4))
print("Path Length:", round(hb_length, 4))
print("Iterations:", hb_iterations)
print("Waypoints:")
print(np.round(X_hb.reshape(-1, 3), 4))

print(
    "\n3D Energy Proxy Reduction:",
    round(energy_reduction, 2),
    "%"
)

print(
    "\nGD Battery:",
    "VALID"
    if gd_energy <= battery_capacity
    else "INVALID"
)

print(
    "HB Battery:",
    "VALID"
    if hb_energy <= battery_capacity
    else "INVALID"
)

print(
    "GD Constraints:",
    "VALID"
    if np.all(
        (X_gd >= lower)
        & (X_gd <= upper)
    )
    else "INVALID"
)

print(
    "HB Constraints:",
    "VALID"
    if np.all(
        (X_hb >= lower)
        & (X_hb <= upper)
    )
    else "INVALID"
)


# ------------------------------------------------------------
# Create results folder
# ------------------------------------------------------------
results = Path("results")
results.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Build paths
# ------------------------------------------------------------
initial_path = np.vstack([
    start,
    X_initial.reshape(-1, 3),
    destination
])

gd_path = np.vstack([
    start,
    X_gd.reshape(-1, 3),
    destination
])

hb_path = np.vstack([
    start,
    X_hb.reshape(-1, 3),
    destination
])


# ------------------------------------------------------------
# 3D PATH VISUALIZATION
# ------------------------------------------------------------
fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

# Initial path
ax.plot(
    initial_path[:, 0],
    initial_path[:, 1],
    initial_path[:, 2],
    marker="o",
    linestyle="--",
    label="Initial 3D Path"
)

# GD
ax.plot(
    gd_path[:, 0],
    gd_path[:, 1],
    gd_path[:, 2],
    marker="o",
    linewidth=2,
    label="3D Gradient Descent"
)

# Heavy-Ball
ax.plot(
    hb_path[:, 0],
    hb_path[:, 1],
    hb_path[:, 2],
    marker="s",
    linewidth=2,
    label="3D Heavy-Ball"
)

# Start
ax.scatter(
    start[0],
    start[1],
    start[2],
    s=120,
    marker="o",
    label="Start"
)

# Destination
ax.scatter(
    destination[0],
    destination[1],
    destination[2],
    s=180,
    marker="*",
    label="Destination"
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
    [vertices[0], vertices[1],
     vertices[2], vertices[3]],

    [vertices[4], vertices[5],
     vertices[6], vertices[7]],

    [vertices[0], vertices[1],
     vertices[5], vertices[4]],

    [vertices[1], vertices[2],
     vertices[6], vertices[5]],

    [vertices[2], vertices[3],
     vertices[7], vertices[6]],

    [vertices[3], vertices[0],
     vertices[4], vertices[7]]
]

nfz = Poly3DCollection(
    faces,
    alpha=0.2
)

ax.add_collection3d(nfz)


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 45)

ax.set_xlabel("X - Field Length (m)")
ax.set_ylabel("Y - Field Width (m)")
ax.set_zlabel("Altitude (m)")

ax.set_title(
    "3D Drone Path Optimization"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    results / "final_3d_paths.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CONVERGENCE GRAPH
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    gd_history,
    linewidth=2,
    label="3D Gradient Descent"
)

plt.plot(
    hb_history,
    linewidth=2,
    label="3D Heavy-Ball"
)

plt.xlabel("Iteration")
plt.ylabel("3D Energy Proxy")

plt.title(
    "3D Convergence Comparison"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    results / "3d_convergence.png",
    dpi=300
)

plt.close()


print("\nGraphs saved:")
print("results/final_3d_paths.png")
print("results/3d_convergence.png")