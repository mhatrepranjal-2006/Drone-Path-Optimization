import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# ENERGY-EFFICIENT 2D DRONE PATH OPTIMIZATION
# ============================================================

# All horizontal coordinates are treated as simulated metres.

# ------------------------------------------------------------
# 1. FIXED ENVIRONMENT
# ------------------------------------------------------------

start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# Rectangular No-Fly Zone
NFZ_X_MIN = 35.0
NFZ_X_MAX = 55.0
NFZ_Y_MIN = 30.0
NFZ_Y_MAX = 65.0

# Battery capacity in simulation energy units
BATTERY_CAPACITY = 4500.0


# ------------------------------------------------------------
# 2. INITIAL DECISION VECTOR
#
# X = [x1, y1, x2, y2, x3, y3, x4, y4]
# ------------------------------------------------------------

X_initial = np.array([
    25.0, 25.0,
    35.0, 72.0,
    60.0, 72.0,
    75.0, 80.0
])


# ------------------------------------------------------------
# 3. FEASIBLE SAFE-CORRIDOR BOUNDS
# ------------------------------------------------------------

lower = np.array([
     0.0,   0.0,
     0.0,  70.0,
    55.0,  70.0,
    60.0,   0.0
])

upper = np.array([
     30.0, 100.0,
     35.0, 100.0,
    100.0, 100.0,
    100.0, 100.0
])


# ------------------------------------------------------------
# 4. ENERGY OBJECTIVE
#
# E = Sum[(dx)^2 + (dy)^2]
# ------------------------------------------------------------

def energy_objective(X):

    waypoints = X.reshape(-1, 2)

    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    differences = path[1:] - path[:-1]

    return np.sum(differences ** 2)


# ------------------------------------------------------------
# 5. PATH LENGTH
# ------------------------------------------------------------

def path_length(X):

    waypoints = X.reshape(-1, 2)

    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    differences = path[1:] - path[:-1]

    segment_lengths = np.linalg.norm(
        differences,
        axis=1
    )

    return np.sum(segment_lengths)


# ------------------------------------------------------------
# 6. ANALYTICAL GRADIENT
# ------------------------------------------------------------

def energy_gradient(X):

    waypoints = X.reshape(-1, 2)

    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    gradient = []

    for i in range(1, len(path) - 1):

        grad_point = (
            4 * path[i]
            - 2 * path[i - 1]
            - 2 * path[i + 1]
        )

        gradient.extend(grad_point)

    return np.array(gradient)


# ------------------------------------------------------------
# 7. SPATIAL PROJECTION
# ------------------------------------------------------------

def project_spatial_constraints(X):

    return np.minimum(
        np.maximum(X, lower),
        upper
    )


# ------------------------------------------------------------
# 8. COMPLETE CONSTRAINT HANDLING
# ------------------------------------------------------------

def enforce_constraints(X_current, X_candidate):

    # Project candidate into safe spatial region
    X_feasible = project_spatial_constraints(
        X_candidate
    )

    # Battery feasibility safeguard
    while energy_objective(X_feasible) > BATTERY_CAPACITY:

        X_feasible = 0.5 * (
            X_current + X_feasible
        )

    return X_feasible


# ------------------------------------------------------------
# 9. GRADIENT DESCENT
# ------------------------------------------------------------

def gradient_descent(
    X_initial,
    alpha=0.1,
    tolerance=1e-6,
    max_iterations=10000
):

    X = X_initial.copy()

    history = [
        energy_objective(X)
    ]

    for iteration in range(
        1,
        max_iterations + 1
    ):

        gradient = energy_gradient(X)

        X_candidate = (
            X - alpha * gradient
        )

        X_new = enforce_constraints(
            X,
            X_candidate
        )

        history.append(
            energy_objective(X_new)
        )

        if np.linalg.norm(
            X_new - X
        ) < tolerance:

            return (
                X_new,
                history,
                iteration
            )

        X = X_new

    return X, history, max_iterations


# ------------------------------------------------------------
# 10. HEAVY-BALL MOMENTUM
# ------------------------------------------------------------

def heavy_ball(
    X_initial,
    alpha=0.1,
    beta=0.4,
    tolerance=1e-6,
    max_iterations=10000
):

    X_previous = X_initial.copy()
    X = X_initial.copy()

    history = [
        energy_objective(X)
    ]

    for iteration in range(
        1,
        max_iterations + 1
    ):

        gradient = energy_gradient(X)

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
            energy_objective(X_new)
        )

        if np.linalg.norm(
            X_new - X
        ) < tolerance:

            return (
                X_new,
                history,
                iteration
            )

        X_previous = X
        X = X_new

    return X, history, max_iterations


# ------------------------------------------------------------
# 11. HESSIAN
# ------------------------------------------------------------

def build_hessian():

    H = np.zeros((8, 8))

    # Diagonal second derivatives
    np.fill_diagonal(H, 4.0)

    # Connections between neighbouring waypoints
    for i in range(6):

        # Connect only same coordinate:
        # x -> x or y -> y
        if i + 2 < 8:

            H[i, i + 2] = -2.0
            H[i + 2, i] = -2.0

    return H


# ------------------------------------------------------------
# 12. CONSTRAINT VERIFICATION
# ------------------------------------------------------------

def check_constraints(X):

    spatial_valid = np.all(
        (X >= lower)
        &
        (X <= upper)
    )

    battery_valid = (
        energy_objective(X)
        <= BATTERY_CAPACITY
    )

    return spatial_valid, battery_valid


# ------------------------------------------------------------
# 13. RUN BOTH ALGORITHMS
# ------------------------------------------------------------

X_gd, gd_history, gd_iterations = (
    gradient_descent(X_initial)
)

X_hb, hb_history, hb_iterations = (
    heavy_ball(X_initial)
)


# ------------------------------------------------------------
# 14. ENERGY + PATH LENGTH RESULTS
# ------------------------------------------------------------

initial_energy = energy_objective(
    X_initial
)

gd_energy = energy_objective(
    X_gd
)

hb_energy = energy_objective(
    X_hb
)

initial_length = path_length(
    X_initial
)

gd_length = path_length(
    X_gd
)

hb_length = path_length(
    X_hb
)


# ------------------------------------------------------------
# 15. HESSIAN + CONDITION NUMBER
# ------------------------------------------------------------

H = build_hessian()

eigenvalues = np.linalg.eigvalsh(H)

lambda_min = np.min(eigenvalues)
lambda_max = np.max(eigenvalues)

condition_number = (
    lambda_max / lambda_min
)


# ------------------------------------------------------------
# 16. CONSTRAINT CHECKS
# ------------------------------------------------------------

gd_spatial, gd_battery = (
    check_constraints(X_gd)
)

hb_spatial, hb_battery = (
    check_constraints(X_hb)
)


# ------------------------------------------------------------
# 17. PRINT FINAL RESULTS
# ------------------------------------------------------------

print("\n====================================")
print("2D DRONE PATH OPTIMIZATION RESULTS")
print("====================================")

print("\nInitial Path")
print(
    "Energy:",
    round(initial_energy, 4)
)
print(
    "Path Length:",
    round(initial_length, 4)
)

print("\nGradient Descent")
print(
    "Optimized X:",
    np.round(X_gd, 4)
)
print(
    "Energy:",
    round(gd_energy, 4)
)
print(
    "Path Length:",
    round(gd_length, 4)
)
print(
    "Iterations:",
    gd_iterations
)
print(
    "Spatial Constraints:",
    "VALID" if gd_spatial else "INVALID"
)
print(
    "Battery:",
    "VALID" if gd_battery else "INVALID"
)

print("\nHeavy-Ball Momentum")
print(
    "Optimized X:",
    np.round(X_hb, 4)
)
print(
    "Energy:",
    round(hb_energy, 4)
)
print(
    "Path Length:",
    round(hb_length, 4)
)
print(
    "Iterations:",
    hb_iterations
)
print(
    "Spatial Constraints:",
    "VALID" if hb_spatial else "INVALID"
)
print(
    "Battery:",
    "VALID" if hb_battery else "INVALID"
)

energy_reduction = (
    (initial_energy - gd_energy)
    / initial_energy
) * 100

print("\nEnergy Proxy Reduction:")
print(
    round(energy_reduction, 2),
    "%"
)

print("\nHessian Eigenvalues:")
print(
    np.round(eigenvalues, 6)
)

print(
    "\nCondition Number:",
    round(condition_number, 4)
)


# ------------------------------------------------------------
# 18. CREATE RESULTS DIRECTORY
# ------------------------------------------------------------

results_directory = Path("results")

results_directory.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 19. PATH COMPARISON GRAPH
# ------------------------------------------------------------

initial_path = np.vstack([
    start,
    X_initial.reshape(-1, 2),
    destination
])

gd_path = np.vstack([
    start,
    X_gd.reshape(-1, 2),
    destination
])

hb_path = np.vstack([
    start,
    X_hb.reshape(-1, 2),
    destination
])

plt.figure(figsize=(9, 8))

# No-Fly Zone
no_fly_zone = plt.Rectangle(
    (NFZ_X_MIN, NFZ_Y_MIN),
    NFZ_X_MAX - NFZ_X_MIN,
    NFZ_Y_MAX - NFZ_Y_MIN,
    fill=False,
    hatch="//",
    linewidth=2,
    label="No-Fly Zone"
)

plt.gca().add_patch(
    no_fly_zone
)

# Initial
plt.plot(
    initial_path[:, 0],
    initial_path[:, 1],
    marker="o",
    linestyle="--",
    label="Initial Path"
)

# Gradient Descent
plt.plot(
    gd_path[:, 0],
    gd_path[:, 1],
    marker="o",
    linewidth=2,
    label="Gradient Descent"
)

# Heavy-Ball
plt.plot(
    hb_path[:, 0],
    hb_path[:, 1],
    marker="s",
    linewidth=2,
    label="Heavy-Ball"
)

plt.scatter(
    start[0],
    start[1],
    s=120,
    marker="o",
    label="Start"
)

plt.scatter(
    destination[0],
    destination[1],
    s=180,
    marker="*",
    label="Destination"
)

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")

plt.title(
    "Initial vs Optimized Drone Paths"
)

plt.grid(True)
plt.legend()
plt.axis("equal")

plt.tight_layout()

plt.savefig(
    results_directory
    / "final_2d_paths.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 20. CONVERGENCE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    gd_history,
    linewidth=2,
    label="Gradient Descent"
)

plt.plot(
    hb_history,
    linewidth=2,
    label="Heavy-Ball"
)

plt.xlabel("Iteration")
plt.ylabel("Energy Proxy")

plt.title(
    "Convergence Comparison"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    results_directory
    / "convergence_comparison.png",
    dpi=300
)

plt.close()


print("\nGraphs saved inside:")
print("results/final_2d_paths.png")
print("results/convergence_comparison.png")