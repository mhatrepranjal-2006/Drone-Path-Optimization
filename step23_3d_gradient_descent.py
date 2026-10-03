import numpy as np

# -------------------------------------------------
# Fixed start and destination
# -------------------------------------------------
start = np.array([5.0, 10.0, 10.0])
destination = np.array([90.0, 85.0, 10.0])

# -------------------------------------------------
# Initial 3D decision vector
# [x1, y1, z1,
#  x2, y2, z2,
#  x3, y3, z3,
#  x4, y4, z4]
# -------------------------------------------------
X_initial = np.array([
    20.0, 40.0, 20.0,
    35.0, 70.0, 25.0,
    55.0, 75.0, 25.0,
    72.5, 80.0, 20.0
])

gamma = 2.0
battery_capacity = 4500.0

# -------------------------------------------------
# Feasible-region bounds
#
# Intermediate waypoints use:
# 20 <= z <= 40
#
# Start and destination remain fixed at z = 10
# -------------------------------------------------
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


# -------------------------------------------------
# 3D Energy Objective
# -------------------------------------------------
def energy_3d(X):

    path = np.vstack([
        start,
        X.reshape(-1, 3),
        destination
    ])

    differences = path[1:] - path[:-1]

    dx = differences[:, 0]
    dy = differences[:, 1]
    dz = differences[:, 2]

    return np.sum(
        dx**2
        + dy**2
        + gamma * dz**2
    )


# -------------------------------------------------
# 3D Gradient
# -------------------------------------------------
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


# -------------------------------------------------
# Projection onto spatial + altitude constraints
# -------------------------------------------------
def project_constraints(X):

    return np.minimum(
        np.maximum(X, lower),
        upper
    )


# -------------------------------------------------
# Complete constraint handling
# -------------------------------------------------
def enforce_constraints(X_current, X_candidate):

    X_feasible = project_constraints(
        X_candidate
    )

    # Battery safeguard
    while energy_3d(X_feasible) > battery_capacity:

        X_feasible = 0.5 * (
            X_current + X_feasible
        )

    return X_feasible


# -------------------------------------------------
# Projected Gradient Descent
# -------------------------------------------------
def gradient_descent_3d(
    X_initial,
    alpha=0.1,
    tolerance=1e-6,
    max_iterations=10000
):

    X = X_initial.copy()

    energy_history = [
        energy_3d(X)
    ]

    for iteration in range(
        1,
        max_iterations + 1
    ):

        gradient = gradient_3d(X)

        # Normal GD update
        X_candidate = (
            X - alpha * gradient
        )

        # Constraint handling
        X_new = enforce_constraints(
            X,
            X_candidate
        )

        energy_history.append(
            energy_3d(X_new)
        )

        # Convergence check
        if np.linalg.norm(
            X_new - X
        ) < tolerance:

            return (
                X_new,
                energy_history,
                iteration
            )

        X = X_new

    return X, energy_history, max_iterations


# -------------------------------------------------
# Run 3D Gradient Descent
# -------------------------------------------------
X_gd, history, iterations = (
    gradient_descent_3d(X_initial)
)

initial_energy = energy_3d(
    X_initial
)

final_energy = energy_3d(
    X_gd
)

print("Initial 3D Energy:")
print(initial_energy)

print("\nOptimized 3D Decision Vector:")
print(np.round(X_gd, 4))

print("\nOptimized 3D Waypoints:")
print(
    np.round(
        X_gd.reshape(-1, 3),
        4
    )
)

print("\nFinal 3D Energy:")
print(round(final_energy, 4))

print("\nIterations:")
print(iterations)

print(
    "\nBattery:",
    "VALID"
    if final_energy <= battery_capacity
    else "INVALID"
)

print(
    "All spatial/altitude constraints:",
    "VALID"
    if np.all(
        (X_gd >= lower)
        & (X_gd <= upper)
    )
    else "INVALID"
)