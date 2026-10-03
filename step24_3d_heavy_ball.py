import numpy as np

# -------------------------------------------------
# Fixed start and destination
# -------------------------------------------------
start = np.array([5.0, 10.0, 10.0])
destination = np.array([90.0, 85.0, 10.0])

# -------------------------------------------------
# Initial 3D decision vector
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
# 3D Energy
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
# Projection
# -------------------------------------------------
def project_constraints(X):

    return np.minimum(
        np.maximum(X, lower),
        upper
    )


# -------------------------------------------------
# Constraint Handling
# -------------------------------------------------
def enforce_constraints(X_current, X_candidate):

    X_feasible = project_constraints(
        X_candidate
    )

    while energy_3d(X_feasible) > battery_capacity:

        X_feasible = 0.5 * (
            X_current + X_feasible
        )

    return X_feasible


# -------------------------------------------------
# 3D Heavy-Ball
# -------------------------------------------------
def heavy_ball_3d(
    X_initial,
    alpha=0.1,
    beta=0.4,
    tolerance=1e-6,
    max_iterations=10000
):

    X_previous = X_initial.copy()
    X = X_initial.copy()

    energy_history = [
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

        energy_history.append(
            energy_3d(X_new)
        )

        if np.linalg.norm(
            X_new - X
        ) < tolerance:

            return (
                X_new,
                energy_history,
                iteration
            )

        X_previous = X
        X = X_new

    return X, energy_history, max_iterations


# -------------------------------------------------
# Run Heavy-Ball
# -------------------------------------------------
X_hb, history, iterations = heavy_ball_3d(
    X_initial
)

initial_energy = energy_3d(
    X_initial
)

final_energy = energy_3d(
    X_hb
)

print("Initial 3D Energy:")
print(initial_energy)

print("\nOptimized 3D Decision Vector:")
print(np.round(X_hb, 4))

print("\nOptimized 3D Waypoints:")
print(
    np.round(
        X_hb.reshape(-1, 3),
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
        (X_hb >= lower)
        & (X_hb <= upper)
    )
    else "INVALID"
)