import numpy as np

# -----------------------------------
# Fixed start and destination
# -----------------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# -----------------------------------
# Initial decision vector
# -----------------------------------
X_initial = np.array([
    25.0, 25.0,   # P1
    35.0, 72.0,   # P2
    60.0, 72.0,   # P3
    75.0, 80.0    # P4
])

# -----------------------------------
# Feasible-region bounds
# Variable order:
# [x1, y1, x2, y2, x3, y3, x4, y4]
# -----------------------------------
lower = np.array([
    0.0, 0.0,
    0.0, 70.0,
    55.0, 70.0,
    60.0, 0.0
])

upper = np.array([
    30.0, 100.0,
    35.0, 100.0,
    100.0, 100.0,
    100.0, 100.0
])

battery_capacity = 4500.0


# -----------------------------------
# Energy objective
# -----------------------------------
def energy_objective(X):

    waypoints = X.reshape(-1, 2)

    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    differences = path[1:] - path[:-1]

    return np.sum(differences**2)


# -----------------------------------
# Gradient
# -----------------------------------
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


# -----------------------------------
# Projection onto feasible region
# -----------------------------------
def project_to_feasible_region(X):

    return np.minimum(
        np.maximum(X, lower),
        upper
    )


# -----------------------------------
# Projected Gradient Descent
# -----------------------------------
def gradient_descent(
    X_initial,
    alpha=0.1,
    tolerance=1e-6,
    max_iterations=10000
):

    X = X_initial.copy()

    # Store energy at every iteration
    energy_history = [energy_objective(X)]

    for iteration in range(1, max_iterations + 1):

        gradient = energy_gradient(X)

        # Normal Gradient Descent step
        X_candidate = X - alpha * gradient

        # Constraint handling
        X_new = project_to_feasible_region(X_candidate)

        energy_history.append(
            energy_objective(X_new)
        )

        # Convergence check
        if np.linalg.norm(X_new - X) < tolerance:

            return X_new, energy_history, iteration

        X = X_new

    return X, energy_history, max_iterations


# -----------------------------------
# Run Gradient Descent
# -----------------------------------
X_gd, gd_history, gd_iterations = gradient_descent(
    X_initial
)

initial_energy = energy_objective(X_initial)
final_energy = energy_objective(X_gd)

print("Initial energy:", initial_energy)

print("\nOptimized decision vector:")
print(np.round(X_gd, 4))

print("\nOptimized waypoints:")
print(np.round(X_gd.reshape(-1, 2), 4))

print("\nFinal energy:", round(final_energy, 4))

print("Iterations:", gd_iterations)

print(
    "Battery constraint:",
    "VALID"
    if final_energy <= battery_capacity
    else "INVALID"
)