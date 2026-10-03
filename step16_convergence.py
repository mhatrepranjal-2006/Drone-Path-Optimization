import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Fixed points
# -----------------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

X_initial = np.array([
    25.0, 25.0,
    35.0, 72.0,
    60.0, 72.0,
    75.0, 80.0
])

# -----------------------------------
# Constraint bounds
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
# Energy function
# -----------------------------------
def energy_objective(X):

    path = np.vstack([
        start,
        X.reshape(-1, 2),
        destination
    ])

    differences = path[1:] - path[:-1]

    return np.sum(differences**2)


# -----------------------------------
# Gradient
# -----------------------------------
def energy_gradient(X):

    path = np.vstack([
        start,
        X.reshape(-1, 2),
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
# Constraint handling
# -----------------------------------
def project_spatial(X):

    return np.minimum(
        np.maximum(X, lower),
        upper
    )


def enforce_constraints(X_current, X_candidate):

    X_feasible = project_spatial(X_candidate)

    while energy_objective(X_feasible) > battery_capacity:

        X_feasible = 0.5 * (
            X_current + X_feasible
        )

    return X_feasible


# -----------------------------------
# Gradient Descent
# -----------------------------------
def gradient_descent(
    X_initial,
    alpha=0.1,
    tolerance=1e-6,
    max_iterations=10000
):

    X = X_initial.copy()

    history = [energy_objective(X)]

    for iteration in range(1, max_iterations + 1):

        gradient = energy_gradient(X)

        X_candidate = X - alpha * gradient

        X_new = enforce_constraints(
            X,
            X_candidate
        )

        history.append(
            energy_objective(X_new)
        )

        if np.linalg.norm(X_new - X) < tolerance:
            return X_new, history, iteration

        X = X_new

    return X, history, max_iterations


# -----------------------------------
# Heavy-Ball
# -----------------------------------
def heavy_ball(
    X_initial,
    alpha=0.1,
    beta=0.4,
    tolerance=1e-6,
    max_iterations=10000
):

    X_previous = X_initial.copy()
    X = X_initial.copy()

    history = [energy_objective(X)]

    for iteration in range(1, max_iterations + 1):

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

        if np.linalg.norm(X_new - X) < tolerance:
            return X_new, history, iteration

        X_previous = X
        X = X_new

    return X, history, max_iterations


# -----------------------------------
# Run algorithms
# -----------------------------------
X_gd, gd_history, gd_iterations = gradient_descent(
    X_initial
)

X_hb, hb_history, hb_iterations = heavy_ball(
    X_initial
)


# -----------------------------------
# Convergence graph
# -----------------------------------
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
plt.ylabel("Energy")

plt.title(
    "Convergence Comparison: "
    "Gradient Descent vs Heavy-Ball"
)

plt.grid(True)
plt.legend()

plt.show()


# -----------------------------------
# Print comparison
# -----------------------------------
print("Gradient Descent iterations:", gd_iterations)
print("Heavy-Ball iterations:", hb_iterations)

print(
    "GD final energy:",
    round(gd_history[-1], 4)
)

print(
    "Heavy-Ball final energy:",
    round(hb_history[-1], 4)
)