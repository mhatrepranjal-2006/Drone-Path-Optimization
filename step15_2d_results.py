import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Fixed points
# -----------------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# -----------------------------------
# Initial path
# -----------------------------------
X_initial = np.array([
    25.0, 25.0,
    35.0, 72.0,
    60.0, 72.0,
    75.0, 80.0
])

# -----------------------------------
# Spatial constraint bounds
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
# Path length
# -----------------------------------
def path_length(X):

    path = np.vstack([
        start,
        X.reshape(-1, 2),
        destination
    ])

    differences = path[1:] - path[:-1]

    segment_lengths = np.linalg.norm(
        differences,
        axis=1
    )

    return np.sum(segment_lengths)


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
# Spatial projection
# -----------------------------------
def project_spatial(X):

    return np.minimum(
        np.maximum(X, lower),
        upper
    )


# -----------------------------------
# Complete constraint handling
# -----------------------------------
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
# Run both algorithms
# -----------------------------------
X_gd, gd_history, gd_iterations = gradient_descent(
    X_initial
)

X_hb, hb_history, hb_iterations = heavy_ball(
    X_initial
)


# -----------------------------------
# Results
# -----------------------------------
initial_energy = energy_objective(X_initial)
gd_energy = energy_objective(X_gd)
hb_energy = energy_objective(X_hb)

initial_length = path_length(X_initial)
gd_length = path_length(X_gd)
hb_length = path_length(X_hb)


print("\n----- 2D RESULTS -----")

print("\nInitial Path")
print("Energy:", round(initial_energy, 4))
print("Path Length:", round(initial_length, 4))

print("\nGradient Descent")
print("Energy:", round(gd_energy, 4))
print("Path Length:", round(gd_length, 4))
print("Iterations:", gd_iterations)

print("\nHeavy-Ball")
print("Energy:", round(hb_energy, 4))
print("Path Length:", round(hb_length, 4))
print("Iterations:", hb_iterations)


# -----------------------------------
# Plot all three paths
# -----------------------------------
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

# No-fly zone
no_fly_zone = plt.Rectangle(
    (35, 30),
    20,
    35,
    fill=False,
    hatch="//",
    linewidth=2,
    label="No-Fly Zone"
)

plt.gca().add_patch(no_fly_zone)

# Initial path
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

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

plt.title(
    "Initial vs Optimized 2D Drone Paths"
)

plt.grid(True)
plt.legend()
plt.axis("equal")

plt.show()