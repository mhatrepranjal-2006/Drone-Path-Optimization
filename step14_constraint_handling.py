import numpy as np

# -----------------------------------
# Fixed points
# -----------------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# Initial feasible solution
X_initial = np.array([
    25.0, 25.0,
    35.0, 72.0,
    60.0, 72.0,
    75.0, 80.0
])

# -----------------------------------
# Spatial feasible-region bounds
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
# Projection onto spatial constraints
# -----------------------------------
def project_spatial_constraints(X):

    return np.minimum(
        np.maximum(X, lower),
        upper
    )


# -----------------------------------
# Complete constraint handling
# -----------------------------------
def enforce_constraints(X_current, X_candidate):

    # Step 1: Spatial projection
    X_feasible = project_spatial_constraints(
        X_candidate
    )

    # Step 2: Battery safeguard
    while energy_objective(X_feasible) > battery_capacity:

        # Move halfway back toward
        # the previous feasible solution
        X_feasible = 0.5 * (
            X_current + X_feasible
        )

    return X_feasible


# -----------------------------------
# Example of an invalid candidate
# -----------------------------------
X_invalid = np.array([
    40.0, -10.0,
    45.0, 60.0,
    50.0, 65.0,
    55.0, 110.0
])

print("Invalid candidate:")
print(X_invalid)

# Apply only spatial projection
X_projected = project_spatial_constraints(
    X_invalid
)

print("\nAfter spatial projection:")
print(X_projected)

print(
    "\nEnergy after spatial projection:",
    energy_objective(X_projected)
)

# Apply complete constraint handling
X_safe = enforce_constraints(
    X_initial,
    X_invalid
)

print("\nAfter complete constraint handling:")
print(np.round(X_safe, 4))

print(
    "Final energy:",
    round(energy_objective(X_safe), 4)
)

print(
    "Battery:",
    "VALID"
    if energy_objective(X_safe) <= battery_capacity
    else "INVALID"
)

print(
    "Spatial constraints:",
    "VALID"
    if np.all((X_safe >= lower) & (X_safe <= upper))
    else "INVALID"
)