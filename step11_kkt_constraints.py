import numpy as np

# -----------------------------------
# Initial decision vector
# -----------------------------------
X = np.array([
    25.0, 25.0,
    35.0, 72.0,
    60.0, 72.0,
    75.0, 80.0
])

# Variable names
names = [
    "x1", "y1",
    "x2", "y2",
    "x3", "y3",
    "x4", "y4"
]

# -----------------------------------
# Combined lower and upper bounds
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

# -----------------------------------
# Fixed points
# -----------------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

battery_capacity = 4500.0


def energy_objective(X):

    waypoints = X.reshape(-1, 2)

    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    differences = path[1:] - path[:-1]

    return np.sum(differences**2)


energy = energy_objective(X)

tolerance = 1e-9

print("Energy:", energy)

# Battery activity
if abs(energy - battery_capacity) <= tolerance:
    print("Battery constraint: ACTIVE")
else:
    print("Battery constraint: INACTIVE")


print("\nWaypoint constraint activity:")

for i in range(len(X)):

    lower_active = abs(X[i] - lower[i]) <= tolerance
    upper_active = abs(X[i] - upper[i]) <= tolerance

    print(
        f"{names[i]} = {X[i]:.1f} | "
        f"Lower: {'ACTIVE' if lower_active else 'inactive'} | "
        f"Upper: {'ACTIVE' if upper_active else 'inactive'}"
    )