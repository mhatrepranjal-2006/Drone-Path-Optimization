import numpy as np

# -----------------------------------
# Fixed start and destination
# -----------------------------------
start = np.array([5.0, 10.0, 10.0])
destination = np.array([90.0, 85.0, 10.0])

# -----------------------------------
# Initial 3D decision vector
# -----------------------------------
X = np.array([
    20.0, 40.0, 20.0,
    35.0, 70.0, 25.0,
    55.0, 75.0, 25.0,
    72.5, 80.0, 20.0
])

gamma = 2.0
battery_capacity = 4500.0

# -----------------------------------
# Lower and upper bounds
# -----------------------------------
lower = np.array([
     0.0,   0.0, 10.0,
     0.0,  70.0, 10.0,
    55.0,  70.0, 10.0,
    60.0,   0.0, 10.0
])

upper = np.array([
     30.0, 100.0, 40.0,
     35.0, 100.0, 40.0,
    100.0, 100.0, 40.0,
    100.0, 100.0, 40.0
])


# -----------------------------------
# 3D energy objective
# -----------------------------------
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


# -----------------------------------
# Constraint checks
# -----------------------------------
spatial_valid = np.all(
    (X >= lower) & (X <= upper)
)

energy = energy_3d(X)

battery_valid = (
    energy <= battery_capacity
)

print("Initial 3D energy:", energy)

print(
    "Spatial + altitude + corridor constraints:",
    "VALID" if spatial_valid else "INVALID"
)

print(
    "Battery constraint:",
    "VALID" if battery_valid else "INVALID"
)

overall_valid = (
    spatial_valid and battery_valid
)

print(
    "\nOverall 3D initial solution:",
    "FEASIBLE" if overall_valid else "INFEASIBLE"
)