import numpy as np

# -----------------------------------
# Fixed start and destination
# -----------------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# -----------------------------------
# Initial decision variables
# -----------------------------------
X = np.array([
    25.0, 25.0,   # P1
    35.0, 72.0,   # P2
    60.0, 72.0,   # P3
    75.0, 80.0    # P4
])

# Battery capacity
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
# Complete feasibility check
# -----------------------------------
waypoints = X.reshape(-1, 2)

P1, P2, P3, P4 = waypoints

energy = energy_objective(X)

# Battery
battery_valid = energy <= battery_capacity

# Environment boundaries
boundary_valid = np.all(
    (waypoints >= 0) & (waypoints <= 100)
)

# Safe corridor
corridor_valid = (
    P1[0] <= 30
    and P2[0] <= 35
    and P2[1] >= 70
    and P3[0] >= 55
    and P3[1] >= 70
    and P4[0] >= 60
)

print("Decision vector X:")
print(X)

print("\nEnergy:", energy)

print(
    "Battery constraint:",
    "VALID" if battery_valid else "INVALID"
)

print(
    "Boundary constraints:",
    "VALID" if boundary_valid else "INVALID"
)

print(
    "No-fly safe corridor:",
    "VALID" if corridor_valid else "INVALID"
)

all_valid = (
    battery_valid
    and boundary_valid
    and corridor_valid
)

print(
    "\nOverall initial solution:",
    "FEASIBLE" if all_valid else "INFEASIBLE"
)