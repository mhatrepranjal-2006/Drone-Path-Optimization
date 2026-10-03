import numpy as np

# -----------------------------
# Fixed points
# -----------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# -----------------------------
# Decision variables
# -----------------------------
X = np.array([
    25.0, 25.0,   # P1
    35.0, 72.0,   # P2
    60.0, 72.0,   # P3
    75.0, 80.0    # P4
])

# -----------------------------
# Battery capacity
# -----------------------------
battery_capacity = 4500.0


# -----------------------------
# Energy objective function
# -----------------------------
def energy_objective(X):
    waypoints = X.reshape(-1, 2)

    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    total_energy = 0.0

    for i in range(len(path) - 1):

        dx = path[i + 1, 0] - path[i, 0]
        dy = path[i + 1, 1] - path[i, 1]

        total_energy += dx**2 + dy**2

    return total_energy


# -----------------------------
# Battery constraint check
# -----------------------------
energy = energy_objective(X)

print("Path energy:", energy)
print("Battery capacity:", battery_capacity)

if energy <= battery_capacity:
    print("Battery constraint: VALID")
else:
    print("Battery constraint: INVALID")