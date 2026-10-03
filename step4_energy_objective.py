import numpy as np

# -----------------------------
# Fixed points
# -----------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# -----------------------------
# Decision variable vector
# -----------------------------
X = np.array([
    25.0, 25.0,   # P1
    35.0, 72.0,   # P2
    60.0, 72.0,   # P3
    75.0, 80.0    # P4
])


# -----------------------------
# Energy objective function
# -----------------------------
def energy_objective(X):
    # Convert X into four (x, y) waypoints
    waypoints = X.reshape(-1, 2)

    # Complete path:
    # Start -> waypoints -> Destination
    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    total_energy = 0.0

    # Calculate energy for each path segment
    for i in range(len(path) - 1):

        dx = path[i + 1, 0] - path[i, 0]
        dy = path[i + 1, 1] - path[i, 1]

        segment_energy = dx**2 + dy**2

        total_energy += segment_energy

        print(
            f"Segment {i + 1}: "
            f"dx = {dx:.2f}, "
            f"dy = {dy:.2f}, "
            f"energy = {segment_energy:.2f}"
        )

    return total_energy


# Calculate initial path energy
initial_energy = energy_objective(X)

print("\nInitial total energy:", initial_energy)