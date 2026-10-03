import numpy as np

# -----------------------------------
# Fixed start and destination
# -----------------------------------
start = np.array([5.0, 10.0, 10.0])
destination = np.array([90.0, 85.0, 10.0])

# -----------------------------------
# Initial 3D decision vector
# Variable order:
# [x1, y1, z1,
#  x2, y2, z2,
#  x3, y3, z3,
#  x4, y4, z4]
# -----------------------------------
X = np.array([
    20.0, 40.0, 20.0,
    35.0, 70.0, 25.0,
    55.0, 75.0, 25.0,
    72.5, 80.0, 20.0
])

# Relative altitude-change cost
gamma = 2.0


# -----------------------------------
# 3D Energy Objective
# -----------------------------------
def energy_3d(X):

    waypoints = X.reshape(-1, 3)

    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    total_energy = 0.0

    for i in range(len(path) - 1):

        dx = path[i + 1, 0] - path[i, 0]
        dy = path[i + 1, 1] - path[i, 1]
        dz = path[i + 1, 2] - path[i, 2]

        segment_energy = (
            dx**2
            + dy**2
            + gamma * dz**2
        )

        total_energy += segment_energy

        print(
            f"Segment {i + 1}: "
            f"dx={dx:.2f}, "
            f"dy={dy:.2f}, "
            f"dz={dz:.2f}, "
            f"energy={segment_energy:.2f}"
        )

    return total_energy


# -----------------------------------
# Calculate total 3D energy
# -----------------------------------
total_energy = energy_3d(X)

print("\nNumber of decision variables:", len(X))
print("Gamma:", gamma)
print("Initial 3D energy:", total_energy)