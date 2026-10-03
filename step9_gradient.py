import numpy as np

# -----------------------------------
# Fixed points
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


# -----------------------------------
# Gradient of energy objective
# -----------------------------------
def energy_gradient(X):

    waypoints = X.reshape(-1, 2)

    # Complete path
    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    gradient = []

    # Gradient for each intermediate waypoint
    for i in range(1, len(path) - 1):

        previous_point = path[i - 1]
        current_point = path[i]
        next_point = path[i + 1]

        grad_point = (
            4 * current_point
            - 2 * previous_point
            - 2 * next_point
        )

        gradient.extend(grad_point)

    return np.array(gradient)


gradient = energy_gradient(X)

print("Initial decision vector X:")
print(X)

print("\nGradient:")
print(gradient)