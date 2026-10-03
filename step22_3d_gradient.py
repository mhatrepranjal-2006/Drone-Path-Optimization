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
    20.0, 40.0, 20.0,    # P1
    35.0, 70.0, 25.0,    # P2
    55.0, 75.0, 25.0,    # P3
    72.5, 80.0, 20.0     # P4
])

# Relative altitude cost
gamma = 2.0


# -----------------------------------
# 3D Gradient
# -----------------------------------
def energy_gradient_3d(X):

    waypoints = X.reshape(-1, 3)

    path = np.vstack([
        start,
        waypoints,
        destination
    ])

    gradient = []

    for i in range(1, len(path) - 1):

        previous_point = path[i - 1]
        current_point = path[i]
        next_point = path[i + 1]

        # X-coordinate gradient
        grad_x = (
            4 * current_point[0]
            - 2 * previous_point[0]
            - 2 * next_point[0]
        )

        # Y-coordinate gradient
        grad_y = (
            4 * current_point[1]
            - 2 * previous_point[1]
            - 2 * next_point[1]
        )

        # Z-coordinate gradient
        grad_z = gamma * (
            4 * current_point[2]
            - 2 * previous_point[2]
            - 2 * next_point[2]
        )

        gradient.extend([
            grad_x,
            grad_y,
            grad_z
        ])

    return np.array(gradient)


gradient = energy_gradient_3d(X)

print("Initial 3D decision vector:")
print(X)

print("\n3D Gradient:")
print(gradient)

print("\nGradient by waypoint:")
print(gradient.reshape(-1, 3))