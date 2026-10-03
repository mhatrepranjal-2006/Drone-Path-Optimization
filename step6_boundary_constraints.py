import numpy as np

# -----------------------------
# Environment boundaries
# -----------------------------
ENV_MIN = 0.0
ENV_MAX = 100.0

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
# Boundary constraint check
# -----------------------------
def check_boundary_constraints(X):

    waypoints = X.reshape(-1, 2)

    all_valid = True

    for i, point in enumerate(waypoints, start=1):

        x = point[0]
        y = point[1]

        valid = (
            ENV_MIN <= x <= ENV_MAX
            and
            ENV_MIN <= y <= ENV_MAX
        )

        print(
            f"P{i} = ({x:.1f}, {y:.1f}) -> "
            f"{'VALID' if valid else 'INVALID'}"
        )

        if not valid:
            all_valid = False

    return all_valid


# Check all waypoint boundaries
boundary_valid = check_boundary_constraints(X)

print("\nOverall boundary constraint:",
      "VALID" if boundary_valid else "INVALID")