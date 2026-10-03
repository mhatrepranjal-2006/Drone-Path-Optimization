import numpy as np

# -----------------------------
# Initial decision variables
# -----------------------------
X = np.array([
    25.0, 25.0,   # P1
    35.0, 72.0,   # P2
    60.0, 72.0,   # P3
    75.0, 80.0    # P4
])

# -----------------------------
# Safe-corridor limits
# -----------------------------
LEFT_SAFE_X = 30.0
NFZ_LEFT_X = 35.0
NFZ_RIGHT_X = 55.0
RIGHT_SAFE_X = 60.0
UPPER_SAFE_Y = 70.0


def check_no_fly_corridor(X):

    waypoints = X.reshape(-1, 2)

    P1 = waypoints[0]
    P2 = waypoints[1]
    P3 = waypoints[2]
    P4 = waypoints[3]

    # Safe-corridor constraints
    c1 = P1[0] <= LEFT_SAFE_X

    c2 = P2[0] <= NFZ_LEFT_X
    c3 = P2[1] >= UPPER_SAFE_Y

    c4 = P3[0] >= NFZ_RIGHT_X
    c5 = P3[1] >= UPPER_SAFE_Y

    c6 = P4[0] >= RIGHT_SAFE_X

    checks = [c1, c2, c3, c4, c5, c6]

    print("P1 left approach:", "VALID" if c1 else "INVALID")
    print("P2 left position:", "VALID" if c2 else "INVALID")
    print("P2 above NFZ:", "VALID" if c3 else "INVALID")
    print("P3 right position:", "VALID" if c4 else "INVALID")
    print("P3 above NFZ:", "VALID" if c5 else "INVALID")
    print("P4 right exit:", "VALID" if c6 else "INVALID")

    return all(checks)


# Check no-fly-zone constraints
no_fly_valid = check_no_fly_corridor(X)

print(
    "\nNo-fly-zone safe corridor:",
    "VALID" if no_fly_valid else "INVALID"
)