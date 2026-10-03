import numpy as np

# -------------------------------------------------
# Fixed start and destination
# -------------------------------------------------
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# -------------------------------------------------
# Final optimized solution
# Order:
# [x1, y1, x2, y2, x3, y3, x4, y4]
# -------------------------------------------------
X_star = np.array([
    20.0, 40.0,
    35.0, 70.0,
    55.0, 75.0,
    72.5, 80.0
])

battery_capacity = 4500.0

# -------------------------------------------------
# Lower and upper bounds
# -------------------------------------------------
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

names = [
    "x1", "y1",
    "x2", "y2",
    "x3", "y3",
    "x4", "y4"
]


# -------------------------------------------------
# Objective function
# -------------------------------------------------
def energy_objective(X):

    path = np.vstack([
        start,
        X.reshape(-1, 2),
        destination
    ])

    differences = path[1:] - path[:-1]

    return np.sum(differences**2)


# -------------------------------------------------
# Gradient
# -------------------------------------------------
def energy_gradient(X):

    path = np.vstack([
        start,
        X.reshape(-1, 2),
        destination
    ])

    gradient = []

    for i in range(1, len(path) - 1):

        grad_point = (
            4 * path[i]
            - 2 * path[i - 1]
            - 2 * path[i + 1]
        )

        gradient.extend(grad_point)

    return np.array(gradient)


# -------------------------------------------------
# Evaluate final solution
# -------------------------------------------------
energy = energy_objective(X_star)
gradient = energy_gradient(X_star)

print("FINAL OPTIMIZED SOLUTION")
print(X_star)

print("\nFinal Energy:")
print(energy)

print("\nGradient at X*:")
print(gradient)


# -------------------------------------------------
# 1. PRIMAL FEASIBILITY
# -------------------------------------------------
battery_residual = energy - battery_capacity

lower_residual = lower - X_star
upper_residual = X_star - upper

battery_valid = battery_residual <= 0

bounds_valid = np.all(
    (lower_residual <= 0)
    &
    (upper_residual <= 0)
)

print("\n------------------------------")
print("1. PRIMAL FEASIBILITY")
print("------------------------------")

print("Battery residual:", battery_residual)
print(
    "Battery:",
    "VALID" if battery_valid else "INVALID"
)

print(
    "Bounds:",
    "VALID" if bounds_valid else "INVALID"
)


# -------------------------------------------------
# 2. ACTIVE / INACTIVE CONSTRAINTS
# -------------------------------------------------
tolerance = 1e-8

print("\n------------------------------")
print("2. ACTIVE CONSTRAINTS")
print("------------------------------")

for i in range(len(X_star)):

    if abs(lower_residual[i]) <= tolerance:

        print(
            f"{names[i]} lower bound ACTIVE"
        )

    if abs(upper_residual[i]) <= tolerance:

        print(
            f"{names[i]} upper bound ACTIVE"
        )


if abs(battery_residual) <= tolerance:
    print("Battery constraint ACTIVE")
else:
    print("Battery constraint INACTIVE")


# -------------------------------------------------
# 3. KKT MULTIPLIERS
#
# Stationarity:
#
# grad(E)
# - mu
# + nu
# + lambda * grad(E)
# = 0
#
# Battery is inactive, therefore lambda = 0.
# -------------------------------------------------

lambda_battery = 0.0

mu = np.zeros(8)   # lower-bound multipliers
nu = np.zeros(8)   # upper-bound multipliers

# Active lower-bound multipliers
mu[3] = 50.0    # y2 >= 70
mu[4] = 5.0     # x3 >= 55

# Active upper-bound multiplier
nu[2] = 10.0    # x2 <= 35


# -------------------------------------------------
# 4. DUAL FEASIBILITY
# -------------------------------------------------
dual_valid = (
    lambda_battery >= 0
    and np.all(mu >= 0)
    and np.all(nu >= 0)
)

print("\n------------------------------")
print("3. DUAL FEASIBILITY")
print("------------------------------")

print("Battery multiplier:", lambda_battery)
print("Lower multipliers:", mu)
print("Upper multipliers:", nu)

print(
    "Dual feasibility:",
    "SATISFIED" if dual_valid else "NOT SATISFIED"
)


# -------------------------------------------------
# 5. STATIONARITY
# -------------------------------------------------
stationarity = (
    (1 + lambda_battery) * gradient
    - mu
    + nu
)

print("\n------------------------------")
print("4. STATIONARITY")
print("------------------------------")

print("Stationarity residual:")
print(stationarity)

stationarity_valid = np.allclose(
    stationarity,
    0,
    atol=1e-8
)

print(
    "Stationarity:",
    "SATISFIED"
    if stationarity_valid
    else "NOT SATISFIED"
)


# -------------------------------------------------
# 6. COMPLEMENTARY SLACKNESS
# -------------------------------------------------

battery_cs = (
    lambda_battery
    *
    battery_residual
)

lower_cs = (
    mu
    *
    lower_residual
)

upper_cs = (
    nu
    *
    upper_residual
)

cs_valid = (
    abs(battery_cs) <= tolerance
    and np.allclose(lower_cs, 0, atol=tolerance)
    and np.allclose(upper_cs, 0, atol=tolerance)
)

print("\n------------------------------")
print("5. COMPLEMENTARY SLACKNESS")
print("------------------------------")

print(
    "Battery:",
    battery_cs
)

print(
    "Lower-bound products:",
    lower_cs
)

print(
    "Upper-bound products:",
    upper_cs
)

print(
    "Complementary slackness:",
    "SATISFIED"
    if cs_valid
    else "NOT SATISFIED"
)


# -------------------------------------------------
# FINAL KKT CHECK
# -------------------------------------------------
kkt_valid = (
    battery_valid
    and bounds_valid
    and dual_valid
    and stationarity_valid
    and cs_valid
)

print("\n==============================")
print("FINAL KKT RESULT")
print("==============================")

print(
    "ALL KKT CONDITIONS SATISFIED"
    if kkt_valid
    else "KKT CONDITIONS NOT FULLY SATISFIED"
)