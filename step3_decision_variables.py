import numpy as np

# Fixed points
start = np.array([5.0, 10.0])
destination = np.array([90.0, 85.0])

# Initial intermediate waypoints
P1 = np.array([25.0, 25.0])
P2 = np.array([35.0, 72.0])
P3 = np.array([60.0, 72.0])
P4 = np.array([75.0, 80.0])

# Decision-variable vector
X = np.array([
    P1[0], P1[1],
    P2[0], P2[1],
    P3[0], P3[1],
    P4[0], P4[1]
])

print("Start point:", start)
print("Destination:", destination)

print("\nDecision variable vector X:")
print(X)

print("\nNumber of optimization variables:", len(X))