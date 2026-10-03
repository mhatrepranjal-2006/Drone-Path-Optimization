import numpy as np

# -----------------------------------
# Hessian matrix
# Variable order:
# [x1, y1, x2, y2, x3, y3, x4, y4]
# -----------------------------------

H = np.array([
    [ 4,  0, -2,  0,  0,  0,  0,  0],
    [ 0,  4,  0, -2,  0,  0,  0,  0],

    [-2,  0,  4,  0, -2,  0,  0,  0],
    [ 0, -2,  0,  4,  0, -2,  0,  0],

    [ 0,  0, -2,  0,  4,  0, -2,  0],
    [ 0,  0,  0, -2,  0,  4,  0, -2],

    [ 0,  0,  0,  0, -2,  0,  4,  0],
    [ 0,  0,  0,  0,  0, -2,  0,  4]
], dtype=float)

print("Hessian Matrix:")
print(H)

# -----------------------------------
# Eigenvalues
# -----------------------------------
eigenvalues = np.linalg.eigvalsh(H)

print("\nEigenvalues:")
print(np.round(eigenvalues, 6))

# -----------------------------------
# Convexity test
# -----------------------------------
if np.all(eigenvalues > 0):
    print("\nHessian is POSITIVE DEFINITE.")
    print("Objective function is STRONGLY CONVEX.")

elif np.all(eigenvalues >= 0):
    print("\nHessian is POSITIVE SEMIDEFINITE.")
    print("Objective function is CONVEX.")

else:
    print("\nHessian is not positive semidefinite.")
    print("Objective function is NOT convex.")