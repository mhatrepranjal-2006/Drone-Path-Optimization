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

# -----------------------------------
# Hessian eigenvalues
# -----------------------------------
eigenvalues = np.linalg.eigvalsh(H)

lambda_min = np.min(eigenvalues)
lambda_max = np.max(eigenvalues)

# -----------------------------------
# Condition number
# -----------------------------------
condition_number = lambda_max / lambda_min

print("Eigenvalues:")
print(np.round(eigenvalues, 6))

print("\nSmallest eigenvalue:",
      round(lambda_min, 6))

print("Largest eigenvalue:",
      round(lambda_max, 6))

print("Condition number:",
      round(condition_number, 4))