# 🚁 Drone Path Optimization

A mathematical and algorithmic framework for optimizing drone paths under spatial, energy, battery, and no-fly-zone constraints.

The project models a drone moving from a fixed start point to a fixed destination through intermediate waypoints. Gradient Descent and Heavy-Ball Momentum are used to optimize the waypoint positions, with the formulation extended from 2D to 3D.

---

## 👥 Authors

### Aryan Dubey
GitHub: [Aryan2624](https://github.com/Aryan2624)

### Pranjal Mhatre
GitHub: [mhatrepranjal-2006](https://github.com/mhatrepranjal-2006)

Department of Artificial Intelligence & Machine Learning  
Universal Skill Tech University, Mumbai, India

---

## 📌 Project Overview

Drone path planning is an optimization problem where a suitable trajectory must be found between a start point and a destination while satisfying practical constraints.

This project formulates the path using four intermediate waypoints and minimizes a quadratic energy proxy based on the squared displacement between consecutive points.

The project includes:

- 2D drone path optimization
- 3D drone path optimization
- Gradient Descent
- Heavy-Ball Momentum
- Quadratic energy objective
- Spatial waypoint constraints
- Battery-capacity constraint
- No-fly-zone validation
- Hessian and convexity analysis
- KKT optimality validation
- Scenario-based testing
- Algorithm convergence comparison
- 2D and 3D path visualization

---

## 🎯 Objectives

The main objectives of the project are:

1. Formulate drone path planning as a constrained optimization problem.
2. Minimize a mathematical energy proxy for the drone trajectory.
3. Respect waypoint and spatial constraints.
4. Enforce the available battery-energy limit.
5. Validate paths against a rectangular no-fly zone.
6. Compare Gradient Descent and Heavy-Ball Momentum.
7. Extend the optimization problem from 2D to 3D.
8. Analyze convexity using the Hessian.
9. Verify optimality using KKT conditions.
10. Test the optimization under different initial-path scenarios.

---

# 🧮 Mathematical Formulation

## 2D Path Representation

The fixed start and destination points are:

\[
S=(5,10)
\]

\[
D=(90,85)
\]

Four intermediate waypoints are represented as:

\[
P_i=(x_i,y_i), \qquad i=1,2,3,4
\]

The complete path is:

\[
S \rightarrow P_1 \rightarrow P_2 \rightarrow P_3
\rightarrow P_4 \rightarrow D
\]

The optimization vector is:

\[
X=
[x_1,y_1,x_2,y_2,x_3,y_3,x_4,y_4]^T
\]

---

## ⚡ Energy Objective

The project uses a simplified quadratic energy proxy:

\[
E(X)=
\sum_{i=0}^{4}
\left\|Q_{i+1}-Q_i\right\|_2^2
\]

where:

- \(Q_0=S\)
- \(Q_i=P_i\)
- \(Q_5=D\)

The objective penalizes large movements between consecutive points.

> **Note:** This is a mathematical optimization proxy, not a physically calibrated drone battery-power model.

---

## 🔋 Battery Constraint

The battery capacity is:

\[
E(X)\leq4500
\]

The optimized solutions remain within this capacity.

---

## 📐 Spatial Constraints

Waypoint coordinates are restricted using lower and upper bounds.

### Lower bounds

```text
[0, 0, 0, 70, 55, 70, 60, 0]
