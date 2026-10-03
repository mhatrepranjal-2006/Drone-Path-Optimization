# 🚁 Drone Path Optimization

## Energy-Efficient 2D and 3D Drone Delivery Path Optimization  
### Under No-Fly-Zone and Battery Constraints Using Gradient Descent and Heavy-Ball Momentum

This project is a **software-based optimization simulation** developed in Python.

The goal is to find an energy-efficient path for a delivery drone while satisfying:

- No-fly-zone constraints
- Battery constraints
- Environment boundaries
- Altitude constraints in 3D

The project compares two optimization algorithms:

- Gradient Descent
- Heavy-Ball Momentum

---

# 🎯 Objective

The main objective is:

> Minimize the simulated energy required for the drone to travel from a fixed start point to a fixed destination.

The drone must also:

- Reach the destination
- Avoid the no-fly zone
- Stay within the allowed environment
- Stay within the battery limit
- Respect altitude limits in 3D

---

# 💻 Project Type

This project is completely **software and simulation based**.

There is no:

- Physical drone
- Arduino
- Raspberry Pi
- GPS hardware
- Motors
- Sensors
- Real flight testing

Everything is simulated using Python.

---

# 🧠 Optimization Model

The drone travels through four intermediate waypoints.

## 2D Waypoints

```text
P1 = (x1, y1)
P2 = (x2, y2)
P3 = (x3, y3)
P4 = (x4, y4)
