import numpy as np
import matplotlib.pyplot as plt

# Time parameters
time_steps = 100  # Number of time steps to simulate
dt = 0.1  # Time increment
time = np.arange(0, time_steps * dt, dt)

# Initialize variables
P = np.zeros_like(time)  # Population
E = np.zeros_like(time)  # Energy level
T = np.zeros_like(time)  # Technological advancement
R = np.zeros_like(time)  # Resource availability
C = np.zeros_like(time)  # Risk factor
S = np.zeros_like(time)  # Survivability

# Initial conditions
P[0] = 1000  # Initial population size
E[0] = 1     # Initial energy level
T[0] = 1     # Initial technological level
R[0] = 1000  # Initial resource availability
C[0] = 0.01  # Initial risk factor
S[0] = 1     # Initial survivability (100%)

# Parameters
alpha = 0.01  # Energy growth factor from technology
beta = 0.001  # Risk impact on energy growth
gamma = 0.005  # Population driving technological growth
delta = 0.0001  # Risk impact on technology growth
eta = 0.01  # Resource depletion rate
zeta = 0.0001  # Risk scaling factor based on energy

# Simulation loop
for t in range(1, len(time)):
    # Energy growth
    dE_dt = alpha * T[t-1] - beta * C[t-1]
    E[t] = E[t-1] + dE_dt * dt

    # Technological growth
    dT_dt = gamma * P[t-1] - delta * C[t-1]
    T[t] = T[t-1] + dT_dt * dt

    # Resource depletion
    dR_dt = -eta * E[t-1]
    R[t] = R[t-1] + dR_dt * dt

    # Risk factor
    C[t] = zeta * E[t-1]**2

    # Survivability
    S[t] = S[t-1] * np.exp(-C[t] * dt)  # Exponential decay based on risk

    # Population (simplified: declines if survivability drops significantly)
    if S[t] < 0.1:
        P[t] = P[t-1] * 0.99  # Decline population if survival is low
    else:
        P[t] = P[t-1]  # Keep population constant for now

# Plot results
plt.figure(figsize=(12, 8))

# Energy level
plt.subplot(2, 2, 1)
plt.plot(time, E, label="Energy Level")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy Level Over Time")
plt.grid(True)

# Technology level
plt.subplot(2, 2, 2)
plt.plot(time, T, label="Technology Level")
plt.xlabel("Time")
plt.ylabel("Technology")
plt.title("Technology Level Over Time")
plt.grid(True)

# Resources
plt.subplot(2, 2, 3)
plt.plot(time, R, label="Resources")
plt.xlabel("Time")
plt.ylabel("Resources")
plt.title("Resource Availability Over Time")
plt.grid(True)

# Survivability
plt.subplot(2, 2, 4)
plt.plot(time, S, label="Survivability")
plt.xlabel("Time")
plt.ylabel("Survivability")
plt.title("Survivability Over Time")
plt.grid(True)

plt.tight_layout()
plt.show()
