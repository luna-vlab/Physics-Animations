'''!    @file       double_pendulum.py
        
        @brief      A Python simulation of a double pendulum system.
        @details    This script numerically solves the equations of motion for 
                    a double pendulum and visualizes their motion using Matplotlib. 
                    
                    The script displays two pendulums side by side with slightly 
                    different initial angles, demonstrating how even tiny changes 
                    in starting conditions lead to dramatically different behavior. 
                    Trails, labels, and adjustable damping are included to help 
                    illustrate the concept of sensitive dependence on initial 
                    conditions and chaotic motion. 
        
        @author     Luna v Lab
        @date       2025-11-13 Original file
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
g = 9.81 # gravity (m/s^2)
L1 = 1.0 # length of first rod (m)
L2 = 1.0 # length of second rod (m)
m1 = 1.0 # mass 1 (kg)
m2 = 1.0 # mass 2 (kg)

DAMPING = 0.02  # try 0.0, 0.01, 0.02,...

# Initial conditions for Pendulum A
theta1_A0 = np.pi / 2           # 90 deg
theta2_A0 = np.pi / 2 + 0.01    # slightly offset
omega1_A0 = 0.0
omega2_A0 = 0.0

# Initial conditions for Pendulum B
theta1_B0 = np.pi / 2
theta2_B0 = np.pi / 2 + 0.03
omega1_B0 = 0.0
omega2_B0 = 0.0

# Time settings
dt = 1 / 120
frames = 3000
trail_length = 200

# Horizontal offset so they appear side by side
arm_len = L1 + L2
offset = arm_len + 0.5   # distance from center to each base (left/right)


# Equations of motion with damping
# state = [theta1, omega1, theta2, omega2]
def derivatives(state, damping=0.0):
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1

    dydt = np.zeros_like(state)

    dydt[0] = omega1
    dydt[2] = omega2

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2

    # d2theta1/dt2
    dydt[1] = (
        m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * L2 * omega2**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / den1

    # denominator for theta2
    den2 = (L2 / L1) * den1

    # d2theta2/dt2
    dydt[3] = (
        -m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
        - (m1 + m2) * L1 * omega1**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta2)
    ) / den2

    # simple viscous damping
    if damping != 0.0:
        dydt[1] -= damping * omega1
        dydt[3] -= damping * omega2

    return dydt

def rk4_step(state, dt, damping=0.0):
    k1 = derivatives(state, damping)
    k2 = derivatives(state + 0.5 * dt * k1, damping)
    k3 = derivatives(state + 0.5 * dt * k2, damping)
    k4 = derivatives(state + dt * k3, damping)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# States and trails
state_A = np.array([theta1_A0, omega1_A0, theta2_A0, omega2_A0], dtype=float)
state_B = np.array([theta1_B0, omega1_B0, theta2_B0, omega2_B0], dtype=float)

trail_A_x, trail_A_y = [], []
trail_B_x, trail_B_y = [], []

# Figure setup
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect("equal", "box")

# Both pendulums side by side
x_max = offset + arm_len + 0.3
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-(arm_len + 0.5), arm_len + 0.8)

ax.set_xticks([])
ax.set_yticks([])

ax.set_title("DOUBLE PENDULUM", fontsize=20, pad=10)

# Pendulum A (left, cyan-ish)
rod_A, = ax.plot([], [], lw=2.5, marker="o", markersize=5)
bob2_A, = ax.plot([], [], "o", markersize=7)
trail_A, = ax.plot([], [], lw=1.2, alpha=0.8)

rod_A.set_color("cyan")
bob2_A.set_color("cyan")
trail_A.set_color("cyan")

# Pendulum B (right, orange-ish)
rod_B, = ax.plot([], [], lw=2.5, marker="o", markersize=5)
bob2_B, = ax.plot([], [], "o", markersize=7)
trail_B, = ax.plot([], [], lw=1.2, alpha=0.8)

rod_B.set_color("orange")
bob2_B.set_color("orange")
trail_B.set_color("orange")

# Base positions (left and right)
base_A = (-offset, 0.0)
base_B = (+offset, 0.0)

# Labels above each pendulum
deg = 180 / np.pi

label_A = ax.text(
    0.25, 0.95,
    "",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=14
)

label_B = ax.text(
    0.75, 0.95,
    "",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=14
)

def init():
    rod_A.set_data([], [])
    bob2_A.set_data([], [])
    trail_A.set_data([], [])

    rod_B.set_data([], [])
    bob2_B.set_data([], [])
    trail_B.set_data([], [])

    # Plain, readable text with degrees
    label_A.set_text(
        f"Pendulum A: \nθ1 = {theta1_A0 * deg:.0f}° \nθ2 = {theta2_A0 * deg:.2f}°"
    )
    label_B.set_text(
        f"Pendulum B: \nθ1 = {theta1_B0 * deg:.0f}° \nθ2 = {theta2_B0 * deg:.2f}°"
    )

    return rod_A, bob2_A, trail_A, rod_B, bob2_B, trail_B, label_A, label_B


def update(frame):
    global state_A, state_B, trail_A_x, trail_A_y, trail_B_x, trail_B_y

    # Step both systems
    state_A[:] = rk4_step(state_A, dt, DAMPING)
    state_B[:] = rk4_step(state_B, dt, DAMPING)

    theta1_A, omega1_A, theta2_A, omega2_A = state_A
    theta1_B, omega1_B, theta2_B, omega2_B = state_B

    # --- Pendulum A (left) relative coordinates
    x1A_rel = L1 * np.sin(theta1_A)
    y1A_rel = -L1 * np.cos(theta1_A)
    x2A_rel = x1A_rel + L2 * np.sin(theta2_A)
    y2A_rel = y1A_rel - L2 * np.cos(theta2_A)

    # Shift by base_A
    x1_A = base_A[0] + x1A_rel
    y1_A = base_A[1] + y1A_rel
    x2_A = base_A[0] + x2A_rel
    y2_A = base_A[1] + y2A_rel

    # --- Pendulum B (right) relative coordinates
    x1B_rel = L1 * np.sin(theta1_B)
    y1B_rel = -L1 * np.cos(theta1_B)
    x2B_rel = x1B_rel + L2 * np.sin(theta2_B)
    y2B_rel = y1B_rel - L2 * np.cos(theta2_B)

    # Shift by base_B
    x1_B = base_B[0] + x1B_rel
    y1_B = base_B[1] + y1B_rel
    x2_B = base_B[0] + x2B_rel
    y2_B = base_B[1] + y2B_rel

    # Update rods
    rod_A.set_data([base_A[0], x1_A, x2_A], [base_A[1], y1_A, y2_A])
    rod_B.set_data([base_B[0], x1_B, x2_B], [base_B[1], y1_B, y2_B])

    # Update bobs
    bob2_A.set_data([x2_A], [y2_A])
    bob2_B.set_data([x2_B], [y2_B])

    # Trails (second bob only)
    trail_A_x.append(x2_A)
    trail_A_y.append(y2_A)
    trail_B_x.append(x2_B)
    trail_B_y.append(y2_B)

    if len(trail_A_x) > trail_length:
        trail_A_x = trail_A_x[-trail_length:]
        trail_A_y = trail_A_y[-trail_length:]

    if len(trail_B_x) > trail_length:
        trail_B_x = trail_B_x[-trail_length:]
        trail_B_y = trail_B_y[-trail_length:]

    trail_A.set_data(trail_A_x, trail_A_y)
    trail_B.set_data(trail_B_x, trail_B_y)

    return rod_A, bob2_A, trail_A, rod_B, bob2_B, trail_B, label_A, label_B

anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=frames,
    interval=1000 * dt,
    blit=True
)

plt.tight_layout()
plt.show()
