'''!    @file       lorenz_attractor.py
        
        @brief      Visualization of Lorenz attractor
        @details    Implements the Lorenz system using a fourth-order
                    Runge–Kutta (RK4) integrator and generates a 3D animation
                    of both trajectories.
        
        @author     Luna v Lab
        @date       2025-11-17 Original file
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Parameters and setup
sigma = 10.0
rho   = 28.0
beta  = 8.0 / 3.0

dt = 0.01
num_steps = 15000   # slightly fewer steps (less lag)

# Initial conditions (almost the same)
x0_1, y0_1, z0_1 = 0.1,   0.0, 0.0
x0_2, y0_2, z0_2 = 0.101, 0.0, 0.0   # tiny offset in x

# Lorenz step (RK4)
def lorenz_step(x, y, z, dt, sigma, beta, rho):
    # Lorenz system derivatives: dx/dt, dy/dt, dz/dt
    def f(x, y, z):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz
    
    # Runge–Kutta 4th order integration:
    # evaluate slopes at 4 different points
    k1x, k1y, k1z = f(x, y, z)
    k2x, k2y, k2z = f(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z)
    k3x, k3y, k3z = f(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z)
    k4x, k4y, k4z = f(x + dt*k3x,     y + dt*k3y,     z + dt*k3z)

    # Weighted average of slopes → next point in trajectory
    x_next = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    y_next = y + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
    z_next = z + (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return x_next, y_next, z_next

# Integrate both trajectories
xs1 = np.empty(num_steps)
ys1 = np.empty(num_steps)
zs1 = np.empty(num_steps)

xs2 = np.empty(num_steps)
ys2 = np.empty(num_steps)
zs2 = np.empty(num_steps)

xs1[0], ys1[0], zs1[0] = x0_1, y0_1, z0_1
xs2[0], ys2[0], zs2[0] = x0_2, y0_2, z0_2

for i in range(1, num_steps):
    xs1[i], ys1[i], zs1[i] = lorenz_step(xs1[i-1], ys1[i-1], zs1[i-1],
                                         dt, sigma, beta, rho)
    xs2[i], ys2[i], zs2[i] = lorenz_step(xs2[i-1], ys2[i-1], zs2[i-1],
                                         dt, sigma, beta, rho)

# Figure & axis
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_pane_color((0.2, 0.2, 0.2, 0.7))  # (R, G, B, alpha)

# Dark background
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

all_x = np.concatenate([xs1, xs2])
all_y = np.concatenate([ys1, ys2])
all_z = np.concatenate([zs1, zs2])

# Extra spacing so trajectories don't touch the edges
pad = 2
ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
ax.set_zlim(all_z.min() - pad, all_z.max() + pad)

ax.set_xlabel("X", color="white", fontsize=11)
ax.set_ylabel("Y", color="white", fontsize=11)
ax.set_zlabel("Z", color="white", fontsize=11)

# Remove axis tick numbers
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Lines for trajectories
label1 = f"IC 1: x0={x0_1:.3f}, y0={y0_1:.0f}, z0={z0_1:.0f}"
label2 = f"IC 2: x0={x0_2:.3f}, y0={y0_2:.0f}, z0={z0_2:.0f}"

line1, = ax.plot(xs1[0:1], ys1[0:1], zs1[0:1],
                 lw=1, label=label1, color="purple")
line2, = ax.plot(xs2[0:1], ys2[0:1], zs2[0:1],
                 lw=1, label=label2, color="deepskyblue")

# Balls at the tip of each trajectory
ball1, = ax.plot(xs1[0:1], ys1[0:1], zs1[0:1],
                 marker="o", markersize=4, color="purple")
ball2, = ax.plot(xs2[0:1], ys2[0:1], zs2[0:1],
                 marker="o", markersize=4, color="deepskyblue")

# Legend in lower-left, no overlap with parameter box
leg = ax.legend(loc="lower left", fontsize=15)
leg.get_frame().set_facecolor("black")
leg.get_frame().set_edgecolor("white")
for text in leg.get_texts():
    text.set_color("white")

# Display parameters
param_text = (
    f"sigma = {sigma}\n"
    f"rho   = {rho}\n"
    f"beta  = {beta:.3f}"
)
ax.text2D(
    0.02, 0.95, param_text,
    transform=ax.transAxes,
    fontsize=14,
    va="top",
    color="white",
    bbox=dict(boxstyle="round", facecolor="black",
              edgecolor="white", alpha=0.7)
)

points_per_frame = 5
num_frames = num_steps // points_per_frame

# Camera motion parameters
base_elev = 20
base_azim = 45
azim_speed = 0.3 # degrees per frame (subtle)

# Animation callbacks
def init():
    # Start each trajectory with the very first point only
    line1.set_data(xs1[0:1], ys1[0:1])
    line1.set_3d_properties(zs1[0:1])

    line2.set_data(xs2[0:1], ys2[0:1])
    line2.set_3d_properties(zs2[0:1])

    # Place the tip markers (balls) at the starting points
    ball1.set_data(xs1[0:1], ys1[0:1])
    ball1.set_3d_properties(zs1[0:1])

    ball2.set_data(xs2[0:1], ys2[0:1])
    ball2.set_3d_properties(zs2[0:1])

    # Set the initial camera angle before animation begins
    ax.view_init(elev=base_elev, azim=base_azim)
    return line1, line2, ball1, ball2


def update(frame):
    idx = 1 + frame * points_per_frame
    if idx > num_steps:
        idx = num_steps

    # Update lines
    line1.set_data(xs1[:idx], ys1[:idx])
    line1.set_3d_properties(zs1[:idx])

    line2.set_data(xs2[:idx], ys2[:idx])
    line2.set_3d_properties(zs2[:idx])

    # Update balls at the tips
    ball1.set_data(xs1[idx-1:idx], ys1[idx-1:idx])
    ball1.set_3d_properties(zs1[idx-1:idx])

    ball2.set_data(xs2[idx-1:idx], ys2[idx-1:idx])
    ball2.set_3d_properties(zs2[idx-1:idx])

    # Subtle camera rotation
    azim = base_azim + azim_speed * frame
    ax.view_init(elev=base_elev, azim=azim)

    return line1, line2, ball1, ball2


anim = FuncAnimation(
    fig,
    update, # updates the lines each frame
    init_func = init, # what to draw before animation starts
    frames = num_frames, # total number of animation frames
    interval = 30, # delay between frames (slower and smoother)
    blit = False,
)

plt.tight_layout()
plt.show()