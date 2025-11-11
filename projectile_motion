import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse, FancyBboxPatch
import matplotlib as mpl

# System Parameters
g = 9.81                 # gravity (m/s^2)
v0 = 10.0                # initial speed (m/s)
angle_deg = 50           # launch angle (degrees)
angle = np.deg2rad(angle_deg)
vx0 = v0 * np.cos(angle)
vy0 = v0 * np.sin(angle)

x0, y0 = 0.0, 0.5        # initial position (m)
dt = 1/60                # seconds per frame
trail_len = 45           # number of past points to show in trail
shadow_scale = 0.6       # scale of the shadow ellipse

# Visualization parameters
PROJECTILE_PIXEL_RADIUS = 30   # visual size in pixels (adjust to taste)
HUD_ALPHA = 0.65               # HUD box alpha
FPS = int(1 / dt)

# Estimate flight time to set axis limits
t_flight = (vy0 + np.sqrt(vy0**2 + 2*g*y0)) / g
t_total = t_flight * 1.05
x_max = x0 + vx0 * t_total
y_max = max(y0, (vy0**2) / (2*g)) * 1.45

# Prepare figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(-0.1 * x_max, 1.05 * x_max)
ax.set_ylim(0, 1.05 * y_max)
ax.axis('off')  # clean: no axes

# Background gradient (sky)
nrows = 512
gradient = np.linspace(1.0, 0.85, nrows)[:, None]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'sky', ['#bfe9ff', '#88d0ff', '#5bb2ff']
)
ax.imshow(
    gradient,
    extent=[-0.1 * x_max, 1.05 * x_max, 0, 1.05 * y_max],
    origin='lower',
    cmap=cmap,
    alpha=1.0,
    interpolation='bicubic',
    aspect='equal'
)

# Ensure equal aspect so circles map correctly (we keep projectile radius in pixels so it's robust)
fig.canvas.draw()
ax.set_aspect('auto')  # Let figure resizing scale both axes uniformly (circle handled via pixel->data)

# Ground 
ground_y = 0.0
ax.fill_between(
    [-0.2*x_max, 1.2*x_max],
    -0.1*y_max,
    ground_y,
    color='#3b7a3b',
    alpha=0.95
)

# Converts pixel radius to data coords at a point (xdata, ydata)
def pixel_to_data_radius(ax, px_radius, xdata, ydata):
    # transform data -> display
    disp0 = ax.transData.transform((xdata, ydata))
    # a small horizontal step in display coords
    disp1 = (disp0[0] + px_radius, disp0[1])
    # transform back to data coords
    data0 = ax.transData.inverted().transform(disp0)
    data1 = ax.transData.inverted().transform(disp1)
    return abs(data1[0] - data0[0])

# Plot elements
# projectile (we'll update proj.center and proj.radius each frame)
proj = Circle(
    (x0, y0),
    0.01, # initial tiny radius in data coords; will be updated per-frame
    facecolor="#c43a3a",
    edgecolor="#4a0000",
    lw=1.6,
    zorder=6
)
ax.add_patch(proj)

# Shadow as flattened ellipse
shadow = Ellipse((x0, 0.02*y_max), width=0.5*x_max*0.02, height=0.02*y_max, color='black', alpha=0.18, zorder=3)
ax.add_patch(shadow)

# Velocity arrow
vel_arrow = FancyArrowPatch(
    (x0, y0), (x0 + 0.7, y0),
    arrowstyle='-|>',
    mutation_scale=20,
    linewidth=3.0,
    zorder=7
)
ax.add_patch(vel_arrow)

# HUD box (rounded rectangle) behind texts
hud_w = 0.28 * x_max
hud_h = 0.12 * y_max
hud_x = 0.02 * x_max
hud_y = 0.86 * y_max
hud_box = FancyBboxPatch(
    (hud_x - 0.01 * x_max, hud_y - 0.01 * y_max),
    hud_w, hud_h,
    boxstyle="round,pad=0.02,rounding_size=6",
    linewidth=0,
    facecolor="#ffffff",
    alpha=HUD_ALPHA,
    zorder=8
)
ax.add_patch(hud_box)

# Speed and height text on top of the HUD
speed_text = ax.text(hud_x + 6, hud_y + 0.035 * y_max, '', fontsize=14, weight='bold', color='#222', zorder=10)
height_text = ax.text(hud_x + 6, hud_y - 0.01 * y_max, '', fontsize=12, weight='bold', color='#222', zorder=10)

# Trail scatter (fading)
trail_scatter = ax.scatter([], [], s=28, cmap='plasma', alpha=0.9, zorder=5)

# Launch label
ax.text(x0, y0 + 0.08*y_max, f'launch {v0:.0f} m/s @ {angle_deg}°', fontsize=10, color='#333', zorder=6)

# Simulation state
positions = []
t = 0.0

def pos_at(t):
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    return x, y

# Animation functions
def init():
    trail_scatter.set_offsets([])
    proj.center = (x0, y0)
    # set initial pixel radius mapped to data units
    data_r = pixel_to_data_radius(ax, PROJECTILE_PIXEL_RADIUS, x0, y0)
    proj.radius = data_r
    vel_arrow.set_positions((x0, y0), (x0+0.5, y0))
    shadow.center = (x0, 0.02*y_max)
    speed_text.set_text('')
    height_text.set_text('')
    return proj, trail_scatter, vel_arrow, shadow, speed_text, height_text, hud_box

def update(frame):
    global t, positions
    t = frame * dt
    x, y = pos_at(t)

    # Ground check
    if y <= ground_y + 1e-4:
        y = ground_y

    # Update projectile position
    proj.center = (x, y)
    # Update pixel->data radius so it stays circular on screen
    proj.radius = pixel_to_data_radius(ax, PROJECTILE_PIXEL_RADIUS, x, y)

    # Update shadow: center and flatten based on height (smaller when high)
    shadow.center = (x, 0.02*y_max)
    max_h = max(1e-3, y_max)
    flat = 0.18 * (1 - 0.6 * (y/max_h))
    shadow.width = 0.5 * max(0.02*x_max, 0.12) * (1 - 0.6*(y/max_h))
    shadow.height = 0.02 * y_max * (0.6 + 0.4*(1 - y/max_h))
    shadow.set_alpha(0.18 * (1 - 0.4 * (y/max_h)))

    # Velocity arrow
    vx = vx0
    vy = vy0 - g * t
    v_display_scale = 0.12 * x_max / max(1e-3, v0)
    arrow_dx = vx * v_display_scale
    arrow_dy = vy * v_display_scale
    vel_arrow.set_positions((x, y), (x + arrow_dx, y + arrow_dy))
    vel_arrow.set_color('#ff6f61' if vy >= 0 else '#3b3bff')

    # HUD texts: speed and height
    speed = np.hypot(vx, vy)
    speed_text.set_text(f'speed: {speed:.1f} m/s')
    height_text.set_text(f'height: {y:.2f} m')

    # Trail
    positions.append((x, y))
    if len(positions) > trail_len:
        positions = positions[-trail_len:]

    if positions:
        offsets = np.array(positions)
        alphas = np.linspace(0.12, 0.95, len(offsets))
        sizes = np.linspace(40, 8, len(offsets))
        trail_scatter.set_offsets(offsets)
        trail_scatter.set_sizes(sizes)
        rgba = np.zeros((len(offsets), 4))
        base = mpl.colors.to_rgba('#ffd166')
        for i in range(len(offsets)):
            rgba[i, :] = base[:3] + (alphas[i],)
        trail_scatter.set_facecolors(rgba)
    else:
        trail_scatter.set_offsets([])

    return proj, trail_scatter, vel_arrow, shadow, speed_text, height_text, hud_box

# Total frames
frames = int((t_total + 1.0) / dt)

anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=dt*1000, blit=False)

plt.tight_layout()
plt.show()

# To save the animation as MP4 (requires ffmpeg)
try:
    anim.save('projectile.mp4', fps=FPS, dpi=150, bitrate=4000)
    print("Saved projectile.mp4")
except Exception as e:
    print("Could not save animation automatically. Ensure ffmpeg is installed.")
    print("Error:", e)
