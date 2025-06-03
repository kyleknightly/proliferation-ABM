import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# MARK: Functions

def morse(distances, A = 0.5, alpha = 0.5, B = 1, beta = 2):
    # A = 0.5, alpha = 0.5, B = 1, beta = 2
    # A = 2.5, alpha = 0.25, B = 5, beta = 1
    distances = np.asarray(distances)
    return A*np.exp(-alpha*distances)-B*np.exp(-beta*distances)

def exp_decay(distances, radius = 2):
    return np.exp(-distances/radius)

def normal(distances, magnitude = 0.5, radius = 0.2):
    return magnitude * np.exp(-((distances/radius)**2))

def linear(distances, radius = 20):
    values = distances*(-1/radius)+1
    return np.maximum(values, 0)

def density(x, y, radius = 15, threshold = 6):
    x = np.asarray(x)
    y = np.asarray(y)
    positions = np.stack((x, y), axis=1)
    dist_matrix = squareform(pdist(positions))
    neighbor_mask = (dist_matrix <= radius)
    np.fill_diagonal(neighbor_mask, False)
    neighbor_counts = np.sum(neighbor_mask, axis=1)

    # densities = 1 - np.exp(-neighbor_counts / threshold)
    densities = 1-linear(neighbor_counts, threshold)
    return densities


# MARK: Forces

def repulsion(x, y, lam=0.2, func=linear, radius=10):
    positions = np.column_stack((x, y))
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    mask = ~np.eye(len(x), dtype=bool)

    weights = func(dists, radius)
    weights[~mask] = 0
    force_matrix = (weights[..., np.newaxis] * diffs).sum(axis=1)
    
    return lam * force_matrix[:, 0], lam * force_matrix[:, 1]

def vel_adhesion(x, y, dx, dy, lam=0.5, dist_func=linear, d_r=5):
    N = len(x)
    pos = np.column_stack((x, y))
    vel = np.column_stack((dx, dy))  # shape (N, 2)
    # Pairwise differences and distances
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # shape (N, N, 2)
    dists = np.linalg.norm(diffs, axis=2)                  # shape (N, N)
    # Get distance-based weights
    weights = dist_func(dists, d_r)                        # shape (N, N)
    # Zero out self-contributions
    np.fill_diagonal(weights, 0)
    # Multiply each neighbor's velocity by its weight
    weighted_velocities = weights @ vel  # matrix multiplication (N, N) x (N, 2) → (N, 2)
    # Scale by lambda
    return lam * weighted_velocities[:, 0], lam * weighted_velocities[:, 1]

def dist_adhesion(x, y, lam=0.5, dist_func=linear, d_r=5):
    N = len(x)
    pos = np.column_stack((x, y))

    # Compute pairwise position differences and distances
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # shape (N, N, 2)
    dists = np.linalg.norm(diffs, axis=2)                  # shape (N, N)

    # Get distance-based weights
    weights = dist_func(dists, d_r)                        # shape (N, N)
    np.fill_diagonal(weights, 0)  # no self-influence

    # Compute unit vectors toward each neighbor (avoid div-by-zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_diffs = np.divide(diffs, dists[..., np.newaxis], where=dists[..., np.newaxis] != 0)

    # Multiply weights with unit vectors and sum over neighbors
    forces = (weights[..., np.newaxis] * -unit_diffs).sum(axis=1)  # sum over neighbors

    return lam * forces[:, 0], lam * forces[:, 1]

def active(dx, dy, alpha):
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    return alpha* dx, alpha*dy

def friction(dx, dy, gamma):
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    v_mag_sq = dx**2 + dy**2
    return -gamma * v_mag_sq * dx, -gamma * v_mag_sq * dy
    

def cell_cycle(x, y, h=0.1):
    return h * (1 - density(x, y))


# MARK: Simulation
radius = 100
steps = 200

def init(num_cells = 8, spawn_radius = 1):
    angles = 2 * np.pi * np.random.rand(num_cells)
    r_dist = spawn_radius * np.sqrt(np.random.rand(num_cells))
    x = r_dist * np.cos(angles)
    y = r_dist * np.sin(angles)
    p = np.random.rand(num_cells)
    d = density(x, y)
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    ddx = np.zeros_like(dx)
    ddy = np.zeros_like(dy)
    dp = np.zeros_like(p)
    return x, y, dx, dy, ddx, ddy, p, dp, d

def simulate(x, y, dx, dy, p, dp):
    # Compute forces
    # ad_ddx, ad_ddy = dist_adhesion(x, y, 0.2, linear, 10)
    ad_ddx, ad_ddy = vel_adhesion(x, y, dx, dy, 0.3, linear, 10)
    r_ddx, r_ddy = repulsion(x, y, 0.3, linear, 10)
    ac_ddx, ac_ddy = active(dx, dy, 0.1)
    f_ddx, f_ddy = friction(dx, dy, 0.1)

    ddx = ad_ddx + r_ddx + ac_ddx + f_ddx
    ddy = ad_ddy + r_ddy + ac_ddy + f_ddy

    dx += ddx
    dy += ddy

    # Cap velocity magnitude
    max_speed = 2.0  # adjust as needed
    speed = np.sqrt(dx**2 + dy**2)
    too_fast = speed > max_speed

    # Scale down dx and dy for fast cells
    dx[too_fast] *= max_speed / speed[too_fast]
    dy[too_fast] *= max_speed / speed[too_fast]

    x += dx
    y += dy

    dp = cell_cycle(x, y)
    p += dp

    dividing = np.where(p >= 1)[0]
    new_x = []
    new_y = []
    new_dx = []
    new_dy = []
    new_ddx = []
    new_ddy = []
    new_p = []
    new_dp = []

    for idx in dividing:
        p[idx] = 0
        dp[idx] = 0

        eps_dist = 1
        rand_angle = 2 * np.pi * np.random.rand()
        offset = eps_dist * np.array([np.cos(rand_angle), np.sin(rand_angle)])

        new_x.append(x[idx] + offset[0])
        new_y.append(y[idx] + offset[1])
        new_dx.append(dx[idx])
        new_dy.append(dy[idx])
        new_ddx.append(ddx[idx])
        new_ddy.append(ddy[idx])
        new_p.append(0.0)
        new_dp.append(0.0)

    if new_x:
        x = np.concatenate([x, new_x])
        y = np.concatenate([y, new_y])
        dx = np.concatenate([dx, new_dx])
        dy = np.concatenate([dy, new_dy])
        ddx = np.concatenate([ddx, new_ddx])
        ddy = np.concatenate([ddy, new_ddy])
        p = np.concatenate([p, new_p])
        dp = np.concatenate([dp, new_dp])

    # Enforce boundary via elastic collisions
    dist_from_center = np.sqrt(x**2 + y**2)
    outside = dist_from_center > radius

    if np.any(outside):
        # Normalize radial direction vectors
        nx = x[outside] / dist_from_center[outside]
        ny = y[outside] / dist_from_center[outside]

        # Radial component of velocity (dot product with normal)
        v_radial = dx[outside] * nx + dy[outside] * ny

        # Reflect: subtract 2 * (v · n) * n from velocity
        dx[outside] -= 2 * v_radial * nx
        dy[outside] -= 2 * v_radial * ny

        # Move particles just inside the boundary to prevent sticking
        x[outside] = nx * (radius - 1e-3)
        y[outside] = ny * (radius - 1e-3)


    d = density(x, y)
    return x, y, dx, dy, p, dp, d


x, y, dx, dy, ddx, ddy, p, dp, d = init()
history = []
history.append(np.column_stack((x, y, dx, dy, p, dp, d)))
for i in range(steps):
    print(i/steps)
    print(len(x))
    x, y, dx, dy, p, dp, d = simulate(x, y, dx, dy, p, dp)
    history.append(np.column_stack((x, y, dx, dy, p, dp, d)))

# MARK: GUI

def radial_colors(dx, dy):
    angles = np.arctan2(dy, dx)  # range [-π, π]
    hues = (angles % (2 * np.pi)) / (2 * np.pi)  # normalize to [0,1)
    # Create HSV color array and convert to RGB
    hsv_colors = np.zeros((len(hues), 3))
    hsv_colors[:, 0] = hues        # hue
    hsv_colors[:, 1] = 1.0         # full saturation
    hsv_colors[:, 2] = 1.0         # full brightness
    return mcolors.hsv_to_rgb(hsv_colors)

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Custom diverging colormap: blue → black → red
black_center_cmap = LinearSegmentedColormap.from_list(
    'blue_black_red',
    [(0, 'blue'), (0.5, 'black'), (1, 'red')]
)

def skew_colors(x, y, dx, dy, cmap=black_center_cmap):
    r = np.column_stack((x, y))
    v = np.column_stack((dx, dy))

    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    r_unit = np.divide(r, r_norm, out=np.zeros_like(r), where=r_norm != 0)
    v_unit = np.divide(v, v_norm, out=np.zeros_like(v), where=v_norm != 0)

    inward = -r_unit

    dot = np.sum(inward * v_unit, axis=1)
    cross = inward[:, 0] * v_unit[:, 1] - inward[:, 1] * v_unit[:, 0]
    angles = np.arctan2(cross, dot)

    norm_angles = angles / np.pi  # range [-1, 1]

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    rgb_colors = cmap(norm(norm_angles))

    return rgb_colors


def run_gui(history):
    num_steps = len(history)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)

    scatter = ax.scatter([], [], facecolors=[], cmap="viridis", vmin=0, vmax=1, s=10, edgecolor='k', linewidth=0)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_title("Cell Simulation")
    
    # Draw boundary circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(radius*np.cos(theta), radius*np.sin(theta), 'k--', linewidth=1)

    # Colorbar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("phase")

    # Slider
    ax_slider = plt.axes([0.15, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Timestep', 0, num_steps - 1, valinit=0, valstep=1)

    # Play/Pause Button
    playing = [False]  # wrapped in list to mutate from inner function
    ax_button = plt.axes([0.78, 0.05, 0.1, 0.04])
    button = Button(ax_button, "Play")

    def update(frame):
        data = history[frame]
        x = data[:, 0]
        y = data[:, 1]
        dx = data[:, 2]
        dy = data[:, 3]
        p = data[:, 4]
        # d = data[:, 6]

        # Compute angle in radians [-π, π], then normalize to [0, 1) for hue
        rgb_colors = skew_colors(x, y, dx, dy) 

        # Update scatter plot
        scatter.set_offsets(np.c_[x, y])
        # scatter.set_facecolors(rgb_colors)
        scatter.set_array(p)
        scatter.set_edgecolor('k')
        fig.canvas.draw_idle()


    def on_slider_change(val):
        update(int(val))

    def on_button_clicked(event):
        playing[0] = not playing[0]
        button.label.set_text("Pause" if playing[0] else "Play")

    def playback_loop(event):
        if playing[0]:
            current = int(slider.val)
            if current < num_steps - 1:
                slider.set_val(current + 1)
            else:
                slider.set_val(0)

    slider.on_changed(on_slider_change)
    button.on_clicked(on_button_clicked)

    # Timer for auto-play
    timer = fig.canvas.new_timer(interval=1)
    timer.add_callback(playback_loop, None)
    timer.start()

    update(0)
    plt.show()


# MARK: Trajectories

def plot_trajectories(history):
    # Determine the max number of cells
    max_cells = max(frame.shape[0] for frame in history)
    num_frames = len(history)

    # Initialize trajectory storage
    x_traj = [[] for _ in range(max_cells)]
    y_traj = [[] for _ in range(max_cells)]

    # Fill in trajectories
    for frame in history:
        x = frame[:, 0]
        y = frame[:, 1]
        for i in range(len(x)):
            x_traj[i].append(x[i])
            y_traj[i].append(y[i])

    # Plot trajectories
    fig, ax = plt.subplots(figsize=(20, 20))
    for xt, yt in zip(x_traj, y_traj):
        if len(xt) > 1:
            ax.plot(xt, yt, linewidth=0.8, alpha=0.6)

    # Draw boundary
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k--', linewidth=1)

    ax.set_title("Cell Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('trajectories.png')

plot_trajectories(history)
# Launch GUI
run_gui(history)

all_data = []
for t, frame in enumerate(history):
    df = pd.DataFrame(frame, columns=["x", "y", "dx", "dy", "p", "dp", "d"])
    df["timestep"] = t  # Add timestep info
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)
full_df.to_csv("cell_simulation_history.csv", index=False)