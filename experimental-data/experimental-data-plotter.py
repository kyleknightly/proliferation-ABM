import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Load and configure
data = np.load('/Users/kyleknightly/Documents/GitHub/proliferation-ABM/experimental-data/dict_cell_cycle_data.npy', allow_pickle=True).item()
experiment_index = 0  # Choose from 0 to 11
phases = ['red', 'green', 'yellow', 'dark']
colors = {'red': 'red', 'green': 'green', 'yellow': 'gold', 'dark': 'black'}

# Extract position histories for all phases
phase_histories = {phase: data[phase][experiment_index] for phase in phases}
num_steps = min(len(hist) for hist in phase_histories.values())  # Ensure equal length

# Compute global radius
all_points = np.vstack([np.vstack(phase_histories[phase]) for phase in phases])
max_radius = np.max(np.linalg.norm(all_points, axis=1)) * 1.1

# GUI viewer
def run_multiphase_gui(phase_histories, radius=500):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)

    # Create a scatter plot for each phase
    scatter_objs = {
        phase: ax.scatter([], [], c=color, s=15, edgecolor='k', linewidth=0, label=phase)
        for phase, color in colors.items()
    }

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_title(f"Multi-Phase Cell Positions (Experiment {experiment_index})")
    # ax.legend(loc='upper right')

    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(radius*np.cos(theta), radius*np.sin(theta), 'k--', linewidth=1)

    ax_slider = plt.axes([0.15, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Timestep', 0, num_steps - 1, valinit=0, valstep=1)

    playing = [False]
    ax_button = plt.axes([0.78, 0.05, 0.1, 0.04])
    button = Button(ax_button, "Play")

    def update(frame):
        for phase in phases:
            pos = phase_histories[phase][frame]
            scatter_objs[phase].set_offsets(pos)
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

    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(playback_loop, None)
    timer.start()

    update(0)
    plt.show()

# Run viewer
run_multiphase_gui(phase_histories, radius=max_radius)
