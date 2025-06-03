import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- Configuration ---
FILE = '/Users/kyleknightly/Documents/GitHub/proliferation-ABM/experimental-data/dict_cell_cycle_data.npy'
PHASES = ['red', 'green', 'yellow', 'dark']
COLORS = {'red': 'red', 'green': 'green', 'yellow': 'gold', 'dark': 'black'}
EXPERIMENT = 0
NUM_BINS = 50
FPS = 20

# --- LOAD DATA ---
data = np.load(FILE, allow_pickle=True).item()
phase_histories = {phase: data[phase][EXPERIMENT] for phase in PHASES}
num_frames = min(len(hist) for hist in phase_histories.values())

# --- DETERMINE GLOBAL MAX RADIUS ---
all_points = np.vstack([np.vstack(phase_histories[phase]) for phase in PHASES])
radii = np.linalg.norm(all_points, axis=1)
MAX_RADIUS = np.max(radii) * 1.1

# --- BINNING SETUP ---
r_bins = np.linspace(0, MAX_RADIUS, NUM_BINS + 1)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)  # area of annular bins

# --- COMPUTE DENSITY HISTORY FOR EACH PHASE ---
density_histories = {}

for phase in PHASES:
    history = []
    for frame in phase_histories[phase][:num_frames]:
        r = np.linalg.norm(frame, axis=1)
        counts, _ = np.histogram(r, bins=r_bins)
        density = counts / areas
        history.append(density)
    density_histories[phase] = history

# --- GUI PLOT ---
def run_radial_density_multiphase_gui():
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)

    # Plot one line per phase
    lines = {}
    for phase in PHASES:
        line, = ax.plot(r_centers, density_histories[phase][0], label=phase.capitalize(), color=COLORS[phase])
        lines[phase] = line

    ax.set_xlabel("Radial Distance (mm)")
    ax.set_ylabel("Cell Density (cells/mm²)")
    ax.set_title(f"Radial Density Profiles (Experiment {EXPERIMENT})")
    ax.set_xlim(0, MAX_RADIUS)
    max_density = max(np.max(hist) for hist in density_histories.values())
    ax.set_ylim(0, max_density * 1.1)
    ax.legend(loc='upper right')

    # Slider
    ax_slider = plt.axes([0.15, 0.08, 0.6, 0.03])
    slider = Slider(ax_slider, 'Timestep', 0, num_frames - 1, valinit=0, valstep=1)

    # Play button
    playing = [False]
    ax_button = plt.axes([0.78, 0.08, 0.1, 0.04])
    button = Button(ax_button, "Play")

    def update(frame_idx):
        for phase in PHASES:
            lines[phase].set_ydata(density_histories[phase][frame_idx])
        fig.canvas.draw_idle()

    def on_slider_change(val):
        update(int(val))

    def on_button_click(event):
        playing[0] = not playing[0]
        button.label.set_text("Pause" if playing[0] else "Play")

    def playback_loop(event):
        if playing[0]:
            current = int(slider.val)
            if current < num_frames - 1:
                slider.set_val(current + 1)
            else:
                slider.set_val(0)

    slider.on_changed(on_slider_change)
    button.on_clicked(on_button_click)

    timer = fig.canvas.new_timer(interval=int(1000 / FPS))
    timer.add_callback(playback_loop, None)
    timer.start()

    update(0)
    plt.show()

# --- RUN ---
if __name__ == '__main__':
    run_radial_density_multiphase_gui()