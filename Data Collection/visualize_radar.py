import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

def visualize_radar(radar_data_list):
    plt.ion()
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], c='r')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Radar Point Cloud')

    for frame in radar_data_list:
        x = np.array(frame["x"])
        y = np.array(frame["y"])

        scatter.set_offsets(np.c_[x, y])
        fig.canvas.draw_idle()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the .npz file")
    args = parser.parse_args()

    data = np.load(args.file, allow_pickle=True)
    radar_data = data['radar']

    print(f"[INFO] Loaded {len(radar_data)} radar frames.")
    visualize_radar(radar_data)
