# Step 3 — plots for the report 

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

HDF5_PATH = Path("data/hoppers_vs_walkers.h5")
OUTPUT_DIR = Path("figures/step3")
RAW_SAMPLE_DURATION_SECONDS = 8.0

AXIS_COLORS = {
    "x": "tab:blue",
    "y": "tab:orange",
    "z": "tab:green",
    "abs": "tab:red",
}


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def decode(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def plot_raw_signal_examples(hdf):
    examples = [("Darcy", "walking"), ("Darcy", "jumping")]
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 7), sharex=False)

    if len(examples) == 1:
        axes = [axes]

    k = 0
    while k < len(examples):
        ax = axes[k]
        participant, activity = examples[k]
        g = hdf["raw"][participant][activity]
        time = g["time"][:]
        xyz = g["acceleration_xyz"][:]
        abs_acc = g["absolute_acceleration"][:]

        end_t = time[0] + RAW_SAMPLE_DURATION_SECONDS
        mask = time <= end_t

        ax.plot(time[mask], xyz[mask, 0], color=AXIS_COLORS["x"], linewidth=1.2, label="x-axis")
        ax.plot(time[mask], xyz[mask, 1], color=AXIS_COLORS["y"], linewidth=1.2, label="y-axis")
        ax.plot(time[mask], xyz[mask, 2], color=AXIS_COLORS["z"], linewidth=1.2, label="z-axis")
        ax.plot(
            time[mask],
            abs_acc[mask],
            color=AXIS_COLORS["abs"],
            linewidth=1.0,
            alpha=0.8,
            label="absolute acceleration",
        )

        ax.set_title("Raw accelerometer sample: " + participant + " - " + activity)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        k += 1

    save_figure(fig, "raw_signal_examples.png")


def plot_dataset_metadata(hdf):
    participants = sorted(hdf["raw"].keys())
    activities = ["walking", "jumping"]

    sample_counts = {}
    durations = {}
    for act in activities:
        sample_counts[act] = []
        durations[act] = []

    for person in participants:
        for act in activities:
            g = hdf["raw"][person][act]
            sample_counts[act].append(int(g.attrs["sample_count"]))
            durations[act].append(float(g.attrs["duration_seconds"]))

    x = np.arange(len(participants))
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(x - w / 2, sample_counts["walking"], w, color="tab:blue", label="walking")
    axes[0].bar(x + w / 2, sample_counts["jumping"], w, color="tab:orange", label="jumping")
    axes[0].set_title("Recorded samples per participant")
    axes[0].set_ylabel("Number of samples")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - w / 2, durations["walking"], w, color="tab:blue", label="walking")
    axes[1].bar(x + w / 2, durations["jumping"], w, color="tab:orange", label="jumping")
    axes[1].set_title("Recording duration per participant")
    axes[1].set_xlabel("Participant")
    axes[1].set_ylabel("Duration (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(participants)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    save_figure(fig, "dataset_metadata.png")


def plot_window_split_summary(hdf):
    train_labels = decode(hdf["splits"]["train"]["labels"][:])
    test_labels = decode(hdf["splits"]["test"]["labels"][:])
    train_parts = decode(hdf["splits"]["train"]["participants"][:])
    test_parts = decode(hdf["splits"]["test"]["participants"][:])

    activities = ["walking", "jumping"]
    participants = sorted(hdf["raw"].keys())

    train_label_counts = []
    test_label_counts = []
    for act in activities:
        train_label_counts.append(train_labels.count(act))
        test_label_counts.append(test_labels.count(act))

    train_pc = []
    test_pc = []
    for person in participants:
        train_pc.append(train_parts.count(person))
        test_pc.append(test_parts.count(person))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.35

    xa = np.arange(len(activities))
    axes[0].bar(xa - width / 2, train_label_counts, width, color="tab:green", label="train")
    axes[0].bar(xa + width / 2, test_label_counts, width, color="tab:red", label="test")
    axes[0].set_title("Window counts by class")
    axes[0].set_xticks(xa)
    axes[0].set_xticklabels(activities)
    axes[0].set_ylabel("Number of 5-second windows")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    xp = np.arange(len(participants))
    axes[1].bar(xp - width / 2, train_pc, width, color="tab:green", label="train")
    axes[1].bar(xp + width / 2, test_pc, width, color="tab:red", label="test")
    axes[1].set_title("Window counts by participant")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(participants)
    axes[1].set_ylabel("Number of 5-second windows")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    save_figure(fig, "window_split_summary.png")


def plot_absolute_acceleration_distributions(hdf):
    participants = sorted(hdf["raw"].keys())
    walking_means = []
    jumping_means = []
    walking_stds = []
    jumping_stds = []

    for person in participants:
        w_abs = hdf["raw"][person]["walking"]["absolute_acceleration"][:]
        j_abs = hdf["raw"][person]["jumping"]["absolute_acceleration"][:]
        walking_means.append(float(np.mean(w_abs)))
        jumping_means.append(float(np.mean(j_abs)))
        walking_stds.append(float(np.std(w_abs)))
        jumping_stds.append(float(np.std(j_abs)))

    x_positions = np.arange(len(participants))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(x_positions - width / 2, walking_means, width, color="tab:blue", label="walking")
    axes[0].bar(x_positions + width / 2, jumping_means, width, color="tab:orange", label="jumping")
    axes[0].set_title("Mean absolute acceleration by participant")
    axes[0].set_ylabel("Mean acceleration (m/s^2)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x_positions - width / 2, walking_stds, width, color="tab:blue", label="walking")
    axes[1].bar(x_positions + width / 2, jumping_stds, width, color="tab:orange", label="jumping")
    axes[1].set_title("Absolute acceleration variability by participant")
    axes[1].set_xlabel("Participant")
    axes[1].set_ylabel("Standard deviation (m/s^2)")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(participants)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    save_figure(fig, "absolute_acceleration_summary.png")


def plot_window_examples(hdf):
    train_signals = hdf["splits"]["train"]["signals"][:]
    train_times = hdf["splits"]["train"]["time"][:]
    train_labels = decode(hdf["splits"]["train"]["labels"][:])

    idx_walk = train_labels.index("walking")
    idx_jump = train_labels.index("jumping")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    pairs = [(idx_walk, "walking"), (idx_jump, "jumping")]
    for ax, (idx, lab) in zip(axes, pairs):
        tv = train_times[idx]
        sv = train_signals[idx]
        ax.plot(tv, sv[:, 0], color=AXIS_COLORS["x"], linewidth=1.2, label="x-axis")
        ax.plot(tv, sv[:, 1], color=AXIS_COLORS["y"], linewidth=1.2, label="y-axis")
        ax.plot(tv, sv[:, 2], color=AXIS_COLORS["z"], linewidth=1.2, label="z-axis")
        ax.plot(tv, sv[:, 3], color=AXIS_COLORS["abs"], linewidth=1.0, label="absolute")
        ax.set_title("Resampled 5-second training window: " + lab)
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time within window (s)")
    save_figure(fig, "window_examples.png")


def main():
    ensure_output_dir()

    with h5py.File(HDF5_PATH, "r") as hdf:
        plot_raw_signal_examples(hdf)
        plot_dataset_metadata(hdf)
        plot_window_split_summary(hdf)
        plot_absolute_acceleration_distributions(hdf)
        plot_window_examples(hdf)

    print("Saved step 3 figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
