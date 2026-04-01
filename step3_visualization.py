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


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def plot_raw_signal_examples(hdf: h5py.File) -> None:
    examples = [("Darcy", "walking"), ("Darcy", "jumping")]
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 7), sharex=False)

    if len(examples) == 1:
        axes = [axes]

    for ax, (participant, activity) in zip(axes, examples, strict=False):
        group = hdf["raw"][participant][activity]
        time = group["time"][:]
        xyz = group["acceleration_xyz"][:]
        absolute_acceleration = group["absolute_acceleration"][:]

        end_time = time[0] + RAW_SAMPLE_DURATION_SECONDS
        mask = time <= end_time

        ax.plot(time[mask], xyz[mask, 0], color=AXIS_COLORS["x"], linewidth=1.2, label="x-axis")
        ax.plot(time[mask], xyz[mask, 1], color=AXIS_COLORS["y"], linewidth=1.2, label="y-axis")
        ax.plot(time[mask], xyz[mask, 2], color=AXIS_COLORS["z"], linewidth=1.2, label="z-axis")
        ax.plot(
            time[mask],
            absolute_acceleration[mask],
            color=AXIS_COLORS["abs"],
            linewidth=1.0,
            alpha=0.8,
            label="absolute acceleration",
        )

        ax.set_title(f"Raw accelerometer sample: {participant} - {activity}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    save_figure(fig, "raw_signal_examples.png")


def plot_dataset_metadata(hdf: h5py.File) -> None:
    participants = sorted(hdf["raw"].keys())
    activities = ["walking", "jumping"]

    sample_counts = {activity: [] for activity in activities}
    durations = {activity: [] for activity in activities}

    for participant in participants:
        for activity in activities:
            group = hdf["raw"][participant][activity]
            sample_counts[activity].append(int(group.attrs["sample_count"]))
            durations[activity].append(float(group.attrs["duration_seconds"]))

    x_positions = np.arange(len(participants))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(
        x_positions - width / 2,
        sample_counts["walking"],
        width,
        color="tab:blue",
        label="walking",
    )
    axes[0].bar(
        x_positions + width / 2,
        sample_counts["jumping"],
        width,
        color="tab:orange",
        label="jumping",
    )
    axes[0].set_title("Recorded samples per participant")
    axes[0].set_ylabel("Number of samples")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        x_positions - width / 2,
        durations["walking"],
        width,
        color="tab:blue",
        label="walking",
    )
    axes[1].bar(
        x_positions + width / 2,
        durations["jumping"],
        width,
        color="tab:orange",
        label="jumping",
    )
    axes[1].set_title("Recording duration per participant")
    axes[1].set_xlabel("Participant")
    axes[1].set_ylabel("Duration (s)")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(participants)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    save_figure(fig, "dataset_metadata.png")


def plot_window_split_summary(hdf: h5py.File) -> None:
    train_labels = decode(hdf["splits"]["train"]["labels"][:])
    test_labels = decode(hdf["splits"]["test"]["labels"][:])
    train_participants = decode(hdf["splits"]["train"]["participants"][:])
    test_participants = decode(hdf["splits"]["test"]["participants"][:])

    activities = ["walking", "jumping"]
    participants = sorted(hdf["raw"].keys())

    train_label_counts = [train_labels.count(activity) for activity in activities]
    test_label_counts = [test_labels.count(activity) for activity in activities]
    train_participant_counts = [train_participants.count(participant) for participant in participants]
    test_participant_counts = [test_participants.count(participant) for participant in participants]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.35

    x_label = np.arange(len(activities))
    axes[0].bar(x_label - width / 2, train_label_counts, width, color="tab:green", label="train")
    axes[0].bar(x_label + width / 2, test_label_counts, width, color="tab:red", label="test")
    axes[0].set_title("Window counts by class")
    axes[0].set_xticks(x_label)
    axes[0].set_xticklabels(activities)
    axes[0].set_ylabel("Number of 5-second windows")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    x_participants = np.arange(len(participants))
    axes[1].bar(
        x_participants - width / 2,
        train_participant_counts,
        width,
        color="tab:green",
        label="train",
    )
    axes[1].bar(
        x_participants + width / 2,
        test_participant_counts,
        width,
        color="tab:red",
        label="test",
    )
    axes[1].set_title("Window counts by participant")
    axes[1].set_xticks(x_participants)
    axes[1].set_xticklabels(participants)
    axes[1].set_ylabel("Number of 5-second windows")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    save_figure(fig, "window_split_summary.png")


def plot_absolute_acceleration_distributions(hdf: h5py.File) -> None:
    participants = sorted(hdf["raw"].keys())
    walking_means = []
    jumping_means = []
    walking_stds = []
    jumping_stds = []

    for participant in participants:
        walking_abs = hdf["raw"][participant]["walking"]["absolute_acceleration"][:]
        jumping_abs = hdf["raw"][participant]["jumping"]["absolute_acceleration"][:]
        walking_means.append(float(np.mean(walking_abs)))
        jumping_means.append(float(np.mean(jumping_abs)))
        walking_stds.append(float(np.std(walking_abs)))
        jumping_stds.append(float(np.std(jumping_abs)))

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


def plot_window_examples(hdf: h5py.File) -> None:
    train_signals = hdf["splits"]["train"]["signals"][:]
    train_times = hdf["splits"]["train"]["time"][:]
    train_labels = decode(hdf["splits"]["train"]["labels"][:])

    walking_index = train_labels.index("walking")
    jumping_index = train_labels.index("jumping")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, window_index, label in zip(
        axes,
        [walking_index, jumping_index],
        ["walking", "jumping"],
        strict=False,
    ):
        time_values = train_times[window_index]
        signal_values = train_signals[window_index]

        ax.plot(time_values, signal_values[:, 0], color=AXIS_COLORS["x"], linewidth=1.2, label="x-axis")
        ax.plot(time_values, signal_values[:, 1], color=AXIS_COLORS["y"], linewidth=1.2, label="y-axis")
        ax.plot(time_values, signal_values[:, 2], color=AXIS_COLORS["z"], linewidth=1.2, label="z-axis")
        ax.plot(time_values, signal_values[:, 3], color=AXIS_COLORS["abs"], linewidth=1.0, label="absolute")
        ax.set_title(f"Resampled 5-second training window: {label}")
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time within window (s)")
    save_figure(fig, "window_examples.png")


def main() -> None:
    ensure_output_dir()

    with h5py.File(HDF5_PATH, "r") as hdf:
        plot_raw_signal_examples(hdf)
        plot_dataset_metadata(hdf)
        plot_window_split_summary(hdf)
        plot_absolute_acceleration_distributions(hdf)
        plot_window_examples(hdf)

    print(f"Saved step 3 figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
