from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
HDF5_PATH = BASE_DIR / "data" / "hoppers_vs_walkers.h5"
OUTPUT_DIR = BASE_DIR / "figures" / "step4"
MOVING_AVERAGE_WINDOW = 9
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


def fill_missing_1d(values: np.ndarray) -> tuple[np.ndarray, int]:
    filled = values.astype(np.float64, copy=True)
    nan_mask = np.isnan(filled)
    missing_count = int(np.sum(nan_mask))

    if missing_count == 0:
        return filled, 0

    valid_indices = np.where(~nan_mask)[0]
    if len(valid_indices) == 0:
        return np.zeros_like(filled), missing_count

    first_valid = valid_indices[0]
    filled[:first_valid] = filled[first_valid]

    for index in range(first_valid + 1, len(filled)):
        if np.isnan(filled[index]):
            filled[index] = filled[index - 1]

    return filled, missing_count


def preprocess_signal_matrix(signal_matrix: np.ndarray, window_size: int) -> tuple[np.ndarray, int]:
    preprocessed = np.zeros_like(signal_matrix, dtype=np.float64)
    total_missing = 0

    for column_index in range(signal_matrix.shape[1]):
        filled_column, missing_count = fill_missing_1d(signal_matrix[:, column_index])
        total_missing += missing_count
        preprocessed[:, column_index] = moving_average(filled_column, window_size)

    return preprocessed, total_missing


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    filtered = np.zeros_like(values, dtype=np.float64)

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        filtered[index] = np.mean(values[start_index : index + 1])

    return filtered


def reset_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def write_preprocessed_raw_group(
    group: h5py.Group,
    time_values: np.ndarray,
    signal_values: np.ndarray,
    source_group: h5py.Group,
    missing_count: int,
) -> None:
    group.create_dataset("time", data=time_values, compression="gzip")
    group.create_dataset("acceleration_xyz", data=signal_values[:, :3], compression="gzip")
    group.create_dataset("absolute_acceleration", data=signal_values[:, 3], compression="gzip")
    group.attrs["source_file"] = source_group.attrs["source_file"]
    group.attrs["sample_count"] = int(source_group.attrs["sample_count"])
    group.attrs["duration_seconds"] = float(source_group.attrs["duration_seconds"])
    group.attrs["moving_average_window"] = MOVING_AVERAGE_WINDOW
    group.attrs["missing_values_filled"] = missing_count


def write_preprocessed_split_group(
    group: h5py.Group,
    source_group: h5py.Group,
    filtered_signals: np.ndarray,
) -> None:
    group.create_dataset("signals", data=filtered_signals, compression="gzip")
    group.create_dataset("time", data=source_group["time"][:], compression="gzip")
    group.create_dataset("labels", data=source_group["labels"][:], compression="gzip")
    group.create_dataset("participants", data=source_group["participants"][:], compression="gzip")
    group.create_dataset("source_files", data=source_group["source_files"][:], compression="gzip")
    group.attrs["window_count"] = int(source_group.attrs["window_count"])
    group.attrs["moving_average_window"] = MOVING_AVERAGE_WINDOW


def preprocess_hdf5() -> dict[str, int]:
    summary = {
        "raw_missing_values_filled": 0,
        "train_windows": 0,
        "test_windows": 0,
    }

    with h5py.File(HDF5_PATH, "r+") as hdf:
        preprocessed_root = hdf["preprocessed"]
        preprocessed_raw_root = reset_group(preprocessed_root, "raw_signals")
        preprocessed_train = reset_group(preprocessed_root, "train")
        preprocessed_test = reset_group(preprocessed_root, "test")

        for participant in sorted(hdf["raw"].keys()):
            participant_group = preprocessed_raw_root.create_group(participant)

            for activity in sorted(hdf["raw"][participant].keys()):
                source_group = hdf["raw"][participant][activity]
                time_values = source_group["time"][:]
                xyz = source_group["acceleration_xyz"][:]
                absolute_acceleration = source_group["absolute_acceleration"][:]
                signal_matrix = np.column_stack([xyz, absolute_acceleration])

                filtered_matrix, missing_count = preprocess_signal_matrix(
                    signal_matrix,
                    MOVING_AVERAGE_WINDOW,
                )
                summary["raw_missing_values_filled"] += missing_count

                destination_group = participant_group.create_group(activity)
                write_preprocessed_raw_group(
                    destination_group,
                    time_values,
                    filtered_matrix,
                    source_group,
                    missing_count,
                )

        for split_name, destination_group in [("train", preprocessed_train), ("test", preprocessed_test)]:
            source_group = hdf["splits"][split_name]
            source_signals = source_group["signals"][:]
            filtered_signals = np.zeros_like(source_signals, dtype=np.float64)

            for window_index in range(source_signals.shape[0]):
                filtered_signals[window_index], _ = preprocess_signal_matrix(
                    source_signals[window_index],
                    MOVING_AVERAGE_WINDOW,
                )

            write_preprocessed_split_group(destination_group, source_group, filtered_signals)
            summary[f"{split_name}_windows"] = int(filtered_signals.shape[0])

        preprocessed_root.attrs["moving_average_window"] = MOVING_AVERAGE_WINDOW

    return summary


def plot_raw_vs_preprocessed_examples() -> None:
    examples = [("Darcy", "walking"), ("Darcy", "jumping")]

    with h5py.File(HDF5_PATH, "r") as hdf:
        fig, axes = plt.subplots(len(examples), 1, figsize=(12, 7), sharex=False)
        if len(examples) == 1:
            axes = [axes]

        for ax, (participant, activity) in zip(axes, examples, strict=False):
            raw_group = hdf["raw"][participant][activity]
            processed_group = hdf["preprocessed"]["raw_signals"][participant][activity]

            time_values = raw_group["time"][:]
            raw_abs = raw_group["absolute_acceleration"][:]
            processed_abs = processed_group["absolute_acceleration"][:]

            end_time = time_values[0] + RAW_SAMPLE_DURATION_SECONDS
            mask = time_values <= end_time

            ax.plot(time_values[mask], raw_abs[mask], color="tab:gray", linewidth=1.0, label="raw")
            ax.plot(time_values[mask], processed_abs[mask], color="tab:red", linewidth=1.5, label="preprocessed")
            ax.set_title(f"Absolute acceleration before and after preprocessing: {participant} - {activity}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration (m/s^2)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        save_figure(fig, "raw_vs_preprocessed_absolute.png")


def plot_axis_comparison() -> None:
    participant = "Darcy"
    activity = "walking"

    with h5py.File(HDF5_PATH, "r") as hdf:
        raw_group = hdf["raw"][participant][activity]
        processed_group = hdf["preprocessed"]["raw_signals"][participant][activity]

        time_values = raw_group["time"][:]
        raw_xyz = raw_group["acceleration_xyz"][:]
        processed_xyz = processed_group["acceleration_xyz"][:]

        end_time = time_values[0] + RAW_SAMPLE_DURATION_SECONDS
        mask = time_values <= end_time

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axis_names = ["x", "y", "z"]

        for axis_index, axis_name in enumerate(axis_names):
            axes[axis_index].plot(
                time_values[mask],
                raw_xyz[mask, axis_index],
                color="tab:gray",
                linewidth=1.0,
                label="raw",
            )
            axes[axis_index].plot(
                time_values[mask],
                processed_xyz[mask, axis_index],
                color=AXIS_COLORS[axis_name],
                linewidth=1.5,
                label="preprocessed",
            )
            axes[axis_index].set_ylabel(f"{axis_name}-axis")
            axes[axis_index].grid(True, alpha=0.3)
            axes[axis_index].legend(loc="upper right")

        axes[0].set_title(f"Axis-wise preprocessing comparison: {participant} - {activity}")
        axes[-1].set_xlabel("Time (s)")
        save_figure(fig, "axis_preprocessing_comparison.png")


def plot_preprocessed_window_examples() -> None:
    with h5py.File(HDF5_PATH, "r") as hdf:
        raw_signals = hdf["splits"]["train"]["signals"][:]
        processed_signals = hdf["preprocessed"]["train"]["signals"][:]
        time_values = hdf["splits"]["train"]["time"][:]
        labels = decode(hdf["splits"]["train"]["labels"][:])

        walking_index = labels.index("walking")
        jumping_index = labels.index("jumping")

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        for ax, index, label in zip(
            axes,
            [walking_index, jumping_index],
            ["walking", "jumping"],
            strict=False,
        ):
            ax.plot(time_values[index], raw_signals[index, :, 3], color="tab:gray", linewidth=1.0, label="raw")
            ax.plot(
                time_values[index],
                processed_signals[index, :, 3],
                color="tab:red",
                linewidth=1.5,
                label="preprocessed",
            )
            ax.set_title(f"5-second training window before and after preprocessing: {label}")
            ax.set_ylabel("Absolute acceleration")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time within window (s)")
        save_figure(fig, "window_preprocessing_comparison.png")


def main() -> None:
    ensure_output_dir()
    summary = preprocess_hdf5()
    plot_raw_vs_preprocessed_examples()
    plot_axis_comparison()
    plot_preprocessed_window_examples()

    print(f"Updated preprocessed data in: {HDF5_PATH}")
    print(f"Saved step 4 figures to: {OUTPUT_DIR}")
    print(f"Missing values filled: {summary['raw_missing_values_filled']}")
    print(f"Preprocessed training windows: {summary['train_windows']}")
    print(f"Preprocessed testing windows: {summary['test_windows']}")


if __name__ == "__main__":
    main()
