# Step 4 — smooth the signals 

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


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def decode(values):
    res = []
    for value in values:
        if isinstance(value, bytes):
            res.append(value.decode())
        else:
            res.append(str(value))
    return res


def fill_missing_1d(values):
    filled = values.astype(np.float64, copy=True)
    nan_mask = np.isnan(filled)
    missing_count = int(np.sum(nan_mask))

    if missing_count == 0:
        return filled, 0

    ok_idx = np.where(~nan_mask)[0]
    if len(ok_idx) == 0:
        return np.zeros_like(filled), missing_count

    first = ok_idx[0]
    filled[:first] = filled[first]

    i = first + 1
    while i < len(filled):
        if np.isnan(filled[i]):
            filled[i] = filled[i - 1]
        i += 1

    return filled, missing_count


def moving_average(values, window_size):
    filtered = np.zeros_like(values, dtype=np.float64)
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        filtered[i] = np.mean(values[start : i + 1])
    return filtered


def preprocess_signal_matrix(signal_matrix, window_size):
    out = np.zeros_like(signal_matrix, dtype=np.float64)
    total_missing = 0

    for col in range(signal_matrix.shape[1]):
        col_filled, nmiss = fill_missing_1d(signal_matrix[:, col])
        total_missing = total_missing + nmiss
        out[:, col] = moving_average(col_filled, window_size)

    return out, total_missing


def reset_group(parent, name):
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def write_preprocessed_raw_group(group, time_values, signal_values, source_group, missing_count):
    group.create_dataset("time", data=time_values, compression="gzip")
    group.create_dataset("acceleration_xyz", data=signal_values[:, :3], compression="gzip")
    group.create_dataset("absolute_acceleration", data=signal_values[:, 3], compression="gzip")
    group.attrs["source_file"] = source_group.attrs["source_file"]
    group.attrs["sample_count"] = int(source_group.attrs["sample_count"])
    group.attrs["duration_seconds"] = float(source_group.attrs["duration_seconds"])
    group.attrs["moving_average_window"] = MOVING_AVERAGE_WINDOW
    group.attrs["missing_values_filled"] = missing_count


def write_preprocessed_split_group(group, source_group, filtered_signals):
    group.create_dataset("signals", data=filtered_signals, compression="gzip")
    group.create_dataset("time", data=source_group["time"][:], compression="gzip")
    group.create_dataset("labels", data=source_group["labels"][:], compression="gzip")
    group.create_dataset("participants", data=source_group["participants"][:], compression="gzip")
    group.create_dataset("source_files", data=source_group["source_files"][:], compression="gzip")
    group.attrs["window_count"] = int(source_group.attrs["window_count"])
    group.attrs["moving_average_window"] = MOVING_AVERAGE_WINDOW


def preprocess_hdf5():
    summary = {
        "raw_missing_values_filled": 0,
        "train_windows": 0,
        "test_windows": 0,
    }

    with h5py.File(HDF5_PATH, "r+") as hdf:
        pre_root = hdf["preprocessed"]
        raw_sig_root = reset_group(pre_root, "raw_signals")
        tr_g = reset_group(pre_root, "train")
        te_g = reset_group(pre_root, "test")

        for person in sorted(hdf["raw"].keys()):
            pg = raw_sig_root.create_group(person)

            for act in sorted(hdf["raw"][person].keys()):
                src = hdf["raw"][person][act]
                time_values = src["time"][:]
                xyz = src["acceleration_xyz"][:]
                abs_acc = src["absolute_acceleration"][:]
                M = np.column_stack([xyz, abs_acc])

                filt, nmiss = preprocess_signal_matrix(M, MOVING_AVERAGE_WINDOW)
                summary["raw_missing_values_filled"] += nmiss

                dest = pg.create_group(act)
                write_preprocessed_raw_group(dest, time_values, filt, src, nmiss)

        for split_name, dest_group in [("train", tr_g), ("test", te_g)]:
            src = hdf["splits"][split_name]
            src_sig = src["signals"][:]
            filt_sig = np.zeros_like(src_sig, dtype=np.float64)

            for wi in range(src_sig.shape[0]):
                tmp, _ = preprocess_signal_matrix(src_sig[wi], MOVING_AVERAGE_WINDOW)
                filt_sig[wi] = tmp

            write_preprocessed_split_group(dest_group, src, filt_sig)
            summary[split_name + "_windows"] = int(filt_sig.shape[0])

        pre_root.attrs["moving_average_window"] = MOVING_AVERAGE_WINDOW

    return summary


def plot_raw_vs_preprocessed_examples():
    examples = [("Darcy", "walking"), ("Darcy", "jumping")]

    with h5py.File(HDF5_PATH, "r") as hdf:
        fig, axes = plt.subplots(len(examples), 1, figsize=(12, 7), sharex=False)
        if len(examples) == 1:
            axes = [axes]

        for ax, (person, act) in zip(axes, examples):
            raw_g = hdf["raw"][person][act]
            proc_g = hdf["preprocessed"]["raw_signals"][person][act]

            t = raw_g["time"][:]
            raw_abs = raw_g["absolute_acceleration"][:]
            proc_abs = proc_g["absolute_acceleration"][:]

            end_t = t[0] + RAW_SAMPLE_DURATION_SECONDS
            mask = t <= end_t

            ax.plot(t[mask], raw_abs[mask], color="tab:gray", linewidth=1.0, label="raw")
            ax.plot(t[mask], proc_abs[mask], color="tab:red", linewidth=1.5, label="preprocessed")
            ax.set_title("Absolute acceleration before/after preprocessing: " + person + " - " + act)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration (m/s^2)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        save_figure(fig, "raw_vs_preprocessed_absolute.png")


def plot_axis_comparison():
    person = "Darcy"
    act = "walking"

    with h5py.File(HDF5_PATH, "r") as hdf:
        raw_g = hdf["raw"][person][act]
        proc_g = hdf["preprocessed"]["raw_signals"][person][act]

        t = raw_g["time"][:]
        raw_xyz = raw_g["acceleration_xyz"][:]
        proc_xyz = proc_g["acceleration_xyz"][:]

        end_t = t[0] + RAW_SAMPLE_DURATION_SECONDS
        mask = t <= end_t

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        names = ["x", "y", "z"]

        for j in range(3):
            axes[j].plot(t[mask], raw_xyz[mask, j], color="tab:gray", linewidth=1.0, label="raw")
            axes[j].plot(
                t[mask],
                proc_xyz[mask, j],
                color=AXIS_COLORS[names[j]],
                linewidth=1.5,
                label="preprocessed",
            )
            axes[j].set_ylabel(names[j] + "-axis")
            axes[j].grid(True, alpha=0.3)
            axes[j].legend(loc="upper right")

        axes[0].set_title("Axis-wise preprocessing comparison: " + person + " - " + act)
        axes[-1].set_xlabel("Time (s)")
        save_figure(fig, "axis_preprocessing_comparison.png")


def plot_preprocessed_window_examples():
    with h5py.File(HDF5_PATH, "r") as hdf:
        raw_signals = hdf["splits"]["train"]["signals"][:]
        proc_signals = hdf["preprocessed"]["train"]["signals"][:]
        tvals = hdf["splits"]["train"]["time"][:]
        labels = decode(hdf["splits"]["train"]["labels"][:])

        iw = labels.index("walking")
        ij = labels.index("jumping")

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        for ax, idx, lab in zip(axes, [iw, ij], ["walking", "jumping"]):
            ax.plot(tvals[idx], raw_signals[idx, :, 3], color="tab:gray", linewidth=1.0, label="raw")
            ax.plot(tvals[idx], proc_signals[idx, :, 3], color="tab:red", linewidth=1.5, label="preprocessed")
            ax.set_title("5s training window before/after preprocessing: " + lab)
            ax.set_ylabel("Absolute acceleration")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time within window (s)")
        save_figure(fig, "window_preprocessing_comparison.png")


def main():
    ensure_output_dir()
    summary = preprocess_hdf5()
    plot_raw_vs_preprocessed_examples()
    plot_axis_comparison()
    plot_preprocessed_window_examples()

    print("Updated preprocessed data in:", HDF5_PATH)
    print("Saved step 4 figures to:", OUTPUT_DIR)
    print("Missing values filled:", summary["raw_missing_values_filled"])
    print("Preprocessed training windows:", summary["train_windows"])
    print("Preprocessed testing windows:", summary["test_windows"])


if __name__ == "__main__":
    main()
