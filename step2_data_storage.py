from pathlib import Path

import h5py
import numpy as np
import pandas as pd


RAW_DATA_DIR = Path("raw_data/raw_data")
OUTPUT_PATH = Path("data/hoppers_vs_walkers.h5")
WINDOW_SECONDS = 5.0
TARGET_SAMPLES_PER_WINDOW = 498
TRAIN_RATIO = 0.9
RANDOM_SEED = 292

TIME_COLUMN = "Time (s)"
AXIS_COLUMNS = [
    "Acceleration x (m/s^2)",
    "Acceleration y (m/s^2)",
    "Acceleration z (m/s^2)",
]
ABS_COLUMN = "Absolute acceleration (m/s^2)"


def load_signal(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = [TIME_COLUMN, *AXIS_COLUMNS, ABS_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

    df = df[required_columns].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        raise ValueError(f"{csv_path} contains missing or non-numeric values.")

    time_values = df[TIME_COLUMN].to_numpy()
    if len(time_values) < 2:
        raise ValueError(f"{csv_path} does not contain enough samples.")

    time_deltas = np.diff(time_values)
    if np.any(time_deltas <= 0):
        raise ValueError(f"{csv_path} has non-increasing timestamps.")

    return df


def create_windows(df: pd.DataFrame, window_seconds: float, target_samples: int) -> tuple[np.ndarray, np.ndarray]:
    time_values = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    signal_values = df[AXIS_COLUMNS + [ABS_COLUMN]].to_numpy(dtype=np.float64)

    start_time = float(time_values[0])
    end_time = float(time_values[-1])
    window_starts = np.arange(start_time, end_time - window_seconds, window_seconds)

    if len(window_starts) == 0:
        return np.empty((0, target_samples, 4)), np.empty((0, target_samples))

    reference_time = np.linspace(0.0, window_seconds, num=target_samples, endpoint=False)
    signal_windows = []

    for window_start in window_starts:
        window_end = window_start + window_seconds
        window_mask = (time_values >= window_start) & (time_values < window_end)
        window_time = time_values[window_mask]
        window_signal = signal_values[window_mask]

        if len(window_time) < 2:
            continue

        window_time = window_time - window_time[0]
        resampled_axes = []
        for axis_index in range(window_signal.shape[1]):
            resampled_axis = np.interp(reference_time, window_time, window_signal[:, axis_index])
            resampled_axes.append(resampled_axis)

        signal_windows.append(np.column_stack(resampled_axes))

    if not signal_windows:
        return np.empty((0, target_samples, 4)), np.empty((0, target_samples))

    return np.stack(signal_windows), np.tile(reference_time, (len(signal_windows), 1))


def write_raw_group(group: h5py.Group, df: pd.DataFrame, csv_path: Path) -> None:
    group.create_dataset("time", data=df[TIME_COLUMN].to_numpy(dtype=np.float64), compression="gzip")
    group.create_dataset(
        "acceleration_xyz",
        data=df[AXIS_COLUMNS].to_numpy(dtype=np.float64),
        compression="gzip",
    )
    group.create_dataset(
        "absolute_acceleration",
        data=df[ABS_COLUMN].to_numpy(dtype=np.float64),
        compression="gzip",
    )

    group.attrs["source_file"] = str(csv_path.as_posix())
    group.attrs["sample_count"] = len(df)
    group.attrs["duration_seconds"] = float(df[TIME_COLUMN].iloc[-1] - df[TIME_COLUMN].iloc[0])


def write_split_group(split_group: h5py.Group, signals: np.ndarray, times: np.ndarray, labels: np.ndarray) -> None:
    split_group.create_dataset("signals", data=signals, compression="gzip")
    split_group.create_dataset("time", data=times, compression="gzip")
    split_group.create_dataset("labels", data=labels.astype("S16"), compression="gzip")


def main() -> None:
    csv_paths = sorted(RAW_DATA_DIR.glob("*/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {RAW_DATA_DIR}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    split_records: list[dict[str, object]] = []

    with h5py.File(OUTPUT_PATH, "w") as hdf:
        hdf.attrs["project"] = "HoppersVsWalkers"
        hdf.attrs["window_seconds"] = WINDOW_SECONDS
        hdf.attrs["target_samples_per_window"] = TARGET_SAMPLES_PER_WINDOW
        hdf.attrs["train_ratio"] = TRAIN_RATIO
        hdf.attrs["random_seed"] = RANDOM_SEED

        raw_root = hdf.create_group("raw")
        preprocessed_root = hdf.create_group("preprocessed")
        splits_root = hdf.create_group("splits")

        preprocessed_root.create_group("raw_signals")
        preprocessed_root.create_group("train")
        preprocessed_root.create_group("test")

        for csv_path in csv_paths:
            participant = csv_path.parent.name
            activity = csv_path.stem

            df = load_signal(csv_path)
            participant_group = raw_root.require_group(participant)
            activity_group = participant_group.create_group(activity)
            write_raw_group(activity_group, df, csv_path)

            signal_windows, time_windows = create_windows(
                df,
                WINDOW_SECONDS,
                TARGET_SAMPLES_PER_WINDOW,
            )

            for signal_window, time_window in zip(signal_windows, time_windows, strict=False):
                split_records.append(
                    {
                        "participant": participant,
                        "activity": activity,
                        "source_file": str(csv_path.as_posix()),
                        "signals": signal_window,
                        "time": time_window,
                    }
                )

        if not split_records:
            raise ValueError("No 5-second windows could be created from the raw CSV files.")

        rng.shuffle(split_records)
        train_count = int(len(split_records) * TRAIN_RATIO)
        train_records = split_records[:train_count]
        test_records = split_records[train_count:]

        for split_name, records in [("train", train_records), ("test", test_records)]:
            split_group = splits_root.create_group(split_name)

            signals = np.stack([record["signals"] for record in records])
            times = np.stack([record["time"] for record in records])
            labels = np.array([record["activity"] for record in records])
            participants = np.array([record["participant"] for record in records], dtype="S32")
            source_files = np.array([record["source_file"] for record in records], dtype="S256")

            write_split_group(split_group, signals, times, labels)
            split_group.create_dataset("participants", data=participants, compression="gzip")
            split_group.create_dataset("source_files", data=source_files, compression="gzip")
            split_group.attrs["window_count"] = len(records)

    print(f"Wrote HDF5 file to: {OUTPUT_PATH}")
    print(f"Total windows: {len(split_records)}")
    print(f"Training windows: {len(train_records)}")
    print(f"Testing windows: {len(test_records)}")


if __name__ == "__main__":
    main()
