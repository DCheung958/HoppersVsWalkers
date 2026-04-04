# Step 2 — data storage put everyone's CSVs into one HDF5 file for the rest of the project

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


def load_signal(csv_path):
    df = pd.read_csv(csv_path)

    needed_cols = [TIME_COLUMN] + AXIS_COLUMNS + [ABS_COLUMN]
    missing = []
    for c in needed_cols:
        if c not in df.columns:
            missing.append(c)
    if len(missing) > 0:
        raise ValueError(str(csv_path) + " is missing columns: " + str(missing))

    df = df[needed_cols].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        raise ValueError(str(csv_path) + " has NaNs or bad numbers.")

    t = df[TIME_COLUMN].to_numpy()
    if len(t) < 2:
        raise ValueError(str(csv_path) + " is too short.")

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError(str(csv_path) + " time column is not strictly increasing.")

    return df


def create_windows(df, window_seconds, target_samples):
    t = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    sig = df[AXIS_COLUMNS + [ABS_COLUMN]].to_numpy(dtype=np.float64)

    t0 = float(t[0])
    t1 = float(t[-1])
    starts = np.arange(t0, t1 - window_seconds, window_seconds)

    if len(starts) == 0:
        return np.empty((0, target_samples, 4)), np.empty((0, target_samples))

    ref_t = np.linspace(0.0, window_seconds, num=target_samples, endpoint=False)
    out = []

    for ws in starts:
        we = ws + window_seconds
        m = (t >= ws) & (t < we)
        wt = t[m]
        wsig = sig[m]

        if len(wt) < 2:
            continue

        wt = wt - wt[0]
        cols = []
        for j in range(wsig.shape[1]):
            cols.append(np.interp(ref_t, wt, wsig[:, j]))
        out.append(np.column_stack(cols))

    if len(out) == 0:
        return np.empty((0, target_samples, 4)), np.empty((0, target_samples))

    stacked = np.stack(out)
    times_out = np.tile(ref_t, (len(out), 1))
    return stacked, times_out


def write_raw_group(group, df, csv_path):
    group.create_dataset("time", data=df[TIME_COLUMN].to_numpy(dtype=np.float64), compression="gzip")
    group.create_dataset("acceleration_xyz", data=df[AXIS_COLUMNS].to_numpy(dtype=np.float64), compression="gzip")
    group.create_dataset("absolute_acceleration", data=df[ABS_COLUMN].to_numpy(dtype=np.float64), compression="gzip")

    group.attrs["source_file"] = str(csv_path.as_posix())
    group.attrs["sample_count"] = len(df)
    dur = float(df[TIME_COLUMN].iloc[-1] - df[TIME_COLUMN].iloc[0])
    group.attrs["duration_seconds"] = dur


def write_split_group(split_group, signals, times, labels):
    split_group.create_dataset("signals", data=signals, compression="gzip")
    split_group.create_dataset("time", data=times, compression="gzip")
    split_group.create_dataset("labels", data=labels.astype("S16"), compression="gzip")


def main():
    csv_paths = []
    for p in RAW_DATA_DIR.glob("*/*.csv"):
        if p.name in ("walking.csv", "jumping.csv"):
            csv_paths.append(p)
    csv_paths.sort()

    if len(csv_paths) == 0:
        raise FileNotFoundError("No CSV files under " + str(RAW_DATA_DIR))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    split_records = []

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
            person = csv_path.parent.name
            activity = csv_path.stem

            df = load_signal(csv_path)
            pg = raw_root.require_group(person)
            ag = pg.create_group(activity)
            write_raw_group(ag, df, csv_path)

            wins, win_times = create_windows(df, WINDOW_SECONDS, TARGET_SAMPLES_PER_WINDOW)

            for i in range(len(wins)):
                split_records.append(
                    {
                        "participant": person,
                        "activity": activity,
                        "source_file": str(csv_path.as_posix()),
                        "signals": wins[i],
                        "time": win_times[i],
                    }
                )

        if len(split_records) == 0:
            raise ValueError("Could not build any 5s windows from the CSVs.")

        rng.shuffle(split_records)
        n_train = int(len(split_records) * TRAIN_RATIO)
        train_recs = split_records[:n_train]
        test_recs = split_records[n_train:]

        for split_name, records in [("train", train_recs), ("test", test_recs)]:
            sg = splits_root.create_group(split_name)

            sig_list = []
            time_list = []
            lab_list = []
            part_list = []
            file_list = []
            for rec in records:
                sig_list.append(rec["signals"])
                time_list.append(rec["time"])
                lab_list.append(rec["activity"])
                part_list.append(rec["participant"])
                file_list.append(rec["source_file"])

            signals = np.stack(sig_list)
            times = np.stack(time_list)
            labels = np.array(lab_list)
            participants = np.array(part_list, dtype="S32")
            source_files = np.array(file_list, dtype="S256")

            write_split_group(sg, signals, times, labels)
            sg.create_dataset("participants", data=participants, compression="gzip")
            sg.create_dataset("source_files", data=source_files, compression="gzip")
            sg.attrs["window_count"] = len(records)

    print("Wrote HDF5 file to:", OUTPUT_PATH)
    print("Total windows:", len(split_records))

    print("Training windows:", len(train_recs))
    print("Testing windows:", len(test_recs))


if __name__ == "__main__":
    main()
