from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
HDF5_PATH = BASE_DIR / "data" / "hoppers_vs_walkers.h5"
OUTPUT_DIR = BASE_DIR / "figures" / "step5"

CHANNEL_NAMES = ["x", "y", "z", "abs"]
STAT_NAMES = [
    "max",
    "min",
    "mean",
    "std",
    "median",
    "range",
    "variance",
    "energy",
    "skewness",
    "kurtosis",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def reset_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def compute_channel_features(channel_values: np.ndarray) -> np.ndarray:
    mean_value = float(np.mean(channel_values))
    std_value = float(np.std(channel_values))
    centered = channel_values - mean_value

    if std_value == 0.0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        standardized = centered / std_value
        skewness = float(np.mean(standardized**3))
        kurtosis = float(np.mean(standardized**4))

    return np.array(
        [
            float(np.max(channel_values)),
            float(np.min(channel_values)),
            mean_value,
            std_value,
            float(np.median(channel_values)),
            float(np.max(channel_values) - np.min(channel_values)),
            float(np.var(channel_values)),
            float(np.mean(channel_values**2)),
            skewness,
            kurtosis,
        ],
        dtype=np.float64,
    )


def extract_feature_matrix(signal_windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    feature_rows = []
    feature_names = []

    for channel_name in CHANNEL_NAMES:
        for stat_name in STAT_NAMES:
            feature_names.append(f"{channel_name}_{stat_name}")

    for window in signal_windows:
        channel_feature_vectors = []
        for channel_index in range(window.shape[1]):
            channel_feature_vectors.append(compute_channel_features(window[:, channel_index]))
        feature_rows.append(np.concatenate(channel_feature_vectors))

    return np.vstack(feature_rows), feature_names


def z_score_normalize(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)
    safe_std = np.where(train_std == 0.0, 1.0, train_std)

    normalized_train = (train_features - train_mean) / safe_std
    normalized_test = (test_features - train_mean) / safe_std

    return normalized_train, normalized_test, train_mean, safe_std


def write_feature_group(
    group: h5py.Group,
    feature_matrix: np.ndarray,
    normalized_feature_matrix: np.ndarray,
    labels: np.ndarray,
    participants: np.ndarray,
    source_files: np.ndarray,
    feature_names: list[str],
) -> None:
    group.create_dataset("features", data=feature_matrix, compression="gzip")
    group.create_dataset("normalized_features", data=normalized_feature_matrix, compression="gzip")
    group.create_dataset("labels", data=labels, compression="gzip")
    group.create_dataset("participants", data=participants, compression="gzip")
    group.create_dataset("source_files", data=source_files, compression="gzip")
    group.create_dataset(
        "feature_names",
        data=np.array(feature_names, dtype="S64"),
        compression="gzip",
    )
    group.attrs["feature_count"] = int(feature_matrix.shape[1])
    group.attrs["window_count"] = int(feature_matrix.shape[0])


def store_features() -> dict[str, int]:
    with h5py.File(HDF5_PATH, "r+") as hdf:
        features_root = reset_group(hdf["preprocessed"], "features")

        train_group = hdf["preprocessed"]["train"]
        test_group = hdf["preprocessed"]["test"]

        train_signals = train_group["signals"][:]
        test_signals = test_group["signals"][:]

        train_features, feature_names = extract_feature_matrix(train_signals)
        test_features, _ = extract_feature_matrix(test_signals)

        normalized_train, normalized_test, train_mean, train_std = z_score_normalize(
            train_features,
            test_features,
        )

        train_feature_group = features_root.create_group("train")
        test_feature_group = features_root.create_group("test")

        write_feature_group(
            train_feature_group,
            train_features,
            normalized_train,
            train_group["labels"][:],
            train_group["participants"][:],
            train_group["source_files"][:],
            feature_names,
        )
        write_feature_group(
            test_feature_group,
            test_features,
            normalized_test,
            test_group["labels"][:],
            test_group["participants"][:],
            test_group["source_files"][:],
            feature_names,
        )

        features_root.create_dataset("normalization_mean", data=train_mean, compression="gzip")
        features_root.create_dataset("normalization_std", data=train_std, compression="gzip")
        features_root.create_dataset(
            "feature_names",
            data=np.array(feature_names, dtype="S64"),
            compression="gzip",
        )
        features_root.attrs["normalization_method"] = "z_score"
        features_root.attrs["feature_count"] = int(len(feature_names))

    return {
        "train_windows": int(train_features.shape[0]),
        "test_windows": int(test_features.shape[0]),
        "feature_count": int(train_features.shape[1]),
    }


def plot_feature_mean_comparison() -> None:
    with h5py.File(HDF5_PATH, "r") as hdf:
        feature_names = decode(hdf["preprocessed"]["features"]["feature_names"][:])
        train_features = hdf["preprocessed"]["features"]["train"]["features"][:]
        train_labels = decode(hdf["preprocessed"]["features"]["train"]["labels"][:])

    selected_feature_names = [
        "abs_mean",
        "abs_std",
        "abs_max",
        "abs_range",
        "z_std",
        "z_energy",
    ]
    selected_indices = [feature_names.index(name) for name in selected_feature_names]

    walking_mask = np.array([label == "walking" for label in train_labels])
    jumping_mask = np.array([label == "jumping" for label in train_labels])

    walking_means = np.mean(train_features[walking_mask][:, selected_indices], axis=0)
    jumping_means = np.mean(train_features[jumping_mask][:, selected_indices], axis=0)

    x_positions = np.arange(len(selected_feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_positions - width / 2, walking_means, width, color="tab:blue", label="walking")
    ax.bar(x_positions + width / 2, jumping_means, width, color="tab:orange", label="jumping")
    ax.set_title("Average feature values for selected training features")
    ax.set_ylabel("Feature value")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(selected_feature_names, rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_figure(fig, "selected_feature_means.png")


def plot_normalization_effect() -> None:
    with h5py.File(HDF5_PATH, "r") as hdf:
        feature_names = decode(hdf["preprocessed"]["features"]["feature_names"][:])
        raw_features = hdf["preprocessed"]["features"]["train"]["features"][:]
        normalized_features = hdf["preprocessed"]["features"]["train"]["normalized_features"][:]

    selected_feature_names = ["abs_mean", "abs_std", "abs_range", "z_energy"]
    selected_indices = [feature_names.index(name) for name in selected_feature_names]

    raw_means = np.mean(raw_features[:, selected_indices], axis=0)
    raw_stds = np.std(raw_features[:, selected_indices], axis=0)
    normalized_means = np.mean(normalized_features[:, selected_indices], axis=0)
    normalized_stds = np.std(normalized_features[:, selected_indices], axis=0)

    x_positions = np.arange(len(selected_feature_names))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(x_positions - width / 2, raw_means, width, color="tab:gray", label="raw feature mean")
    axes[0].bar(
        x_positions + width / 2,
        normalized_means,
        width,
        color="tab:green",
        label="normalized feature mean",
    )
    axes[0].set_title("Effect of z-score normalization on selected features")
    axes[0].set_ylabel("Mean across training windows")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x_positions - width / 2, raw_stds, width, color="tab:gray", label="raw feature std")
    axes[1].bar(
        x_positions + width / 2,
        normalized_stds,
        width,
        color="tab:green",
        label="normalized feature std",
    )
    axes[1].set_ylabel("Standard deviation")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(selected_feature_names, rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    save_figure(fig, "normalization_effect.png")


def main() -> None:
    ensure_output_dir()
    summary = store_features()
    plot_feature_mean_comparison()
    plot_normalization_effect()

    print(f"Updated feature data in: {HDF5_PATH}")
    print(f"Saved step 5 figures to: {OUTPUT_DIR}")
    print(f"Training windows processed: {summary['train_windows']}")
    print(f"Testing windows processed: {summary['test_windows']}")
    print(f"Features per window: {summary['feature_count']}")


if __name__ == "__main__":
    main()
