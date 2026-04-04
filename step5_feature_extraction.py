# Step 5 — feature extraction

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


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def decode(values):
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode())
        else:
            out.append(str(v))
    return out


def reset_group(parent, name):
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def compute_channel_features(channel_values):
    mu = float(np.mean(channel_values))
    sd = float(np.std(channel_values))
    centered = channel_values - mu

    if sd == 0.0:
        sk = 0.0
        ku = 0.0
    else:
        z = centered / sd
        sk = float(np.mean(z**3))
        ku = float(np.mean(z**4))

    return np.array(
        [
            float(np.max(channel_values)),
            float(np.min(channel_values)),
            mu,
            sd,
            float(np.median(channel_values)),
            float(np.max(channel_values) - np.min(channel_values)),
            float(np.var(channel_values)),
            float(np.mean(channel_values**2)),
            sk,
            ku,
        ],
        dtype=np.float64,
    )


def extract_feature_matrix(signal_windows):
    feature_rows = []
    feature_names = []

    for ch in CHANNEL_NAMES:
        for st in STAT_NAMES:
            feature_names.append(ch + "_" + st)

    for win in signal_windows:
        vecs = []
        for c in range(win.shape[1]):
            vecs.append(compute_channel_features(win[:, c]))
        feature_rows.append(np.concatenate(vecs))

    return np.vstack(feature_rows), feature_names


def z_score_normalize(train_features, test_features):
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)
    safe_std = np.where(train_std == 0.0, 1.0, train_std)

    nt = (train_features - train_mean) / safe_std
    nv = (test_features - train_mean) / safe_std
    return nt, nv, train_mean, safe_std


def write_feature_group(group, feature_matrix, normalized_feature_matrix, labels, participants, source_files, feature_names):
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


def store_features():
    with h5py.File(HDF5_PATH, "r+") as hdf:
        feat_root = reset_group(hdf["preprocessed"], "features")

        tr = hdf["preprocessed"]["train"]
        te = hdf["preprocessed"]["test"]

        Xtr = tr["signals"][:]
        Xte = te["signals"][:]

        train_features, names = extract_feature_matrix(Xtr)
        test_features, _ = extract_feature_matrix(Xte)

        ntr, nte, mu, sig = z_score_normalize(train_features, test_features)

        g_tr = feat_root.create_group("train")
        g_te = feat_root.create_group("test")

        write_feature_group(
            g_tr,
            train_features,
            ntr,
            tr["labels"][:],
            tr["participants"][:],
            tr["source_files"][:],
            names,
        )
        write_feature_group(
            g_te,
            test_features,
            nte,
            te["labels"][:],
            te["participants"][:],
            te["source_files"][:],
            names,
        )

        feat_root.create_dataset("normalization_mean", data=mu, compression="gzip")
        feat_root.create_dataset("normalization_std", data=sig, compression="gzip")
        feat_root.create_dataset(
            "feature_names",
            data=np.array(names, dtype="S64"),
            compression="gzip",
        )
        feat_root.attrs["normalization_method"] = "z_score"
        feat_root.attrs["feature_count"] = int(len(names))

    return {
        "train_windows": int(train_features.shape[0]),
        "test_windows": int(test_features.shape[0]),
        "feature_count": int(train_features.shape[1]),
    }


def plot_feature_mean_comparison():
    with h5py.File(HDF5_PATH, "r") as hdf:
        fnames = decode(hdf["preprocessed"]["features"]["feature_names"][:])
        X = hdf["preprocessed"]["features"]["train"]["features"][:]
        y = decode(hdf["preprocessed"]["features"]["train"]["labels"][:])

    pick = [
        "abs_mean",
        "abs_std",
        "abs_max",
        "abs_range",
        "z_std",
        "z_energy",
    ]
    idxs = []
    for name in pick:
        idxs.append(fnames.index(name))

    walk_mask = np.array([lab == "walking" for lab in y])
    jump_mask = np.array([lab == "jumping" for lab in y])

    m_walk = np.mean(X[walk_mask][:, idxs], axis=0)
    m_jump = np.mean(X[jump_mask][:, idxs], axis=0)

    xpos = np.arange(len(pick))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xpos - w / 2, m_walk, w, color="tab:blue", label="walking")
    ax.bar(xpos + w / 2, m_jump, w, color="tab:orange", label="jumping")
    ax.set_title("Average feature values for a few training features")
    ax.set_ylabel("Feature value")
    ax.set_xticks(xpos)
    ax.set_xticklabels(pick, rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_figure(fig, "selected_feature_means.png")


def plot_normalization_effect():
    with h5py.File(HDF5_PATH, "r") as hdf:
        fnames = decode(hdf["preprocessed"]["features"]["feature_names"][:])
        raw = hdf["preprocessed"]["features"]["train"]["features"][:]
        norm = hdf["preprocessed"]["features"]["train"]["normalized_features"][:]

    pick = ["abs_mean", "abs_std", "abs_range", "z_energy"]
    idxs = [fnames.index(n) for n in pick]

    rm = np.mean(raw[:, idxs], axis=0)
    rs = np.std(raw[:, idxs], axis=0)
    nm = np.mean(norm[:, idxs], axis=0)
    ns = np.std(norm[:, idxs], axis=0)

    xpos = np.arange(len(pick))
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(xpos - w / 2, rm, w, color="tab:gray", label="raw feature mean")
    axes[0].bar(xpos + w / 2, nm, w, color="tab:green", label="normalized feature mean")
    axes[0].set_title("What z-score normalization does (selected features)")
    axes[0].set_ylabel("Mean across training windows")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(xpos - w / 2, rs, w, color="tab:gray", label="raw feature std")
    axes[1].bar(xpos + w / 2, ns, w, color="tab:green", label="normalized feature std")
    axes[1].set_ylabel("Standard deviation")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(pick, rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    save_figure(fig, "normalization_effect.png")


def main():
    ensure_output_dir()
    s = store_features()
    plot_feature_mean_comparison()
    plot_normalization_effect()

    print("Updated feature data in:", HDF5_PATH)
    print("Saved step 5 figures to:", OUTPUT_DIR)
    print("Training windows processed:", s["train_windows"])
    print("Testing windows processed:", s["test_windows"])
    print("Features per window:", s["feature_count"])


if __name__ == "__main__":
    main()
