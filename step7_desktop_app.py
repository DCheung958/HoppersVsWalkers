# Step 7 — desktop app

from pathlib import Path
import csv
import pickle
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"
TIME_COLUMN = "Time (s)"
AXIS_COLUMNS = [
    "Acceleration x (m/s^2)",
    "Acceleration y (m/s^2)",
    "Acceleration z (m/s^2)",
]
ABS_COLUMN = "Absolute acceleration (m/s^2)"
WINDOW_SECONDS = 5.0
TARGET_SAMPLES_PER_WINDOW = 498
MOVING_AVERAGE_WINDOW = 9

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


def load_model_bundle():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def sigmoid(values):
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_signal(csv_path):
    df = pd.read_csv(csv_path)
    need = [TIME_COLUMN] + AXIS_COLUMNS + [ABS_COLUMN]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + str(missing))

    df = df[need].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError("Input CSV contains missing or non-numeric values.")

    t = df[TIME_COLUMN].to_numpy()
    if len(t) < 2:
        raise ValueError("Input CSV does not contain enough samples.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Input CSV timestamps must be strictly increasing.")

    return df


def create_windows(df, window_seconds, target_samples):
    t = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    sig = df[AXIS_COLUMNS + [ABS_COLUMN]].to_numpy(dtype=np.float64)

    t0 = float(t[0])
    t1 = float(t[-1])
    starts = np.arange(t0, t1 - window_seconds, window_seconds)
    if len(starts) == 0:
        return (
            np.empty((0, target_samples, 4), dtype=np.float64),
            np.empty((0, target_samples), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    ref_t = np.linspace(0.0, window_seconds, num=target_samples, endpoint=False)
    wins = []
    tw = []
    valid_starts = []

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

        wins.append(np.column_stack(cols))
        tw.append(ref_t.copy())
        valid_starts.append(ws)

    if len(wins) == 0:
        return (
            np.empty((0, target_samples, 4), dtype=np.float64),
            np.empty((0, target_samples), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    return np.stack(wins), np.stack(tw), np.array(valid_starts, dtype=np.float64)


def fill_missing_1d(values):
    filled = values.astype(np.float64, copy=True)
    nan_mask = np.isnan(filled)
    if not np.any(nan_mask):
        return filled

    ok = np.where(~nan_mask)[0]
    if len(ok) == 0:
        return np.zeros_like(filled)

    first = ok[0]
    filled[:first] = filled[first]

    for i in range(first + 1, len(filled)):
        if np.isnan(filled[i]):
            filled[i] = filled[i - 1]

    return filled


def moving_average(values, window_size):
    out = np.zeros_like(values, dtype=np.float64)
    for i in range(len(values)):
        s = max(0, i - window_size + 1)
        out[i] = np.mean(values[s : i + 1])
    return out


def preprocess_signal_matrix(signal_matrix, window_size):
    out = np.zeros_like(signal_matrix, dtype=np.float64)
    for c in range(signal_matrix.shape[1]):
        col = fill_missing_1d(signal_matrix[:, c])
        out[:, c] = moving_average(col, window_size)
    return out


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
    rows = []
    for win in signal_windows:
        bits = []
        for c in range(win.shape[1]):
            bits.append(compute_channel_features(win[:, c]))
        rows.append(np.concatenate(bits))
    return np.vstack(rows)


def normalize_features(feature_matrix, normalization_mean, normalization_std):
    return (feature_matrix - normalization_mean) / normalization_std


def predict_probabilities(normalized_features, model_bundle):
    z = normalized_features @ model_bundle["coefficients"] + model_bundle["intercept"]
    return sigmoid(z)


def classify_file(csv_path, model_bundle):
    df = load_signal(csv_path)
    signal_windows, time_windows, window_starts = create_windows(df, WINDOW_SECONDS, TARGET_SAMPLES_PER_WINDOW)
    if len(signal_windows) == 0:
        raise ValueError("The input file is too short to form any 5-second windows.")

    proc = np.zeros_like(signal_windows, dtype=np.float64)
    for i in range(signal_windows.shape[0]):
        proc[i] = preprocess_signal_matrix(signal_windows[i], MOVING_AVERAGE_WINDOW)

    feats = extract_feature_matrix(proc)
    normed = normalize_features(
        feats,
        model_bundle["normalization_mean"],
        model_bundle["normalization_std"],
    )

    probs = predict_probabilities(normed, model_bundle)
    thr = model_bundle["classification_threshold"]
    pred_bin = (probs >= thr).astype(int)
    pred_labels = []
    for v in pred_bin:
        if v == 1:
            pred_labels.append(model_bundle["positive_label"])
        else:
            pred_labels.append(model_bundle["negative_label"])

    return {
        "time_windows": time_windows,
        "window_starts": window_starts,
        "predicted_labels": pred_labels,
        "probabilities": probs,
    }


def write_predictions_csv(output_path, predictions):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_index", "window_start_s", "window_end_s", "predicted_label", "jumping_probability"])

        for idx, (ws, lab, p) in enumerate(
            zip(
                predictions["window_starts"],
                predictions["predicted_labels"],
                predictions["probabilities"],
            )
        ):
            w.writerow(
                [
                    idx,
                    "%.4f" % ws,
                    "%.4f" % (ws + WINDOW_SECONDS),
                    lab,
                    "%.6f" % p,
                ]
            )


def save_prediction_plot(plot_path, predictions):
    starts = predictions["window_starts"]
    probs = predictions["probabilities"]
    labels = predictions["predicted_labels"]
    enc = np.array([1 if lab == "jumping" else 0 for lab in labels], dtype=np.int64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(starts, probs, marker="o", color="tab:red", linewidth=1.8)
    axes[0].axhline(0.5, linestyle="--", color="tab:gray", linewidth=1.0)
    axes[0].set_ylabel("Jumping probability")
    axes[0].set_title("Predicted probability per 5-second window")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(starts, enc, where="post", color="tab:blue", linewidth=1.8)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["walking", "jumping"])
    axes[1].set_xlabel("Window start time (s)")
    axes[1].set_ylabel("Predicted label")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


class ActivityClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HoppersVsWalkers Classifier")
        self.root.geometry("980x760")

        self.model_bundle = load_model_bundle()
        self.input_path = None
        self.last_predictions = None

        self.status_var = tk.StringVar(value="Select a CSV file to begin.")
        self.input_var = tk.StringVar(value="No file selected")

        self._build_ui()

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding=12)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Input CSV:").grid(row=0, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.input_var, width=90).grid(row=0, column=1, padx=8, sticky="ew")
        ttk.Button(ctrl, text="Browse", command=self.select_file).grid(row=0, column=2, padx=4)
        ttk.Button(ctrl, text="Classify", command=self.run_classification).grid(row=0, column=3, padx=4)
        ttk.Button(ctrl, text="Save Output CSV", command=self.save_output_csv).grid(row=0, column=4, padx=4)
        ttk.Button(ctrl, text="Save Plot", command=self.save_plot).grid(row=0, column=5, padx=4)
        ctrl.columnconfigure(1, weight=1)

        ttk.Label(self.root, textvariable=self.status_var, padding=(12, 0, 12, 8)).pack(fill="x")

        results = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        results.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            results,
            columns=("start", "end", "label", "probability"),
            show="headings",
            height=8,
        )
        self.tree.heading("start", text="Window Start (s)")
        self.tree.heading("end", text="Window End (s)")
        self.tree.heading("label", text="Predicted Label")
        self.tree.heading("probability", text="Jumping Probability")
        for col, width in [("start", 130), ("end", 130), ("label", 160), ("probability", 160)]:
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(fill="x", pady=(0, 12))

        self.figure, self.axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=results)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._clear_plot()

    def _clear_plot(self):
        for ax in self.axes:
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.axes[0].set_title("Predicted probability per 5-second window")
        self.axes[0].set_ylabel("Jumping probability")
        self.axes[1].set_ylabel("Predicted label")
        self.axes[1].set_xlabel("Window start time (s)")
        self.axes[1].set_yticks([0, 1])
        self.axes[1].set_yticklabels(["walking", "jumping"])
        self.canvas.draw()

    def select_file(self):
        selected = filedialog.askopenfilename(
            title="Select accelerometer CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.input_path = Path(selected)
            self.input_var.set(str(self.input_path))
            self.status_var.set("File selected. Click Classify to run the model.")

    def run_classification(self):
        if self.input_path is None:
            messagebox.showerror("No file selected", "Please choose a CSV file first.")
            return

        try:
            predictions = classify_file(self.input_path, self.model_bundle)
        except Exception as e:
            messagebox.showerror("Classification failed", str(e))
            return

        self.last_predictions = predictions
        self._populate_table(predictions)
        self._update_plot(predictions)
        n = len(predictions["predicted_labels"])
        self.status_var.set("Classification complete. Generated " + str(n) + " windows.")

    def _populate_table(self, predictions):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for ws, lab, p in zip(
            predictions["window_starts"],
            predictions["predicted_labels"],
            predictions["probabilities"],
        ):
            self.tree.insert(
                "",
                "end",
                values=(
                    "%.4f" % ws,
                    "%.4f" % (ws + WINDOW_SECONDS),
                    lab,
                    "%.4f" % p,
                ),
            )

    def _update_plot(self, predictions):
        self._clear_plot()

        starts = predictions["window_starts"]
        probs = predictions["probabilities"]
        enc = np.array([1 if lab == "jumping" else 0 for lab in predictions["predicted_labels"]])

        self.axes[0].plot(starts, probs, marker="o", color="tab:red", linewidth=1.8)
        self.axes[0].axhline(0.5, linestyle="--", color="tab:gray", linewidth=1.0)
        self.axes[1].step(starts, enc, where="post", color="tab:blue", linewidth=1.8)
        self.canvas.draw()

    def save_output_csv(self):
        if self.last_predictions is None:
            messagebox.showerror("No predictions", "Run classification before saving outputs.")
            return

        out = filedialog.asksaveasfilename(
            title="Save prediction CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not out:
            return

        write_predictions_csv(Path(out), self.last_predictions)
        self.status_var.set("Saved prediction CSV to: " + out)

    def save_plot(self):
        if self.last_predictions is None:
            messagebox.showerror("No predictions", "Run classification before saving the plot.")
            return

        out = filedialog.asksaveasfilename(
            title="Save prediction plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
        )
        if not out:
            return

        save_prediction_plot(Path(out), self.last_predictions)
        self.status_var.set("Saved prediction plot to: " + out)


def run_cli(csv_path):
    bundle = load_model_bundle()
    predictions = classify_file(csv_path, bundle)
    out_csv = csv_path.with_name(csv_path.stem + "_predictions.csv")
    out_png = csv_path.with_name(csv_path.stem + "_predictions.png")

    write_predictions_csv(out_csv, predictions)
    save_prediction_plot(out_png, predictions)

    print("Saved prediction CSV to:", out_csv)
    print("Saved prediction plot to:", out_png)
    print("Generated", len(predictions["predicted_labels"]), "predictions.")


def main():
    if len(sys.argv) > 1:
        run_cli(Path(sys.argv[1]))
        return

    root = tk.Tk()
    ActivityClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
