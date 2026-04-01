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


def load_model_bundle() -> dict:
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_signal(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = [TIME_COLUMN, *AXIS_COLUMNS, ABS_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df[required_columns].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError("Input CSV contains missing or non-numeric values.")

    time_values = df[TIME_COLUMN].to_numpy()
    if len(time_values) < 2:
        raise ValueError("Input CSV does not contain enough samples.")
    if np.any(np.diff(time_values) <= 0):
        raise ValueError("Input CSV timestamps must be strictly increasing.")

    return df


def create_windows(df: pd.DataFrame, window_seconds: float, target_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_values = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    signal_values = df[AXIS_COLUMNS + [ABS_COLUMN]].to_numpy(dtype=np.float64)

    start_time = float(time_values[0])
    end_time = float(time_values[-1])
    window_starts = np.arange(start_time, end_time - window_seconds, window_seconds)
    if len(window_starts) == 0:
        return (
            np.empty((0, target_samples, 4), dtype=np.float64),
            np.empty((0, target_samples), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    reference_time = np.linspace(0.0, window_seconds, num=target_samples, endpoint=False)
    signal_windows = []
    time_windows = []
    valid_window_starts = []

    for window_start in window_starts:
        window_end = window_start + window_seconds
        mask = (time_values >= window_start) & (time_values < window_end)
        window_time = time_values[mask]
        window_signal = signal_values[mask]

        if len(window_time) < 2:
            continue

        window_time = window_time - window_time[0]
        resampled_axes = []
        for axis_index in range(window_signal.shape[1]):
            resampled_axis = np.interp(reference_time, window_time, window_signal[:, axis_index])
            resampled_axes.append(resampled_axis)

        signal_windows.append(np.column_stack(resampled_axes))
        time_windows.append(reference_time.copy())
        valid_window_starts.append(window_start)

    if not signal_windows:
        return (
            np.empty((0, target_samples, 4), dtype=np.float64),
            np.empty((0, target_samples), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    return np.stack(signal_windows), np.stack(time_windows), np.array(valid_window_starts, dtype=np.float64)


def fill_missing_1d(values: np.ndarray) -> np.ndarray:
    filled = values.astype(np.float64, copy=True)
    nan_mask = np.isnan(filled)
    if not np.any(nan_mask):
        return filled

    valid_indices = np.where(~nan_mask)[0]
    if len(valid_indices) == 0:
        return np.zeros_like(filled)

    first_valid = valid_indices[0]
    filled[:first_valid] = filled[first_valid]

    for index in range(first_valid + 1, len(filled)):
        if np.isnan(filled[index]):
            filled[index] = filled[index - 1]

    return filled


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    filtered = np.zeros_like(values, dtype=np.float64)
    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        filtered[index] = np.mean(values[start_index : index + 1])
    return filtered


def preprocess_signal_matrix(signal_matrix: np.ndarray, window_size: int) -> np.ndarray:
    preprocessed = np.zeros_like(signal_matrix, dtype=np.float64)
    for column_index in range(signal_matrix.shape[1]):
        filled_column = fill_missing_1d(signal_matrix[:, column_index])
        preprocessed[:, column_index] = moving_average(filled_column, window_size)
    return preprocessed


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


def extract_feature_matrix(signal_windows: np.ndarray) -> np.ndarray:
    feature_rows = []
    for window in signal_windows:
        channel_feature_vectors = []
        for channel_index in range(window.shape[1]):
            channel_feature_vectors.append(compute_channel_features(window[:, channel_index]))
        feature_rows.append(np.concatenate(channel_feature_vectors))
    return np.vstack(feature_rows)


def normalize_features(feature_matrix: np.ndarray, normalization_mean: np.ndarray, normalization_std: np.ndarray) -> np.ndarray:
    return (feature_matrix - normalization_mean) / normalization_std


def predict_probabilities(normalized_features: np.ndarray, model_bundle: dict) -> np.ndarray:
    linear_output = normalized_features @ model_bundle["coefficients"] + model_bundle["intercept"]
    return sigmoid(linear_output)


def classify_file(csv_path: Path, model_bundle: dict) -> dict[str, object]:
    df = load_signal(csv_path)
    signal_windows, time_windows, window_starts = create_windows(df, WINDOW_SECONDS, TARGET_SAMPLES_PER_WINDOW)
    if len(signal_windows) == 0:
        raise ValueError("The input file is too short to form any 5-second windows.")

    preprocessed_windows = np.zeros_like(signal_windows, dtype=np.float64)
    for window_index in range(signal_windows.shape[0]):
        preprocessed_windows[window_index] = preprocess_signal_matrix(signal_windows[window_index], MOVING_AVERAGE_WINDOW)

    feature_matrix = extract_feature_matrix(preprocessed_windows)
    normalized_features = normalize_features(
        feature_matrix,
        model_bundle["normalization_mean"],
        model_bundle["normalization_std"],
    )

    probabilities = predict_probabilities(normalized_features, model_bundle)
    predicted_binary = (probabilities >= model_bundle["classification_threshold"]).astype(int)
    predicted_labels = [
        model_bundle["positive_label"] if value == 1 else model_bundle["negative_label"]
        for value in predicted_binary
    ]

    return {
        "time_windows": time_windows,
        "window_starts": window_starts,
        "predicted_labels": predicted_labels,
        "probabilities": probabilities,
    }


def write_predictions_csv(output_path: Path, predictions: dict[str, object]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["window_index", "window_start_s", "window_end_s", "predicted_label", "jumping_probability"])

        for index, (window_start, label, probability) in enumerate(
            zip(
                predictions["window_starts"],
                predictions["predicted_labels"],
                predictions["probabilities"],
                strict=False,
            )
        ):
            writer.writerow(
                [
                    index,
                    f"{window_start:.4f}",
                    f"{window_start + WINDOW_SECONDS:.4f}",
                    label,
                    f"{probability:.6f}",
                ]
            )


def save_prediction_plot(plot_path: Path, predictions: dict[str, object]) -> None:
    starts = predictions["window_starts"]
    probabilities = predictions["probabilities"]
    labels = predictions["predicted_labels"]
    encoded_labels = np.array([1 if label == "jumping" else 0 for label in labels], dtype=np.int64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(starts, probabilities, marker="o", color="tab:red", linewidth=1.8)
    axes[0].axhline(0.5, linestyle="--", color="tab:gray", linewidth=1.0)
    axes[0].set_ylabel("Jumping probability")
    axes[0].set_title("Predicted probability per 5-second window")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(starts, encoded_labels, where="post", color="tab:blue", linewidth=1.8)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["walking", "jumping"])
    axes[1].set_xlabel("Window start time (s)")
    axes[1].set_ylabel("Predicted label")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


class ActivityClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HoppersVsWalkers Classifier")
        self.root.geometry("980x760")

        self.model_bundle = load_model_bundle()
        self.input_path: Path | None = None
        self.last_predictions: dict[str, object] | None = None

        self.status_var = tk.StringVar(value="Select a CSV file to begin.")
        self.input_var = tk.StringVar(value="No file selected")

        self._build_ui()

    def _build_ui(self) -> None:
        control_frame = ttk.Frame(self.root, padding=12)
        control_frame.pack(fill="x")

        ttk.Label(control_frame, text="Input CSV:").grid(row=0, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.input_var, width=90).grid(row=0, column=1, padx=8, sticky="ew")
        ttk.Button(control_frame, text="Browse", command=self.select_file).grid(row=0, column=2, padx=4)
        ttk.Button(control_frame, text="Classify", command=self.run_classification).grid(row=0, column=3, padx=4)
        ttk.Button(control_frame, text="Save Output CSV", command=self.save_output_csv).grid(row=0, column=4, padx=4)
        ttk.Button(control_frame, text="Save Plot", command=self.save_plot).grid(row=0, column=5, padx=4)
        control_frame.columnconfigure(1, weight=1)

        ttk.Label(self.root, textvariable=self.status_var, padding=(12, 0, 12, 8)).pack(fill="x")

        results_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        results_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            results_frame,
            columns=("start", "end", "label", "probability"),
            show="headings",
            height=8,
        )
        self.tree.heading("start", text="Window Start (s)")
        self.tree.heading("end", text="Window End (s)")
        self.tree.heading("label", text="Predicted Label")
        self.tree.heading("probability", text="Jumping Probability")
        for column, width in [("start", 130), ("end", 130), ("label", 160), ("probability", 160)]:
            self.tree.column(column, width=width, anchor="center")
        self.tree.pack(fill="x", pady=(0, 12))

        self.figure, self.axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=results_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._clear_plot()

    def _clear_plot(self) -> None:
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

    def select_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select accelerometer CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.input_path = Path(selected)
            self.input_var.set(str(self.input_path))
            self.status_var.set("File selected. Click Classify to run the model.")

    def run_classification(self) -> None:
        if self.input_path is None:
            messagebox.showerror("No file selected", "Please choose a CSV file first.")
            return

        try:
            predictions = classify_file(self.input_path, self.model_bundle)
        except Exception as error:
            messagebox.showerror("Classification failed", str(error))
            return

        self.last_predictions = predictions
        self._populate_table(predictions)
        self._update_plot(predictions)
        self.status_var.set(f"Classification complete. Generated {len(predictions['predicted_labels'])} windows.")

    def _populate_table(self, predictions: dict[str, object]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for window_start, label, probability in zip(
            predictions["window_starts"],
            predictions["predicted_labels"],
            predictions["probabilities"],
            strict=False,
        ):
            self.tree.insert(
                "",
                "end",
                values=(
                    f"{window_start:.4f}",
                    f"{window_start + WINDOW_SECONDS:.4f}",
                    label,
                    f"{probability:.4f}",
                ),
            )

    def _update_plot(self, predictions: dict[str, object]) -> None:
        self._clear_plot()

        starts = predictions["window_starts"]
        probabilities = predictions["probabilities"]
        encoded_labels = np.array([1 if label == "jumping" else 0 for label in predictions["predicted_labels"]])

        self.axes[0].plot(starts, probabilities, marker="o", color="tab:red", linewidth=1.8)
        self.axes[0].axhline(0.5, linestyle="--", color="tab:gray", linewidth=1.0)
        self.axes[1].step(starts, encoded_labels, where="post", color="tab:blue", linewidth=1.8)
        self.canvas.draw()

    def save_output_csv(self) -> None:
        if self.last_predictions is None:
            messagebox.showerror("No predictions", "Run classification before saving outputs.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save prediction CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not output_path:
            return

        write_predictions_csv(Path(output_path), self.last_predictions)
        self.status_var.set(f"Saved prediction CSV to: {output_path}")

    def save_plot(self) -> None:
        if self.last_predictions is None:
            messagebox.showerror("No predictions", "Run classification before saving the plot.")
            return

        plot_path = filedialog.asksaveasfilename(
            title="Save prediction plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
        )
        if not plot_path:
            return

        save_prediction_plot(Path(plot_path), self.last_predictions)
        self.status_var.set(f"Saved prediction plot to: {plot_path}")


def run_cli(csv_path: Path) -> None:
    model_bundle = load_model_bundle()
    predictions = classify_file(csv_path, model_bundle)
    output_csv = csv_path.with_name(f"{csv_path.stem}_predictions.csv")
    output_plot = csv_path.with_name(f"{csv_path.stem}_predictions.png")

    write_predictions_csv(output_csv, predictions)
    save_prediction_plot(output_plot, predictions)

    print(f"Saved prediction CSV to: {output_csv}")
    print(f"Saved prediction plot to: {output_plot}")
    print(f"Generated {len(predictions['predicted_labels'])} predictions.")


def main() -> None:
    if len(sys.argv) > 1:
        run_cli(Path(sys.argv[1]))
        return

    root = tk.Tk()
    app = ActivityClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
