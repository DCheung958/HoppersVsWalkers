from collections import deque
from pathlib import Path
import csv
import pickle
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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

DEFAULT_BUFFERS = {
    "time": "acc_time",
    "x": "accX",
    "y": "accY",
    "z": "accZ",
    "abs": "acc",
}


def load_model_bundle() -> dict:
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


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


def extract_feature_vector(signal_window: np.ndarray) -> np.ndarray:
    channel_feature_vectors = []
    for channel_index in range(signal_window.shape[1]):
        channel_feature_vectors.append(compute_channel_features(signal_window[:, channel_index]))
    return np.concatenate(channel_feature_vectors)


def prepare_window(time_values: np.ndarray, signal_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    relative_time = time_values - time_values[0]
    reference_time = np.linspace(0.0, WINDOW_SECONDS, num=TARGET_SAMPLES_PER_WINDOW, endpoint=False)

    resampled_axes = []
    for axis_index in range(signal_values.shape[1]):
        resampled_axes.append(np.interp(reference_time, relative_time, signal_values[:, axis_index]))

    resampled_signal = np.column_stack(resampled_axes)
    preprocessed_signal = preprocess_signal_matrix(resampled_signal, MOVING_AVERAGE_WINDOW)
    return reference_time, preprocessed_signal


def classify_latest_window(time_values: np.ndarray, signal_values: np.ndarray, model_bundle: dict) -> dict[str, float | str] | None:
    if len(time_values) < 2:
        return None
    if float(time_values[-1] - time_values[0]) < WINDOW_SECONDS:
        return None

    end_time = float(time_values[-1])
    start_time = end_time - WINDOW_SECONDS
    mask = time_values >= start_time
    selected_time = time_values[mask]
    selected_signal = signal_values[mask]

    if len(selected_time) < 2:
        return None

    _, preprocessed_signal = prepare_window(selected_time, selected_signal)
    feature_vector = extract_feature_vector(preprocessed_signal)
    normalized = (feature_vector - model_bundle["normalization_mean"]) / model_bundle["normalization_std"]
    probability = float(sigmoid(normalized @ model_bundle["coefficients"] + model_bundle["intercept"]))
    label = (
        model_bundle["positive_label"]
        if probability >= model_bundle["classification_threshold"]
        else model_bundle["negative_label"]
    )

    return {
        "window_start": start_time,
        "window_end": end_time,
        "jumping_probability": probability,
        "predicted_label": label,
    }


class RealtimeSignalStore:
    def __init__(self, history_seconds: float = 20.0) -> None:
        self.history_seconds = history_seconds
        self.time = deque()
        self.x = deque()
        self.y = deque()
        self.z = deque()
        self.abs = deque()

    def append_many(self, time_values: list[float], x_values: list[float], y_values: list[float], z_values: list[float], abs_values: list[float]) -> None:
        for values in zip(time_values, x_values, y_values, z_values, abs_values, strict=False):
            self.time.append(values[0])
            self.x.append(values[1])
            self.y.append(values[2])
            self.z.append(values[3])
            self.abs.append(values[4])

        self.trim_history()

    def trim_history(self) -> None:
        if not self.time:
            return
        newest_time = self.time[-1]
        while self.time and newest_time - self.time[0] > self.history_seconds:
            self.time.popleft()
            self.x.popleft()
            self.y.popleft()
            self.z.popleft()
            self.abs.popleft()

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        time_values = np.array(self.time, dtype=np.float64)
        signal_values = np.column_stack(
            [
                np.array(self.x, dtype=np.float64),
                np.array(self.y, dtype=np.float64),
                np.array(self.z, dtype=np.float64),
                np.array(self.abs, dtype=np.float64),
            ]
        )
        return time_values, signal_values


class PhyphoxClient:
    def __init__(self, base_url: str, buffers: dict[str, str]) -> None:
        self.base_url = base_url.rstrip("/")
        self.buffers = buffers
        self.last_time_value: float | None = None

    def control(self, command: str) -> None:
        requests.get(f"{self.base_url}/control", params={"cmd": command}, timeout=5).raise_for_status()

    def poll(self) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        time_buffer = self.buffers["time"]
        x_buffer = self.buffers["x"]
        y_buffer = self.buffers["y"]
        z_buffer = self.buffers["z"]
        abs_buffer = self.buffers["abs"]

        if self.last_time_value is None:
            params = {
                time_buffer: "full",
                x_buffer: "full",
                y_buffer: "full",
                z_buffer: "full",
                abs_buffer: "full",
            }
        else:
            threshold = f"{self.last_time_value:.6f}"
            params = {
                time_buffer: threshold,
                x_buffer: f"{threshold}|{time_buffer}",
                y_buffer: f"{threshold}|{time_buffer}",
                z_buffer: f"{threshold}|{time_buffer}",
                abs_buffer: f"{threshold}|{time_buffer}",
            }

        response = requests.get(f"{self.base_url}/get", params=params, timeout=5)
        response.raise_for_status()
        payload = response.json()["buffer"]

        time_values = payload[time_buffer]["buffer"]
        x_values = payload[x_buffer]["buffer"]
        y_values = payload[y_buffer]["buffer"]
        z_values = payload[z_buffer]["buffer"]
        abs_values = payload[abs_buffer]["buffer"]

        if time_values:
            self.last_time_value = float(time_values[-1])

        return time_values, x_values, y_values, z_values, abs_values


class CsvReplayClient:
    def __init__(self, csv_path: Path, chunk_seconds: float = 0.5) -> None:
        df = pd.read_csv(csv_path)
        self.time = df[TIME_COLUMN].to_numpy(dtype=np.float64)
        self.x = df[AXIS_COLUMNS[0]].to_numpy(dtype=np.float64)
        self.y = df[AXIS_COLUMNS[1]].to_numpy(dtype=np.float64)
        self.z = df[AXIS_COLUMNS[2]].to_numpy(dtype=np.float64)
        self.abs = df[ABS_COLUMN].to_numpy(dtype=np.float64)
        self.chunk_seconds = chunk_seconds
        self.cursor = 0
        self.start_time = float(self.time[0])

    def control(self, command: str) -> None:
        if command == "clear":
            self.cursor = 0

    def poll(self) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        if self.cursor >= len(self.time):
            return [], [], [], [], []

        chunk_start_time = self.time[self.cursor]
        chunk_end_time = chunk_start_time + self.chunk_seconds
        next_cursor = self.cursor
        while next_cursor < len(self.time) and self.time[next_cursor] < chunk_end_time:
            next_cursor += 1

        time_values = (self.time[self.cursor:next_cursor] - self.start_time).tolist()
        x_values = self.x[self.cursor:next_cursor].tolist()
        y_values = self.y[self.cursor:next_cursor].tolist()
        z_values = self.z[self.cursor:next_cursor].tolist()
        abs_values = self.abs[self.cursor:next_cursor].tolist()
        self.cursor = next_cursor
        time.sleep(self.chunk_seconds)
        return time_values, x_values, y_values, z_values, abs_values


class BonusRealtimeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HoppersVsWalkers Bonus Realtime App")
        self.root.geometry("1100x820")

        self.model_bundle = load_model_bundle()
        self.signal_store = RealtimeSignalStore()
        self.prediction_history: list[dict[str, float | str]] = []
        self.client = None
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pending_error: str | None = None

        self.base_url_var = tk.StringVar(value="http://192.168.0.42:8080")
        self.time_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["time"])
        self.x_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["x"])
        self.y_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["y"])
        self.z_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["z"])
        self.abs_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["abs"])
        self.mode_var = tk.StringVar(value="phyphox")
        self.simulation_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Configure the data source and click Start.")
        self.current_label_var = tk.StringVar(value="Current action: waiting")
        self.current_probability_var = tk.StringVar(value="Jumping probability: --")

        self._build_ui()
        self._schedule_ui_refresh()

    def _build_ui(self) -> None:
        top_frame = ttk.Frame(self.root, padding=12)
        top_frame.pack(fill="x")

        source_frame = ttk.LabelFrame(top_frame, text="Data Source", padding=10)
        source_frame.pack(fill="x")

        ttk.Radiobutton(source_frame, text="Phyphox Remote Access", variable=self.mode_var, value="phyphox").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(source_frame, text="CSV Simulation", variable=self.mode_var, value="simulation").grid(row=0, column=1, sticky="w")

        ttk.Label(source_frame, text="Base URL").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_frame, textvariable=self.base_url_var, width=42).grid(row=1, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(source_frame, text="Simulation CSV").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_frame, textvariable=self.simulation_path_var, width=42).grid(row=2, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(source_frame, text="Browse", command=self._browse_simulation_file).grid(row=2, column=2, padx=(6, 0), pady=(8, 0))

        buffer_frame = ttk.LabelFrame(top_frame, text="Phyphox Buffer Names", padding=10)
        buffer_frame.pack(fill="x", pady=(10, 0))
        entries = [
            ("Time", self.time_buffer_var),
            ("X", self.x_buffer_var),
            ("Y", self.y_buffer_var),
            ("Z", self.z_buffer_var),
            ("Absolute", self.abs_buffer_var),
        ]
        for index, (label, variable) in enumerate(entries):
            ttk.Label(buffer_frame, text=label).grid(row=0, column=index * 2, sticky="w")
            ttk.Entry(buffer_frame, textvariable=variable, width=14).grid(row=0, column=index * 2 + 1, padx=(0, 10))

        button_frame = ttk.Frame(top_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        ttk.Button(button_frame, text="Start Stream", command=self.start_stream).pack(side="left")
        ttk.Button(button_frame, text="Stop Stream", command=self.stop_stream).pack(side="left", padx=(8, 0))
        ttk.Button(button_frame, text="Save Session CSV", command=self.save_session_csv).pack(side="left", padx=(8, 0))

        ttk.Label(self.root, textvariable=self.status_var, padding=(12, 6, 12, 0)).pack(fill="x")
        ttk.Label(self.root, textvariable=self.current_label_var, padding=(12, 6, 12, 0), font=("Segoe UI", 12, "bold")).pack(fill="x")
        ttk.Label(self.root, textvariable=self.current_probability_var, padding=(12, 2, 12, 8)).pack(fill="x")

        middle_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        middle_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            middle_frame,
            columns=("start", "end", "label", "probability"),
            show="headings",
            height=8,
        )
        for name, heading, width in [
            ("start", "Window Start (s)", 130),
            ("end", "Window End (s)", 130),
            ("label", "Predicted Label", 160),
            ("probability", "Jumping Probability", 160),
        ]:
            self.tree.heading(name, text=heading)
            self.tree.column(name, width=width, anchor="center")
        self.tree.pack(fill="x", pady=(0, 12))

        self.figure, self.axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=False)
        self.canvas = FigureCanvasTkAgg(self.figure, master=middle_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._clear_plot()

    def _browse_simulation_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select simulation CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.simulation_path_var.set(selected)

    def _clear_plot(self) -> None:
        for ax in self.axes:
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.axes[0].set_title("Realtime absolute acceleration")
        self.axes[0].set_xlabel("Time (s)")
        self.axes[0].set_ylabel("Acceleration (m/s^2)")
        self.axes[1].set_title("Realtime classifier output")
        self.axes[1].set_xlabel("Window end time (s)")
        self.axes[1].set_ylabel("Jumping probability")
        self.axes[1].axhline(0.5, linestyle="--", color="tab:gray", linewidth=1.0)
        self.canvas.draw()

    def _build_client(self):
        if self.mode_var.get() == "simulation":
            simulation_path = Path(self.simulation_path_var.get())
            if not simulation_path.exists():
                raise FileNotFoundError("Simulation CSV file not found.")
            return CsvReplayClient(simulation_path)

        buffers = {
            "time": self.time_buffer_var.get().strip(),
            "x": self.x_buffer_var.get().strip(),
            "y": self.y_buffer_var.get().strip(),
            "z": self.z_buffer_var.get().strip(),
            "abs": self.abs_buffer_var.get().strip(),
        }
        return PhyphoxClient(self.base_url_var.get().strip(), buffers)

    def start_stream(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Already running", "The realtime stream is already running.")
            return

        try:
            self.client = self._build_client()
            self.client.control("clear")
            if self.mode_var.get() == "phyphox":
                self.client.control("start")
        except Exception as error:
            messagebox.showerror("Could not start stream", str(error))
            return

        self.signal_store = RealtimeSignalStore()
        self.prediction_history = []
        self.stop_event.clear()
        self.pending_error = None
        self.status_var.set("Realtime stream started.")
        self.current_label_var.set("Current action: collecting data")
        self.current_probability_var.set("Jumping probability: --")

        self.worker_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.worker_thread.start()

    def stop_stream(self) -> None:
        self.stop_event.set()
        if self.client is not None and self.mode_var.get() == "phyphox":
            try:
                self.client.control("stop")
            except Exception:
                pass
        self.status_var.set("Realtime stream stopped.")

    def _stream_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                time_values, x_values, y_values, z_values, abs_values = self.client.poll()
                if not time_values:
                    continue

                if not abs_values:
                    abs_array = np.sqrt(np.array(x_values) ** 2 + np.array(y_values) ** 2 + np.array(z_values) ** 2)
                    abs_values = abs_array.tolist()

                self.signal_store.append_many(time_values, x_values, y_values, z_values, abs_values)
                signal_time, signal_matrix = self.signal_store.to_arrays()
                prediction = classify_latest_window(signal_time, signal_matrix, self.model_bundle)

                if prediction is not None:
                    if not self.prediction_history or prediction["window_end"] > self.prediction_history[-1]["window_end"]:
                        self.prediction_history.append(prediction)
            except Exception as error:
                self.pending_error = str(error)
                self.stop_event.set()

    def _schedule_ui_refresh(self) -> None:
        self._refresh_ui()
        self.root.after(500, self._schedule_ui_refresh)

    def _refresh_ui(self) -> None:
        if self.pending_error:
            messagebox.showerror("Realtime stream error", self.pending_error)
            self.pending_error = None
            self.status_var.set("Realtime stream stopped due to an error.")

        self._refresh_prediction_table()
        self._refresh_plot()

        if self.prediction_history:
            latest = self.prediction_history[-1]
            self.current_label_var.set(f"Current action: {latest['predicted_label']}")
            self.current_probability_var.set(f"Jumping probability: {latest['jumping_probability']:.4f}")

    def _refresh_prediction_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for prediction in self.prediction_history[-12:]:
            self.tree.insert(
                "",
                "end",
                values=(
                    f"{prediction['window_start']:.2f}",
                    f"{prediction['window_end']:.2f}",
                    prediction["predicted_label"],
                    f"{prediction['jumping_probability']:.4f}",
                ),
            )

    def _refresh_plot(self) -> None:
        self._clear_plot()

        time_values = np.array(self.signal_store.time, dtype=np.float64)
        abs_values = np.array(self.signal_store.abs, dtype=np.float64)
        if len(time_values) > 0:
            self.axes[0].plot(time_values, abs_values, color="tab:red", linewidth=1.6)

        if self.prediction_history:
            prediction_times = [prediction["window_end"] for prediction in self.prediction_history]
            probabilities = [prediction["jumping_probability"] for prediction in self.prediction_history]
            self.axes[1].plot(prediction_times, probabilities, marker="o", color="tab:blue", linewidth=1.6)

        self.canvas.draw()

    def save_session_csv(self) -> None:
        if not self.prediction_history:
            messagebox.showerror("No predictions", "No realtime predictions are available yet.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save realtime session CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not output_path:
            return

        with Path(output_path).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["window_start_s", "window_end_s", "predicted_label", "jumping_probability"])
            for prediction in self.prediction_history:
                writer.writerow(
                    [
                        f"{prediction['window_start']:.4f}",
                        f"{prediction['window_end']:.4f}",
                        prediction["predicted_label"],
                        f"{prediction['jumping_probability']:.6f}",
                    ]
                )

        self.status_var.set(f"Saved realtime session CSV to: {output_path}")


def main() -> None:
    root = tk.Tk()
    app = BonusRealtimeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
