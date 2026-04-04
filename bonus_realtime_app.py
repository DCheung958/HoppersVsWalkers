# Bonus — realtime-ish classifier hooked up to Phyphox (or a CSV replay)

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


def fetch_phyphox_config(base_url):
    base = base_url.rstrip("/")
    r = requests.get(base + "/config", timeout=5)
    r.raise_for_status()
    return r.json()


def load_model_bundle():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def sigmoid(values):
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


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


def extract_feature_vector(signal_window):
    parts = []
    for c in range(signal_window.shape[1]):
        parts.append(compute_channel_features(signal_window[:, c]))
    return np.concatenate(parts)


def prepare_window(time_values, signal_values):
    rel = time_values - time_values[0]
    ref_t = np.linspace(0.0, WINDOW_SECONDS, num=TARGET_SAMPLES_PER_WINDOW, endpoint=False)

    cols = []
    for j in range(signal_values.shape[1]):
        cols.append(np.interp(ref_t, rel, signal_values[:, j]))

    stacked = np.column_stack(cols)
    smoothed = preprocess_signal_matrix(stacked, MOVING_AVERAGE_WINDOW)
    return ref_t, smoothed


def classify_latest_window(time_values, signal_values, model_bundle):
    if len(time_values) < 2:
        return None
    if float(time_values[-1] - time_values[0]) < WINDOW_SECONDS:
        return None

    end_t = float(time_values[-1])
    start_t = end_t - WINDOW_SECONDS
    m = time_values >= start_t
    sel_t = time_values[m]
    sel_s = signal_values[m]

    if len(sel_t) < 2:
        return None

    _, proc = prepare_window(sel_t, sel_s)
    feat = extract_feature_vector(proc)
    mu = model_bundle["normalization_mean"]
    sd = model_bundle["normalization_std"]
    normed = (feat - mu) / sd
    w = model_bundle["coefficients"]
    b = model_bundle["intercept"]
    p = float(sigmoid(normed @ w + b))
    if p >= model_bundle["classification_threshold"]:
        lab = model_bundle["positive_label"]
    else:
        lab = model_bundle["negative_label"]

    return {
        "window_start": start_t,
        "window_end": end_t,
        "jumping_probability": p,
        "predicted_label": lab,
    }


class RealtimeSignalStore:
    def __init__(self, history_seconds=20.0):
        self.history_seconds = history_seconds
        self.time = deque()
        self.x = deque()
        self.y = deque()
        self.z = deque()
        self.abs = deque()

    def append_many(self, time_values, x_values, y_values, z_values, abs_values):
        for tup in zip(time_values, x_values, y_values, z_values, abs_values):
            self.time.append(tup[0])
            self.x.append(tup[1])
            self.y.append(tup[2])
            self.z.append(tup[3])
            self.abs.append(tup[4])

        self.trim_history()

    def trim_history(self):
        if not self.time:
            return
        newest = self.time[-1]
        while self.time and newest - self.time[0] > self.history_seconds:
            self.time.popleft()
            self.x.popleft()
            self.y.popleft()
            self.z.popleft()
            self.abs.popleft()

    def to_arrays(self):
        t = np.array(self.time, dtype=np.float64)
        sig = np.column_stack(
            [
                np.array(self.x, dtype=np.float64),
                np.array(self.y, dtype=np.float64),
                np.array(self.z, dtype=np.float64),
                np.array(self.abs, dtype=np.float64),
            ]
        )
        return t, sig


class PhyphoxClient:
    def __init__(self, base_url, buffers):
        self.base_url = base_url.rstrip("/")
        self.buffers = buffers
        self.last_time_value = None

    def control(self, command):
        requests.get(self.base_url + "/control", params={"cmd": command}, timeout=5).raise_for_status()

    def poll(self):
        tb = self.buffers["time"]
        xb = self.buffers["x"]
        yb = self.buffers["y"]
        zb = self.buffers["z"]
        ab = self.buffers["abs"]

        if self.last_time_value is None:
            params = {
                tb: "full",
                xb: "full",
                yb: "full",
                zb: "full",
                ab: "full",
            }
        else:
            thr = "%.6f" % self.last_time_value
            params = {
                tb: thr,
                xb: thr + "|" + tb,
                yb: thr + "|" + tb,
                zb: thr + "|" + tb,
                ab: thr + "|" + tb,
            }

        r = requests.get(self.base_url + "/get", params=params, timeout=5)
        r.raise_for_status()
        payload = r.json()["buffer"]

        try:
            time_values = payload[tb]["buffer"]
            x_values = payload[xb]["buffer"]
            y_values = payload[yb]["buffer"]
            z_values = payload[zb]["buffer"]
            abs_values = payload[ab]["buffer"]
        except KeyError as e:
            missing = e.args[0]
            raise ValueError("Unknown Phyphox buffer %r." % missing) from e

        if time_values:
            self.last_time_value = float(time_values[-1])

        return time_values, x_values, y_values, z_values, abs_values


class CsvReplayClient:
    def __init__(self, csv_path, chunk_seconds=0.5):
        df = pd.read_csv(csv_path)
        self.time = df[TIME_COLUMN].to_numpy(dtype=np.float64)
        self.x = df[AXIS_COLUMNS[0]].to_numpy(dtype=np.float64)
        self.y = df[AXIS_COLUMNS[1]].to_numpy(dtype=np.float64)
        self.z = df[AXIS_COLUMNS[2]].to_numpy(dtype=np.float64)
        self.abs = df[ABS_COLUMN].to_numpy(dtype=np.float64)
        self.chunk_seconds = chunk_seconds
        self.cursor = 0
        self.start_time = float(self.time[0])

    def control(self, command):
        if command == "clear":
            self.cursor = 0

    def poll(self):
        if self.cursor >= len(self.time):
            return [], [], [], [], []

        chunk_start = self.time[self.cursor]
        chunk_end = chunk_start + self.chunk_seconds
        nxt = self.cursor
        while nxt < len(self.time) and self.time[nxt] < chunk_end:
            nxt += 1

        time_values = (self.time[self.cursor:nxt] - self.start_time).tolist()
        x_values = self.x[self.cursor:nxt].tolist()
        y_values = self.y[self.cursor:nxt].tolist()
        z_values = self.z[self.cursor:nxt].tolist()
        abs_values = self.abs[self.cursor:nxt].tolist()
        self.cursor = nxt
        time.sleep(self.chunk_seconds)
        return time_values, x_values, y_values, z_values, abs_values


class BonusRealtimeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HoppersVsWalkers Bonus Realtime (Phyphox / CSV replay)")
        self.root.geometry("1100x820")

        self.model_bundle = load_model_bundle()
        self.signal_store = RealtimeSignalStore()
        self.prediction_history = []
        self.client = None
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.pending_error = None

        self.base_url_var = tk.StringVar(value="http://192.168.1.98:8080")
        self.time_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["time"])
        self.x_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["x"])
        self.y_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["y"])
        self.z_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["z"])
        self.abs_buffer_var = tk.StringVar(value=DEFAULT_BUFFERS["abs"])
        self.mode_var = tk.StringVar(value="phyphox")
        self.simulation_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready.")
        self.current_label_var = tk.StringVar(value="Current action: waiting")
        self.current_probability_var = tk.StringVar(value="Jumping probability: --")

        self._build_ui()
        self._schedule_ui_refresh()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        src = ttk.LabelFrame(top, text="Data Source", padding=10)
        src.pack(fill="x")

        ttk.Radiobutton(src, text="Phyphox Remote Access", variable=self.mode_var, value="phyphox").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(src, text="CSV Simulation", variable=self.mode_var, value="simulation").grid(row=0, column=1, sticky="w")

        ttk.Label(src, text="Base URL").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        url_col = ttk.Frame(src)
        url_col.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Entry(url_col, textvariable=self.base_url_var, width=42).pack(fill="x")

        ttk.Label(src, text="Simulation CSV").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(src, textvariable=self.simulation_path_var, width=42).grid(row=3, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(src, text="Browse", command=self._browse_simulation_file).grid(row=3, column=2, padx=(6, 0), pady=(8, 0))
        src.columnconfigure(1, weight=1)

        buf = ttk.LabelFrame(top, text="Phyphox Buffer Names", padding=10)
        buf.pack(fill="x", pady=(10, 0))
        rows = [
            ("Time", self.time_buffer_var),
            ("X", self.x_buffer_var),
            ("Y", self.y_buffer_var),
            ("Z", self.z_buffer_var),
            ("Absolute", self.abs_buffer_var),
        ]
        for i, (lbl, var) in enumerate(rows):

            ttk.Label(buf, text=lbl).grid(row=0, column=i * 2, sticky="w")

            ttk.Entry(buf, textvariable=var, width=14).grid(row=0, column=i * 2 + 1, padx=(0, 10))

        btns = ttk.Frame(top)

        btns.pack(fill="x", pady=(10, 0))

        ttk.Button(btns, text="Test Phyphox connection", command=self._test_phyphox_connection).pack(side="left")
        ttk.Button(btns, text="Start Stream", command=self.start_stream).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Stop Stream", command=self.stop_stream).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Save Session CSV", command=self.save_session_csv).pack(side="left", padx=(8, 0))

        ttk.Label(self.root, textvariable=self.status_var, padding=(12, 6, 12, 0)).pack(fill="x")
        ttk.Label(self.root, textvariable=self.current_label_var, padding=(12, 6, 12, 0), font=("Segoe UI", 12, "bold")).pack(fill="x")
        ttk.Label(self.root, textvariable=self.current_probability_var, padding=(12, 2, 12, 8)).pack(fill="x")

        mid = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        mid.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            mid,
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
        self.canvas = FigureCanvasTkAgg(self.figure, master=mid)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._clear_plot()

    def _browse_simulation_file(self):
        selected = filedialog.askopenfilename(
            title="Select simulation CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.simulation_path_var.set(selected)

    def _test_phyphox_connection(self):
        base = self.base_url_var.get().strip().rstrip("/")
        if not base:
            messagebox.showerror("Empty URL", "Base URL is required.")
            return
        try:
            cfg = fetch_phyphox_config(base)
        except requests.RequestException as e:
            messagebox.showerror("Cannot reach Phyphox", "URL: %s\n%s" % (base, e))
            return

        title = cfg.get("localTitle") or cfg.get("title") or "(unknown experiment)"
        lines = ["Experiment: " + title, "", "Buffers:", ""]

        for entry in cfg.get("buffers") or []:
            if isinstance(entry, dict):
                name = entry.get("name", "?")
                size = entry.get("size", "")
                if size != "":
                    lines.append("  " + str(name) + "  (size " + str(size) + ")")
                else:
                    lines.append("  " + str(name))
            else:
                lines.append("  " + str(entry))

        export_sets = cfg.get("export") or []
        if export_sets:
            lines.extend(["", "Export buffer sets:", ""])
            for item in export_sets:
                if isinstance(item, dict):
                    for k, v in item.items():
                        lines.append("  " + str(k) + ": " + str(v))
                else:
                    lines.append("  " + str(item))

        dialog = tk.Toplevel(self.root)
        dialog.title("Phyphox connection OK")
        dialog.geometry("520x400")
        txt = tk.Text(dialog, wrap="word", font=("Consolas", 10))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 8))
        self.status_var.set("Phyphox OK: " + title)

    def _clear_plot(self):
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
            p = Path(self.simulation_path_var.get())
            if not p.exists():
                raise FileNotFoundError("Simulation CSV file not found.")
            return CsvReplayClient(p)

        buffers = {
            "time": self.time_buffer_var.get().strip(),
            "x": self.x_buffer_var.get().strip(),
            "y": self.y_buffer_var.get().strip(),
            "z": self.z_buffer_var.get().strip(),
            "abs": self.abs_buffer_var.get().strip(),
        }
        return PhyphoxClient(self.base_url_var.get().strip(), buffers)

    def start_stream(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Already running", "The realtime stream is already running.")
            return

        if self.mode_var.get() == "phyphox" and not self.base_url_var.get().strip():
            messagebox.showerror("Base URL missing", "Base URL is required.")
            return

        try:
            self.client = self._build_client()
            self.client.control("clear")
            if self.mode_var.get() == "phyphox":
                self.client.control("start")
        except Exception as e:
            messagebox.showerror("Could not start stream", str(e))
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

    def stop_stream(self):
        self.stop_event.set()
        if self.client is not None and self.mode_var.get() == "phyphox":
            try:
                self.client.control("stop")
            except Exception:
                pass
        self.status_var.set("Realtime stream stopped.")

    def _stream_loop(self):
        while not self.stop_event.is_set():
            try:
                tvals, xv, yv, zv, av = self.client.poll()
                if not tvals:
                    continue

                if not av:
                    xa = np.sqrt(np.array(xv) ** 2 + np.array(yv) ** 2 + np.array(zv) ** 2)
                    av = xa.tolist()

                self.signal_store.append_many(tvals, xv, yv, zv, av)
                sig_t, sig_m = self.signal_store.to_arrays()
                pred = classify_latest_window(sig_t, sig_m, self.model_bundle)

                if pred is not None:
                    if len(self.prediction_history) == 0:
                        self.prediction_history.append(pred)
                    elif pred["window_end"] > self.prediction_history[-1]["window_end"]:
                        self.prediction_history.append(pred)
            except Exception as e:
                self.pending_error = str(e)
                self.stop_event.set()

    def _schedule_ui_refresh(self):
        self._refresh_ui()
        self.root.after(500, self._schedule_ui_refresh)

    def _refresh_ui(self):
        if self.pending_error:
            messagebox.showerror("Realtime stream error", self.pending_error)
            self.pending_error = None
            self.status_var.set("Realtime stream stopped due to an error.")

        self._refresh_prediction_table()
        self._refresh_plot()

        if self.prediction_history:
            latest = self.prediction_history[-1]
            self.current_label_var.set("Current action: " + str(latest["predicted_label"]))
            self.current_probability_var.set("Jumping probability: %.4f" % latest["jumping_probability"])

    def _refresh_prediction_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        tail = self.prediction_history[-12:]
        for pred in tail:
            self.tree.insert(
                "",
                "end",
                values=(
                    "%.2f" % pred["window_start"],
                    "%.2f" % pred["window_end"],
                    pred["predicted_label"],
                    "%.4f" % pred["jumping_probability"],
                ),
            )

    def _refresh_plot(self):
        self._clear_plot()

        tvals = np.array(self.signal_store.time, dtype=np.float64)
        avals = np.array(self.signal_store.abs, dtype=np.float64)
        if len(tvals) > 0:
            self.axes[0].plot(tvals, avals, color="tab:red", linewidth=1.6)

        if self.prediction_history:
            ends = [p["window_end"] for p in self.prediction_history]
            ps = [p["jumping_probability"] for p in self.prediction_history]
            self.axes[1].plot(ends, ps, marker="o", color="tab:blue", linewidth=1.6)

        self.canvas.draw()

    def save_session_csv(self):
        if not self.prediction_history:
            messagebox.showerror("No predictions", "No realtime predictions are available yet.")
            return

        out = filedialog.asksaveasfilename(
            title="Save realtime session CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not out:
            return

        with Path(out).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["window_start_s", "window_end_s", "predicted_label", "jumping_probability"])
            for pred in self.prediction_history:
                w.writerow(
                    [
                        "%.4f" % pred["window_start"],
                        "%.4f" % pred["window_end"],
                        pred["predicted_label"],
                        "%.6f" % pred["jumping_probability"],
                    ]
                )

        self.status_var.set("Saved realtime session CSV to: " + out)


def main():
    root = tk.Tk()
    BonusRealtimeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
