from pathlib import Path
import json
import pickle
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    log_loss,
)
from sklearn.exceptions import ConvergenceWarning


BASE_DIR = Path(__file__).resolve().parent
HDF5_PATH = BASE_DIR / "data" / "hoppers_vs_walkers.h5"
OUTPUT_DIR = BASE_DIR / "figures" / "step6"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
METRICS_PATH = MODEL_DIR / "step6_metrics.json"

POSITIVE_LABEL = "jumping"
NEGATIVE_LABEL = "walking"
ITERATION_STEPS = [1, 2, 5, 10, 20, 40, 80, 120, 160, 200]


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def labels_to_binary(labels: list[str]) -> np.ndarray:
    return np.array([1 if label == POSITIVE_LABEL else 0 for label in labels], dtype=np.int64)


def load_feature_data() -> dict[str, np.ndarray]:
    with h5py.File(HDF5_PATH, "r") as hdf:
        feature_root = hdf["preprocessed"]["features"]
        train_group = feature_root["train"]
        test_group = feature_root["test"]

        data = {
            "x_train": train_group["normalized_features"][:],
            "x_test": test_group["normalized_features"][:],
            "y_train_labels": decode(train_group["labels"][:]),
            "y_test_labels": decode(test_group["labels"][:]),
            "feature_names": decode(feature_root["feature_names"][:]),
            "normalization_mean": feature_root["normalization_mean"][:],
            "normalization_std": feature_root["normalization_std"][:],
        }

    data["y_train"] = labels_to_binary(data["y_train_labels"])
    data["y_test"] = labels_to_binary(data["y_test_labels"])
    return data


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        random_state=292,
    )
    model.fit(x_train, y_train)
    return model


def build_training_curve(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, list[float]]:
    curve_model = LogisticRegression(
        max_iter=1,
        solver="saga",
        warm_start=True,
        random_state=292,
    )

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    previous_iterations = 0
    for total_iterations in ITERATION_STEPS:
        curve_model.max_iter = total_iterations - previous_iterations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            curve_model.fit(x_train, y_train)
        previous_iterations = total_iterations

        train_prob = curve_model.predict_proba(x_train)[:, 1]
        test_prob = curve_model.predict_proba(x_test)[:, 1]
        train_pred = (train_prob >= 0.5).astype(int)
        test_pred = (test_prob >= 0.5).astype(int)

        train_loss.append(float(log_loss(y_train, train_prob, labels=[0, 1])))
        test_loss.append(float(log_loss(y_test, test_prob, labels=[0, 1])))
        train_accuracy.append(float(accuracy_score(y_train, train_pred)))
        test_accuracy.append(float(accuracy_score(y_test, test_pred)))

    return {
        "iterations": ITERATION_STEPS,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


def plot_training_curves(curve_data: dict[str, list[float]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(curve_data["iterations"], curve_data["train_loss"], color="tab:blue", linewidth=1.8, label="train")
    axes[0].plot(curve_data["iterations"], curve_data["test_loss"], color="tab:orange", linewidth=1.8, label="test")
    axes[0].set_title("Logistic regression training curves")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(curve_data["iterations"], curve_data["train_accuracy"], color="tab:blue", linewidth=1.8, label="train")
    axes[1].plot(curve_data["iterations"], curve_data["test_accuracy"], color="tab:orange", linewidth=1.8, label="test")
    axes[1].set_xlabel("Training iterations")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    save_figure(fig, "training_curves.png")


def plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([NEGATIVE_LABEL, POSITIVE_LABEL])
    ax.set_yticklabels([NEGATIVE_LABEL, POSITIVE_LABEL])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test confusion matrix")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")

    save_figure(fig, "confusion_matrix.png")


def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray, roc_auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="tab:red", linewidth=2.0, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", linewidth=1.2, label="chance")
    ax.set_title("ROC curve on the test set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    save_figure(fig, "roc_curve.png")


def plot_top_coefficients(model: LogisticRegression, feature_names: list[str]) -> None:
    coefficients = model.coef_[0]
    sorted_indices = np.argsort(np.abs(coefficients))[-10:]
    sorted_indices = sorted_indices[np.argsort(np.abs(coefficients[sorted_indices]))]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:orange" if coefficients[index] > 0 else "tab:blue" for index in sorted_indices]
    ax.barh(np.arange(len(sorted_indices)), coefficients[sorted_indices], color=colors)
    ax.set_yticks(np.arange(len(sorted_indices)))
    ax.set_yticklabels([feature_names[index] for index in sorted_indices])
    ax.set_xlabel("Coefficient value")
    ax.set_title("Most influential logistic regression features")
    ax.grid(True, axis="x", alpha=0.3)

    save_figure(fig, "top_feature_coefficients.png")


def save_model_artifacts(model: LogisticRegression, metrics: dict[str, object], feature_names: list[str], normalization_mean: np.ndarray, normalization_std: np.ndarray) -> None:
    with MODEL_PATH.open("wb") as file:
        pickle.dump(
            {
                "coefficients": model.coef_[0],
                "intercept": float(model.intercept_[0]),
                "feature_names": feature_names,
                "normalization_mean": normalization_mean,
                "normalization_std": normalization_std,
                "positive_label": POSITIVE_LABEL,
                "negative_label": NEGATIVE_LABEL,
                "classification_threshold": 0.5,
            },
            file,
        )

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    ensure_output_dirs()
    data = load_feature_data()

    model = train_model(data["x_train"], data["y_train"])

    train_prob = model.predict_proba(data["x_train"])[:, 1]
    test_prob = model.predict_proba(data["x_test"])[:, 1]
    train_pred = (train_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)

    train_metrics = compute_metrics(data["y_train"], train_pred, train_prob)
    test_metrics = compute_metrics(data["y_test"], test_pred, test_prob)
    cm = confusion_matrix(data["y_test"], test_pred, labels=[0, 1])
    curve_data = build_training_curve(data["x_train"], data["y_train"], data["x_test"], data["y_test"])

    plot_training_curves(curve_data)
    plot_confusion_matrix(cm)
    plot_roc_curve(data["y_test"], test_prob, test_metrics["roc_auc"])
    plot_top_coefficients(model, data["feature_names"])

    metrics = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "model_parameters": {
            "model_type": "logistic_regression",
            "solver": "lbfgs",
            "max_iter": 200,
            "classification_threshold": 0.5,
            "positive_label": POSITIVE_LABEL,
        },
        "training_curve": curve_data,
    }
    save_model_artifacts(
        model,
        metrics,
        data["feature_names"],
        data["normalization_mean"],
        data["normalization_std"],
    )

    print(f"Saved trained model to: {MODEL_PATH}")
    print(f"Saved step 6 metrics to: {METRICS_PATH}")
    print(f"Saved step 6 figures to: {OUTPUT_DIR}")
    print(f"Training accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
