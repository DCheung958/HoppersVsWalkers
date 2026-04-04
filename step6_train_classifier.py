# Step 6 — train classifier logistic regression

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


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def labels_to_binary(labels):
    y = []
    for lab in labels:
        if lab == POSITIVE_LABEL:
            y.append(1)
        else:
            y.append(0)
    return np.array(y, dtype=np.int64)


def load_feature_data():
    with h5py.File(HDF5_PATH, "r") as hdf:
        root = hdf["preprocessed"]["features"]
        tr = root["train"]
        te = root["test"]

        data = {
            "x_train": tr["normalized_features"][:],
            "x_test": te["normalized_features"][:],
            "y_train_labels": decode(tr["labels"][:]),
            "y_test_labels": decode(te["labels"][:]),
            "feature_names": decode(root["feature_names"][:]),
            "normalization_mean": root["normalization_mean"][:],
            "normalization_std": root["normalization_std"][:],
        }

    data["y_train"] = labels_to_binary(data["y_train_labels"])
    data["y_test"] = labels_to_binary(data["y_test_labels"])
    return data


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def train_model(x_train, y_train):
    m = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        random_state=292,
    )
    m.fit(x_train, y_train)
    return m


def build_training_curve(x_train, y_train, x_test, y_test):
    curve_model = LogisticRegression(
        max_iter=1,
        solver="saga",
        warm_start=True,
        random_state=292,
    )

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    prev = 0
    for total in ITERATION_STEPS:
        curve_model.max_iter = total - prev
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            curve_model.fit(x_train, y_train)
        prev = total

        p_tr = curve_model.predict_proba(x_train)[:, 1]
        p_te = curve_model.predict_proba(x_test)[:, 1]
        pred_tr = (p_tr >= 0.5).astype(int)
        pred_te = (p_te >= 0.5).astype(int)

        train_loss.append(float(log_loss(y_train, p_tr, labels=[0, 1])))
        test_loss.append(float(log_loss(y_test, p_te, labels=[0, 1])))
        train_acc.append(float(accuracy_score(y_train, pred_tr)))
        test_acc.append(float(accuracy_score(y_test, pred_te)))

    return {
        "iterations": ITERATION_STEPS,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }


def plot_training_curves(curve_data):
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


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([NEGATIVE_LABEL, POSITIVE_LABEL])
    ax.set_yticklabels([NEGATIVE_LABEL, POSITIVE_LABEL])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test confusion matrix")

    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center", color="black")

    save_figure(fig, "confusion_matrix.png")


def plot_roc_curve(y_test, y_prob, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="tab:red", linewidth=2.0, label="AUC = %.3f" % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", linewidth=1.2, label="chance")
    ax.set_title("ROC curve on the test set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    save_figure(fig, "roc_curve.png")


def plot_top_coefficients(model, feature_names):
    coef = model.coef_[0]
    idx = np.argsort(np.abs(coef))[-10:]
    idx = idx[np.argsort(np.abs(coef[idx]))]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = []
    for i in idx:
        if coef[i] > 0:
            colors.append("tab:orange")
        else:
            colors.append("tab:blue")
    ax.barh(np.arange(len(idx)), coef[idx], color=colors)
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Coefficient value")
    ax.set_title("Biggest logistic regression weights")
    ax.grid(True, axis="x", alpha=0.3)

    save_figure(fig, "top_feature_coefficients.png")


def save_model_artifacts(model, metrics, feature_names, normalization_mean, normalization_std):
    bundle = {
        "coefficients": model.coef_[0],
        "intercept": float(model.intercept_[0]),
        "feature_names": feature_names,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "classification_threshold": 0.5,
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump(bundle, f)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    ensure_output_dirs()
    data = load_feature_data()

    model = train_model(data["x_train"], data["y_train"])

    p_tr = model.predict_proba(data["x_train"])[:, 1]
    p_te = model.predict_proba(data["x_test"])[:, 1]
    pred_tr = (p_tr >= 0.5).astype(int)
    pred_te = (p_te >= 0.5).astype(int)

    train_metrics = compute_metrics(data["y_train"], pred_tr, p_tr)
    test_metrics = compute_metrics(data["y_test"], pred_te, p_te)
    cm = confusion_matrix(data["y_test"], pred_te, labels=[0, 1])
    curve_data = build_training_curve(data["x_train"], data["y_train"], data["x_test"], data["y_test"])

    plot_training_curves(curve_data)
    plot_confusion_matrix(cm)
    plot_roc_curve(data["y_test"], p_te, test_metrics["roc_auc"])
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

    print("Saved trained model to:", MODEL_PATH)
    print("Saved step 6 metrics to:", METRICS_PATH)
    print("Saved step 6 figures to:", OUTPUT_DIR)
    print("Training accuracy: %.4f" % train_metrics["accuracy"])
    print("Test accuracy: %.4f" % test_metrics["accuracy"])
    print("Test ROC AUC: %.4f" % test_metrics["roc_auc"])


if __name__ == "__main__":
    main()
