import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


# 1. Grafik Accuracy dan Loss selama training
def plot_training_history(history, model_name="Model", save_path=None):

    history_data = history.history if hasattr(history, "history") else history

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Grafik accuracy
    axes[0].plot(history_data["accuracy"], label="Training Accuracy")

    if "val_accuracy" in history_data:
        axes[0].plot(history_data["val_accuracy"], label="Validation Accuracy")

    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Grafik loss
    axes[1].plot(history_data["loss"], label="Training Loss")

    if "val_loss" in history_data:
        axes[1].plot(history_data["val_loss"], label="Validation Loss")

    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# 2. Menampilkan Confusion Matrix
def plot_confusion_matrix(results, class_names, model_name="Model", save_path=None):

    cm = results["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# 3. Membandingkan performa beberapa model
def plot_model_comparison(results_dict, save_path=None):

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    model_names = list(results_dict.keys())

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):

        values = [
            results_dict[model_name][metric]
            for metric in metrics
        ]

        ax.bar(
            x + i * width,
            values,
            width,
            label=model_name
        )

    ax.set_title("Perbandingan Performa Model")
    ax.set_xlabel("Evaluation Metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
