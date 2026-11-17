
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_true, y_pred, class_names):

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    # Menampilkan hasil
    print("\n===== HASIL EVALUASI MODEL =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }


import numpy as np


def main():

    # Membaca hasil testing
    data = np.load("models/testing_results.npz")

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    class_names = data["class_names"].tolist()

    # Menjalankan evaluasi
    results = evaluate_model(
        y_true,
        y_pred,
        class_names
    )


if __name__ == "__main__":
    main()
