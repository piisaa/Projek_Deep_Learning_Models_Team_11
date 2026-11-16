import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, test_ds, class_names):
    """
    Mengevaluasi model menggunakan test dataset.
    """

    y_true = []
    y_pred = []

    # Mengambil label asli dan prediksi model
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred
    )

    # Menampilkan hasil
    print("===== HASIL EVALUASI MODEL =====")
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }
