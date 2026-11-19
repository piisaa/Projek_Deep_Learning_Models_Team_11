
import torch
import numpy as np

from pathlib import Path
from Model.CNN2 import DeepCNN
from utils.preprocessing import prepare_datasets


def main():

    # Pengaturan dasar
    PROJECT_DIR = Path(__file__).resolve().parent
    MODEL_DIR = PROJECT_DIR / "models"

    # Mengambil hasil preprocessing
    prepared_data = prepare_datasets()

    class_names = prepared_data["class_names"]
    test_loader = prepared_data["test_ds"]

    # Memuat checkpoint model terbaik
    checkpoint = torch.load(
        MODEL_DIR / "DeepCNN_fold_2.pt",
        map_location="cpu"
    )

    # Membuat arsitektur model
    model = DeepCNN(
        input_dim=224,
        input_c=3,
        output=len(class_names),
        hidden_dim=128,
        dropout=0.5,
        device="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Mode evaluasi
    model.eval()

    y_true = []
    y_pred = []

    correct = 0
    total = 0

    # Proses testing
    with torch.no_grad():

        for batch, (src, trg) in enumerate(test_loader):

            pred = model(src)

            predicted = torch.argmax(pred, dim=1)

            correct += (predicted == trg).sum().item()
            total += trg.size(0)

            y_true.extend(trg.numpy())
            y_pred.extend(predicted.numpy())

    # Menghitung accuracy
    accuracy = correct / total

    print("\n========== TESTING RESULT ==========")
    print("Total data:", total)
    print("Correct prediction:", correct)
    print("Incorrect prediction:", total - correct)
    print("Test Accuracy:", accuracy)

    # Menyimpan hasil prediksi
    np.savez(
        MODEL_DIR / "testing_results.npz",
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        class_names=np.array(class_names)
    )

    # Menyimpan hasil prediksi
    np.savez(
        MODEL_DIR / "testing_results.npz",
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        class_names=np.array(class_names)
    )

    print("\nHasil prediksi berhasil disimpan!")


if __name__ == "__main__":
    main()

