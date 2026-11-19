
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from torch.optim import Adam

from utils.preprocessing import prepare_datasets
from Model.CNN2 import DeepCNN


# ==============================
# TRAINING FUNCTION
# ==============================

def train_one_epoch(model, train_loader, optimizer, loss_function):

    model.train()
    train_loss = 0

    for batch, (src, trg) in enumerate(train_loader):

        pred = model(src)

        loss = loss_function(pred, trg)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(train_loader)


# ==============================
# VALIDATION FUNCTION
# ==============================

def validate(model, val_loader, loss_function):

    model.eval()
    val_loss = 0

    with torch.no_grad():

        for batch, (src, trg) in enumerate(val_loader):

            pred = model(src)

            loss = loss_function(pred, trg)
            val_loss += loss.item()

    return val_loss / len(val_loader)


# ==============================
# FINAL TEST FUNCTION
# ==============================

def evaluate_test(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for src, trg in test_loader:

            pred = model(src)
            predicted = torch.argmax(pred, dim=1)

            correct += (predicted == trg).sum().item()
            total += trg.size(0)

    accuracy = correct / total

    return accuracy


# ==============================
# MAIN TRAINING
# ==============================

def main():

    BATCH_SIZE = 32
    EPOCH = 10
    LEARNING_RATE = 0.001

    PROJECT_DIR = Path(__file__).resolve().parent
    MODEL_DIR = PROJECT_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    # Mengambil hasil preprocessing
    prepared_data = prepare_datasets()

    class_names = prepared_data["class_names"]
    fold_datasets = prepared_data["fold_datasets"]
    test_loader = prepared_data["test_ds"]

    loss_function = nn.CrossEntropyLoss()

    all_fold_results = []

    # ==============================
    # 5-FOLD CROSS VALIDATION
    # ==============================

    for fold_data in fold_datasets:

        fold = fold_data["fold"]

        train_loader = fold_data["train_ds"]
        val_loader = fold_data["val_ds"]

        print(f"\n========== FOLD {fold} ==========")

        model = DeepCNN(
            input_dim=224,
            input_c=3,
            output=len(class_names),
            hidden_dim=128,
            dropout=0.5,
            device="cpu"
        )

        optimizer = Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        loss_train_all = []
        loss_val_all = []

        best_val_loss = float("inf")

        # ==============================
        # EPOCH TRAINING
        # ==============================

        for epoch in range(EPOCH):

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_function
            )

            val_loss = validate(
                model,
                val_loader,
                loss_function
            )

            loss_train_all.append(train_loss)
            loss_val_all.append(val_loss)

            print(
                "epoch =", epoch + 1,
                "train loss =", train_loss,
                "validation loss =", val_loss
            )

            # Simpan model terbaik berdasarkan validation loss
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                    "class_names": class_names
                }, MODEL_DIR / f"DeepCNN_fold_{fold}.pt")

        # ==============================
        # SAVE LOSS GRAPH
        # ==============================

        plt.figure()

        plt.plot(
            range(1, EPOCH + 1),
            loss_train_all,
            label="Training"
        )

        plt.plot(
            range(1, EPOCH + 1),
            loss_val_all,
            label="Validation"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - Fold {fold}")
        plt.legend()

        plt.savefig(MODEL_DIR / f"training_fold_{fold}.png")
        plt.close()

        all_fold_results.append({
            "fold": fold,
            "best_val_loss": best_val_loss
        })

    # ==============================
    # SELECT BEST FOLD MODEL
    # ==============================

    best_fold = min(
        all_fold_results,
        key=lambda x: x["best_val_loss"]
    )

    print("\nBest fold:", best_fold["fold"])
    print("Best validation loss:", best_fold["best_val_loss"])

    # Load best model
    best_checkpoint = torch.load(
        MODEL_DIR / f"DeepCNN_fold_{best_fold['fold']}.pt",
        map_location="cpu"
    )

    final_model = DeepCNN(
        input_dim=224,
        input_c=3,
        output=len(class_names),
        hidden_dim=128,
        dropout=0.5,
        device="cpu"
    )

    final_model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    # ==============================
    # FINAL TEST
    # ==============================

    test_accuracy = evaluate_test(
        final_model,
        test_loader
    )

    print("\n========== FINAL TEST ==========")
    print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
