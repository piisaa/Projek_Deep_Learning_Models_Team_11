
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold


# 1. PENGATURAN DASAR

# preprocessing.py berada di Flower-Classification/CNN/Utils/
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.15


# 2. MEMBACA NAMA KELAS, GAMBAR, DAN LABEL

def get_image_data():
    """Membaca path gambar dan label dari setiap folder kelas."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {DATA_DIR}"
        )

    class_names = sorted([
        folder.name
        for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    ])

    if not class_names:
        raise ValueError("Tidak ada folder kelas di dalam dataset.")

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    image_paths = []
    labels = []

    valid_extensions = {".jpg", ".jpeg", ".png"}

    for class_name in class_names:
        class_folder = DATA_DIR / class_name

        files = [
            file
            for file in class_folder.rglob("*")
            if file.is_file() and file.suffix.lower() in valid_extensions
        ]

        for file in files:
            image_paths.append(str(file))
            labels.append(class_to_index[class_name])

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    if len(image_paths) == 0:
        raise ValueError("Tidak ada gambar yang ditemukan!")

    print("Total gambar:", len(image_paths))
    print("\nJumlah gambar setiap kelas:")

    for index, name in enumerate(class_names):
        print(name, ":", int(np.sum(labels == index)))

    return image_paths, labels, class_names


# 3. LOAD IMAGE, RESIZE, DAN NORMALIZATION

def load_image(path):
    image = Image.open(path).convert("RGB")

    image = image.resize(
        (IMG_SIZE[1], IMG_SIZE[0])
    )

    image = transforms.ToTensor()(image)

    return image


# 4. DATA AUGMENTATION

def create_augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(contrast=0.1)
    ])

augmentation = create_augmentation()

# 5. MEMBUAT PIPELINE DATASET

class FlowerDataset(Dataset):

    def __init__(self, paths, labels, training=False):
        self.paths = paths
        self.labels = labels
        self.training = training

        self.resize_normalization = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image = Image.open(self.paths[index]).convert("RGB")

        image = self.resize_normalization(image)

        if self.training:
            image = augmentation(image)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return image, label


def create_dataset(paths, labels, training=False):

    dataset = FlowerDataset(
        paths,
        labels,
        training=training
    )

    if training:
        generator = torch.Generator().manual_seed(SEED)
    else:
        generator = None

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=training,
        generator=generator
    )

    return loader


# 6. MEMBAGI DATA: FINAL TEST + 5-FOLD CV

def prepare_datasets():

    image_paths, labels, class_names = get_image_data()

    # Sisihkan 15% data sebagai final test.
    # Sisa 85% digunakan untuk 5-fold cross-validation.

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels
    )

    print("\nJumlah data:")
    print("Data untuk 5-fold CV:", len(X_trainval))
    print("Final test:", len(X_test))

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_datasets = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_trainval, y_trainval),
        start=1
    ):

        print(f"\n========== FOLD {fold} ==========")

        train_paths = X_trainval[train_idx]
        train_labels = y_trainval[train_idx]

        val_paths = X_trainval[val_idx]
        val_labels = y_trainval[val_idx]

        train_ds = create_dataset(
            train_paths,
            train_labels,
            training=True
        )

        val_ds = create_dataset(
            val_paths,
            val_labels,
            training=False
        )

        fold_datasets.append({
            "fold": fold,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "train_paths": train_paths,
            "train_labels": train_labels,
            "val_paths": val_paths,
            "val_labels": val_labels
        })

        print("Jumlah training:", len(train_paths))
        print("Jumlah validation:", len(val_paths))

    # Final test tidak digunakan dalam proses cross-validation.

    test_ds = create_dataset(
        X_test,
        y_test,
        training=False
    )

    print("\nFinal test dataset berhasil dibuat!")
    print("Jumlah data testing:", len(X_test))

    return {
        "class_names": class_names,
        "fold_datasets": fold_datasets,
        "test_ds": test_ds,
        "X_test": X_test,
        "y_test": y_test
    }


# 7. CEK HASIL PREPROCESSING

def check_preprocessing(data):

    print("\nPreprocessing selesai!")

    for fold_data in data["fold_datasets"]:

        print(f"\n========== FOLD {fold_data['fold']} ==========")

        images, batch_labels = next(iter(fold_data["train_ds"]))

        print("Training images:", images.shape)
        print("Training labels:", batch_labels.shape)

        print(
            "Rentang pixel:",
            float(images.min()),
            "-",
            float(images.max())
        )

    test_images, test_labels = next(iter(data["test_ds"]))

    print("\nFinal test shape:", test_images.shape)
    print("Final test labels:", test_labels.shape)
    print("\nSEMUA PIPELINE BERHASIL DIBUAT!")


# 8. MENJALANKAN PREPROCESSING

if __name__ == "__main__":

    prepared_data = prepare_datasets()
    check_preprocessing(prepared_data)
