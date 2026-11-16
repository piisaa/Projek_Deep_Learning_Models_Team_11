import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


class FlowerDataset(Dataset):

    def __init__(self, folder="../dataset", img_size=(224, 224)):

        self.dataset = []
        self.class_names = sorted([
            folder.name
            for folder in Path(folder).iterdir()
            if folder.is_dir()
        ])

        # One-hot encoding untuk 5 kelas
        to_one_hot = np.eye(len(self.class_names))

        dataset_dir = Path(folder)

        for i, class_name in enumerate(self.class_names):

            class_folder = dataset_dir / class_name

            for image_path in class_folder.rglob("*"):

                if image_path.suffix.lower() not in [
                    ".jpg", ".jpeg", ".png"
                ]:
                    continue

                # Membaca gambar
                image = cv2.imread(str(image_path))

                if image is None:
                    continue

                # Mengubah BGR menjadi RGB
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                # Resize gambar
                image = cv2.resize(image, img_size)

                # Normalisasi pixel 0-255 menjadi 0-1
                image = image.astype(np.float32) / 255.0

                # One-hot label
                label = to_one_hot[i]

                self.dataset.append((image, label))

    def __getitem__(self, item):

        feature, label = self.dataset[item]

        # HWC menjadi CHW untuk PyTorch
        feature = np.transpose(feature, (2, 0, 1))

        return (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.dataset)
