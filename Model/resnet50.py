import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=2, pretrained=True):
    # Menggunakan ResNet50 pretrained dari torchvision[cite: 9]
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Mengganti layer fully connected terakhir sesuai jumlah kelas[cite: 9]
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model