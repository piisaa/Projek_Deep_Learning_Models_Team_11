import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=200, input_c=3, output=5, hidden_dim=128, dropout=0.5, device='cpu'):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kkernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.fc = nn.Linear(in_features=100 * 10 * 10, out_features=output)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)

        x = self.fc(torch.flatten(x, 1))
        x = self.Softmax(x)

        return x