import torch
import torch.nn as nn


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(PrototypicalNetwork, self).__init__()

        # Define the layers of the encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Forward pass through the encoder
        encoded_features = self.encoder(x)
        return encoded_features