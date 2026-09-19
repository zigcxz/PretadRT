import torch
from torch import nn


class GradientAE(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

        self.to(device)


    def forward(self, x):

        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat
