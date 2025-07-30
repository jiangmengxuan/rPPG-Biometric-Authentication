
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class BVPGenerator(nn.Module):
    def __init__(self, in_channels=3, signal_length=600):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        self.output_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, signal_length)),
            nn.Tanh()
        )

    def forward(self, x):
        feat = self.encoder(x)
        upsampled = self.decoder(feat)
        out = self.output_conv(upsampled)
        return out.squeeze(1).squeeze(1)
