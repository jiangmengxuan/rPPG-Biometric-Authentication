
import torch
import torch.nn as nn

class GphyNet(nn.Module):
    def __init__(self, signal_length=600):
        super().__init__()
        self.fc = nn.Linear(signal_length, 64 * 9 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, s_gt):
        B, T = s_gt.shape
        x = self.fc(s_gt)
        x = x.view(B, 64, 9, 5)
        x = self.deconv(x)
        return x


class GnoiseNet(nn.Module):
    def __init__(self, noise_dim=64, signal_length=600):
        super().__init__()
        self.fc = nn.Linear(noise_dim, 64 * 9 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, z):
        B = z.shape[0]
        x = self.fc(z)
        x = x.view(B, 64, 9, 5)
        x = self.deconv(x)
        return x


class NoiseGenerator(nn.Module):
    def __init__(self, signal_length=600, noise_dim=64):
        super().__init__()
        self.signal_length = signal_length
        self.gphy = GphyNet(signal_length)
        self.gnoise = GnoiseNet(noise_dim, signal_length)

    def get_phy(self, s_gt):
        return self.gphy(s_gt)

    def get_noise(self, z, T):
        return self.gnoise(z)

    def forward(self, s_gt, z):
        m_phy = self.get_phy(s_gt)
        m_noise = self.get_noise(z, self.signal_length)
        return m_phy + m_noise
