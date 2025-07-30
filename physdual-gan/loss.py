
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import euclidean

# 1. Pearson Correlation Loss
def pearson_loss(x, y):
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    corr = (x_centered * y_centered).sum(dim=1) / (
        torch.norm(x_centered, dim=1) * torch.norm(y_centered, dim=1) + 1e-8
    )
    return 1 - corr.mean()

# 2. DTW Morphology Loss with Normalization
def dtw_loss(x, y):
    loss = 0.0
    for i in range(x.shape[0]):
        xi = x[i].detach().cpu().numpy().flatten()
        yi = y[i].detach().cpu().numpy().flatten()
        # normalize
        xi = (xi - np.mean(xi)) / (np.std(xi) + 1e-6)
        yi = (yi - np.mean(yi)) / (np.std(yi) + 1e-6)
        dist, _ = fastdtw(xi, yi, dist=lambda a, b: abs(a - b))
        loss += dist
    return loss / x.shape[0]



# 3. GAN Loss
def gan_loss(dis_real, dis_fake):
    criterion = nn.BCELoss()
    real_labels = torch.ones_like(dis_real)
    fake_labels = torch.zeros_like(dis_fake)
    loss_real = criterion(dis_real, real_labels)
    loss_fake = criterion(dis_fake, fake_labels)
    return loss_real + loss_fake

# 4. Full Generator Loss: L_gen = λ1*L_dtw + λ2*L_pearson + λ3*L_gan
def full_generator_loss(s_fake, s_gt, dis_fake, lambda1=0.05, lambda2=1.0, lambda3=0.1):
    l_dtw = dtw_loss(s_fake, s_gt)
    l_p = pearson_loss(s_fake, s_gt)
    l_gan = torch.mean(torch.log(1 - dis_fake + 1e-6))
    return lambda1 * l_dtw + lambda2 * l_p + lambda3 * l_gan

# 5. L_r constraint
def recon_loss(s_recon, s_gt):
    return pearson_loss(s_recon, s_gt)

# 6. Discriminator loss
def joint_discriminator_loss(D, s_gt, m_real, m_syn, m_noise):
    real = D(s_gt)
    fake1 = D(m_syn)
    fake2 = D(m_noise)
    return -torch.mean(torch.log(real + 1e-6)) - torch.mean(torch.log(1 - fake1 + 1e-6)) - torch.mean(torch.log(1 - fake2 + 1e-6))
