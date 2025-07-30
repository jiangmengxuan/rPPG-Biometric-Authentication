
# train_dualgan.py - Stable PhysDual-GAN Training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dual_gan.fb_generator import BVPGenerator
from dual_gan.noise_gan import NoiseGenerator
from dual_gan.discriminator import BVPDiscriminator
from dual_gan.loss import full_generator_loss, recon_loss, joint_discriminator_loss
from utils_data import H5DatasetWithBVP
import glob
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 600
noise_dim = 64
lr = 3e-4
epochs = 150
batch_size = 4

fb = BVPGenerator().to(device)
noise_gan = NoiseGenerator(signal_length=T, noise_dim=noise_dim).to(device)
d = BVPDiscriminator(signal_length=T).to(device)

torch.autograd.set_detect_anomaly(True)

opt_fb = torch.optim.Adam(fb.parameters(), lr=lr)
opt_gphy = torch.optim.Adam(noise_gan.gphy.parameters(), lr=lr)
opt_gnoise = torch.optim.Adam(noise_gan.gnoise.parameters(), lr=lr)
opt_d = torch.optim.Adam(d.parameters(), lr=lr)

train_list = glob.glob('../data_example/h5_obf/*1.h5')
print(f"[INFO] 加载到 {len(train_list)} 个 _1.h5 文件")
assert len(train_list) > 0, "未找到训练文件！"
dataset = H5DatasetWithBVP(train_list, T)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

os.makedirs("./weights", exist_ok=True)

for epoch in range(epochs):
    for i, (m_real, s_gt) in enumerate(dataloader):
        m_real = m_real.to(device)
        s_gt = s_gt.to(device)
        z = torch.randn(m_real.shape[0], noise_dim).to(device)

        # Step 1: Train Fb
        fb.train()
        opt_fb.zero_grad()
        s_fake = fb(m_real)
        d_fake = d(s_fake)
        L_gen = full_generator_loss(s_fake, s_gt, d_fake)
        L_gen.backward()
        opt_fb.step()

        # Step 2: Train Gphy (freeze Fb)
        for p in fb.parameters(): p.requires_grad = False
        opt_gphy.zero_grad()
        m_phy = noise_gan.get_phy(s_gt)
        s_recon = fb(m_phy)
        L_r = recon_loss(s_recon, s_gt)
        L_r.backward(retain_graph=True)
        opt_gphy.step()
        for p in fb.parameters(): p.requires_grad = True

        # Step 3: Train Fb + Gnoise
        fb.train()
        opt_fb.zero_grad()
        opt_gnoise.zero_grad()
        m_noise = noise_gan.get_noise(z, s_gt.shape[1])
        m_phy = m_phy.detach()  # 🔥 必加，确保 Gnoise 可训练
        m_syn = m_phy + m_noise
        s_syn = fb(m_syn)
        m_real_rppg = fb(m_real).detach()
        m_syn_rppg = fb(m_syn).detach()
        m_noise_rppg = fb(m_noise).detach()
        L_joint = joint_discriminator_loss(d, s_gt, m_real_rppg, m_syn_rppg, m_noise_rppg)

        L_joint.backward()
        opt_fb.step()
        opt_gnoise.step()

        # Step 4: Train Discriminator
        d.train()
        opt_d.zero_grad()
        d_real = d(s_gt)
        d_fake1 = d(s_syn.detach())
        d_fake2 = d(s_fake.detach())
        L_d = -torch.mean(torch.log(d_real + 1e-6)) - torch.mean(torch.log(1 - d_fake1 + 1e-6)) - torch.mean(torch.log(1 - d_fake2 + 1e-6))
        L_d.backward()
        opt_d.step()

        if i % 10 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Step {i+1}/{len(dataloader)} | "
                  f"L_gen: {L_gen.item():.4f} | L_r: {L_r.item():.4f} | "
                  f"L_joint: {L_joint.item():.4f} | L_D: {L_d.item():.4f}")

    torch.save(fb.state_dict(), f"./weights/fb_epoch{epoch+1}.pt")
    torch.save(noise_gan.state_dict(), f"./weights/noise_gan_epoch{epoch+1}.pt")
    torch.save(d.state_dict(), f"./weights/discriminator_epoch{epoch+1}.pt")
