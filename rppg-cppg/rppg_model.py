import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from utils_sig import *
from dual_gan.fb_generator import BVPGenerator as rppg_model


class rppg_model(nn.Module):
    def __init__(self, fs, in_channels=3, out_channels=1):
        super(rppg_model, self).__init__()
        self.fs = fs

        #编码器部分
        self.inc = (DoubleConv(in_channels, 32))# 初始双卷积层

        self.down1 = (DownS(32, 64))  # 空间下采样（高度减半）
        self.down2 = (DownS(64, 128))  # 空间下采样（高度再减半）
        self.down3 = (DownT(128, 256))  # 时间下采样（时间维度减半）
        self.down4 = (DownT(256, 512))  # 时间下采样（时间维度再减半）

        #解码器部分
        self.up1 = (UpT(512, 256))  # 时间上采样（时间维度加倍）
        self.up2 = (UpT(256, 128))  # 时间上采样（时间维度加倍）
        self.outc = (OutConv(128, out_channels))  # 输出层

    def forward(self, x):
        # 输入形状(B, 3, N, T)→ (批次, 通道, 空间高度, 时间帧数)
        #标准化
        means = torch.mean(x, dim=(2, 3), keepdim=True)  # 计算均值和标准差
        stds = torch.std(x, dim=(2, 3), keepdim=True)
        x = (x - means) / stds # (B, 3, N, T)# 标准化处理

        #编码器路径
        x = self.inc(x) # (B, C, 36, 600)假设输入高度为36
        x = self.down1(x) # (B, C, 18, 600)空间高度减半
        x = self.down2(x) # (B, C, 9, 600)
        x = self.down3(x) # (B, C, 9, 300)时间维度减半
        x = self.down4(x) # (B, C, 9, 150)时间维度再减半

        #解码器路径
        x = self.up1(x) # (B, C, 9, 300)时间维度加倍
        x = self.up2(x) # (B, C, 9, 600)时间维度加倍
        x = self.outc(x) # (2, 1, 4, 600)空间高度降采样到4
        x = x[:, 0] # (2, 4, 600)提取通道0

        # filtering巴特沃斯滤波
        filter_b, filter_a = butter_ba(lowcut=40/60, highcut=250/60, fs=self.fs)
        filter_a = torch.tensor(filter_a.astype('float32')).to(x.device)
        filter_b = torch.tensor(filter_b.astype('float32')).to(x.device)

        x = torchaudio.functional.filtfilt(x, filter_a, filter_b, clamp=False) # (2, 4, 600)
        return x, torch.mean(x, 1) # (2, 4, 600), (2, 600)


#双卷积模块：包含两个连续的卷积层，每层后接批量归一化和ELU激活。
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


#空间下采样模块；示例：输入形状 (B, C, H, T) → 输出 (B, out_channels, H/2, T)。
class DownS(nn.Module):
    """Downscaling with avgpool then double conv"""
    #空间下采样：高度减半
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d((2, 1)),# 在高度方向（第2维）平均池化，步长(2,1)
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


#时间下采样模块；示例：输入形状 (B, C, H, T) → 输出 (B, out_channels, H, T/2)
class DownT(nn.Module):
    """Downscaling with avgpool then double conv"""
    #时间下采样：时间维度减半
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d((1, 2)),# 在时间方向（第3维）平均池化，步长(1,2)
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


#时间上采样模块
class UpT(nn.Module):
    """Upscaling then double conv"""
    #时间上采样：时间维度加倍
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2)) # 时间维度插值扩展
        return self.conv(x)


#输出层模块
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.avg_down = nn.AvgPool2d((2, 1)) # 空间高度减半
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.avg_down(x)
        return self.conv(x)
 