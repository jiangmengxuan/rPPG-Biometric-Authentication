import sys

import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import glob
from numpy.random import default_rng



# 第一阶段训练数据集（仅返回ST maps）
class H5Dataset(Dataset):
    # this dataset is used in the 1st training stage, only returning ST maps.
    #用于模型第一训练阶段的数据加载器，返回时空图（ST maps）
    def __init__(self, train_list, T):
        # 参数说明：
        # train_list - 包含所有.h5训练文件路径的列表
        # T - 每个视频片段的帧数（时间维度长度）
        self.train_list = np.random.permutation(train_list) # list of .h5 file paths for training
        self.T = T # video clip length



    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):

        with h5py.File(self.train_list[idx], 'r') as f:#打开HDF5文件
            img_length = int(f['imgs'].shape[0])#获取视频总帧数


            #随机选择起始帧
            idx_start = np.random.choice(img_length-self.T)
            idx_end = idx_start+self.T#计算结束帧

            #提取视频片段【T,H,W,C】
            img_seq = f['imgs'][idx_start:idx_end]
            #维度转置：从 [T, H, W, C] -> [C, T, H, W]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
            #空间置换操作
            hw = img_seq.shape[2]# 获取空间维度大小（假设H=W）
            img_seq = img_seq.reshape(3, self.T, -1) #展开空间维度 [C, T, H*W]
            #对空间位置进行随机排列（破坏局部空间结构）
            img_seq = img_seq[:,:,np.random.permutation(hw*hw)] #输出shape (3, T, S)
            img_seq = np.transpose(img_seq, (0,2,1)) # 最终转为形状(3, S, T)的ST map
        return img_seq


# 第二阶段训练数据集（返回ST maps + ID标签）
class H5Dataset_id(Dataset):
    # this dataset is used in the 2nd training stage, returning ST maps and ID labels.
    def __init__(self, train_list, T):
        self.train_list = train_list # list of .h5 file paths for training#保持原始文件顺序
        self.T = T # video clip length#时间维度长度

    def __len__(self):
        return len(self.train_list)#数据集样本总数

    def __getitem__(self, idx):
        # f_name = self.train_list[idx].split('/')[-1][:-3]
        # id_label = int(f_name[:3])-1

        # 获取完整文件名（例如："001_Jingang_1.h5"）
        full_name = os.path.basename(self.train_list[idx])  # 确保从完整路径提取文件名

        # 提取 ID 部分（从文件名开头到第一个下划线前的3位数字）
        id_part = full_name.split('_')[0]  # 示例："001_Jingang_1.h5" → "001"

        # 验证并转换 ID
        try:
            if len(id_part) != 3 or not id_part.isdigit():
                raise ValueError
            id_label = int(id_part) - 1  # ID 减1适配标签（001 → 0）
        except ValueError:
            raise ValueError(f"文件名 {full_name} 的ID格式错误，应为3位数字前缀（如 001_xxx_1.h5）")


        #数据加载部分（与H5Dataset类似但有不同之处）
        with h5py.File(self.train_list[idx], 'r') as f:
            # 取imgs和bvp的最小长度，并只使用前60%的帧，限制训练数据范围
            img_length = int(np.min([f['imgs'].shape[0], f['bvp'].shape[0]])*0.6) # first 60% for training
            #随机选择起始帧
            print(img_length, self.T)
            sys.exit()
            idx_start = np.random.choice(img_length-self.T)
            idx_end = idx_start+self.T
            #提取图像序列[T, H, W, C]
            img_seq = f['imgs'][idx_start:idx_end]
            #维度转换：从 [T, H, W, C] -> [C, T, H, W]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
            # 空间置换操作
            hw = img_seq.shape[2]#获取空间维度大小
            img_seq = img_seq.reshape(3, self.T, -1) #展开空间维度 [C, T, H*W]
            #对空间位置随机排列（破坏局部空间结构）
            img_seq = img_seq[:,:,np.random.permutation(hw*hw)] #输出(3, T, S)
            img_seq = np.transpose(img_seq, (0,2,1)) # 最终转换为(3, S, T)的ST map
        return img_seq, id_label