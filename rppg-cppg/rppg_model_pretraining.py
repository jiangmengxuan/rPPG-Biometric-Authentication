import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import h5py
import torch
from rppg_model import rppg_model
from rppg_model_loss import ContrastLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio

from utils_data import *
from utils_sig import *
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('model_train', save_git_info=False)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #print("Using GPU:")
else:
    device = torch.device('cpu')
    #print("Using CPU:")

@ex.config
def my_config():
    # hyperparameters

    total_epoch = 30 # total number of epoch总训练轮数
    lr = 1e-5 # learning rate学习率
    in_ch = 3 #number of input video channels, in_ch=3 for RGB videos输入视频通道数

    fs = 60 # video frame rate视频帧率，每秒60帧
    T = fs * 10 # input video length, default 10s.输入视频长度（默认10秒）,T代表总帧数

    # hyperparams for rPPG spatiotemporal sampling#rPPG时空采样的超参数
    delta_t = int(T/2) # time length of each rPPG sample#每个rPPG样本的时间长度
    K = 4  # the number of rPPG samples per row of an rPPG ST map#每个rPPG时空图每行的样本数
    
    train_exp_name = 'default'
    result_dir = './results/%s'%(train_exp_name) # store checkpoints and training recording#结果保存目录
    os.makedirs(result_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(result_dir))

@ex.automain
def my_main(_run, total_epoch, T, lr, result_dir, fs, delta_t, K, in_ch):
    #实验记录目录
    exp_dir = result_dir + '/%d'%(int(_run._id)) # store experiment recording to the path

    # training list.训练数据列表
    #在预练习视频上进行训练。在加载数据时，每个视频的前60%（5分钟中的3分钟）时长用于训练。
    train_list = glob.glob('./data_example/h5_obf/*1.h5') # train on pre-exercise videos. During loading data, the first 60% (3min out of 5min) length of each video is used for training.
    np.save(exp_dir+'/train_list.npy', train_list)#保存训练文件列表

    # define the dataloader#定义数据集和数据加载器
    dataset = H5Dataset(train_list, T)#使用自定义H5数据集在utils_data里
    dataloader = DataLoader(dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # define the model and loss
    model = rppg_model(fs).to(device).train()#创建rPPG模型并设置为训练模式
    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)#定义对比损失函数

    # define irrelevant power ratio#初始化无关功率比计算器
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer#定义优化器
    opt = optim.AdamW(model.parameters(), lr=lr)


    # for e in range(total_epoch):
    #     for it in range(np.round(180/(T/fs)).astype('int')): # 180 means the video length of each video is 180s (3min).
    #         for imgs in dataloader: # dataloader randomly samples a video clip with length T
    #             imgs = imgs.to(device)
    #
    #             # model forward propagation
    #             model_output, rppg = model(imgs) # model_output is the rPPG ST map
    #
    #             # define the loss functions
    #             loss, p_loss, n_loss = loss_func(model_output)
    #
    #             # optimize
    #             opt.zero_grad()
    #             loss.backward()
    #             opt.step()
    #
    #             # evaluate irrelevant power ratio during training
    #             ipr = torch.mean(IPR(rppg.clone().detach()))
    #
    #             # save loss values and IPR
    #             ex.log_scalar("loss", loss.item())
    #             ex.log_scalar("p_loss", p_loss.item())
    #             ex.log_scalar("n_loss", n_loss.item())
    #             ex.log_scalar("ipr", ipr.item())
    #
    #
    #     # save model checkpoints
    #     torch.save(model.state_dict(), exp_dir+'/epoch%d.pt'%e)


# 外层循环（Iter）：
    # 将180秒视频分割为18个10秒的片段。
    # 每个片段包含600帧（10秒×60帧/秒）。
# 内层循环（Batch）：
    # 对每个10秒片段（600帧），进一步分批次（如32帧/批次）。600/32=19
    # 每个批次通过模型训练，共处理19个批次。
# 完整训练过程：
    # 每个 Epoch遍历所有18个Iter（覆盖180秒视频）。
    # 每个 Iter 遍历所有19个Batch（覆盖600帧）。
    # 每个 Batch 更新一次模型参数。


    #训练循环
    for e in range(total_epoch):#遍历每个epoch
        for it in range(np.round(180 / (T / fs)).astype('int')):  # 180秒的视频长度，将一个总长度为180秒的视频分割成多个长度为T帧（即10秒）的片段
            for batch_idx, imgs in enumerate(dataloader):  # dataloader随机采样长度为T的视频片段
                imgs = imgs.to(device)

                # 模型前向传播
                model_output, rppg = model(imgs)  # model_output是rPPG时空图

                # 计算损失函数
                loss, p_loss, n_loss = loss_func(model_output)

                # 反向传播和优化
                opt.zero_grad()#清空梯度
                loss.backward()#反向计算梯度
                opt.step()#更新参数

                # 计算无关功率比
                ipr = torch.mean(IPR(rppg.clone().detach()))

                # 记录损失值和IPR
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("p_loss", p_loss.item())
                ex.log_scalar("n_loss", n_loss.item())
                ex.log_scalar("ipr", ipr.item())

                # 打印当前的损失值和IPR
                #Epoch：显示当前训练轮次和总轮次；Iter：当前视频片段和总片段;Batch:当前处理批次和总批次
                #Loss：总损失；P_Loss:正损失；N_Loss：负损失
                print(f"Epoch [{e + 1}/{total_epoch}], "
                      f"Iter [{it + 1}/{np.round(36 / (T / fs)).astype('int')}], "
                      f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"P_Loss: {p_loss.item():.4f}, "
                      f"N_Loss: {n_loss.item():.4f}, "
                      f"IPR: {ipr.item():.4f}")
        # save model checkpoints保存模型检查点
        torch.save(model.state_dict(), exp_dir + '/epoch%d.pt' % e)