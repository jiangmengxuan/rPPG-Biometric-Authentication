
import numpy as np
import h5py
import os
from torch.utils.data import Dataset

class H5DatasetWithBVP(Dataset):
    def __init__(self, train_list, T):
        self.train_list = train_list
        self.T = T

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = int(min(f['imgs'].shape[0], f['bvp'].shape[0]) * 0.6)

            idx_start = np.random.choice(img_length - self.T)
            idx_end = idx_start + self.T

            # 视频片段 (T, H, W, C) -> (3, S, T)
            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
            hw = img_seq.shape[2]
            img_seq = img_seq.reshape(3, self.T, -1)
            img_seq = img_seq[:, :, np.random.permutation(hw * hw)]
            img_seq = np.transpose(img_seq, (0, 2, 1))  # (3, S, T)

            # BVP 信号 (T,)
            bvp = f['bvp'][idx_start:idx_end].astype('float32')

        return img_seq, bvp
