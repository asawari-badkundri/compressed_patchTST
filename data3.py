import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ETTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset="ETTh1", mode="train", scale=True, seq_len=336, pred_len=96):
        super().__init__()
        df = pd.read_csv("dataset/ETTh1.csv".format(dataset))
        x_y = df.iloc[:,1:]
        time_stamp = df.iloc[:,0]

        assert mode in ['train', 'val', 'finetune']
        type_map = {'train': 0, 'val': 1, 'finetune': 2}
        self.set_type = type_map[mode]

        self.seq_len = seq_len
        self.pred_len = pred_len

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # print(border1s)
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        # print(border2s)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if scale:
            train_x_y = x_y.iloc[border1s[0]: border2s[0]]
            self.ss = StandardScaler()
            self.ss.fit(train_x_y.to_numpy(dtype=np.float32))
            x_y = self.ss.transform(x_y.to_numpy(dtype=np.float32))
        else:
            x_y = x_y.to_numpy(dtype=np.float32)

        time_stamp = time_stamp.to_numpy()

        self.data_x = x_y[border1: border2, :]
        # print("border 1 :", border1)
        # print("border 2 :", border2)
        # print("data_x", self.data_x)
        self.data_y = x_y[border1: border2, -1]

        self.data_stamp = time_stamp[border1: border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.ss.inverse_transform(data)