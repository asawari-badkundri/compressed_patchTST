import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, trainmode=False, scale=True, seq_len=336, pred_len=96):
        super().__init__()

        x_y = pd.read_csv(data_path)

        self.seq_len = seq_len
        self.pred_len = pred_len

        if scale and trainmode:
            self.ss = StandardScaler()
            self.ss.fit(x_y.to_numpy(dtype=np.float32))
            x_y = self.ss.transform(x_y.to_numpy(dtype=np.float32))
        else:
            x_y = x_y.to_numpy(dtype=np.float32)

        self.data_x = x_y[:, :]
        self.data_y = x_y[: , -1]


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

# Script to split data files into train, finetune, val, test

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset/ETTh1.csv", help="Path to dataset")
    parser.add_argument("--seq-len", type=int, default=336, help="Sequence length")
    args = parser.parse_args()

    assert os.path.exists(args.data_path), f"{args.data_path} does not exist"

    df = pd.read_csv(args.data_path)
    n = df.shape[0]

    set_ranges = {
        "train": [0, int(0.6 * n)],
        "finetune": [int(0.6 * n) - args.seq_len, int(0.8 * n)],
        "val": [int(0.8 * n) - args.seq_len, int(0.9 * n)],
        "test": [int(0.9 * n) - args.seq_len, n-1]
    }

    for mode in ["train", "finetune", "val", "test"]:
        start, end = set_ranges[mode]
        df.iloc[start:end,1:].to_csv(f"dataset/{mode}.csv", index=False)