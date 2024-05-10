import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class ETTDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, trainmode=False, scale=True, seq_len=336, pred_len=96):
        super().__init__()

        x_y = pd.read_csv(data_path)
        # x_y = df.iloc[:,:]

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset/ETTh1.csv", help="Path to dataset")
    parser.add_argument("--seq-len", type=int, default=336, help="Sequence length")
    args = parser.parse_args()

    assert os.path.exists(args.data_path), f"{args.data_path} does not exist"

    df = pd.read_csv(args.data_path)

    set_ranges = {
        "train": [0, 10_000],
        "val": [10_000 - args.seq_len, 14_000],
        "test": [14_000 - args.seq_len, 16_000],
        "test2": [16_000 - args.seq_len, df.shape[0]-1]
    }

    for mode in ["train", "val", "test", "test2"]:
        start, end = set_ranges[mode]
        df.iloc[start:end,1:].to_csv(f"dataset/{mode}.csv", index=False)


