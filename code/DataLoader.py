import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic


class EmDataset(Dataset):

    def __init__(self, csv_name, root_dir, length, forecast_window):
        """
        Args:
            csv_name (string): 数据集文件名。
            root_dir (string): 数据集所在目录。
            length (int): 输入序列长度。
            forecast_window (int): 预测窗口长度。
        """
        # 加载数据集
        csv_file = f"{root_dir}/{csv_name}"
        self.df = pd.read_csv(csv_file)
        self.T = length
        self.S = forecast_window
        self.transform = MinMaxScaler()

    def __len__(self):
        # 数据集长度为唯一的 server-up 数量
        return self.df["server-up"].nunique()

    def __getitem__(self, idx):

        if len(self.df) < self.T + self.S:
            raise ValueError(f"Dataset length {len(self.df)} is less than required length {self.T + self.S}")
        
        start = torch.randint(0, len(self.df) - self.T - self.S + 1, (1,)).item()
        
        index_in = torch.tensor([i for i in range(start, start + self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        
        _input = torch.tensor(self.df.iloc[start:start + self.T][[
            "load-1m",
            "sys-mem-swap-total", "sys-mem-swap-free", "sys-mem-free", "sys-mem-cache", 
            "sys-mem-buffered", "sys-mem-available", "sys-mem-total", 
            "sys-fork-rate", "sys-interrupt-rate", "sys-context-switch-rate", 
            "sys-thermal", "disk-io-time", "disk-bytes-read", "disk-bytes-written", 
            "disk-io-read", "disk-io-write", "cpu-iowait", "cpu-system", "cpu-user"
        ]].values)
        
        target = torch.tensor(self.df.iloc[start + self.T:start + self.T + self.S][[
            "load-1m",
            "sys-mem-swap-total", "sys-mem-swap-free", "sys-mem-free", "sys-mem-cache", 
            "sys-mem-buffered", "sys-mem-available", "sys-mem-total", 
            "sys-fork-rate", "sys-interrupt-rate", "sys-context-switch-rate", 
            "sys-thermal", "disk-io-time", "disk-bytes-read", "disk-bytes-written", 
            "disk-io-read", "disk-io-write", "cpu-iowait", "cpu-system", "cpu-user"
        ]].values)

        for i in range(_input.shape[1]):
            feature_scaler = MinMaxScaler()
            feature_scaler.fit(_input[:, i].reshape(-1, 1))
            _input[:, i] = torch.tensor(feature_scaler.transform(_input[:, i].reshape(-1, 1)).squeeze(-1))
            target[:, i] = torch.tensor(feature_scaler.transform(target[:, i].reshape(-1, 1)).squeeze(-1))
            # 保存用于第一列的 scaler
            if i == 0:
                dump(feature_scaler, 'scalar_item_feature0.joblib')

        return index_in, index_tar, _input, target