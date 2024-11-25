import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic


# class EmDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, csv_name, root_dir, length, forecast_window):
#         """
#         Args:
#             csv_file (string): Path to the csv file.
#             root_dir (string): Directory
#         """
        
#         # load raw data file
#         csv_file = os.path.join(root_dir, csv_name)
#         self.df = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.T = length
#         self.S = forecast_window

#     def __len__(self):
#         # Since the data is obtained randomly, this data is iterated only once
#         return 1

#     # Will pull an index between 0 and __len__. 
#     def __getitem__(self, idx):
#         # conbime multiple cores into one tensor 

#         #start = np.random.randint(0, len(self.df.groupby(by=["group_index"])) - self.T - self.S) 
#         start = np.random.randint(0, len(self.df) - self.T - self.S)
#         index_in = torch.tensor([i for i in range(start, start+self.T)])
#         index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        
#         _input_tensor_list = []
#         target_tensor_list = []

#         feature_cols = [
#             "cpu_load", "sin_ms", "cos_ms", "sin_sec", "cos_sec"
#         ]

#         for core_idx in self.df['core_id'].unique().tolist():
#             _input = torch.tensor(self.df[self.df["core_id"]==core_idx][feature_cols][start : start + self.T].values)
#             target = torch.tensor(self.df[self.df["core_id"]==core_idx][feature_cols][start + self.T : start + self.T + self.S].values)

#             # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
#             # scalar is fit only for em_load, as the timestamps are already scaled
#             # scalar input/output of shape: [n_samples, n_features].
    
#             # scalar for other features (except timestamps)
#             for i, value in enumerate(feature_cols[:-4]):
#                 if "em_load" == value:
#                     scaler = MinMaxScaler()
#                 else:
#                     scaler = StandardScaler()

#                 scaler.fit(_input[:,i].unsqueeze(-1))
#                 _input[:,i] = torch.tensor(scaler.transform(_input[:,i].unsqueeze(-1)).squeeze(-1))
#                 target[:,i] = torch.tensor(scaler.transform(target[:,i].unsqueeze(-1)).squeeze(-1))

#                 if "em_load" == value:
#                     # save the scalar to be used later when inverse translating the data for plotting.
#                     dump(scaler, 'scalar_item_core%s.joblib'%core_idx)

#             _input_tensor_list.append(_input)
#             target_tensor_list.append(target)

#         _input = torch.stack(_input_tensor_list)
#         target = torch.stack(target_tensor_list)

#         return index_in, index_tar, _input, target

class EmDataset(Dataset):
    def __init__(self, csv_name, root_dir, length, forecast_window):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory of the file
        """
        # Load the raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.T = length
        self.S = forecast_window

    def __len__(self):
        # Since data is randomly sampled, iterated only once
        return 1

    def __getitem__(self, idx):
        # Define start index to ensure enough data for both input (T) and target (S) windows
        start = np.random.randint(0, len(self.df) - self.T - self.S)
        
        # Define input and target indices
        index_in = torch.tensor([i for i in range(start, start + self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])

        # Define features to be used
        #feature_cols = ["cpu_load", "cgroupfs_user_cpu_usage_us", "sin_ms", "cos_ms", "sin_sec", "cos_sec","sin_min", "cos_min", "sin_hour", "cos_hour"]
        feature_cols = ["cpu_load", "cgroupfs_user_cpu_usage_us", "sin_ms", "cos_ms", "sin_sec", "cos_sec","sin_min", "cos_min"]

        # Extract input and target data without filtering by core_id
        _input = torch.tensor(self.df[feature_cols][start : start + self.T].values)
        target = torch.tensor(self.df[feature_cols][start + self.T : start + self.T + self.S].values)
        
        # Scale data
        for i, feature in enumerate(feature_cols):
            if feature == "cpu_load":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            # Fit scaler to the input data only to avoid target information leakage
            scaler.fit(_input[:, i].unsqueeze(-1))
            
            # Apply scaling
            _input[:, i] = torch.tensor(scaler.transform(_input[:, i].unsqueeze(-1)).squeeze(-1))
            target[:, i] = torch.tensor(scaler.transform(target[:, i].unsqueeze(-1)).squeeze(-1))
            
            # Save scaler for cpu_load to inverse transform later if needed
            if feature == "cpu_load":
                dump(scaler, f'scaler_cpu_load.joblib')

        # Return input and target along with their indices
        return index_in, index_tar, _input, target