from typing import *
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
Vector = np.ndarray
Matrix = np.ndarray
PATH = '/data/seungmin'


def overwrite(data: Matrix) -> Matrix:
    """
    If there is NaN in the i-th data, 
    this function overwrites it with the most recent without-NaN data.

    Parameters:
        data: The data that you want to overwrite
    """
    for i in range(1, data.shape[0]):
        if np.any(np.isnan(data[i])):
            data[i] = data[i - 1]
    return data


class Dataset(object):
    """
    Load data.

    Parameters:
        dataset:     Name of the dataset.
        window_size: Length of the sliding window. Default is 100.
        mode:        Either train or test.
        convert_nan: How to convert the data with NaN value. Default 'nan_to_zero.'
                     If dataset is 'GECCO_2018' or 'CECCO_2019', then set it to 'overwrite.' 
        train_traio: The ratio of train set. For MSL, SMAP, SMD, SWaT, it is unnecessary.
                     For the GECCO_2018 and GECCO_2019, default is 0.8.
    """
    def __init__(
        self,
        dataset: str,
        window_size: int = 100,
        mode: str = 'train',
        convert_nan: str = 'nan_to_zero',
        train_ratio: float = 0.8,
    ) -> None:
        data_path = os.path.join(PATH, dataset)
        scaler = StandardScaler()

        data = None
        labels = None

        if dataset in ['MSL', 'SMAP', 'SMD']:
            if mode == 'train':
                data = np.load(os.path.join(data_path, f'{dataset}_train.npy'))
            elif mode == 'test':
                data = np.load(os.path.join(data_path, f'{dataset}_test.npy'))
                labels = np.load(os.path.join(data_path, f'{dataset}_test_label.npy'))
        elif dataset == 'SWaT':
            if mode == 'train':
                data = pd.read_csv(os.path.join(data_path, 'SWaT_Normal.csv'))
                data.drop(columns=[' Timestamp', 'Normal/Attack'], inplace=True)
                data = data.values[:, 1:]
            elif mode == 'test':
                data = pd.read_csv(os.path.join(data_path, 'SWaT_Abormal.csv'))
                data.drop(columns=[' Timestamp'], inplace=True)
                data = data.values
                labels = data[:, -1]
                data = data[:, :-1]
                labels = np.where(labels == 'Normal', 0, 1)
        elif dataset in ['GECCO_2018', 'GECCO_2019']:
            data = pd.read_csv(os.path.join(PATH, 'GECCO', dataset, f'1_{dataset.lower().replace('_', '')}_water_quality.csv'))
            data.drop(columns=['Time'], inplace=True)
            data = data.values[:, 1:]
            labels = data[:, -1]
            labels = np.where(labels == False, 0, 1)
            data = data[:, :-1]
            train_size = int(data.shape[0] * train_ratio)
            if mode == 'train':
                data = data[:train_size, :]
            elif mode == 'test':
                data = data[train_size:, :]
                labels = labels[train_size:]
        
        if convert_nan == 'overwrite':
            data = overwrite(data)
        elif convert_nan == 'nan_to_zero':
            data = np.nan_to_num(data)

        scaler.fit(data)
        data = scaler.transform(data)

        self.data_shape = data.shape

        self.data = data
        self.labels = labels
        self.mode = mode
        self.window_size = window_size
    
    def __len__(self) -> int:
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx: int) -> Matrix:
        if self.mode == 'train':
            return np.float32(self.data[idx: idx+self.window_size])
        elif self.mode == 'test':
            return np.float32(self.data[idx: idx+self.window_size]), np.float32(self.labels[idx: idx+self.window_size])
