import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, data_path, lookback=30, train=True, transform=None, split_ratio=0.8):
        self.data = pd.read_csv(data_path)
        self.lookback = lookback
        self.train = train
        self.transform = transform

        # Normalize data (using MinMaxScaler for 'Close' price) 收盘价
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data['Close'] = scaler.fit_transform(self.data[['Close']].values)

        # Split data
        x_train, y_train, x_test, y_test = self.split_data(self.data[['Close']], lookback, split_ratio)

        # Convert to PyTorch tensors
        if self.train:
            self.x_data = torch.tensor(x_train, dtype=torch.float32)
            self.y_data = torch.tensor(y_train, dtype=torch.float32)
        else:
            self.x_data = torch.tensor(x_test, dtype=torch.float32)
            self.y_data = torch.tensor(y_test, dtype=torch.float32)

    def split_data(self, stock, lookback, split_ratio):
        data_raw = stock.to_numpy()
        data = []
        for index in range(len(data_raw) - lookback):
            data.append(data_raw[index: index + lookback])
        data = np.array(data)
        train_set_size = int(np.round(split_ratio * data.shape[0]))
        test_set_size = data.shape[0] - train_set_size
        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]
        x_test = data[train_set_size:, :-1, :]
        y_test = data[train_set_size:, -1, :]
        return x_train, y_train, x_test, y_test

    def __len__(self):
        return len(self.x_data)  # 返回对象self中x_data属性的长度

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        if self.transform and not torch.is_tensor(x):
            x = self.transform(x)
        return x, y
