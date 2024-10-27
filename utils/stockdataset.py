import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    # 读取数据
    # data_path = '/root/autodl-tmp/FLShapley/data/amazon-stock-price/AMZN11_data_1999_2022.csv' 

    def __init__(self, data_path, lookback, train, transform=None, split_ratio=0.8):
        self.data = pd.read_csv(data_path)
        self.data = self.data.sort_values('Date')
        self.lookback = lookback
        self.train = train
        self.transform = transform

        # 标准化
        price = self.data[['Close']] # 使用收盘价
        scaler = MinMaxScaler(feature_range=(-1, 1))
        price.loc[:, 'Close'] = scaler.fit_transform(price[['Close']].values)


        # 划分训练集和测试集
        x_train, y_train, x_test, y_test = self.split_data(price[['Close']], lookback, split_ratio)

        # 转换为张量
        if self.train: # 如果使用训练集
            self.x_data = torch.tensor(x_train, dtype=torch.float32)
            self.y_data = torch.tensor(y_train, dtype=torch.float32)
        else: #  如果使用测试集
            self.x_data = torch.tensor(x_test, dtype=torch.float32)
            self.y_data = torch.tensor(y_test, dtype=torch.float32)

    def split_data(self, stock, lookback, split_ratio):
        self.data_raw = stock.to_numpy()
        self.data = []

        for index in range(len(self.data_raw) - lookback): # 至少需要lookback天的数据来进行预测
            self.data.append(self.data_raw[index: index + lookback])
        
        self.data = np.array(self.data)
        train_set_size = int(np.round(split_ratio * self.data.shape[0]))
        test_set_size = self.data.shape[0] - train_set_size

        x_train = self.data[:train_set_size, :-1, :]  # 训练集特征，不包括最后一列（预测目标）
        y_train = self.data[:train_set_size, -1, :]  # 训练集的目标值（最后一列）
        x_test = self.data[train_set_size:, :-1, :]  # 测试集
        y_test = self.data[train_set_size:, -1, :]
        return x_train, y_train, x_test, y_test

    def __len__(self):
        return len(self.x_data)  # 返回对象self中x_data属性的长度

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        if self.transform and not torch.is_tensor(x):
            x = self.transform(x)
        return x, y
