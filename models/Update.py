#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object): # 模拟联邦学习设置中客户端的本地训练过程
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.MSELoss(reduction='mean')
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep): #local_ep本地训练的轮数或称局部时期（w在本地更新的次数） 控制客户端本地数据上进行训练的次数
            batch_loss = [] # 跟踪一个时期内每个批次的损失
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward() # 反向传播来计算梯度
                optimizer.step() # 根据梯度在本地更新模型参数
                if self.args.verbose and batch_idx % 10 == 0: # 每10批次打印一次进度和损失以供监控
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item()) # 存储当前批次的损失
            epoch_loss.append(sum(batch_loss)/len(batch_loss)) # 存储当前时期的平均损失
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) # 返回本地训练后更新的模型参数 返回所有时期的平均损失

