#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pandas as pd
import numpy as np
from torchvision import datasets, transforms


# 数据集为股票数据
def stock_iid(dataset, num_users, lookback=30):  # 独立同分布
    """
    Sample I.I.D. client data from stock dataset
    :param data_path: path to the CSV file containing stock data
    :param num_users: number of users
    :param lookback: number of past days used as features for predicting the next day's value
    :return: dict of user data indices
    """
    prices = dataset.x_data.numpy()
    num_items = int(len(prices) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(prices) - lookback)]  # 使用lookback天数作为输入

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def stock_noniid(dataset, num_users, lookback=30):  # 非独立同分布
    """
    Sample non-I.I.D. client data from stock dataset
    :param data_path: path to the CSV file containing stock data
    :param num_users: number of users
    :param lookback: number of past days used as features for predicting the next day's value
    :return: dict of user data indices
    """
    prices = dataset.x_data.numpy() 
    num_shards = 200  # 分片数量
    num_items = int(len(prices) / num_shards)  # 每个分片的项数
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(prices) - lookback)

    # 将数据按按价格排序
    sorted_idxs = np.argsort(prices[lookback:])
    idxs = idxs[sorted_idxs]

    # 将分片分配给用户
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            start_idx = rand * num_items
            end_idx = (rand + 1) * num_items
            dict_users[i] = np.concatenate((dict_users[i], idxs[start_idx:end_idx].reshape(-1)), axis=0)
    return dict_users



# # 数据集为minist
# def mnist_iid(dataset, num_users): # 分数据 模仿数据在不同设备或位置之间的自然分布方式
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users) # 每个用户将收到的图像数量
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))  # 不可以取相同的 每个索引只被选择一次
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


# def mnist_noniid(dataset, num_users): # 用户之间的数据分布不平衡 模拟不同客户端收集的数据可能存在很大差异的场景
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 300 # 数据集分割成的分片总数为200。分片代表一组具有相似标签的图像。每个分片中图像数量为300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     for i in range(num_users): # 迭代每个用户并为他们分配数据
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # 每个用户从 2 个不同的分片获取数据
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set: # 遍历选定的分片索引
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users


# # 数据集为cifar-10,仅考虑独立同分布
# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users



# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,), (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)
