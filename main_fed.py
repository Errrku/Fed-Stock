#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__': # 确保内部的代码块仅在直接执行脚本时运行
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users 为不同的用户拆分数据集
    if args.dataset == 'mnist': 
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid: # 检查用户是否指定了IID（独立同分布）数据分区
            dict_users = mnist_iid(dataset_train, args.num_users) # IID分布 则随机分训练数据集给各用户
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users) # 非IID分布 则数据的分区方式不是随机分布
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10') # 对于 CIFAR-10，仅考虑 IID 分布
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device) # 保存模型实例的变量CNNCifar
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size: # 计算 MLP 的输入特征总数
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob) # 打印全局模型的结构
    net_glob.train() # 将模型置于训练模式

    # copy weights
    w_glob = net_glob.state_dict() # 获取和存储模型的参数
    #（权重分发给不同的用户，他们用相同的初始全局模型进行本地训练，在本地训练之后，每个用户更新模型的本地副本，聚合回去w_glob以再次更新全局模型）

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: # 检查是否汇总来自所有客户端的更新
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)] # w_locals包含每个用户的初始全局模型权重的副本 确保每个客户端都以相同的权重开始
        
    for iter in range(args.epochs): # 训练 每次迭代代表一轮联邦学习
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1) # 确定本轮选取进行训练的用户数量 args.fra比例 确保至少选择了1个
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) # 随机选择用户，不进行替换，模拟联邦学习中的用户选择过程
        for idx in idxs_users: # 用户 仿真 对选定的用户进行迭代以进行 本地训练
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # 为每个选定的用户创建一个LocalUpdate对象 负责处理用户的本地训练
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device)) # w：本地训练模型的权重 loss：局部损失
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w) # 深拷贝
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals) # 中心服务器聚合参数

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob) # 将聚合权重加载回全局模型（net_glob）

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals) # 计算当前轮次所有选定用户的平均训练损失
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg) # 存储当前轮次的平均损失，loss_train以便跟踪一段时间内的训练进度

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval() # 将全局模型设置net_glob为评估模式
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

