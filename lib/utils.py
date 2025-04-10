#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
from sampling import cifar_iid, cifar100_noniid, cifar10_noniid, cifar100_noniid_lt, cifar10_noniid_lt
import femnist
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_dataset(args, n_list, k_list):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(args, train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups, classes_list = mnist_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = mnist_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
                classes_list_gt = classes_list

    elif args.dataset == 'femnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = femnist.FEMNIST(args, data_dir, train=True, download=True,
                                        transform=apply_transform)
        test_dataset = femnist.FEMNIST(args, data_dir, train=False, download=True,
                                       transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = femnist_iid(train_dataset, args.num_users)
            # print("TBD")
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                # user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                user_groups = femnist_noniid_unequal(args, train_dataset, args.num_users)
                # print("TBD")
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = femnist_noniid(args, args.num_users, n_list, k_list)
                user_groups_lt = femnist_noniid_lt(args, args.num_users, classes_list)

    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = cifar10_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar10_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_cifar100_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_cifar100_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list = cifar100_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar100_noniid_lt(test_dataset, args.num_users, classes_list)

    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != '....':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_sem(w, n_list):
    """
    Returns the average of the weights.
    """
    k = 2
    model_dict = {}
    for i in range(k):
        model_dict[i] = []

    idx = 0
    for i in n_list:
        if i< np.mean(n_list):
            model_dict[0].append(idx)
        else:
            model_dict[1].append(idx)
        idx += 1

    ww = copy.deepcopy(w)
    for cluster_id in model_dict.keys():
        model_id_list = model_dict[cluster_id]
        w_avg = copy.deepcopy(w[model_id_list[0]])
        for key in w_avg.keys():
            for j in range(1, len(model_id_list)):
                w_avg[key] += w[model_id_list[j]][key]
            w_avg[key] = torch.true_divide(w_avg[key], len(model_id_list))
            # w_avg[key] = torch.div(w_avg[key], len(model_id_list))
        for model_id in model_id_list:
            for key in ww[model_id].keys():
                ww[model_id][key] = w_avg[key]

    return ww

def average_weights_per(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:2] != 'fc':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            # w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_het(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc2.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

# def agg_func(protos):
#     """
#     Returns the average of the weights.
#     """

#     for [label, proto_list] in protos.items():
#         if len(proto_list) > 1:
#             proto = 0 * proto_list[0].data
#             for i in proto_list:
#                 proto += i.data
#             protos[label] = proto / len(proto_list)
#         else:
#             protos[label] = proto_list[0]

#     return protos

#def proto_aggregation(local_protos_list):
#     agg_protos_label = dict()
#     for idx in local_protos_list:
#         local_protos = local_protos_list[idx]
#         for label in local_protos.keys():
#             if label in agg_protos_label:
#                 agg_protos_label[label].append(local_protos[label])
#             else:
#                 agg_protos_label[label] = [local_protos[label]]

#     for [label, proto_list] in agg_protos_label.items():
#         if len(proto_list) > 1:
#             proto = 0 * proto_list[0].data
#             for i in proto_list:
#                 proto += i.data
#             agg_protos_label[label] = [proto / len(proto_list)]
#         else:
#             agg_protos_label[label] = [proto_list[0].data]

#     return agg_protos_label
def agg_func(protos):
    """
    兼容处理三种输入情况：
    1. 字典{label: 原型张量}
    2. 字典{label: 原型列表}
    3. 字典{label: {'protos': 原型, 'counts': 数量}}
    """
    aggregated = {}
    for label, proto_data in protos.items():
        # 情况1：直接是张量
        if torch.is_tensor(proto_data):
            aggregated[label] = proto_data
        # 情况2：是列表
        elif isinstance(proto_data, list):
            if len(proto_data) > 0:
                aggregated[label] = torch.stack(proto_data).mean(dim=0)
        # 情况3：是字典结构
        elif isinstance(proto_data, dict) and 'protos' in proto_data:
            if torch.is_tensor(proto_data['protos']):
                aggregated[label] = proto_data['protos']
            elif isinstance(proto_data['protos'], list):
                if proto_data['protos']:
                    aggregated[label] = torch.stack(proto_data['protos']).mean(dim=0)
        else:
            raise ValueError(f"Unexpected proto format for label {label}")
    
    return aggregated

def _weighted_aggregate(local_protos, args):
    """加权聚合实现"""
    global_protos = defaultdict(list)
    total_counts = defaultdict(int)
    
    # 统计全局样本量
    for client_data in local_protos.values():
        for label, count in client_data['counts'].items():
            total_counts[label] += count
    
    # 加权聚合
    for label in total_counts:
        weighted_proto = None
        total_weight = 0
        
        for client_data in local_protos.values():
            if label in client_data['protos']:
                weight = client_data['counts'][label] / total_counts[label]
                proto = client_data['protos'][label]
                if weighted_proto is None:
                    weighted_proto = weight * proto
                else:
                    weighted_proto += weight * proto
                total_weight += weight
        
        if weighted_proto is not None:
            global_protos[label] = [weighted_proto / total_weight]  # 保持列表形式
    
    return dict(global_protos)

def _legacy_aggregate(local_protos):
    """旧式平均聚合"""
    agg_protos = defaultdict(list)
    for client_protos in local_protos.values():
        for label, proto in client_protos.items():
            agg_protos[label].append(proto)
    
    return {
        label: [torch.stack(proto_list).mean(dim=0)]
        for label, proto_list in agg_protos.items()
    }

class DatasetSplit(Dataset):
    """用于分割数据集的辅助类"""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), label.clone().detach()

def save_protos(args, models, test_dataset, user_groups_lt):
    """
    保存原型数据用于可视化
    参数:
        args: 配置参数
        models: 所有客户的模型列表 
        test_dataset: 测试数据集
        user_groups_lt: 每个客户的测试数据索引字典
    """
    protos_dict = defaultdict(list)
    labels_list = []
    
    for idx, model in enumerate(models):
        model.eval()
        # 确保user_groups_lt[idx]是有效的索引
        if idx not in user_groups_lt or len(user_groups_lt[idx]) == 0:
            print(f"Warning: 客户端 {idx} 无测试数据")
            continue
            
        # 创建数据加载器
        test_loader = DataLoader(
            DatasetSplit(test_dataset, user_groups_lt[idx]),
            batch_size=50, 
            shuffle=False
        )
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(args.device)
                _, protos = model(images)  # 假设模型返回(logits, prototypes)
                
                # 收集原型和标签
                for i in range(len(labels)):
                    label = labels[i].item()
                    protos_dict[label].append(protos[i].cpu().numpy())
                    labels_list.append(label)
    
    # 保存为numpy文件
    np.save(f'./{args.alg}_protos.npy', np.concatenate([np.stack(v) for v in protos_dict.values()]))
    np.save(f'./{args.alg}_labels.npy', np.array(labels_list))
    print(f"原型已保存至 {args.alg}_protos.npy 和 {args.alg}_labels.npy")

def proto_aggregation(local_protos, args=None):
    """
    改进的原型聚合函数，兼容两种调用方式：
    1. 旧式：proto_aggregation(local_protos)
    2. 新式：proto_aggregation(local_protos, args)
    """
    if args is None:  # 旧式调用
        return _legacy_aggregate(local_protos)
    else:  # 新式加权聚合
        return _weighted_aggregate(local_protos, args)

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return