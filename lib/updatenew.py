#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import logging
from torch.cuda.amp import autocast, GradScaler

# 设置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        """
        初始化类的构造方法。
        
        参数:
        dataset: 数据集，包含了一系列的数据样本。
        idxs: 数据集索引列表，指定了数据样本的顺序或选择。
        """
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        """返回索引列表的长度。"""
        return len(self.idxs)

    def __getitem__(self, item):
        """
        根据给定的索引item从数据集中获取元素。
    
        参数:
        item (int): 要访问的元素的索引。
    
        返回:
        tuple: 包含两个torch.tensor对象，分别代表图像数据和标签。
        """
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        """
        初始化类的构造函数。
    
        参数:
        - args: 包含程序配置和设置的对象。
        - dataset: 数据集对象，用于训练和测试。
        - idxs: 数据集索引的列表，用于分割数据集。
        """
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)
        self.scaler = GradScaler()  # 混合精度训练

    def train_val_test(self, dataset, idxs):
        """
        根据给定的数据集和用户索引，返回训练、验证和测试数据加载器。
    
        参数:
        - dataset: 要分割成训练、验证和测试集的数据集。
        - idxs: 数据集中特定用户的样本索引。
    
        返回:
        - trainloader: 训练集的数据加载器。
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, 
                                 num_workers=4, drop_last=True)
        return trainloader

    def _train_epoch(self, model, optimizer, global_round, idx, iter):
        """
        通用的训练周期函数，用于更新模型权重。
    
        参数:
        - model: 要训练的模型。
        - optimizer: 优化器。
        - global_round: 当前全局轮次。
        - idx: 用户索引。
        - iter: 当前本地轮次。
    
        返回:
        - batch_loss: 当前批次的损失。
        - acc_val: 当前批次的准确率。
        """
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            
            with autocast():  # 混合精度训练
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            _, y_hat = log_probs.max(1)
            acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

            if self.args.verbose and (batch_idx % 10 == 0):
                logging.info(f'| Global Round : {global_round} | User: {idx} | Local Epoch : {iter} | '
                            f'[{batch_idx * len(images)}/{len(self.trainloader.dataset)} '
                            f'({100. * batch_idx / len(self.trainloader):.0f}%)]\tLoss: {loss.item():.3f} | '
                            f'Acc: {acc_val.item():.3f}')
            batch_loss.append(loss.item())
        return batch_loss, acc_val

    def update_weights(self, idx, model, global_round):
        """
        本地训练模型，更新权重。
        
        参数:
        idx (int): 用户索引。
        model (torch.nn.Module): 全局模型。
        global_round (int): 当前全局轮次。
        
        返回:
        model.state_dict(): 更新后的模型权重。
        sum(epoch_loss) / len(epoch_loss): 平均损失。
        acc_val.item(): 准确率。
        """
        model.train()
        epoch_loss = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss, acc_val = self._train_epoch(model, optimizer, global_round, idx, iter)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_prox(self, idx, local_weights, model, global_round):
        """
        使用Proximal Federated Learning更新模型权重。
    
        参数:
        - idx: 用户索引，对应客户端的标识符。
        - local_weights: 其他客户端的本地权重字典。
        - model: 要训练的全局模型。
        - global_round: 当前的全局训练轮次。
    
        返回:
        - 更新后的模型权重。
        - 所有epoch的平均损失。
        - 最后一批次的准确率。
        """
        model.train()
        epoch_loss = []

        if idx in local_weights.keys():
            w_old = local_weights[idx]

        w_avg = model.state_dict()
        loss_mse = nn.MSELoss().to(self.device)

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss, acc_val = self._train_epoch(model, optimizer, global_round, idx, iter)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def inference(self, model):
        """ 返回模型的推理准确率和损失。"""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct / total
        return accuracy, loss


class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        """
        初始化类的构造函数。
    
        参数:
        - args: 包含程序配置和设置的对象。
        - dataset: 数据集，用于模型的训练和测试。
        - idxs: 指定数据集中的索引列表，用于创建测试集。
        """
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        """
        生成测试数据加载器。
    
        将给定的数据集根据指定的索引划分成测试集，并创建一个数据加载器。
    
        参数:
        dataset: 数据集实例，包含所有数据样本。
        idxs: 用于测试集的样本索引列表。
    
        返回:
        testloader: 测试数据加载器，用于迭代加载测试数据批次。
        """
        idxs_test = idxs[:int(1 * len(idxs))]
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        """
        评估模型在给定数据集上的性能。
    
        参数:
            args: 包含命令行参数的命名空间。
            idx: 模型或数据集的索引。
            classes_list: 类名列表。
            model: 要评估的神经网络模型。
    
        返回:
            loss: 所有批次的平均损失。
            acc: 模型在测试数据集上的准确率。
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, protos = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                outputs = outputs[:, 0:args.num_classes]
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        acc = correct / total
        return loss, acc


def test_inference(args, model, test_dataset, global_protos=[]):
    """ 返回测试准确率和损失。"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs, protos = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct / total
    return accuracy, loss