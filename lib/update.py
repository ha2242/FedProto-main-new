#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
#from models import CNNFemnist
import math
import torch.nn.functional as F

class DatasetSplit(Dataset):#分割子集，将原始数据集分割为子集，根据传入的索引 idxs 返回特定的数据样本
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        """
        初始化类的构造方法。
        
        参数:
        dataset: 数据集，包含了一系列的数据样本。
        idxs: 数据集索引列表，指定了数据样本的顺序或选择。
        
        该构造方法的主要作用是将传入的数据集和索引列表进行初始化，并确保索引列表中的索引值为整数类型。
        """
        # 初始化dataset属性
        self.dataset = dataset
        
        # 将idxs中的索引转换为整数类型，并初始化idxs属性
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        """
        实现对象的长度属性，返回索引列表的长度。
    
        通过重写__len__方法，使得len()函数可以应用于该对象，反映其索引列表的大小。
    
        Returns:
            int: 索引列表的长度。
        """
        return len(self.idxs)

    def __getitem__(self, item):
        """
        根据给定的索引item从数据集中获取元素。
    
        参数:
        item (int): 要访问的元素的索引。
    
        返回:
        tuple: 包含两个torch.tensor对象，分别代表图像数据和标签。
        """
        # 通过item索引从self.idxs中获取对应的数据集索引，并从数据集中提取图像和标签
        image, label = self.dataset[self.idxs[item]]
        # 将图像和标签转换为torch.tensor类型，并以元组的形式返回
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
        # 初始化类的参数
        self.args = args
    
        # 调用自定义方法train_val_test来分割数据集，并将训练数据加载到trainloader
        self.trainloader = self.train_val_test(dataset, list(idxs))
    
        # 设置设备（如CPU或GPU）用于模型训练和推理
        self.device = args.device
    
        # 初始化损失函数，并将其移动到指定设备
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        根据给定的数据集和用户索引，返回训练、验证和测试数据加载器。

        参数:
        - dataset: 要分割成训练、验证和测试集的数据集。
        - idxs: 数据集中特定用户的样本索引。

        返回:
        - trainloader: 训练集的数据加载器。
        """
        # 选择训练集的索引，这里使用了整个数据集作为训练集。
        idxs_train = idxs[:int(1 * len(idxs))]
        # 创建训练集的数据加载器。
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    # 更新模型权重的方法
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
        # 设置模型为训练模式
        model.train()
        epoch_loss = []
    
        # 根据配置选择优化器
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
    
        # 进行本地训练周期
        for iter in range(self.args.train_ep):
            batch_loss = []
            # 遍历数据集中的每个批次
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)
    
                # 模型梯度归零
                model.zero_grad()
                # 前向传播
                log_probs, protos = model(images)
                # 计算损失
                loss = self.criterion(log_probs, labels)
    
                # 反向传播
                loss.backward()
                # 优化器更新模型参数
                optimizer.step()
    
                # 计算准确率
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
    
                # 根据配置打印训练信息
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                # 记录批次损失
                batch_loss.append(loss.item())
            # 记录每个周期的平均损失
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
        # 返回更新后的模型权重，平均损失和准确率
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
        # 设置模型为训练模式
        model.train()
        epoch_loss = []

        # 如果存在，加载旧权重
        if idx in local_weights.keys():
            w_old = local_weights[idx]

        # 获取当前模型权重作为平均权重
        w_avg = model.state_dict()
        loss_mse = nn.MSELoss().to(self.device)

        # 根据配置设置本地更新的优化器
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # 进行多个epoch的训练
        for iter in range(self.args.train_ep):
            batch_loss = []
            # 遍历训练数据加载器中的每个批次
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                # 清零梯度
                model.zero_grad()
                # 前向传播
                log_probs, protos = model(images)
                # 计算损失
                loss = self.criterion(log_probs, labels)
                # 如果存在旧权重，计算额外的损失
                if idx in local_weights.keys():
                    loss2 = 0
                    for para in w_avg.keys():
                        loss2 += loss_mse(w_avg[para].float(), w_old[para].float())
                    loss2 /= len(local_weights)
                    loss += loss2 * 150
                # 反向传播并更新权重
                loss.backward()
                optimizer.step()

                # 计算预测准确率
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                # 打印训练进度和损失、准确率
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| 全局轮次 : {} | 用户: {} | 本地轮次 : {} | [{}/{} ({:.0f}%)]\t损失: {:.3f} | 准确率: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 返回更新后的模型权重、平均损失和最终准确率
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    # def update_weights_het(self, args, idx, global_protos, model, global_round=round):
    #     """
    #     更新异构联邦学习中的模型权重。

    #     参数:
    #     - args: 程序参数
    #     - idx: 用户索引
    #     - global_protos: 全局原型
    #     - model: 待训练的模型
    #     - global_round: 当前全局轮次

    #     返回:
    #     - model.state_dict(): 更新后的模型权重
    #     - epoch_loss: 每个epoch的损失
    #     - acc_val.item(): 最终的准确率
    #     - agg_protos_label: 聚合的原型标签
    #     """
    #     # 设置模型为训练模式
    #     model.train()
    #     epoch_loss = {'total':[], '1':[], '2':[], '3':[]}

    #     # 根据参数设置本地更新的优化器
    #     if self.args.optimizer == 'sgd':
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
    #     elif self.args.optimizer == 'adam':
    #         optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

    #     # 进行多个本地训练周期
    #     for iter in range(self.args.train_ep):
    #         batch_loss = {'total':[], '1':[], '2':[], '3':[]}
    #         agg_protos_label = {}
    #         for batch_idx, (images, label_g) in enumerate(self.trainloader):
    #             images, labels = images.to(self.device), label_g.to(self.device)

    #             # 计算交叉熵损失和原型距离损失
    #             model.zero_grad()
    #             log_probs, protos = model(images)
    #             loss1 = self.criterion(log_probs, labels)

    #             loss_mse = nn.MSELoss()
    #             if len(global_protos) == 0:
    #                 loss2 = 0 * loss1
    #             else:
    #                 proto_new = copy.deepcopy(protos.data)
    #                 i = 0
    #                 for label in labels:
    #                     if label.item() in global_protos.keys():
    #                         proto_new[i, :] = global_protos[label.item()][0].data
    #                     i += 1
    #                 loss2 = loss_mse(proto_new, protos)

    #             # 计算总损失并进行反向传播和优化
    #             loss = loss1 + loss2 * args.ld
    #             loss.backward()
    #             optimizer.step()

    #             # 聚合原型标签
    #             for i in range(len(labels)):
    #                 if label_g[i].item() in agg_protos_label:
    #                     agg_protos_label[label_g[i].item()].append(protos[i,:])
    #                 else:
    #                     agg_protos_label[label_g[i].item()] = [protos[i,:]]

    #             # 计算准确率
    #             log_probs = log_probs[:, 0:args.num_classes]
    #             _, y_hat = log_probs.max(1)
    #             acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

    #             # 打印训练进度信息
    #             if self.args.verbose and (batch_idx % 10 == 0):
    #                 print('| 全局轮次 : {} | 用户: {} | 本地周期 : {} | [{}/{} ({:.0f}%)]\t损失: {:.3f} | 准确率: {:.3f}'.format(
    #                     global_round, idx, iter, batch_idx * len(images),
    #                     len(self.trainloader.dataset),
    #                     100. * batch_idx / len(self.trainloader),
    #                     loss.item(),
    #                     acc_val.item()))
    #             batch_loss['total'].append(loss.item())
    #             batch_loss['1'].append(loss1.item())
    #             batch_loss['2'].append(loss2.item())
    #         epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
    #         epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
    #         epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

    #     # 计算平均损失
    #     epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
    #     epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
    #     epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
    #     return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label
    def update_weights_het(self, args, idx, global_protos, model, global_round):
        model.train()
        epoch_loss = {'total': [], '1': [], '2': []}  # 保持原有键名

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # 动态损失权重
        current_ld = args.ld * (0.5 + 0.5 * math.cos(math.pi * global_round / args.rounds))

        sample_counts = {}
        agg_protos_label = {}

        for iter in range(args.train_ep):
            batch_loss = {'total': [], '1': [], '2': []}

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)

                # 分类损失 -> loss['1']
                loss1 = self.criterion(log_probs, labels)
                batch_loss['1'].append(loss1.item())

                # 原型对齐损失 -> loss['2']
                loss2 = 0.0
                if global_protos:
                    for i, label in enumerate(labels):
                        if label.item() in global_protos:
                            loss2 += F.mse_loss(protos[i], global_protos[label.item()][0])
                    if len(labels) > 0:
                        loss2 /= len(labels)  # 平均损失

                batch_loss['2'].append(loss2)
                total_loss = loss1 + current_ld * loss2
                batch_loss['total'].append(total_loss.item())

                total_loss.backward()
                optimizer.step()

                # 记录原型（简化版）
                for i, label in enumerate(labels):
                    if label.item() not in agg_protos_label:
                        agg_protos_label[label.item()] = []
                    agg_protos_label[label.item()].append(protos[i].detach().clone())

            # 计算epoch平均损失
            for k in batch_loss:
                epoch_loss[k].append(sum(batch_loss[k]) / len(batch_loss[k]))

        # 聚合原型
        mean_protos = {
            label: torch.stack(proto_list).mean(dim=0) 
            for label, proto_list in agg_protos_label.items()
        }

        return (
            model.state_dict(),
            {'total': sum(epoch_loss['total'])/len(epoch_loss['total']),
             '1': sum(epoch_loss['1'])/len(epoch_loss['1']),
             '2': sum(epoch_loss['2'])/len(epoch_loss['2'])},  # 保持键名结构
            torch.mean(torch.tensor([acc for acc in epoch_loss.get('acc', [0.0])])).item(),
            mean_protos,
            {label: len(proto_list) for label, proto_list in agg_protos_label.items()}
        )
    def inference(self, model):
        """ 返回模型的推理准确率和损失。

        将模型设置为评估模式，并遍历测试数据集以计算累计损失和准确率，最后返回整体准确率和损失。

        参数:
        - model: 用于推理的模型，该模型已经经过训练。

        返回:
        - accuracy: 模型在测试数据集上的准确率。
        - loss: 模型在测试数据集上的累计损失。
        """

        # 将模型设置为评估模式
        model.eval()
        # 初始化损失、总样本数和正确预测数为0
        loss, total, correct = 0.0, 0.0, 0.0

        # 遍历测试数据集
        for batch_idx, (images, labels) in enumerate(self.testloader):
            # 将图像和标签移动到指定设备
            images, labels = images.to(self.device), labels.to(self.device)

            # 推理
            outputs = model(images)
            # 计算该批次的损失并加到总损失中
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # 预测
            # 获取预测标签
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            # 计算该批次的正确预测数并加到总正确预测数中
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            # 将该批次的样本数加到总样本数中
            total += len(labels)

        # 计算整体准确率
        accuracy = correct / total
        # 返回整体准确率和总损失
        return accuracy, loss

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        """
        初始化类的构造函数。
    
        参数:
        - args: 包含程序配置和设置的对象。
        - dataset: 数据集，用于模型的训练和测试。
        - idxs: 指定数据集中的索引列表，用于创建测试集。
    
        该构造函数主要负责初始化类的各种属性，包括配置参数、测试数据加载器、设备设置和损失函数。
        """
        # 保存配置参数
        self.args = args
        
        # 调用test_split方法，根据给定的数据集和索引创建测试数据加载器
        self.testloader = self.test_split(dataset, list(idxs))
        
        # 保存设备设置，用于后续模型和数据的设备加载
        self.device = args.device
        
        # 初始化损失函数，并将其移动到指定设备
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
        # 创建测试集索引，这里使用了所有传入的索引
        idxs_test = idxs[:int(1 * len(idxs))]
    
        # 创建测试数据加载器
        # DatasetSplit 类假定为一个自定义的数据集类，它根据给定的索引从数据集中提取样本
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=64, shuffle=False)
        # 返回测试数据加载器
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
        # 将模型设置为评估模式
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        # 遍历测试数据集
        for batch_idx, (images, labels) in enumerate(self.testloader):
            # 将图像和标签移动到指定设备
            images, labels = images.to(self.device), labels.to(self.device)
            # 清除上一批次的梯度
            model.zero_grad()
            # 前向传播：通过将输入传递给模型来计算预测输出
            outputs, protos = model(images)
            # 计算预测输出与真实标签之间的损失
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # 修改输出以匹配预测的类别数
            outputs = outputs[:, 0:args.num_classes]
            # 获取预测标签
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            # 计算正确预测的数量
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        # 计算准确率
        acc = correct / total

        return loss, acc

def fine_tune(self, args, dataset, idxs, model):
    """
    对模型进行微调。

    该函数在数据集的一个子集上对模型进行微调。首先创建数据加载器，根据提供的参数设置损失函数和优化器，
    然后对模型进行指定轮数的训练。

    参数:
        args (object): 包含训练配置参数的对象，包括使用的设备、优化器类型、学习率等。
        dataset (object): 数据集对象。
        idxs (list): 要使用的数据样本的索引列表。
        model (object): 要进行微调的模型对象。

    返回:
        dict: 微调后模型的状态字典。
    """

    trainloader = self.test_split(dataset, list(idxs))
    device = args.device
    criterion = nn.NLLLoss().to(device)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for i in range(args.ft_round):
        for batch_idx, (images, label_g) in enumerate(trainloader):
            images, labels = images.to(device), label_g.to(device)

            # 计算损失
            model.zero_grad()
            log_probs, protos = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def test_inference(args, model, test_dataset, global_protos):
    """ 返回测试准确率和损失。

    参数:
    args (namespace): 包含实验配置信息，包括使用的设备。
    model: 用于测试的模型。
    test_dataset: 测试数据集。
    global_protos: 全局原型，此函数中未使用。

    返回:
    accuracy (float): 测试准确率。
    loss (float): 测试损失。
    """

    # 将模型设置为评估模式
    model.eval()

    # 初始化总损失、样本总数和正确预测数为0
    loss, total, correct = 0.0, 0.0, 0.0

    # 获取用于实验的设备
    device = args.device

    # 定义损失函数并将其移动到指定设备
    criterion = nn.NLLLoss().to(device)

    # 创建测试数据集的数据加载器，批量大小为128，不进行打乱
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    # 遍历测试数据集进行推理
    for batch_idx, (images, labels) in enumerate(testloader):
        # 将图像和标签移动到指定设备
        images, labels = images.to(device), labels.to(device)

        # 模型推理
        outputs, protos = model(images)
        # 计算当前批次的损失并累加到总损失中
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # 预测
        # 获取预测标签并计算正确的预测数量
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    # 计算准确率
    accuracy = correct / total
    return accuracy, loss

def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ 返回测试的损失和准确率。

    参数:
        args: 包含全局参数的对象，如设备信息和用户数量。
        local_model_list: 本地模型列表，用于测试推理。
        test_dataset: 测试数据集。
        classes_list: 每个用户的数据类别列表。
        global_protos: 全局原型集合（默认为空）。

    返回:
        loss: 测试的总损失。
        acc: 测试的准确率。
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        # 初始化输出张量和计数器
        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i] += 1
        for i in range(10):
            if cnt[i] != 0:
                outputs[:, i] = outputs[:,i] / cnt[i]

        # 计算批量损失
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # 预测标签
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    # 计算准确率
    acc = correct / total

    return loss, acc

def test_inference_new_cifar(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ 返回测试的损失和准确率。

    参数:
        args: 包含全局参数的对象，如设备信息和用户数量。
        local_model_list: 本地模型列表，用于测试推理。
        test_dataset: 测试数据集。
        classes_list: 每个用户的数据类别列表。
        global_protos: 全局原型集合（默认为空）。

    返回:
        loss: 测试的总损失。
        acc: 测试的准确率。
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        # 初始化输出张量和计数器
        outputs = torch.zeros(size=(images.shape[0], 100)).to(device)  # outputs 64*100
        cnt = np.zeros(100)
        for i in range(100):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i] += 1
        for i in range(100):
            if cnt[i] != 0:
                outputs[:, i] = outputs[:,i] / cnt[i]

        # 计算批量损失
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # 预测标签
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    # 计算准确率
    acc = correct / total

    return loss, acc


def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ 返回测试的准确率。

    参数:
        args: 包含全局参数的对象，如设备信息和用户数量。
        local_model_list: 本地模型列表，用于测试推理。
        test_dataset: 测试数据集。
        global_protos: 全局原型集合（默认为空）。

    返回:
        acc: 测试的准确率。
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            _, protos = model(images)
            protos_list.append(protos)

        # 原型集成
        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(10):
                if j in global_protos.keys():
                    dist = loss_mse(ensem_proto[i,:], global_protos[j][0])
                    outputs[i,j] = dist

        # 预测标签
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    # 计算准确率
    acc = correct / total

    return acc

# def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
#     """ 返回每个用户的本地测试准确率、使用全局原型的测试准确率和损失列表。

#     参数:
#         args: 包含全局参数的对象，如设备信息、用户数量和类别数量。
#         local_model_list: 本地模型列表，用于测试推理。
#         test_dataset: 测试数据集。
#         classes_list: 每个用户的数据类别列表。
#         user_groups_gt: 每个用户的测试数据索引。
#         global_protos: 全局原型集合（默认为空）。

#     返回:
#         acc_list_l: 每个用户的本地测试准确率列表。
#         acc_list_g: 每个用户使用全局原型的测试准确率列表。
#         loss_list: 每个用户的损失列表。
#     """
#     loss, total, correct = 0.0, 0.0, 0.0
#     loss_mse = nn.MSELoss()

#     device = args.device
#     criterion = nn.NLLLoss().to(device)

#     acc_list_g = []
#     acc_list_l = []
#     loss_list = []
#     for idx in range(args.num_users):
#         model = local_model_list[idx]
#         model.to(args.device)
#         testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

#         # 测试本地模型
#         model.eval()
#         for batch_idx, (images, labels) in enumerate(testloader):
#             images, labels = images.to(device), labels.to(device)
#             model.zero_grad()
#             outputs, protos = model(images)

#             batch_loss = criterion(outputs, labels)
#             loss += batch_loss.item()

#             # 预测标签
#             _, pred_labels = torch.max(outputs, 1)
#             pred_labels = pred_labels.view(-1)
#             correct += torch.sum(torch.eq(pred_labels, labels)).item()
#             total += len(labels)

#         acc = correct / total
#         print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
#         acc_list_l.append(acc)

#         # 使用全局原型进行测试
#         if global_protos != []:
#             correct, total = 0.0, 0.0  # 重置正确预测和总数
#             for batch_idx, (images, labels) in enumerate(testloader):
#                 images, labels = images.to(device), labels.to(device)
#                 model.zero_grad()
#                 outputs, protos = model(images)

#                 # 计算原型与全局原型之间的距离
#                 a_large_num = 100
#                 dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # 初始化距离矩阵
#                 for i in range(images.shape[0]):
#                     for j in range(args.num_classes):
#                         if j in global_protos.keys() and j in classes_list[idx]:
#                             d = loss_mse(protos[i, :], global_protos[j][0])
#                             dist[i, j] = d

#                 # 预测标签
#                 _, pred_labels = torch.min(dist, 1)
#                 pred_labels = pred_labels.view(-1)
#                 correct += torch.sum(torch.eq(pred_labels, labels)).item()
#                 total += len(labels)

#                 # 计算损失
#                 proto_new = copy.deepcopy(protos.data)
#                 i = 0
#                 for label in labels:
#                     if label.item() in global_protos.keys():
#                         proto_new[i, :] = global_protos[label.item()][0].data
#                     i += 1
#                 loss2 = loss_mse(proto_new, protos)
#                 if args.device == 'cuda':
#                     loss2 = loss2.cpu().detach().numpy()
#                 else:
#                     loss2 = loss2.detach().numpy()

#             acc = correct / total
#             print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
#             acc_list_g.append(acc)
#             loss_list.append(loss2)

#     return acc_list_l, acc_list_g, loss_list

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []

    print(f"Total users: {args.num_users}")
    print(f"Global protos keys: {global_protos.keys() if global_protos else 'None'}")

    for idx in range(args.num_users):
        print(f"Processing user {idx}...")
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        # 检查测试数据是否为空
        if len(user_groups_gt[idx]) == 0:
            print(f"Warning: User {idx} has no test data.")
            continue

        # 测试本地模型
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # 预测标签
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)

        # 使用全局原型进行测试
        if global_protos != []:
            correct, total = 0.0, 0.0  # 重置正确预测和总数
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs, protos = model(images)

                # 计算原型与全局原型之间的距离
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # 初始化距离矩阵
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # 预测标签
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # 计算损失
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()

            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list


def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ 计算测试准确率和损失，并保存模型的原型向量及其对应的标签。

    该函数对每个用户的模型在其各自的测试数据集上进行评估，计算整体的测试准确率和损失，
    并保存模型的原型向量及其对应的标签，以便后续分析或可视化。

    参数:
    args (object): 包含实验配置参数的对象，包括使用的设备、算法名称等。
    local_model_list (list): 包含所有用户本地模型的列表。
    test_dataset (object): 全局测试数据集。
    user_groups_gt (dict): 包含每个用户测试数据索引的字典。

    返回值:
    None
    """
    # 初始化损失、总数和正确预测数
    loss, total, correct = 0.0, 0.0, 0.0

    # 设置设备和损失函数
    device = args.device
    criterion = nn.NLLLoss().to(device)

    # 创建一个字典来存储每个用户的原型向量及其对应的标签
    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # 预测
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            # 将每个样本的原型向量按标签分类存储
            for i in range(len(labels)):
                label = labels[i].item()
                if label in agg_protos_label[idx]:
                    agg_protos_label[idx][label].append(protos[i, :])
                else:
                    agg_protos_label[idx][label] = [protos[i, :]]

    # 准备保存的数据
    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    # 转换为 numpy 数组并保存
    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("成功保存原型向量和标签。")

def test_inference_new_het_cifar(args, local_model_list, test_dataset, global_protos=[]):
    """ 返回测试准确率和损失。

    该函数用于测试异构CIFAR模型的推理过程。通过比较模型的预测结果与测试数据集的实际标签，计算测试准确率和损失。

    参数:
    args (argparse.Namespace): 包含各种配置参数的对象。
    local_model_list (list): 本地模型列表。
    test_dataset (torch.utils.data.Dataset): 测试数据集。
    global_protos (dict): 全局原型字典，默认为空列表。

    返回:
    float: 测试准确率。
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        # 初始化输出矩阵为一个较大的数
        a_large_num = 1000
        outputs = a_large_num * torch.ones(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(100):
                if j in global_protos.keys():
                    # 计算原型之间的均方误差
                    dist = loss_mse(protos[i,:], global_protos[j][0])
                    outputs[i,j] = dist

        # 获取预测标签
        _, pred_labels = torch.topk(outputs, 5)
        for i in range(pred_labels.shape[1]):
            correct += torch.sum(torch.eq(pred_labels[:,i], labels)).item()
        total += len(labels)

        cnt += 1
        if cnt == 20:
            break

    acc = correct / total

    return acc