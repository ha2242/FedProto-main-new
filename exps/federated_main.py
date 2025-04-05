#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import math

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    """联邦原型学习主函数 - 任务异构场景"""
    # 初始化日志记录器
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + 
                                 str(args.ways) + 'w' + str(args.shots) + 's' + 
                                 str(args.stdev) + 'e_' + str(args.num_users) + 
                                 'u_' + str(args.rounds) + 'r')
    
    global_protos = {}  # 使用字典存储全局原型
    idxs_users = np.arange(args.num_users)
    train_loss, train_accuracy = [], []

    # 动态调整原型损失权重
    def get_ld_weight(round):
        return args.ld * (0.5 + 0.5 * math.cos(math.pi * round / args.rounds))

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        local_protos = {}
        proto_stats = {}  # 记录每个客户端的原型统计信息
        
        print(f'\n| Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            # 初始化本地更新
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            
            # 动态调整损失权重
            current_ld = get_ld_weight(round)
            
            # 更新本地模型权重
            w, loss, acc, protos, counts = local_model.update_weights_het(
                args, idx, global_protos, 
                model=copy.deepcopy(local_model_list[idx]), 
                global_round=round
            )
            
            # 记录结果
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = {
                'protos': protos,
                'counts': counts  # 记录每个类别的样本数量
            }
            
            # 记录训练指标
            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        # 更新全局模型权重
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[idx], strict=True)
            local_model_list[idx] = local_model

        # 加权聚合全局原型
        global_protos = proto_aggregation(local_protos, args)
        
        # 计算平均损失
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    # 最终测试评估
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args, local_model_list, test_dataset, 
        classes_list, user_groups_lt, global_protos
    )
    
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(
        np.mean(loss_list), np.std(loss_list)))

    # 保存原型可视化数据
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)

def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    """联邦原型学习主函数 - 模型异构场景"""
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + 
                                 str(args.ways) + 'w' + str(args.shots) + 's' + 
                                 str(args.stdev) + 'e_' + str(args.num_users) + 
                                 'u_' + str(args.rounds) + 'r')
    
    global_protos = {}
    idxs_users = np.arange(args.num_users)
    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        local_protos = {}
        proto_stats = {}

        print(f'\n| Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            
            w, loss, acc, protos, counts = local_model.update_weights_het(
                args, idx, global_protos, 
                model=copy.deepcopy(local_model_list[idx]), 
                global_round=round
            )
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = {
                'protos': protos,
                'counts': counts
            }
            
            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        # 更新本地模型
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[idx], strict=True)
            local_model_list[idx] = local_model

        # 加权聚合全局原型
        global_protos = proto_aggregation(local_protos, args)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    # 最终评估
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args, local_model_list, test_dataset, 
        classes_list, user_groups_lt, global_protos
    )
    
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(
        np.mean(loss_list), np.std(loss_list)))

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    exp_details(args)
    
    # 设置随机种子
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 生成任务分配
    n_list = np.random.randint(
        max(2, args.ways - args.stdev), 
        min(args.num_classes, args.ways + args.stdev + 1), 
        args.num_users
    )
    
    # 根据数据集调整shot范围
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)

    # 加载数据集
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    # 初始化模型
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                args.out_channels = 18 if i<7 else 20 if i<14 else 22
            else:
                args.out_channels = 20
            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                args.out_channels = 18 if i<7 else 20 if i<14 else 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                args.stride = [1,4] if i<10 else [2,2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    initial_weight[key] = initial_weight_1[key]
            local_model.load_state_dict(initial_weight)

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    # 选择训练模式
    if args.mode == 'task_heter':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)