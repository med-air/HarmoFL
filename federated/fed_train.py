"""
HarmoFL
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils.dataset import Camelyon17, Prostate
from utils.loss import DiceLoss, JointLoss
from nets.models import DenseNet, UNet

import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import math

def train_perturbation(args, model, data_loader, optimizer, loss_fun, device):
    model.to(device)
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    train_acc = 0.
    segmentation = model.__class__.__name__ == 'UNet'

    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()

        if segmentation:
            train_acc += DiceLoss().dice_coef(output, target).item()
        else:
            total += target.size(0)
            pred = output.data.max(1)[1]
            batch_correct = pred.eq(target.view(-1)).sum().item()
            correct += batch_correct
            if step % math.ceil(len(data_loader)*0.2) == 0:
                print(' [Step-{}|{}]| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(step, len(data_loader), loss.item(), batch_correct/target.size(0)), end='\r')

        loss.backward()
        optimizer.generate_delta(zero_grad=True)
        loss_fun(model(data), target).backward()
        optimizer.step(zero_grad=True)

    loss = loss_all / len(data_loader)
    acc = train_acc/ len(data_loader) if segmentation else correct/total

    model.to('cpu')
    return loss, acc


def test(args, model, data_loader, loss_fun, device):
    model.to(device)
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    test_acc = 0.
    segmentation = model.__class__.__name__ == 'UNet'

    with torch.no_grad():
        for step, (data, target) in enumerate(data_loader):

            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()

            if segmentation:
                test_acc += DiceLoss().dice_coef(output, target).item()
            else:
                total += target.size(0)
                pred = output.data.max(1)[1]
                batch_correct = pred.eq(target.view(-1)).sum().item()
                correct += batch_correct
                if step % math.ceil(len(data_loader)*0.2) == 0:
                    print(' [Step-{}|{}]| Test Acc: {:.4f}'.format(step, len(data_loader), batch_correct/target.size(0)), end='\r')

    loss = loss_all / len(data_loader)
    acc = test_acc/ len(data_loader) if segmentation else correct/total
    model.to('cpu')
    return loss, acc


def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key])
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            if 'running_amp' in key:
                # aggregate at first round only to save communication cost
                server_model.amp_norm.fix_amp = True
                for model in models:
                    model.amp_norm.fix_amp = True
    return server_model, models

def initialize(args):
    train_loaders, test_loaders = [], []
    val_loaders = []
    trainsets, testsets = [], []
    valsets = []
    if args.data == 'camelyon17':
        args.lr = 1e-3
        model = DenseNet(input_shape=[3,96,96]) # Dense121
        loss_fun = nn.CrossEntropyLoss()
        sites = ['1', '2', '3', '4', '5']

        for site in sites:
            trainset = Camelyon17(site=site, split='train', transform=transforms.ToTensor())
            testset = Camelyon17(site=site, split='test', transform=transforms.ToTensor())

            val_len = math.floor(len(trainset)*0.2)
            train_idx = list(range(len(trainset)))[:-val_len]
            val_idx = list(range(len(trainset)))[-val_len:]
            valset   = torch.utils.data.Subset(trainset, val_idx)
            trainset = torch.utils.data.Subset(trainset, train_idx)
            print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
            trainsets.append(trainset)
            valsets.append(valset)
            testsets.append(testset)
    elif args.data=='prostate':
        args.lr = 1e-4
        args.iters = 500
        model = UNet(input_shape=[3, 384, 384])
        loss_fun = JointLoss()
        sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        for site in sites:
            trainset = Prostate(site=site, split='train', transform=transform)
            valset = Prostate(site=site, split='val', transform=transform)
            testset = Prostate(site=site, split='test', transform=transform)

            print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
            trainsets.append(trainset)
            valsets.append(valset)
            testsets.append(testset)

    min_data_len = min([len(s) for s in trainsets])
    for idx in range(len(trainsets)):
        if args.imbalance:
            trainset = trainsets[idx]
            valset = valsets[idx]
            testset = testsets[idx]
        else:
            trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len))))
            valset = valsets[idx]
            testset = testsets[idx]

        train_loaders.append(torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True))
        val_loaders.append(torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False))
        test_loaders.append(torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False))
    return model, loss_fun, sites, trainsets, testsets, train_loaders, val_loaders, test_loaders


if __name__ == '__main__':
    available_datasets = ['camelyon17','prostate']
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch', type = int, default= 128, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local client between communication')
    parser.add_argument('--alpha', type=float, default=0.05, help='The hyper parameter of perturbation in HarmoFL')
    parser.add_argument('--data', type = str, choices=available_datasets, default='camelyon17', help='Different dataset')
    parser.add_argument('--save_path', type = str, default='../checkpoint/', help='path to save the checkpoint')
    parser.add_argument('--test_path', type=str, default='../checkpoint/', help='path to saved model, for testing')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--gpu', type = int, default=0, help = 'gpu device number')
    parser.add_argument('--seed', type = int, default=0, help = 'random seed')
    parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--imbalance', action='store_true', help='do not truncate train data to same length')
    
    args = parser.parse_args()
    args.log = True

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.save_path = '../../harmofl/checkpoint/{}/seed{}'.format(args.data, seed)
    exp_folder = 'HarmoFL_exp'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, 'HarmoFL')

    server_model, loss_fun, datasets, _, _, train_loaders, val_loaders, test_loaders = initialize(args)



    print('# Deive:', device)
    print('# Training Clients:{}'.format(datasets))

    log = args.log
    if log:
        log_path = args.save_path.replace('checkpoint', 'log')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'HarmoFL.log'), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        for k in list(vars(args).keys()):
            logfile.write('{}: {}\n'.format(k, vars(args)[k]))
    # federated client number
    client_num = len(datasets)
    client_weights = [1./client_num for i in range(client_num)]
    # each local client model
    models = [copy.deepcopy(server_model) for idx in range(client_num)]
    if args.test:
        # evaluate performance on testset using model already trained.
        print('Loading snapshots...')
        checkpoint = torch.load(args.test_path, map_location=device)
        server_model.load_state_dict(checkpoint['server_model'], strict=False)
        
        for idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(args, server_model, test_loader, loss_fun, device)
            print('[Client-{}]  Test  Loss: {:.4f}, Test  Acc: {:.4f}'.format(datasets[idx],test_loss,test_acc))
        exit(0)

    best_changed = False
    ''' intialize local client optimizers'''
    from utils.weight_perturbation import WPOptim
    
    if args.data == 'prostate':
        optimizers = [WPOptim(params=models[idx].parameters(), base_optimizer=optim.Adam, lr=args.lr, alpha=args.alpha, weight_decay=1e-4) for idx in range(client_num)]

    else:
        optimizers = [WPOptim(params=models[idx].parameters(), base_optimizer=optim.SGD, lr=args.lr, alpha=args.alpha, momentum=0.9, weight_decay=1e-4) for idx in range(client_num)]

    if args.resume:
        checkpoint = torch.load(SAVE_PATH+'_latest', map_location=device)
        server_model.load_state_dict(checkpoint['server_model'])
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint['server_model'])
            models[client_idx].to(device)

        if 'optim_0' in list(checkpoint.keys()):
            for client_idx in range(client_num):
                optimizers[client_idx].load_state_dict(checkpoint[f'optim_{client_idx}'])
        for client_idx in range(client_num):
            models[client_idx].to('cpu')

        best_epoch, best_acc  = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print(f'Last time best:{best_epoch} acc :{best_acc}')
        print('Resume training from epoch {}'.format(start_iter))

    else:
        best_epoch = 0
        best_acc = [0. for j in range(client_num)]
        start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))
            for client_idx, model in enumerate(models):
                train_loss, train_acc = train_perturbation(args, model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss, train_acc))
                if args.log:
                    logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx], train_loss, train_acc))

        with torch.no_grad():
            # Aggregation      
            server_model, models = communication(args, server_model, models, client_weights)
            # Validation
            val_acc_list = [None for j in range(client_num)]
            print('============== {} =============='.format('Global Validation'))
            if args.log:
                    logfile.write('============== {} ==============\n'.format('Global Validation'))
            for client_idx, model in enumerate(models):
                val_loss, val_acc = test(args, server_model, val_loaders[client_idx], loss_fun, device)
                val_acc_list[client_idx] = val_acc
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                if args.log:
                    logfile.write(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))
                    logfile.flush()
            # Test after each round
            print('============== {} =============='.format('Test'))
            if args.log:
                logfile.write('============== {} ==============\n'.format('Test'))
            for client_idx, datasite in enumerate(datasets):
                _, test_acc = test(args, server_model, test_loaders[client_idx], loss_fun, device)
                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, a_iter, test_acc))
                if args.log:
                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, a_iter, test_acc))

            # Record best acc
            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed=True
                print(' Best Epoch:{}'.format(best_epoch))
                if args.log:
                    logfile.write(' Best Epoch:{}\n'.format(best_epoch))

            if best_changed:
                print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                if args.log: logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
              
                model_dicts = {'server_model': server_model.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter}
                
                for o_idx in range(client_num):
                    model_dicts['optim_{}'.format(o_idx)] = optimizers[o_idx].state_dict()

                torch.save(model_dicts, SAVE_PATH)
                torch.save(model_dicts, SAVE_PATH+'_latest')
                best_changed = False
            else:
                # save the latest model
                print(' Saving the latest checkpoint to {}...'.format(SAVE_PATH))
                if args.log: logfile.write(' Saving the latest checkpoint to {}...\n'.format(SAVE_PATH))
                
                model_dicts = {'server_model': server_model.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter}
                for o_idx in range(client_num):
                    model_dicts['optim_{}'.format(o_idx)] = optimizers[o_idx].state_dict()

                torch.save(model_dicts, SAVE_PATH+'_latest')


    if log:
        logfile.flush()
        logfile.close()
