import argparse
import numpy as np
import os
import csv
import sys
import random
import math
from PIL import Image
# from cvxpy import *

import torch
import foolbox
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# from tools import progress_bar

from model.AttnVGG import RAM_VGG16, RAM_VGG16_v2


parser = argparse.ArgumentParser()
# Directory
parser.add_argument('--data-dir', default='./data/', help='data path')
parser.add_argument('--save-dir', default='./checkpoint/', help='save path')
# Hyper-parameters
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--mu', type=float, default=1, help='Nuclear Norm hyper-param (default: 1)')
parser.add_argument('--svdprob', type=float, default=0.8, help='USVT hyper-param (default: 0.8)')
parser.add_argument('--mask-num', type=int, default=2, help='number of sampled masks (default: 3)')
parser.add_argument('--maskp', type=float, default=0.5, help='probability of mask sampling (default: 0.5)')
parser.add_argument('--startp', type=float, default=0.4, help='start probability of mask sampling (default: 0.5)')
parser.add_argument('--endp', type=float, default=0.6, help='end probability of mask sampling (default: 0.5)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 256)')
parser.add_argument('--epoch', type=int, default=101, help='total epochs (default: 200)')
parser.add_argument('--num_ckpt_steps', type=int, default=10, help='save checkpoint steps (default: 10)')
parser.add_argument('--attack', type=bool, default=True,
                    help='whether use adversarial training/testing (default: True)')
parser.add_argument('--decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epsilon', type=float, default=8, help='The upper bound change of L-inf norm on input pixels')
parser.add_argument('--iter', type=int, default=7, help='The number of iterations for iterative attacks')
# ME parameters
parser.add_argument('--me-channel', type=str, default='concat',
                    choices=['separate', 'concat'],
                    help='handle RGB channels separately as independent matrices, or jointly by concatenating')
parser.add_argument('--me-type', type=str, default='usvt',
                    choices=['usvt', 'softimp', 'nucnorm'],
                    help='method of matrix estimation')
# Utility parameters
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', type=str, default='ResNet18', help='choose model type (default: ResNet18)')
parser.add_argument('--name', type=str, default='advtrain', help='name of the run')

args = parser.parse_args()

# Checkpoint related
START_EPOCH = 0
batch_size = args.batch_size
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'epsilon': args.epsilon / 255.,
    'num_steps': args.iter,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
}

# Normalization param
mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data(train=False):
    data = None
    labels = None
    if train:
        for i in range(1, 6):
            batch = unpickle(args.data_dir + 'cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data, batch[b'data']])
            if i == 1:
                labels = batch[b'labels']
            else:
                labels = np.concatenate([labels, batch[b'labels']])

        data_tmp = data
        labels_tmp = labels
        # repeat n times for different masks
        for i in range(args.mask_num - 1):
            data = np.concatenate([data, data_tmp])
            labels = np.concatenate([labels, labels_tmp])
    else:
        batch = unpickle(args.data_dir + 'cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


class CIFAR10_Dataset(Data.Dataset):
    def __init__(self, train=True, target_transform=None):
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data, self.train_labels = get_data(train)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            self.test_data, self.test_labels = get_data()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        if self.train:
            img = transform_train(img)
        else:
            img = transform_test(img)

        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class usvt(torch.autograd.Function):
    """ME-Net layer with universal singular value thresholding (USVT) approach.
    """

    @staticmethod
    def forward(ctx, input):
        batch_num, c, h, w = input.size()
        output = torch.zeros_like(input).cpu().numpy()

        for i in range(batch_num):
            img = (input[i] * 2 - 1).cpu().numpy()

            if args.me_channel == 'concat':
                img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
                if globe_train:
                    mask = np.random.binomial(1, args.startp + mask_train_cnt*(args.endp-args.startp)/args.mask_num,
                                              h * w * c).reshape(h, w * c)
                else:
                    mask = np.random.binomial(1, random.uniform(args.startp, args.endp), h * w * c).reshape(h, w * c)
                p_obs = len(mask[mask == 1]) / (h * w * c)
                u, sigma, v = np.linalg.svd(img * mask)
                S = np.zeros((h, w))
                for j in range(int(args.svdprob * h)):
                    S[j][j] = sigma[j]
                S = np.concatenate((S, np.zeros((h, w * 2))), axis=1)
                W = np.dot(np.dot(u, S), v) / p_obs
                W[W < -1] = -1
                W[W > 1] = 1
                est_matrix = (W + 1) / 2
                for channel in range(c):
                    output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
            else:
                if globe_train:
                    mask = np.random.binomial(1, args.startp + mask_train_cnt*(args.endp-args.startp)/args.mask_num,
                                              h * w).reshape(h, w)
                else:
                    mask = np.random.binomial(1, random.uniform(args.startp, args.endp), h * w).reshape(h, w)
                p_obs = len(mask[mask == 1]) / (h * w)
                for channel in range(c):
                    u, sigma, v = np.linalg.svd(img[channel] * mask)
                    S = np.zeros((h, w))
                    for j in range(int(args.svdprob * h)):
                        S[j][j] = sigma[j]
                    W = np.dot(np.dot(u, S), v) / p_obs
                    W[W < -1] = -1
                    W[W > 1] = 1
                    output[i, channel] = (W + 1) / 2

        output = output - mean
        output /= std
        output = torch.from_numpy(output).float().to(device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # BPDA, approximate gradients
        return grad_output


class RecosNet(nn.Module):
    """Reconstruction layer.
    It is called by using the 'apply' method of different functions.
    """
    def __init__(self, model):
        super(RecosNet, self).__init__()
        self.model = model

    def forward(self, input):
        x = globals()[args.me_type].apply(input)
        return self.model(x)


class AttackPGD(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Use cross-entropy as loss function.'

    def forward(self, inputs, targets):
        if not args.attack:
            return self.model(inputs), inputs

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.model(x), x


def train(epoch, net, optimizer, criterion, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()
    global globe_train, mask_train_cnt
    globe_train = True
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        mask_train_cnt = math.ceil((batch_idx + 1) / (50000/batch_size))
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, pert_inputs = net(inputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred_idx = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += pred_idx.eq(targets.data).cpu().sum().float()

        # Bar visualization
        print(batch_idx, len(train_loader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss / batch_idx, 100. * correct / total


def test(epoch, net, criterion, test_loader):
    global globe_train
    globe_train = False
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, pert_inputs = net(inputs, targets)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, pred_idx = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += pred_idx.eq(targets.data).cpu().sum().float()

            # Bar visualization
            print(batch_idx, len(test_loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return test_loss / batch_idx, 100. * correct / total


def save_checkpoint(acc, epoch, model):
    print('=====> Saving checkpoint...')
    state = {
        'model': args.model,
        'acc': acc,
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, args.save_dir + args.name + '_epoch' + str(epoch) + '.ckpt')


def adjust_lr(optimizer, epoch):
    lr = args.lr
    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 20
    if epoch >= 150:
        lr /= 40
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def attack_foolbox(RecosNet_model,test_loader):
    fmodel = foolbox.models.PyTorchModel(RecosNet_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
    attack_criteria = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.ProjectedGradientDescentAttack(model=fmodel, criterion=attack_criteria)

    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cpu().numpy()[0], int(targets.cpu().numpy())

        adversarial = attack(inputs.astype(np.float32), targets, epsilon=config['epsilon'],
                             stepsize=config['step_size'], iterations=config['num_steps'])

        if adversarial is None:
            adversarial = inputs.astype(np.float32)

        if np.argmax(fmodel.predictions(adversarial)) == targets:
            correct += 1.

        sys.stdout.write("\rWhite-box attack (toolbox)... Acc: %.3f%% (%d/%d)" %
                         (100. * correct / (batch_idx + 1), correct, batch_idx + 1))
        sys.stdout.flush()

    return 100. * correct / batch_idx


def attack_pgd(net,test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                outputs, pert_inputs = net(inputs, targets)

            _, pred_idx = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += pred_idx.eq(targets.data).cpu().sum().float()

            sys.stdout.write("\rWhite-box pgd attack... Acc: %.3f%% (%d/%d)"
                             % (100. * correct / total, correct, total))
            sys.stdout.flush()

    return 100. * correct / total


if __name__ == '__main__':

    # Data
    print('=====> Preparing data...')
    #
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalization messes with L-inf bounds. Used after ME-Net layer.
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFAR10_Dataset(True, target_transform)
    test_dataset = CIFAR10_Dataset(False, target_transform)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        batch_size = args.batch_size * n_gpu

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=6*n_gpu)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=6*n_gpu)

    # Models
    print('=====> Building model...')
    if args.model == 'ram-vgg16':
        model = RAM_VGG16()
    elif args.model == 'ram-vgg16-v2':
        model = RAM_VGG16_v2()
    else:
        raise Exception('Unsupported Model')

    model = model.to(device)
    RecosNet_model = RecosNet(model)
    net = AttackPGD(RecosNet_model, config)

    if torch.cuda.device_count() > 1:
        print("=====> Use", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)

    if args.resume:
        check_info = torch.load(args.save_dir + args.name + '_epoch' + str(110) + '.ckpt')
        START_EPOCH = check_info['epoch']
        net.load_state_dict(check_info['state_dict'])
        torch.set_rng_state(check_info['rng_state'])

    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('results/log_' + args.name + '.csv')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    for epoch in range(START_EPOCH, args.epoch):
        train_loss, train_acc = train(epoch, net, optimizer, criterion, train_loader)
        test_loss, test_acc = test(epoch, net, criterion, test_loader)
        adjust_lr(optimizer, epoch)

        if epoch % args.num_ckpt_steps == 0:
            save_checkpoint(test_acc, epoch, net)
