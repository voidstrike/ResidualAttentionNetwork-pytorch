from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from torchvision import transforms, datasets, models
import os
import time
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
from art.attacks import DeepFool, BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod
from art.attacks import FastGradientMethod, SaliencyMapMethod, ProjectedGradientDescent
from art.classifiers import PyTorchClassifier

from model.AttnVGG import CBAM_VGG16, LTPA_VGG16, RAM_VGG16, RAM_VGG16_v2
from model.AttnVGG import CBAM_VGG19, LTPA_VGG19, RAM_VGG19, RAM_VGG19_v2
from model.AttnResNet_v2 import CBAM_ResidualNet, RAM_ResNet18, LTPA_ResNet18, RAM_ResNet18_v2

import argparse
import math
import random
from PIL import Image

# Global Variables Area

model_file = 'model_92_sgd.pkl'
_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Image Preprocessing
_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32), padding=4),
    transforms.ToTensor()
])

_test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Normalization param
mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_globe_train= False

class CIFAR10_Dataset(Dataset):
    def __init__(self, config, train=True, target_transform=None):
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data, self.train_labels = get_data(config, train)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            self.test_data, self.test_labels = get_data(config)
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
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class CIFAR10_PH(Dataset):
    def __init__(self, features, labels, feature_tfs, label_tfs):
        self.test_data, self.test_labels = features, labels
        self.ftfs = feature_tfs
        self.ltfs = label_tfs

    def __getitem__(self, index):
        img = self.test_data[index]
        label = self.test_labels[index]

        img = Image.fromarray(img)
        img = self.ftfs(img)

        if self.ltfs is not None:
            label = self.ltfs(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        tensor[:, 0, :, :] = tensor[:, 0, :, :].mul(self.std[0]).add_(self.mean[0])
        tensor[:, 1, :, :] = tensor[:, 1, :, :].mul(self.std[1]).add_(self.mean[1])
        tensor[:, 2, :, :] = tensor[:, 2, :, :].mul(self.std[2]).add_(self.mean[2])

        return tensor


class usvt(torch.autograd.Function):
    """ME-Net layer with universal singular value thresholding (USVT) approach.
    """

    @staticmethod
    def forward(ctx, input, config):
        global _globe_train
        global mean, std, device
        batch_num, c, h, w = input.size()
        output = torch.zeros_like(input).cpu().numpy()

        for i in range(batch_num):
            img = (input[i] * 2 - 1).cpu().numpy()

            if config.me_channel == 'concat':
                img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
                if _globe_train:
                    raise NotImplementedError('Training Mode is not supported in this script')
                else:
                    mask = np.random.binomial(1, random.uniform(config.startp, config.endp), h * w * c).reshape(h, w * c)
                p_obs = len(mask[mask == 1]) / (h * w * c)
                u, sigma, v = np.linalg.svd(img * mask)
                S = np.zeros((h, w))
                for j in range(int(config.svdprob * h)):
                    S[j][j] = sigma[j]
                S = np.concatenate((S, np.zeros((h, w * 2))), axis=1)
                W = np.dot(np.dot(u, S), v) / p_obs
                W[W < -1] = -1
                W[W > 1] = 1
                est_matrix = (W + 1) / 2
                for channel in range(c):
                    output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
            else:
                if _globe_train:
                    raise NotImplementedError('Training Mode is not supported in this script')
                else:
                    mask = np.random.binomial(1, random.uniform(config.startp, config.endp), h * w).reshape(h, w)
                p_obs = len(mask[mask == 1]) / (h * w)
                for channel in range(c):
                    u, sigma, v = np.linalg.svd(img[channel] * mask)
                    S = np.zeros((h, w))
                    for j in range(int(config.svdprob * h)):
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
        global c_opt
        x = globals()['usvt'].apply(input, c_opt)
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

    def forward(self, inputs, targets, config):
        if not config.attack:
            return self.model(inputs), inputs

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.model(x), x

# ------------------------------------------------------------------------------------------------


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data(config, train=False):
    data = None
    labels = None
    if train:
        for i in range(1, 6):
            batch = unpickle(config.data_dir + 'cifar-10-batches-py/data_batch_' + str(i))
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
        for i in range(config.mask_num - 1):
            data = np.concatenate([data, data_tmp])
            labels = np.concatenate([labels, labels_tmp])
    else:
        batch = unpickle(config.data_dir + 'cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


def general_test(model, optimizer, input_shape, nb_classes, test_loader, method, btrain=False,
                  model_file='last_model_92_sgd.pkl'):
    global _classes
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    loss = nn.CrossEntropyLoss()
    warped_model = PyTorchClassifier(model, loss, optimizer, input_shape, nb_classes, clip_values=(.0, 1.))
    if method == 'Deepfool':
        adv_crafter = DeepFool(warped_model)
    elif method == 'BIM':
        adv_crafter = BasicIterativeMethod(warped_model, batch_size=20)
    elif method == 'JSMA':
        adv_crafter = SaliencyMapMethod(warped_model, batch_size=20)
    elif method == 'CW2':
        adv_crafter = CarliniL2Method(warped_model, batch_size=20)
    elif method == 'CWI':
        adv_crafter = CarliniLInfMethod(warped_model, batch_size=20)

    correct, total = 0, 0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    for images, labels in test_loader:
        images = adv_crafter.generate(images.numpy())

        images = Variable(torch.from_numpy(images).cuda())
        labels = Variable(labels.cuda())

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        c = (predicted == labels.data).squeeze()
        for i in range(20):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct) / total)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            _classes[i], 100 * class_correct[i] / class_total[i]))
    return correct / total


def general_test_v2(model, optimizer, input_shape, nb_classes, test_loader, method, conf, btrain=False,
                  model_file='last_model_92_sgd.pkl'):
    global _classes
    if not btrain:
        checked_state = torch.load(model_file)['state_dict']
        model.load_state_dict(checked_state)
        assert isinstance(model, AttackPGD), 'Incorrect Model Configuration'
    model = model.model.eval()
    # model.eval()

    loss = nn.CrossEntropyLoss()
    warped_model = PyTorchClassifier(model, loss, optimizer, input_shape, nb_classes, clip_values=(.0, 1.))
    if method == 'Deepfool':
        adv_crafter = DeepFool(warped_model)
    elif method == 'BIM':
        adv_crafter = BasicIterativeMethod(warped_model, batch_size=32)
    elif method == 'JSMA':
        adv_crafter = SaliencyMapMethod(warped_model, batch_size=32)
    elif method == 'CW2':
        adv_crafter = CarliniL2Method(warped_model, batch_size=32)
    elif method == 'CWI':
        adv_crafter = CarliniLInfMethod(warped_model, batch_size=32)
    elif method == 'FGSM':
        adv_crafter = FastGradientMethod(warped_model, batch_size=32)
    elif method == 'PGD':
        adv_crafter = ProjectedGradientDescent(warped_model, batch_size=32)

    correct, total = 0, 0

    adv_dataset = adv_generalization(test_loader, adv_crafter, conf)
    temp_loader = DataLoader(dataset=adv_dataset, batch_size=32, shuffle=False, drop_last=True)
    # temp_loader = test_loader

    for images, labels in temp_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        outputs = model(images, conf)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct) / total)
    return correct / total


# for test
def test(model, test_loader, btrain=False, model_file='last_model_92_sgd.pkl'):
    global _classes
    if not btrain:
        model.load_state_dict(torch.load(model_file)['state_dict'])
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        #
        c = (predicted == labels.data).squeeze()
        for i in range(20):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct)/total)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            _classes[i], 100 * class_correct[i] / class_total[i]))

    model.train()
    return correct / total


def adv_generalization(tgt_dl, adv, conf):
    fout, lout = None, None
    unnorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    # c_mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
    # c_std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)
    # unnorm = transforms.Normalize((-c_mean / c_std).tolist(), (1.0 / c_std).tolist())

    for batch_idx, (images, labels) in enumerate(tgt_dl):
        images = adv.generate(images.numpy())
        images = torch.from_numpy(images) * 255.
        images = images.clamp(0, 255)
        # images = unnorm(images) * 255.0

        if fout is None:
            fout = images
        else:
            fout = torch.cat((fout, images), dim=0)

        if lout is None:
            lout = labels
        else:
            lout = torch.cat((lout, labels), dim=0)

    global _test_transform
    res_data = CIFAR10_PH(fout.numpy().astype('uint8').transpose(0, 2, 3, 1), lout.numpy(),
                          _test_transform, target_transform)

    return res_data


def main(opt):
    global _transform, _test_transform
    global device
    # Load dataset
    train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=_transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=_test_transform)
    # test_dataset = CIFAR10_Dataset(opt, False, target_transform)

    # Get Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, drop_last=True)

    if opt.model == 'cbam-vgg16':
        model = CBAM_VGG16()
    elif opt.model == 'cbam-vgg19':
        model = CBAM_VGG19()
    elif opt.model == 'ram-vgg16':
        model = RAM_VGG16()
    elif opt.model == 'ram-vgg19':
        model = RAM_VGG19()
    elif opt.model == 'ram-vgg16_v2':
        model = RAM_VGG16_v2()
    elif opt.model == 'ram_vgg19_v2':
        model = RAM_VGG19_v2()
    elif opt.model == 'ltpa-vgg16':
        model = LTPA_VGG16()
    elif opt.model == 'ltpa-vgg19':
        model = LTPA_VGG19()
    elif opt.model == 'cbam-resnet':
        model = CBAM_ResidualNet("CIFAR10", 18, 10)
    elif opt.model == 'ram-resnet':
        model = RAM_ResNet18()
    elif opt.model == 'ram-resnet-v2':
        model = RAM_ResNet18_v2()
    elif opt.model == 'ltpa-resnet':
        model = LTPA_ResNet18()
    elif opt.model == 'vgg16':
        model = models.vgg16(pretrained=False)
    elif opt.model == 'vgg19':
        model = models.vgg19(pretrained=False)
    elif opt.model == 'resnet':
        model = models.resnet18(pretrained=False)
    else:
        raise Exception('Unsupported Model')

    # print(model)
    temp_config = {
        'epsilon': opt.epsilon / 255.,
        'num_steps': opt.iter,
        'step_size': 2. / 255,
        'random_start': True,
        'loss_func': 'xent',
    }
    model = model.to(device)
    RecosNet_model = RecosNet(model)
    net = AttackPGD(RecosNet_model, temp_config)

    lr = opt.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)

    is_train, is_pretrain = opt.train, opt.pretrain
    acc_best = 0
    total_epoch = 300

    if is_train is True:
        if is_pretrain:
            model.load_state_dict((torch.load(model_file)))
        # Training
        for epoch in range(total_epoch):
            model.train()
            tims = time.time()
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                    epoch + 1, total_epoch, i + 1, len(train_loader), loss.item()))

            print('the epoch takes time:', time.time() - tims)
            print('evaluate test set:')
            acc = test(model, test_loader, btrain=True)
            if acc > acc_best:
                acc_best = acc
                print('current best acc,', acc_best)
                torch.save(model.state_dict(), model_file)
            # Decaying Learning Rate
            if (epoch + 1) / float(total_epoch) == 0.3 or (epoch + 1) / float(total_epoch) == 0.6 or (
                    epoch + 1) / float(total_epoch) == 0.9:
                lr /= 10
                print('reset learning rate to:', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print(param_group['lr'])
                # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
        # Save the Model
        torch.save(model.state_dict(), 'last_model_92_sgd.pkl')

    else:
        if opt.method == 'None':
            test(model, test_loader, btrain=False, model_file=opt.model_path)
        elif opt.method == 'fgsm':
            general_test_v2(net, optimizer, (32, 3, 32, 32), 10, test_loader, 'FGSM',
                            conf=opt, btrain=False, model_file=opt.model_path)
        elif opt.method == 'Deepfool':
            general_test_v2(model, optimizer, (32, 3, 32, 32), 10, test_loader, 'Deepfool',
                            conf=opt, btrain=False, model_file=opt.model_path)
        elif opt.method == 'BIM':
            general_test_v2(model, optimizer, (32, 3, 32, 32), 10, test_loader, 'BIM',
                            conf=opt, btrain=False, model_file=opt.model_path)
        elif opt.method == 'JSMA':
            general_test_v2(model, optimizer, (32, 3, 32, 32), 10, test_loader, 'JSMA',
                            conf=opt, btrain=False, model_file=opt.model_path)
        elif opt.method == 'CW2':
            general_test_v2(model, optimizer, (32, 3, 32, 32), 10, test_loader, 'CW2',
                            conf=opt, btrain=False, model_file=opt.model_path)
        elif opt.method == 'CWI':
            general_test_v2(model, optimizer, (32, 3, 32, 32), 10, test_loader, 'CWI',
                            conf=opt, btrain=False, model_file=opt.model_path)
        elif opt.method == 'PGD':
            general_test_v2(net, optimizer, (32, 3, 32, 32), 10, test_loader, 'PGD',
                            conf=opt, btrain=False, model_file=opt.model_path)
        else:
            raise Exception("Unsupported Attack Method")
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='None', help='None|fgsm|BIM|Deepfool|JSMA|CW2|CWI')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', default='data/', help='data path')

    # Default Training Setting, placeholder for some parameters only
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--pretrain', type=bool, default=False)

    # Default ME setting, borrow from train_ori
    parser.add_argument('--startp', type=float, default=0.8)
    parser.add_argument('--endp', type=float, default=1)
    parser.add_argument('--me-type', type=str, default='usvt')
    parser.add_argument('--me-channel', type=str, default='concat')
    parser.add_argument('--svdprob', type=float, default=0.8, help='USVT hyper-param (default: 0.8)')
    parser.add_argument('--epsilon', type=float, default=8.)
    parser.add_argument('--iter', type=int, default=7)
    parser.add_argument('--no-augment', dest='augment', action='store_false')

    c_opt = parser.parse_args()

    if c_opt.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                      (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                      (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    main(c_opt)
