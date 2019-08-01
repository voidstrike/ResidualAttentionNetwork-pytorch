from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from torchvision import transforms, datasets, models
import os
import time
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network_v2 import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
import cv2

import argparse

model_file = 'model_92_sgd.pkl'
_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def fgsm_convert(x_input, y_input, model, criterion):
    eps = list()
    b, c, h, w = x_input.size()

    for _ in range(b):
        while True:
            rand_num = abs(np.random.normal(loc=.0, scale=8, size=None))
            if rand_num <= 16:
                eps.append(float(rand_num) / 255.0)
                break

    for idx in range(b):
        x = Variable(x_input[idx].clone().expand(1, 3, 32, 32).cuda(), requires_grad=True)
        y = Variable(y_input[idx].clone().expand(1).cuda(), requires_grad=False)

        model.zero_grad()
        h, _ = model(x)
        loss = criterion(h, y)

        if x.grad is not None:
            x.grad.data.fill_(0)

        loss.backward()
        # print(x.grad)

        x_adv = x.detach() - eps[idx] * torch.sign(x.grad)
        x_adv = torch.clamp(x_adv, 0, 1)

        if idx == 0:
            adv_all_x = x_adv.clone()
            adv_all_y = y.clone()
        else:
            adv_all_x = torch.cat((adv_all_x, x_adv), 0)
            adv_all_y = torch.cat((adv_all_y, y), 0)

    return adv_all_x, adv_all_y


# for test
def test(model, test_loader, btrain=False, model_file='last_model_92_sgd.pkl'):
    global _classes
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs, _ = model(images)
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


def img_test(model, img, idx=0):
    # original_img = img.squeeze().numpy().transpose(1, 2, 0)
    original_img = img.mul(255).byte()
    original_img_v1 = original_img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    original_img_v2 = original_img_v1
    tmp_img = np.copy(original_img_v1)

    img = Variable(img.cuda())
    _, attn_map_list = model(img)

    for eachKey in attn_map_list:
        current_attn_map = attn_map_list[eachKey].detach()
        heatmap = torch.mean(current_attn_map, dim=1).squeeze()
        heatmap = heatmap.cpu()
        # heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap) - torch.min(heatmap)

        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (32, 32))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_img_v1 = np.concatenate((original_img_v1, heatmap), axis=1)
        tmp_img2 = tmp_img + 0.4 * heatmap
        original_img_v2 = np.concatenate((original_img_v2, tmp_img2), axis=1)

    original_img = np.concatenate((original_img_v1, original_img_v2), axis=0)
    cv2.imwrite('temp_res/map_{}.jpg'.format(idx), original_img)


def fgsm_img_test(model, img, label, criterion, idx=0):
    original_img = img.mul(255).byte()
    original_img_v1 = original_img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    original_img_v2 = original_img_v1
    tmp_img = np.copy(original_img_v1)

    img, label = fgsm_convert(img, label, model, criterion)

    img = Variable(img.cuda())
    _, attn_map_list = model(img)

    for eachKey in attn_map_list:
        current_attn_map = attn_map_list[eachKey].detach()
        heatmap = torch.mean(current_attn_map, dim=1).squeeze()
        heatmap = heatmap.cpu()
        heatmap /= torch.max(heatmap) - torch.min(heatmap)

        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (32, 32))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_img_v1 = np.concatenate((original_img_v1, heatmap), axis=1)
        tmp_img2 = tmp_img + 0.4 * heatmap
        original_img_v2 = np.concatenate((original_img_v2, tmp_img2), axis=1)

    original_img = np.concatenate((original_img_v1, original_img_v2), axis=0)
    cv2.imwrite('temp_res/map_{}.jpg'.format(idx), original_img)


# Image Preprocessing
_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
    # transforms.Scale(224),
    transforms.ToTensor()
])

_test_transform = transforms.Compose([
    transforms.ToTensor()
])


def main(opt):
    # Load dataset
    train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=_transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=_test_transform)

    # Get Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=20, shuffle=False, drop_last=True)

    model = ResidualAttentionModel().cuda()
    print(model)

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
        potential_list = [168, 196, 312, 333, 427, 668, 732, 778, 914, 994, 1141, 1185, 1200, 1323,
                          1327, 1381, 1390, 1537, 1572, 2021, 2045, 2086, 2177, 2263, 2294, 2453, 2642,
                          2657, 2864, 3027, 3181, 3361]
        model.load_state_dict(torch.load(model_file))
        model.eval()

        # for _ in range(100):
        #     idx = random.randint(0, test_loader.dataset.__len__() - 1)
        #     c_img = test_loader.dataset[idx][0].unsqueeze(0)
        #     img_test(model, c_img, idx=idx)
        # test(model, test_loader, btrain=False)
        for idx in potential_list:
            c_img = test_loader.dataset[idx][0].unsqueeze(0)
            # print(type(c_img))
            c_label = torch.tensor(test_loader.dataset[idx][1]).unsqueeze(0)
            # print(c_label)
            fgsm_img_test(model, c_img, c_label,criterion, idx)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='None', help='None|fgsm|BIM|Deepfool|JSMA|CW2|CWI')
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--pretrain', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)