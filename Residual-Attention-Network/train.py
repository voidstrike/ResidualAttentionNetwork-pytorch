from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
from art.attacks import DeepFool
from art.classifiers import PyTorchClassifier

model_file = 'model_92_sgd.pkl'


def register_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] % (message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


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
        h = model(x)
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


def deepfool_test(model, optimizer, input_shape, nb_classes, test_loader, btrain=False, model_file='last_model_92_sgd.pkl'):
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    loss = nn.CrossEntropyLoss()
    warped_model = PyTorchClassifier(model, loss, optimizer, input_shape, nb_classes, clip_values=(.0, 1.))
    tmp_logger = register_logger()

    tmp_logger.info('Create DeepFool attack')
    adv_crafter = DeepFool(warped_model)

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
            classes[i], 100 * class_correct[i] / class_total[i]))
    return correct / total

    pass


def fgsm_test(model, test_loader, btrain=False, model_file='last_model_92_sgd.pkl'):
    # Test
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    #
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    for images, labels in test_loader:
        images, labels = fgsm_convert(images, labels, model, criterion)
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
    print('Accuracy of the model on the test images:', float(correct) / total)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return correct / total


# for test
def test(model, test_loader, btrain=False, model_file='last_model_92_sgd.pkl'):
    # Test
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
            classes[i], 100 * class_correct[i] / class_total[i]))
    return correct / total


# Image Preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
    # transforms.Scale(224),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.ToTensor()
])
# when image is rgb, totensor do the division 255
# CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='./data/',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = datasets.CIFAR10(root='./data/',
                              train=False,
                              transform=test_transform)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, # 64
                                           shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=20,
                                          shuffle=False,
                                          drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = ResidualAttentionModel().cuda()
print(model)

lr = 0.1  # 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
is_train = False
is_pretrain = False
acc_best = 0
total_epoch = 300
if is_train is True:
    if is_pretrain == True:
        model.load_state_dict((torch.load(model_file)))
    # Training
    for epoch in range(total_epoch):
        model.train()
        tims = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            # print(images.data)
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("hello")
            if (i+1) % 100 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.item()))
        print('the epoch takes time:',time.time()-tims)
        print('evaluate test set:')
        acc = test(model, test_loader, btrain=True)
        if acc > acc_best:
            acc_best = acc
            print('current best acc,', acc_best)
            torch.save(model.state_dict(), model_file)
        # Decaying Learning Rate
        if (epoch+1) / float(total_epoch) == 0.3 or (epoch+1) / float(total_epoch) == 0.6 or (epoch+1) / float(total_epoch) == 0.9:
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
    deepfool_test(model, optimizer, (20, 3, 32, 32), 10, test_loader, btrain=False)
    # fgsm_test(model, test_loader, btrain=False)
    # test(model, test_loader, btrain=False)

