# -*- coding: utf-8 -*-
"""
Установить библиотеки:
pip install numpy
pip install -U scikit-learn
pip install torchvision
# cuda 10.0 windows
pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas
pip install opencv-python
pip install torchsummary
pip install efficientnet_pytorch
pip install matplotlib

git clone https://github.com/davidtvs/pytorch-lr-finder

# Если файлы лежат в архиве rar
pip install patool
import patoolib
#patoolib.extract_archive("PATH/slag.rar", outdir="PATH/")
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics
from torchvision import transforms

import cv2
import os
import torchvision
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from torchsummary import summary
from efficientnet_pytorch import EfficientNet
from tqdm.autonotebook import tqdm


PATH_train = 'D:\Работа\slag\slag\slag_train'
PATH_test = 'D:\Работа\slag\slag\slag_test/'

# parameters
batch_size = 8
im_size = 350


# функция для нормализации изображений
def normalization_parameter(dataloader):
    print("Нормализация изображений")
    print()
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    for data, _ in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(), std.numpy()


# Настройка трансформации изображения
train_transforms = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor()])

# Загружаем и трансфармируем изображение
train_data = torchvision.datasets.ImageFolder(root=PATH_train, transform=train_transforms)
#
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
mean, std = normalization_parameter(train_loader)

# преобразования изображений для обучающих и тестовых данных
train_transforms = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.RandomRotation(degrees=10),
    transforms.CenterCrop(size=299),  # Image net standards
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])
test_transforms = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# обратная нормализация для графика изображения
inv_normalize = transforms.Normalize(
    mean=-1 * np.divide(mean, std),
    std=1 / std)


def data_loader(train_data, test_data=None, valid_size=None, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if (test_data == None and valid_size == None):
        dataloaders = {'train': train_loader}
        return dataloaders
    if (test_data == None and valid_size != None):
        data_len = len(train_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(valid_size * data_len))
        valid_idx, test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
        dataloaders = {'train': train_loader, 'val': valid_loader}
        return dataloaders
    if (test_data != None and valid_size != None):
        data_len = len(test_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)  # перемешивание массива
        split1 = int(np.floor(valid_size * data_len))
        valid_idx, test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        valid_loader = DataLoader(test_data, batch_size=batch_size, sampler=valid_sampler)
        test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)
        dataloaders = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
        return dataloaders


# data loader
train_data = torchvision.datasets.ImageFolder(root=PATH_train, transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(root=PATH_test, transform=test_transforms)

dataloaders = data_loader(train_data, test_data, valid_size=0.2, batch_size=batch_size)

# label of classes
classes = train_data.classes

# определение словарей энкодера и декодера, содержащих наименование классов
decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i

encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]


# построение случайных изображений из набора данных
def class_plot(data, encoder, inv_normalize=None, n_figures=12):
    n_row = int(n_figures / 4)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=4)
    for ax in axes.flatten():
        a = random.randint(0, len(data))
        (image, label) = data[a]
        label = int(label)
        l = encoder[label]
        if (inv_normalize != None):
            image = inv_normalize(image)

        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis('off')
    plt.show()


class_plot(train_data, encoder, inv_normalize)


# использование модели efficientnet
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = EfficientNet.from_pretrained('efficientnet-b0') # поменять на свои веса
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256, 6)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier().to(device)

criterion = nn.CrossEntropyLoss()

import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


shutil.copy('./pytorch-lr-finder/torch_lr_finder/lr_finder.py', './lr_finder.py')
from lr_finder import LRFinder

optimizer_ft = optim.Adam(classifier.parameters(), lr=0.0000001)
lr_finder = LRFinder(classifier, optimizer_ft, criterion, device=device)
lr_finder.range_test(dataloaders['train'], end_lr=1, num_iter=500)
lr_finder.reset()
lr_finder.plot()


def train(model, dataloaders, criterion, num_epochs=10, lr=0.00001, batch_size=8, patience=None):
    since = time.time()
    model.to(device)
    best_acc = 0.0
    i = 0
    phase1 = dataloaders.keys()
    losses = list()
    acc = list()

    if (patience != None):
        earlystop = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr = lr * 0.8
        if (epoch % 10 == 0):
            lr = 0.0001

        for phase in phase1:
            if phase == ' train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            j = 0
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                data, target = Variable(data), Variable(target)
                data = data.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.LongTensor)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                _, preds = torch.max(output, 1)
                running_corrects = running_corrects + torch.sum(preds == target.data)
                running_loss += loss.item() * data.size(0)
                j = j + 1
                if (phase == 'train'):
                    loss.backward()
                    optimizer.step()

                if batch_idx % 300 == 0:
                    print(
                        '{} Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}\n'.format(
                            phase,
                            epoch,
                            batch_idx * len(data),
                            len(dataloaders[phase].dataset),
                            100. * batch_idx / len(dataloaders[phase]),
                            running_loss / (j * batch_size),
                            running_corrects.double() / (j * batch_size)))

            epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * batch_size)
            epoch_loss = running_loss / (len(dataloaders[phase]) * batch_size)
            if (phase == 'val'):
                earlystop(epoch_loss, model)

            if (phase == 'train'):
                losses.append(epoch_loss)
                acc.append(epoch_acc)
            print(earlystop.early_stop)
        if earlystop.early_stop:
            print("Early stopping")
            model.load_state_dict(torch.load('./checkpoint.pt'))
            break
        print('{} Accuracy: '.format(phase), epoch_acc.item())
    return losses, acc


def test(dataloader):
    running_corrects = 0
    running_loss = 0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    sm = nn.Softmax(dim=1)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = Variable(data), Variable(target)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        classifier.eval()
        output = classifier(data)
        loss = criterion(output, target)
        output = sm(output)
        _, preds = torch.max(output, 1)
        running_corrects = running_corrects + torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        preds = np.reshape(preds, (len(preds), 1))
        target = np.reshape(target, (len(preds), 1))
        data = data.cpu().numpy()

        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            if (preds[i] != target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])

    epoch_acc = running_corrects.double() / (len(dataloader) * batch_size)
    epoch_loss = running_loss / (len(dataloader) * batch_size)
    print(epoch_acc, epoch_loss)
    return true, pred, image, true_wrong, pred_wrong


def error_plot(loss):
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.title("Training loss plot")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.show()


def acc_plot(acc):
    plt.figure(figsize=(10, 5))
    plt.plot(acc)
    plt.title("Training accuracy plot")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()


# To plot the wrong predictions given by model
def wrong_plot(n_figures, true, ima, pred, encoder, inv_normalize):
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures / 3)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=3)
    for ax in axes.flatten():
        a = random.randint(0, len(true) - 1)

        image, correct, wrong = ima[a], true[a], pred[a]
        image = torch.from_numpy(image)
        correct = int(correct)
        c = encoder[correct]
        wrong = int(wrong)
        w = encoder[wrong]
        f = 'A:' + c + ',' + 'P:' + w
        if inv_normalize != None:
            image = inv_normalize(image)
        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(f)
        ax.axis('off')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100,
                                                                         recall * 100,
                                                                         accuracy * 100,
                                                                         f1_score * 100))


def train_model(model, dataloaders, criterion, num_epochs=10, lr=0.0001, batch_size=8,
                patience=None, classes=None):
    dataloader_train = {}
    losses = list()
    accuracy = list()
    key = dataloaders.keys()
    for phase in key:
        if (phase == 'test'):
            perform_test = True
        else:
            dataloader_train.update([(phase, dataloaders[phase])])
    losses, accuracy = train(model, dataloader_train, criterion, num_epochs, lr, batch_size,
                             patience)
    error_plot(losses)
    # вылетает ошибка
    #acc_plot(accuracy)
    if (perform_test == True):
        true, pred, image, true_wrong, pred_wrong = test(dataloaders['test'])
        wrong_plot(12, true_wrong, image, pred_wrong, encoder, inv_normalize)
        performance_matrix(true, pred)
        if (classes != None):
            plot_confusion_matrix(true, pred, classes=classes,
                                  title='Confusion matrix, without normalization')


train_model(classifier, dataloaders, criterion, 4, patience=3, batch_size=batch_size,
            classes=classes)








# Predict
from PIL import Image
import numpy as np
import cv2


def fast_predict(model, image, device, encoder, transforms=None, inv_normalize=None):
    # model = torch.load('./model.h5')
    model.eval()

    if (isinstance(image, np.ndarray)):
        image = Image.fromarray(image)

    if (transforms != None):
        image = transforms(image)

    data = image.expand(1, -1, -1, -1)
    data = data.type(torch.FloatTensor).to(device)
    sm = nn.Softmax(dim=1)
    output = model(data)
    output = sm(output)
    _, preds = torch.max(output, 1)
    output = output.cpu().detach().numpy()
    a = output.argsort()
    a = a[0][3:6]  # тут может быть ошибка, или показывать
    size = len(a)

    if (size > 5):
        a = np.flip(a[-5:])

    else:
        a = np.flip(a[-1 * size:])

    return a[0]


def predict(model, image, device, encoder, transforms=None, inv_normalize=None):
    # model = torch.load('./model.h5')
    model.eval()

    if (isinstance(image, np.ndarray)):
        image = Image.fromarray(image)

    if (transforms != None):
        image = transforms(image)

    data = image.expand(1, -1, -1, -1)
    data = data.type(torch.FloatTensor).to(device)
    sm = nn.Softmax(dim=1)
    output = model(data)
    output = sm(output)
    _, preds = torch.max(output, 1)
    img_plot(image, inv_normalize)
    prediction_bar(output, encoder)
    return preds


def prediction_bar(output, encoder):
    output = output.cpu().detach().numpy()
    a = output.argsort()
    a = a[0][3:6]  # тут может быть ошибка, или показывать
    # не все классы, но так работает у меня

    size = len(a)
    if (size > 5):
        a = np.flip(a[-5:])
    else:
        a = np.flip(a[-1 * size:])
    prediction = list()
    clas = list()
    for i in a:
        prediction.append(float(output[:, i] * 100))
        clas.append(str(i))
    for i in a:
        # print(encoder[int(i)])
        # print(float(output[:,i]*100))
        print('Class: {} , confidence: {}'.format(encoder[int(i)], float(output[:, i] * 100)))
    plt.bar(clas, prediction)
    plt.title("Confidence score bar graph")
    plt.xlabel("Confidence score")
    plt.ylabel("Class number")


def img_plot(image, inv_normalize=None):
    if (inv_normalize != None):
        image = inv_normalize(image)
    image = image.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()


import os

PATH_test = 'D:\Работа\slag\slag\slag_test/'
PATH_pred = 'D:\Работа\slag\slag\slag_pred/'


# функция возвращает рандомное имя файла лежащего по указанному в аргументе пути
# если flag_clas=True, то в директории находятся папки с классами, которые содержат изображения
def choise_random_file(path_crf, flag_clas=True):
    path = ''
    clas = None
    path = path_crf

    if flag_clas:
        class_list = os.listdir(path_crf)  # список всех папок(классы)
        rand_num = np.random.randint(0,
                                     len(class_list))  # выбор случайного номера из списка классов
        clas = class_list[rand_num]  # сохранение имени класса
        path = path + clas + '/'  # путь с выбранной папкой

    files = os.listdir(path)  # выбор случайного номера из списка файлов
    rand_num = np.random.randint(0, len(files))  # выбор случайного номера из списка файлов
    one_file = files[rand_num]  # сохранение имени файла
    path += one_file  # добавление имени файла к пути

    return path, clas


import re

# выбрать рандомный файл по указанному пути
# в папке нет подпапок обозночающих класс, поэтому flag_clas=False
random_file = choise_random_file(PATH_pred, flag_clas=False)

print('Файл: {}\n\n'.format(random_file[0]))

video_number = re.split(r'-', random_file[0])
clas_digit = int(video_number[2][0])

image = cv2.imread(random_file[0])
print('Истинный класс: {}\n'.format(encoder[clas_digit]))

# предсказать класс на изображении
pred = predict(classifier, image, device, encoder, test_transforms, inv_normalize)

# выбрать рандомный файл по указанному пути
random_file = choise_random_file(PATH_test)

print('Файл: {}\n\n'.format(random_file[0]))

image = cv2.imread(random_file[0])

print('Истинный класс: ', random_file[1])

# предсказать класс на изображении
pred = predict(classifier, image, device, encoder, test_transforms, inv_normalize)

# сравнение реального класса с предсказанным
print(encoder[pred.tolist()[0]] == random_file[1])

accuracy = {0: [0, 0], 1: [0, 0], 2: [0, 0]}

for dir in os.listdir(PATH_test):
    for files in os.listdir(PATH_test + dir + '/'):
        one_file = PATH_test + dir + '/' + files

        image = cv2.imread(one_file)
        clas = fast_predict(classifier, image, device, encoder, test_transforms, inv_normalize)

        if encoder[clas] == dir:
            accuracy[clas][0] += 1
            accuracy[clas][1] += 1
        else:
            accuracy[clas][1] += 1
            print(files, encoder[clas], dir)

accuracy

y_true = []
y_pred = []

for dir in os.listdir(PATH_test):
    for files in os.listdir(PATH_test + dir + '/'):
        one_file = PATH_test + dir + '/' + files

        image = cv2.imread(one_file)
        clas = fast_predict(classifier, image, device, encoder, test_transforms, inv_normalize)
        y_true.append(decoder[dir])
        y_pred.append(clas)

from sklearn.metrics import precision_recall_curve, classification_report

report = classification_report(y_true, y_pred, target_names=['fluid', 'normal', 'viscous'])
print(report)

for files in os.listdir(PATH_pred):
    one_file = PATH_pred + files

    image = cv2.imread(one_file)
    clas = fast_predict(classifier, image, device, encoder, test_transforms, inv_normalize)

    y_true.append(decoder[dir])
    y_pred.append(clas)
