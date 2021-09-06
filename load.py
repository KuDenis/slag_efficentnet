# -*- coding: utf-8 -*-


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


train_data = torch.load('train_data')
test_data = torch.load('test_data')

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
