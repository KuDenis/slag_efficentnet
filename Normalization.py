# -*- coding: utf-8 -*-
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm.autonotebook import tqdm


def find_normalization_parameter(PATH, batch_size=8, im_size=350):
    """
    Определение параметров нормализации по тестовым изображениям
    Return:
         mean - среднее значение
         std - стандартное отклонение
    """
    print("Нормализация изображений")

    def normalization_parameter(dataloader):
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

    # Создание трансформации для обучающего набора
    findn_transforms = transforms.Compose([
        transforms.Resize((640, 384)),
        transforms.CenterCrop(300),
        transforms.ToTensor()])

    # Загружаем и трансфармируем изображение
    train_data = torchvision.datasets.ImageFolder(root=PATH, transform=findn_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    mean, std = normalization_parameter(train_loader)

    return mean, std


def my_transforms(mean, std, im_size=350):
    # преобразования изображений для обучающих и тестовых данных
    train_transforms = transforms.Compose([
        transforms.Resize((640, 384)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    # преобразования изображений для обучающих и тестовых данных
    test_transforms = transforms.Compose([
        transforms.Resize((640, 384)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    # обратная нормализация для графика изображения
    norm_transforms = transforms.Normalize(
        mean=-1 * np.divide(mean, std),
        std=1 / std)

    return train_transforms, test_transforms, norm_transforms


# Функция для нормализации изображений
def image_normalization(PATH, transforms):
    data = torchvision.datasets.ImageFolder(root=PATH, transform=transforms)
    return data


# создание словаря
def data_loader(train_data, test_data=None, valid_size=None, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if test_data is None and valid_size is None:
        dataloaders = {'train': train_loader}
        return dataloaders

    if test_data is None and valid_size is not None:
        data_len = len(train_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(valid_size * data_len))
        valid_idx, test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
        dataloaders = {'train': train_loader, 'val': valid_loader}
        return dataloaders

    if test_data is not None and valid_size is not None:
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


# построение случайных изображений из набора данных
def class_plot(data, encoder, norm_transforms=None, n_figures=12):
    n_row = int(n_figures / 4)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=4)
    for ax in axes.flatten():
        a = random.randint(0, len(data) - 1)
        (image, label) = data[a]
        label = int(label)
        l = encoder[label]
        if norm_transforms is not None:
            image = norm_transforms(image)

        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis('off')
    plt.show()


def main(PATH_train, PATH_test):
    # parameters
    batch_size = 8
    im_size = 350

    # определение параметров нормализации
    mean, std = find_normalization_parameter(PATH_train, batch_size, im_size)

    train_transforms, test_transforms, norm_transforms = my_transforms(mean, std, im_size)

    # нормализация изображений
    train_data = image_normalization(PATH_train, train_transforms)
    test_data = image_normalization(PATH_test, test_transforms)

    # Нормализованный массив данных, с структурой DataLoader
    dataloaders = data_loader(train_data, test_data, valid_size=0.2, batch_size=batch_size)

    # подписи классов
    classes = train_data.classes

    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]

    # отобразить случайные изображения
    class_plot(train_data, encoder, norm_transforms, 4)

    return dataloaders, classes, train_transforms, test_transforms, norm_transforms


if __name__ == "__main__":
    PATH_train = r'D:\Work\slag\slag\slag_train'
    PATH_test = r'D:\Work\slag\slag\slag_test'

    dataloaders, classes, train_transforms, test_transforms, norm_transforms = main(PATH_train, PATH_test)

    norm_data_object = (dataloaders, classes, train_transforms, test_transforms, norm_transforms)

    print(dataloaders)

    with open('./save_point/norm_data_object.pickle', 'wb') as f:
        pickle.dump(norm_data_object, f)
