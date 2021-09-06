import pickle
import matplotlib.pyplot as plt

import torch
from torch import nn

from PIL import Image
import numpy as np
import cv2
from efficientnet_pytorch import EfficientNet


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = EfficientNet.from_pretrained('efficientnet-b0')  # поменять на свои веса
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


with open('./save_point/norm_data_object.pickle', 'rb') as f:
    norm_data_object = pickle.load(f)

dataloaders, classes, train_transforms, test_transforms, norm_transforms = norm_data_object


#with open('./save_point/classifier.pickle', 'rb') as f:
#    classifier = pickle.load(f)

# определение словарей энкодера и декодера, содержащих наименование классов
decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i

encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]


def fast_predict(model, image, device, transforms=None):
    #model = torch.load('./save_point/classifier.pt')
    #model.eval()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if transforms is not None:
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

    if size > 5:
        a = np.flip(a[-5:])
    else:
        a = np.flip(a[-1 * size:])

    return a[0]


def predict(model, image, device, encoder, transforms=None, norm_transforms=None):
    #model = torch.load('./save_point/classifier.pt')
    #model.eval()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if transforms is not None:
        image = transforms(image)

    data = image.expand(1, -1, -1, -1)
    data = data.type(torch.FloatTensor).to(device)
    sm = nn.Softmax(dim=1)
    output = model(data)
    output = sm(output)
    _, preds = torch.max(output, 1)
    img_plot(image, norm_transforms)
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


def img_plot(image, norm_transforms=None):
    if norm_transforms is not None:
        image = norm_transforms(image)
    image = image.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()


import os

# функция возвращает рандомное имя файла лежащего по указанному в аргументе пути
# если flag_clas=True, то в директории находятся папки с классами, которые содержат изображения
def choise_random_file(path_crf, flag_clas=True):
    path = ''
    clas = None
    path = path_crf

    if flag_clas:
        class_list = os.listdir(path_crf)  # список всех папок(классы)
        rand_num = np.random.randint(0, len(class_list))  # выбор случайного номера из списка
        # классов
        clas = class_list[rand_num]  # сохранение имени класса
        path = path + clas + '/'  # путь с выбранной папкой

    files = os.listdir(path)  # выбор случайного номера из списка файлов
    rand_num = np.random.randint(0, len(files))  # выбор случайного номера из списка файлов
    one_file = files[rand_num]  # сохранение имени файла
    path += one_file  # добавление имени файла к пути

    return path, clas


import re

PATH_test = r'D:\Work\slag\slag\slag_test\\'
PATH_pred = r"D:\Work\slag\slag\slag_pred\\"


# загрузка модели
classifier = torch.load('./save_point/classifier.pt')
classifier.eval()

import time

# выбрать рандомный файл по указанному пути
# в папке нет подпапок обозночающих класс, поэтому flag_clas=False
random_file = choise_random_file(PATH_pred, flag_clas=False)
print('Файл: {}\n\n'.format(random_file[0]))
video_number = re.split(r'-', random_file[0])
clas_digit = int(video_number[2][0])
image = cv2.imread(random_file[0])
print('Истинный класс: {}\n'.format(encoder[clas_digit]))

start_time = time.time()
# предсказать класс на изображении
pred = fast_predict(classifier, image, device, test_transforms)
print("--- Время предсказания одного изображения: %s seconds (fast_predict)---" % (time.time() -
                                                                                   start_time))

print(pred)

'''
# выбрать рандомный файл по указанному пути
random_file = choise_random_file(PATH_test)
print('Файл: {}\n\n'.format(random_file[0]))
image = cv2.imread(random_file[0])
print('Истинный класс: ', random_file[1])

start_time = time.time()
# предсказать класс на изображении
pred = predict(classifier, image, device, encoder, test_transforms, norm_transforms)
print("--- Время предсказания одного изображения: %s seconds (predict)---" % (time.time() - start_time))

# сравнение реального класса с предсказанным
print(encoder[pred.tolist()[0]] == random_file[1])
'''
'''
# предсказание для всех файлов в папке test
y_true = []
y_pred = []

for dir in os.listdir(PATH_test):
    for files in os.listdir(PATH_test + dir + '/'):
        one_file = PATH_test + dir + '/' + files

        image = cv2.imread(one_file)
        clas = fast_predict(classifier, image, device, encoder, test_transforms, norm_transforms)
        y_true.append(decoder[dir])
        y_pred.append(clas)

from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=['fluid', 'normal', 'viscous'])
print('---Test---')
print(report)


# предсказание для всех файлов в папке pred
for files in os.listdir(PATH_pred):
    one_file = PATH_pred + files

    image = cv2.imread(one_file)
    clas = fast_predict(classifier, image, device, encoder, test_transforms, norm_transforms)

    y_true.append(decoder[dir])
    y_pred.append(clas)

report = classification_report(y_true, y_pred, target_names=['fluid', 'normal', 'viscous'])
print('---Pred---')
print(report)
'''