import torch.optim as optim
import pickle

import torch

from torch import nn, Tensor
import torchvision.models as models


# функция, отвечающая за обучение сети
def fit(model,
        optimizer,
        loss_function,
        train_loader,
        test_loader,
        epochs,
        device,
        ):
    # определяем количество батчей в тренировочной выборке
    total_step = len(train_loader)

    # пускаем цикл по эпохам
    for epoch in range(epochs):
        train_loss = 0
        # для каждого батча в тренировочном наборе
        for i, batch in enumerate(train_loader):
            # извлекаем изображения и их метки
            images, labels = batch
            # отправляем их на устройство
            images = images.to(device)
            labels = labels.to(device)

            # вычисляем выходы сети
            outputs = model(images)
            # вычисляем потери на батче
            loss = loss_function(outputs, labels)
            # обнуляем значения градиентов
            optimizer.zero_grad()
            # вычисляем значения градиентов на батче
            loss.backward()
            # корректируем веса
            optimizer.step()

            # корректируем значение потерь на эпохе
            train_loss += loss.item()

            # логируем
            if (i + 1) % 500 == 0:
                print('Эпоха [{}/{}], Шаг [{}/{}], Тренировочные потери: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.data.item()))

    # режим тестирования модели
    # для тестирования вычислять градиенты не обязательно, поэтому оборачиваем код
    # для теста в блок with torch.no_grad()
    with torch.no_grad():
        # заводим начальные значения корректно распознанных примеров и общего количества примеров
        correct = 0
        total = 0
        # для каждого батча в тестовой выборкй
        for batch in test_loader:
            # извлекаем изображения и метки
            images, labels = batch
            # помещаем их на устройство
            images = images.to(device)
            labels = labels.to(device)
            # вычисление предсказаний сети
            outputs = model(images)
            # создание тензора предсказаний сети
            _, predicted = torch.max(outputs.data, 1)
            # корректировка общего значения примеров на величину батча
            total += labels.size(0)
            # корректировка значения верно классифицированных примеров
            correct += (predicted == labels).sum().item()

        # логирование
        print('Точность на тестовом наборе {} %'.format(100 * correct / total))


with open('./save_point/norm_data_object.pickle', 'rb') as f:
    norm_data_object = pickle.load(f)

dataloaders, classes, train_transforms, test_transforms, norm_transforms = norm_data_object

# создание загрузчика для тренировочного набора данных
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
train_loader = dataloaders['train']

# создание загрузчика для тестового набора данных
# test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
test_loader = dataloaders['test']

model = models.mobilenet_v3_small(num_classes=3)
print(model)

# определим функцию оптимизации
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# определим функцию потерь
loss_function = nn.CrossEntropyLoss()

# определим устройство, на котором будет идти обучение
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

# перемещение модели на устройство
model.to(device)

# обучение сверточной сети на наборе данных
epochs = 15

fit(model,
    optimizer,
    loss_function,
    train_loader,
    test_loader,
    epochs,
    device)

# сохранение текущих параметров сети
torch.save(model.state_dict(), "./save_point/model_15epochs.pt")

'''
import torchvision.models as models

# получим для начала модель AlexNet
cnn = models.AlexNet(num_classes=3)
print(cnn)

# инициализируем предобученную модель ResNet50
cnn = models.resnet50(pretrained=True)

# замораживаем слои, используя метод requires_grad()
# в этом случае не вычисляются градиенты для слоев
# сделать это надо для всех параметеров сети
for name, param in cnn.named_parameters():
  param.requires_grad = False


# к различным блокам модели в PyTorch легко получить доступ
# заменим блок классификатора на свой, подходящий для решения
# задачи классификации кошек и собак
cnn.fc = nn.Sequential(
    nn.Linear(cnn.fc.in_features, 500),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 2)
)

# выведем модель
print(cnn)

# использование обученной сверочной нейронной сети для классификаций изображений одежды

# список классов
# labels = train_data.classes
labels = classes

# возьмем пример из тестового набора
image, label = next(iter(test_data))
image = image.to(device)

image = image.unsqueeze(0)

# формируем предсказания
predictions = cnn(image)
prediction = predictions.argmax()
print(predictions)
print("Предсказание: ", labels[prediction])
print("Метка:", labels[label])

"""# Новый раздел

Многие из вышеперечисленных моделей представлены в PyTorch "из коробки" в модуле `torchvision.models`
"""

import torchvision.models as models

# получим для начала модель AlexNet
alexnet = models.AlexNet(num_classes=10)
print(alexnet)

# ...а затем взглянем на VGG16
vgg16 = models.vgg16()
print(vgg16)

# ...и, наконец, на ResNet152
resnet152 = models.resnet152()
print(resnet152)

# необходимые импорты
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

"""Многие из вышеперечисленных моделей доступны с предварительно обученными весовыми параметрами (например, на ImageNet). что открывает широкие возможности для их использования  

Например, их можно сразу использовать по прямому назначению, т. е. использовать для классификации изображений  
"""

from PIL import Image

# возьмем для примера предобученную на наборе данных ImageNet модель ResNet152
resnet152 = models.resnet152(pretrained=True)

# создадим новый объект transform
transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# откроем пример изображения, которого необходимо классифицировать
image = Image.open("/content/30090.jpg")
image = transform(image)
image = image.unsqueeze(0)

# переводим модель в режим теста и формируем предсказания
resnet152.eval()
preds = resnet152(image)
pred = preds.argmax()
print(pred)

import json

imagenet_classes_file = r"/content/drive/MyDrive/Colab Notebooks/imagenet_class_index.json"
with open(imagenet_classes_file) as f:
  labels = json.load(f)

# название предсказанного класса
print(labels[str(int(pred))])

"""Также с помощью предобученнных моделей на PyTorch можно выполнять такую высокоэффективную методику, как **перенос обучения**  

Она заключается в использовании предобученных весов для обучения модели на собственном наборе данных  

Это оправдано тем, что, например, архитектура, обученная на ImageNet, уже очень много знает об изображениях, и в этом случае обучение на собственном наборе данных потребует меньше времени и меньше данных  

Попробуем запустить обучение модели ResNet50 на собственном небольшом наборе данных для классификации кошек и собак
"""

# приготовим данные



train_path = r"/content/car_truck/train"
test_path  = r"/content/car_truck/test"

train_data = dataset.ImageFolder(train_path, transform)
test_data = dataset.ImageFolder(test_path, transform)

print(type(train_data))
print(type(test_data))

print(train_data.classes)
print(test_data.classes)

train_loader_1 = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader_1  = DataLoader(train_data, batch_size=16, shuffle=True)

"""Предобученная модель ResNet50 уже содержит в себе много информации для распознавания и классификации изображений. Поэтому вместо переобучения можно немного изменить ее архитектуру, подстроив под текущий набор данных: заменить блок классификации

В этом случае будет происходить **заморозка** всех слоев слоев и **обучение** нового блока классификации
"""

# инициализируем предобученную модель ResNet50
pretrained_resnet50 = models.resnet50(pretrained=True)

# замораживаем слои, используя метод requires_grad()
# в этом случае не вычисляются градиенты для слоев
# сделать это надо для всех параметеров сети
for name, param in pretrained_resnet50.named_parameters():
  param.requires_grad = False


# к различным блокам модели в PyTorch легко получить доступ
# заменим блок классификатора на свой, подходящий для решения
# задачи классификации кошек и собак
pretrained_resnet50.fc = nn.Sequential(
    nn.Linear(pretrained_resnet50.fc.in_features, 500),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 2)
)

# выведем модель
print(pretrained_resnet50)

# попробуем обучить!

epochs = 10
optimizer = optim.Adam(pretrained_resnet50.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

fit(pretrained_resnet50,
    optimizer,
    loss_function,
    train_loader_1,
    test_loader_1,
    epochs,
    device='cpu')

'''
