import pickle
import torch
import torchvision.models as models

model = models.mobilenet_v3_small(num_classes=3)

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
else:
    device = torch.device("cpu")
print(device)

model.load_state_dict(torch.load("./save_point/cnn.pt", map_location=device))
model.eval()

with open('./save_point/norm_data_object.pickle', 'rb') as f:
    norm_data_object = pickle.load(f)

dataloaders, classes, train_transforms, test_transforms, norm_transforms = norm_data_object

# создание загрузчика для тренировочного набора данных
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
train_loader = dataloaders['train']

# создание загрузчика для тестового набора данных
# test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
test_loader = dataloaders['test']
'''
import time

# извлекаем изображения и метки
images, labels = iter(test_loader).next()
image = images[0]
label = labels[0]

# images, labels = images.cuda(), labels.cuda()

# помещаем их на устройство
image = image.to(device)
label = label.to(device)

start_time = time.time()

with torch.no_grad():
    logits = model.forward(image[None, ...])

    ps = torch.exp(logits)
    _, predicted = torch.max(ps,1)

print("--- Время предсказания одного изображения: %s seconds---" % (time.time() - start_time))

print(label)
print(predicted)
'''
import time



with torch.no_grad():
    images, labels = next(iter(test_loader))
    start_time = time.time()
    images = images.to(device, non_blocking=True)
    label = labels.to(device, non_blocking=True)
    print("размер изображения ", images.size())
    predicted = model(images).argmax(1)
    print("--- Время предсказания одного изображения: %s seconds---" % (time.time() - start_time))
    print(label)
    print(predicted)


'''
# для каждого батча в тестовой выборкй
for batch in test_loader:
    # извлекаем изображения и метки
    images, labels = batch

    #images, labels = images.cuda(), labels.cuda()

    # помещаем их на устройство
    images = images.to(device)
    labels = labels.to(device)

    print(type(images))
    print(images)

    # вычисление предсказаний сети
    outputs = model(images)

    # создание тензора предсказаний сети
    _, predicted = torch.max(outputs.data, 1)

    print(predicted)
'''
    
'''
    # корректировка общего значения примеров на величину батча
    total += labels.size(0)
    # корректировка значения верно классифицированных примеров
    correct += (predicted == labels).sum().item()
'''