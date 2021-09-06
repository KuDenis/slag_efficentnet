import os
import numpy as np
import re
import time
import pickle
import torch
import torchvision.models as models
from PIL import Image
from sklearn.metrics import precision_recall_curve, classification_report


def choise_random_file(path_crf, flag_clas=True):
    path = ''
    clas = None
    path = path_crf

    if flag_clas:
        class_list = os.listdir(path_crf)  # список всех папок(классы)
        rand_num = np.random.randint(0, len(class_list))  # выбор случайного номера из списка
        # классов
        clas = class_list[rand_num]  # сохранение имени класса
        path = path + clas + '\\'  # путь с выбранной папкой

    files = os.listdir(path)  # выбор случайного номера из списка файлов
    rand_num = np.random.randint(0, len(files))  # выбор случайного номера из списка файлов
    one_file = files[rand_num]  # сохранение имени файла
    path += one_file  # добавление имени файла к пути

    return path, clas

PATH_test = r'D:\Work\slag\slag\slag_test\\'
PATH_pred = r"D:\Work\slag\slag\slag_pred\\"

model = models.mobilenet_v3_small(num_classes=3)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device is "cuda":
    model.cuda()

print("Using {} device".format(device))

model.load_state_dict(torch.load("./save_point/model_15epochs.pt", map_location=device))
model.eval()

with open('./save_point/norm_data_object.pickle', 'rb') as f:
    norm_data_object = pickle.load(f)

with open('./save_point/norm_data_object.pickle', 'rb') as f:
    norm_data_object = pickle.load(f)

dataloaders, classes, train_transforms, test_transforms, norm_transforms = norm_data_object

classes = ["fluid", "normal", "viscous",]

delay_list = []
y_true = []
y_pred = []

def test_pred(test_count=100):
    for _ in range(test_count):
        # выбрать рандомный файл по указанному пути
        # в папке нет подпапок обозночающих класс, поэтому flag_clas=False
        random_file = choise_random_file(PATH_pred, flag_clas=False)
        print('Файл: {}'.format(random_file[0]))
        video_number = re.split(r'-', random_file[0])
        clas_digit = int(video_number[2][0])

        with Image.open(random_file[0]) as image:
            with torch.no_grad():
                start_time = time.time()

                image_tensor = test_transforms(image).float()
                input = image_tensor.unsqueeze_(0)
                input = input.to(device)
                input.size()
                pred = model(input).argmax(1).cpu().numpy()

                delay = (time.time() - start_time)
                y_true.append(clas_digit)
                y_pred.append(int(pred))
                print(int(pred))
                print(clas_digit)
                print(f'Predicted: "{classes[int(pred)]}", Actual: "{classes[clas_digit]}"')


            print("--- Время предсказания одного изображения: %s seconds ---\n\n" % delay)

            delay_list.append(delay)

    print("Средняя задержка = ", sum(delay_list[1:])/(test_count-1))

    report = classification_report(y_true, y_pred, target_names=['fluid', 'normal', 'viscous'])
    print(report)


def test_test(test_count=100):
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i
    
    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]

    for _ in range(test_count):
        # выбрать рандомный файл по указанному пути
        # в папке нет подпапок обозночающих класс, поэтому flag_clas=False
        random_file = choise_random_file(PATH_test, flag_clas=True)
        print('Файл: {}\n\n'.format(random_file[0]))
        #video_number = re.split(r'-', random_file[0])
        clas_digit = decoder[random_file[1]]
    
        with Image.open(random_file[0]) as image:
            with torch.no_grad():
                start_time = time.time()
    
                image_tensor = test_transforms(image).float()
                input = image_tensor.unsqueeze_(0)
                input = input.to(device)
                input.size()
                pred = model(input).argmax(1).cpu().numpy()
    
                delay = (time.time() - start_time)
                y_true.append(clas_digit)
                y_pred.append(int(pred))
                print(int(pred))
                print(clas_digit)
                print(f'Predicted: "{classes[int(pred)]}", Actual: "{classes[clas_digit]}"')
    
    
            print("--- Время предсказания одного изображения: %s seconds ---\n\n" % delay)
    
            delay_list.append(delay)
    
    print("Средняя задержка = ", sum(delay_list[1:])/(test_count-1))
    
    from sklearn.metrics import precision_recall_curve, classification_report
    
    report = classification_report(y_true, y_pred, target_names=['fluid', 'normal', 'viscous'])
    print(report)

# test_test(1000)
test_pred(1000)
