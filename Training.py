import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import metrics

import shutil

import torch
from torch import nn

from torch.autograd import Variable
import torch.optim as optim

from efficientnet_pytorch import EfficientNet

PATH_train = 'D:\Работа\slag\slag\slag_train'
PATH_test = 'D:\Работа\slag\slag\slag_test/'

with open('./save_point/norm_data_object.pickle', 'rb') as f:
    norm_data_object = pickle.load(f)

dataloaders, classes, train_transforms, test_transforms, norm_transforms = norm_data_object

# определение словарей энкодера и декодера, содержащих наименование классов
decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i

encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]

# parameters
batch_size = 8


# использование модели efficientnet
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
criterion = nn.CrossEntropyLoss()


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
        self.early_stop = True
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


def train(model, dataloaders, criterion, num_epochs=10, lr=0.00001, batch_size=8, patience=None):
    since = time.time()
    model.to(device)
    best_acc = 0.0
    i = 0
    phase1 = dataloaders.keys()
    losses = list()
    acc = list()

    if patience is not None:
        earlystop = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr = lr * 0.8
        if epoch % 10 == 0:
            lr = 0.0001

        for phase in phase1:
            if phase == 'train':
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
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if batch_idx % 300 == 0:
                    print(
                        '{} Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}'.format(phase,
                                                                                             epoch,
                                                                                             batch_idx * len(
                                                                                                 data),
                                                                                             len(
                                                                                                 dataloaders[
                                                                                                     phase].dataset),
                                                                                             100. * batch_idx / len(
                                                                                                 dataloaders[
                                                                                                     phase])
                                                                                             ,
                                                                                             running_loss / (
                                                                                                     j * batch_size),
                                                                                             running_corrects.double() / (
                                                                                                     j * batch_size)))


            epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * batch_size)
            epoch_loss = running_loss / (len(dataloaders[phase]) * batch_size)
            if phase == 'val':
                earlystop(epoch_loss, model)

            if phase == 'train':
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
            if preds[i] != target[i]:
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
def wrong_plot(n_figures, true, ima, pred, encoder, norm_transforms):
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
        if norm_transforms is not None:
            image = norm_transforms(image)
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
        if phase == 'test':
            perform_test = True
        else:
            dataloader_train.update([(phase, dataloaders[phase])])
    losses, accuracy = train(model, dataloader_train, criterion, num_epochs, lr, batch_size,
                             patience)
    error_plot(losses)
    # вылетает ошибка
    #acc_plot(accuracy)
    if perform_test is True:
        true, pred, image, true_wrong, pred_wrong = test(dataloaders['test'])
        #wrong_plot(12, true_wrong, image, pred_wrong, encoder, norm_transforms)
        performance_matrix(true, pred)
        if classes is None:
            plot_confusion_matrix(true, pred, classes=classes,
                                  title='Confusion matrix, without normalization')
    #model.save_weights("./save_point/weights.h0")


def main():
    #shutil.copy('./pytorch-lr-finder/torch_lr_finder/lr_finder.py', './lr_finder.py')
    from lr_finder import LRFinder

    optimizer_ft = optim.Adam(classifier.parameters(), lr=0.0000001)
    lr_finder = LRFinder(classifier, optimizer_ft, criterion, device=device)
    lr_finder.range_test(dataloaders['train'], end_lr=1, num_iter=500)
    lr_finder.reset()
    lr_finder.plot()

    train_model(classifier, dataloaders, criterion, 1, patience=3, batch_size=batch_size,
                classes=classes)

    #with open('./save_point/classifier.pickle', 'wb') as f:
    #    pickle.dump(classifier, f)
    torch.save(classifier, './save_point/classifier.pt')


if __name__ == "__main__":
    main()
