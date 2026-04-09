import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim, Tensor
import numpy as np
from sklearn.utils import class_weight


class TradingCNN(nn.Module):
    def __init__(self, n_features=4, n_classes=2):
        """
        Первый сверточный слой:

        self.conv1 = ...: Мы создаем слой и сохраняем его как атрибут модели.
        in_channels=1: На входе ожидается тензор с 1 каналом. В трейдинге это обычно "картинка", где высота — это время (свечи),
            а ширина — признаки (цена, объем). Канал здесь один (черно-белое изображение).
        out_channels=32: Слой создаст 32 новые карты признаков. Каждая карта будет "видеть" свой уникальный паттерн.
        kernel_size=(3, n_features): Размер фильтра (ядра).
        (3): Свертка по времени (берет 3 свечи).
        (n_features): Свертка по всем признакам сразу. Это позволяет фильтру видеть взаимосвязь между ценой и объемом одновременно.
        padding=(1, 0): Добавляет 1 единицу отступа по высоте (времени), чтобы сохранить длину временного ряда. По ширине отступ 0.

        :param n_features: Числовой параметр, описывающий количество признаков (open, high, low, close)...
        :param n_classes: Числовой параметр, описывающий количество классов в классификации (buy, sell)
        """
        super(TradingCNN, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, n_features), padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), padding=(1, 0))

        # Полносвязные слои (для более сложного анализа)
        self.fc1 = nn.Linear(768, 700)  # тут не понял как правильно считать количество нейронов. Сейчас стоит 768, но это я вставляю после ошибки запуска.
        self.fc2 = nn.Linear(700, 64)

        # Выходной слой
        self.output = nn.Linear(64, n_classes)  # Классификация

    def forward(self, x):
        # x: [batch_size, 1, window_len, n_features]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 1))

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 1))

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(2, 1))

        # Преобразуем для полносвязных слоёв
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        out = self.output(x)
        return out

def train_model(model, train_loader, val_loader, target: Tensor, epochs=20):
    """
    Тут все стандартно! Единственное реализовал развесовку классов
    :param model:
    :param train_loader:
    :param val_loader:
    :param target:
    :param epochs:
    :return:
    """
    unique = np.unique(target)
    l = target.tolist()
    weight = class_weight.compute_class_weight(class_weight='balanced', classes=unique, y=l)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Валидация
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        val_accuracies.append(accuracy)

        print(f'Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f}')

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Обучение модели')
    plt.show()