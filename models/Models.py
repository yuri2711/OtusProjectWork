import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim, Tensor
import numpy as np
from sklearn.utils import class_weight

METRICS_FILE = 'training_metrics.csv'


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

        # Сверточные слои + BatchNorm для стабилизации обучения
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, n_features), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(256)

        # Полносвязные слои
        # 256 каналов * 3 (временных шага после 3x MaxPool) * 1 (ширина) = 768
        self.fc1 = nn.Linear(768, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)

        # Выходной слой
        self.output = nn.Linear(64, n_classes)  # Классификация

    def forward(self, x):
        # x: [batch_size, 1, window_len, n_features]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))

        # Преобразуем для полносвязных слоёв
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, val_accuracies = [], []

    with open(METRICS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_accuracy', 'lr'])

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
        scheduler.step(avg_loss)

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
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f} | LR: {current_lr:.6f}')

        with open(METRICS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f'{avg_loss:.6f}', f'{accuracy:.6f}', f'{current_lr:.6f}'])

    # Сохранение графика
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(range(1, epochs + 1), train_losses, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val Accuracy', color='tab:blue')
    ax2.plot(range(1, epochs + 1), val_accuracies, color='tab:blue', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.suptitle('Training Progress')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2)
    fig.tight_layout()
    fig.savefig('training_curve.png', dpi=150)
    print(f'Graph saved to training_curve.png')
    print(f'Metrics saved to {METRICS_FILE}')