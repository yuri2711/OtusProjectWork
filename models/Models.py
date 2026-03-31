import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingCNN(nn.Module):
    def __init__(self, n_features=5, n_classes=2):
        super(TradingCNN, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, n_features), padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), padding=(1, 0))

        # Полносвязные слои (для более сложного анализа)
        self.fc1 = nn.Linear(128 * 10, 256)  # 10 — длина окна после pooling
        self.fc2 = nn.Linear(256, 64)

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