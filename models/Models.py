import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim, Tensor
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import f1_score


class TradingCNN(nn.Module):
    def __init__(self, n_features=4, n_classes=2):
        super(TradingCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, n_features), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(768, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.output(x)


class TradingLSTM(nn.Module):
    """
    LSTM-модель для классификации торговых сигналов.
    Вход: [batch, seq_len, n_features]
    """
    def __init__(self, n_features=12, hidden_size=128, num_layers=2, n_classes=2, dropout=0.3):
        super(TradingLSTM, self).__init__()

        self.bn_input = nn.BatchNorm1d(n_features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # bidirectional -> hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [batch, seq_len, features]
        # BatchNorm expects [batch, features], transpose -> normalize -> transpose back
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.bn_input(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]

        lstm_out, (h_n, _) = self.lstm(x)

        # Берём последний выход LSTM (конкатенация прямого и обратного)
        out = lstm_out[:, -1, :]

        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        return self.output(out)


def train_model(model, train_loader, val_loader, target: Tensor, epochs=20,
                metrics_file='training_metrics.csv', label_smoothing=0.0,
                early_stopping_patience=0, clip_grad=0.0):
    """
    Обучение модели с поддержкой:
    - label_smoothing
    - early stopping
    - gradient clipping
    """
    unique = np.unique(target)
    y_list = target.tolist()
    weight = class_weight.compute_class_weight(class_weight='balanced', classes=unique, y=y_list)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weight, dtype=torch.float32),
        label_smoothing=label_smoothing
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, val_accuracies, val_f1s = [], [], []
    best_val_f1 = 0.0
    patience_counter = 0

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_accuracy', 'val_f1', 'lr'])

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Валидация
        model.eval()
        correct, total = 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = correct / total
        val_accuracies.append(accuracy)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        val_f1s.append(f1)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f} | F1: {f1:.4f} | LR: {current_lr:.6f}')

        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f'{avg_loss:.6f}', f'{accuracy:.6f}', f'{f1:.6f}', f'{current_lr:.6f}'])

        # Early stopping по F1
        if early_stopping_patience > 0:
            if f1 > best_val_f1:
                best_val_f1 = f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1} (best F1: {best_val_f1:.4f})')
                    break

    actual_epochs = len(train_losses)

    # Сохранение графика
    curve_file = metrics_file.replace('metrics', 'curve').replace('.csv', '.png')
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(range(1, actual_epochs + 1), train_losses, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val Metrics', color='tab:blue')
    ax2.plot(range(1, actual_epochs + 1), val_accuracies, color='tab:blue', label='Val Accuracy')
    ax2.plot(range(1, actual_epochs + 1), val_f1s, color='tab:green', label='Val F1', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.suptitle('Training Progress')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=3)
    fig.tight_layout()
    fig.savefig(curve_file, dpi=150)
    plt.close(fig)
    print(f'Graph saved to {curve_file}')
    print(f'Metrics saved to {metrics_file}')

    return train_losses, val_accuracies, val_f1s
