import torch
from models import Models

import dataset.Dataset as Dataset

if __name__ == "__main__":
    # (X_train, y_train), (X_test, y_test), (X_train_tensor_TWO, y_train_tensor_TWO), (X_test_tensor_TWO, y_test_tensor_TWO), scaler = Dataset.create_dataset('EURUSDrfd')
    (X_train, y_train), (X_test, y_test), scaler = Dataset.create_dataset('EURUSDrfd')
    # print(f'X_train: {X_train.shape}, y_train: {y_train.shape}, X_train_tensor_TWO: {X_train_tensor_TWO.shape}, scaler: {scaler}')


    # # DataLoader для батчей
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # # Инициализация модели
    n_features = X_train.shape[3]
    model = Models.TradingCNN(n_features=n_features)

    print(model)
    #
    # # Обучение
    Models.train_model(model, train_loader, val_loader, y_train, epochs=50)