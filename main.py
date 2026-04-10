import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models import Models

DATA_FILE = 'trading_data.h5'
WINDOW_LEN = 30
PREDICT_LEN = 10
POINT = 0.00001

if __name__ == "__main__":
    df = pd.read_hdf(DATA_FILE, key='data')
    df.drop(columns=['tick_volume', 'real_volume'], inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)

    features = ['open', 'high', 'low', 'close']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled_data) - WINDOW_LEN - PREDICT_LEN):
        X.append(scaled_data[i:i + WINDOW_LEN])
        past_close = df['close'].iloc[i + WINDOW_LEN - 1]
        tmp = -1
        for _y in range(i + WINDOW_LEN, i + WINDOW_LEN + PREDICT_LEN):
            low_diff = (past_close - df['close'].iloc[_y]) / POINT
            high_diff = (df['close'].iloc[_y] - past_close) / POINT
            if high_diff > 200:
                tmp = 0
                break
            elif low_diff > 200:
                tmp = 1
                break
        y.append(tmp)

    X = np.array(X)
    y = np.array(y)

    # BUY/SELL -> 1, WAIT -> 0
    y[y == 0] = 1
    y[y == -1] = 0

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # DataLoader для батчей
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # Инициализация модели
    n_features = X_train.shape[3]
    model = Models.TradingCNN(n_features=n_features)

    print(model)

    # Обучение
    Models.train_model(model, train_loader, val_loader, y_train, epochs=50)
