"""
Эксперимент A+B+C: расширенные фичи + LSTM + стабилизация обучения
- Label smoothing 0.1
- Batch size 128
- Early stopping (patience=7)
- Dropout 0.4
- Gradient clipping (max_norm=1.0)
- WeightedRandomSampler для балансировки батчей
"""
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models import Models

DATA_FILE = 'trading_data.h5'
WINDOW_LEN = 30
PREDICT_LEN = 10
POINT = 0.00001
PIP_THRESHOLD = 100
BATCH_SIZE = 128
EPOCHS = 30
METRICS_FILE = 'training_metrics_abc.csv'


def add_features(df):
    """Добавление технических индикаторов и производных фичей"""
    df['returns'] = df['close'].pct_change()
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    df['ema10'] = df['close'].ewm(span=10).mean()
    df['ema30'] = df['close'].ewm(span=30).mean()
    df['ema_diff'] = df['ema10'] - df['ema30']

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = pd.read_hdf(DATA_FILE, key='data')
    df.drop(columns=['tick_volume', 'real_volume'], inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)

    df = add_features(df)

    features = ['open', 'high', 'low', 'close',
                'returns', 'body', 'upper_shadow', 'lower_shadow',
                'ema_diff', 'rsi', 'atr', 'bb_position']

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
            if high_diff > PIP_THRESHOLD:
                tmp = 0
                break
            elif low_diff > PIP_THRESHOLD:
                tmp = 1
                break
        y.append(tmp)

    X = np.array(X)
    y = np.array(y)

    y[y == 0] = 1
    y[y == -1] = 0

    count_0 = np.sum(y == 0)
    count_1 = np.sum(y == 1)
    print(f'Class distribution: WAIT={count_0} ({count_0/len(y)*100:.1f}%), SIGNAL={count_1} ({count_1/len(y)*100:.1f}%)')

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # WeightedRandomSampler для балансировки классов в батчах
    class_counts = np.bincount(y_train.numpy())
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.numpy()]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # C: усиленный dropout 0.4
    n_features = X_train.shape[2]
    model = Models.TradingLSTM(n_features=n_features, dropout=0.4)

    print(model)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    # C: label_smoothing=0.1, early_stopping=7, gradient clipping=1.0
    Models.train_model(
        model, train_loader, val_loader, y_train,
        epochs=EPOCHS,
        metrics_file=METRICS_FILE,
        label_smoothing=0.1,
        early_stopping_patience=7,
        clip_grad=1.0
    )
