import torch
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init = False

def __init__():
    global init
    init = mt5.initialize('C:/demoalfaforex/terminal64.exe')
    if init:
        print('Initialization complete')
    else:
        print('Initialization failed')

def create_dataset(symbol: str, window_len: int = 30 , predict_len: int = 3):
    """
    Для вызова текущей функции будет проверка на инициализацию и подключение к терминалу.
    Далее идет выгрузка данных из терминала Metatrader5
    Удаляются ненужные колонки.

    После чего происходит обработка данных. Берутся все данные за 30 свечей истории и предсказывается на 3 свечи вперед.
    Проходится в цикле по всей истории и собираются в список кортежей, где первое значение это данные из истории,
    второе значение это таргет для первой модели и третье значение это список таргетов для третьей модели

    :param predict_len:
    :param window_len:
    :param symbol:
    :return:
    """
    global init
    if not init:
        __init__()
    point = 0.00001

    df = pd.DataFrame(mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 99000))
    df.drop(columns=['tick_volume', 'real_volume'], inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    print(df)
    features = ['open', 'high', 'low', 'close']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])


    X, y = [], []
    for i in range(len(scaled_data) - window_len):
        X.append(scaled_data[i:i+window_len])
        past_close = df['close'][i+window_len - 1]
        tmp = -1
        for _y in range(i + window_len, i + window_len + predict_len):
            low_diff = (past_close - df['close'][_y]) / point
            high_diff = (df['close'][_y] - past_close) / point

            if high_diff > 100:
                tmp = 0
                break
            elif low_diff > 100:
                tmp = 1
                break
        y.append(tmp)

    X = np.array(X)
    y = np.array(y)


    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]


    # Преобразование в тензоры PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor), scaler

if __name__ == '__main__':
    create_dataset(symbol='EURUSDrfd', window_len=30, predict_len=3)