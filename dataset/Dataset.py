import torch
import pandas as pd
import MetaTrader5 as mt5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inits = False

def __init__():
    global inits
    inits = mt5.initialize('C:/demoalfaforex/terminal64.exe')
    if inits:
        print('Initialization complete')
    else:
        print('Initialization failed')

def create_dataset(symbol: str) -> list:
    """
    Для вызова текущей функции будет проверка на инициализацию и подключение к терминалу.
    Далее идет выгрузка данных из терминала Metatrader5
    Удаляются не нужные колонки.

    После чего происходит обработка данных. Берутся все данные за 30 свечей истории и предсказывается на 3 свечи вперед.
    Проходится в цикле по всей истории и собираюся в список кортежей, где первое значение это данные из истории,
    второе значение это таргет для первой модели и третье значение это список таргетов для третьей модели

    :param symbol:
    :return:
    """
    global inits
    if not inits:
        __init__()

    df = pd.DataFrame(mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 99000))
    df.drop(columns=['tick_volume', 'real_volume'], inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

    past_bar = 30
    predict = 3

    lst_data = []

    for _i in range(past_bar, len(df) - predict):
        open_lst = list(df['open'][_i - past_bar: _i])
        high_lst = list(df['high'][_i - past_bar: _i])
        low_lst = list(df['low'][_i - past_bar: _i])
        close_lst = list(df['close'][_i - past_bar: _i])

        tmp_data = open_lst+ high_lst+ low_lst+close_lst
        tmp_data = [round(num, 5) for num in tmp_data]

        tmp_target_one = 0
        tmp_target_two = []
        past_close = df['close'][_i]
        past_data = df['time'][_i] # временная строка для тестирования текущей временной метки
        point = 0.00001
        for _y in range(_i, _i + predict):
            low_diff = (past_close - df['close'][_y + 1]) / point
            high_diff = (df['close'][_y + 1] - past_close) / point

            if low_diff > 50 or high_diff > 50:
                tmp_target_one = 1
                tmp_target_two = [1, 0] if high_diff > low_diff else [0, 1]
                break
            else:
                tmp_target_two = [1, 0] if high_diff > low_diff else [0, 1]

        lst_data.append((tmp_data, tmp_target_one, tmp_target_two))

    return lst_data




if __name__ == '__main__':
    dataset = create_dataset("EURUSDrfd")
    print(dataset)
