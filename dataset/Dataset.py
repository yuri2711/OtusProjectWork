import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import MetaTrader5 as mt5


# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inits = False

def __init__():
    global inits
    inits = mt5.initialize('C:/demoalfaforex/terminal64.exe')
    if inits:
        print('Initialization complete')
    else:
        print('Initialization failed')

def create_dataset():
    global inits
    if not inits:
        __init__()




if __name__ == '__main__':
    __init__()
