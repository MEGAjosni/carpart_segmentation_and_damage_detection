import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

from eval_functions import DiceCoefficients
from dataloader import CarDataset
from userpaths import *
import torch
from torch.utils.data import DataLoader
import numpy as np


batchsize = 8
test_data = CarDataset(test_folder,changelabel=True)
test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle = False)

models = [name for name in os.listdir(model_saves) if name[-2:] in ['pt', 'pth']]

print("-"*50)
for model in models:
    overall_dice, nobg_dice, per_class_dice = DiceCoefficients(torch.load(os.path.join("models", model), 'cpu'), test_loader)
    print("Tested Model: " + model[:-3] + "\n")
    print("Per class DiceLoss: {}".format(np.round_(per_class_dice, decimals=4)))
    print("Overall DiceLoss excluding background: {}".format(np.round_(nobg_dice, decimals=4)))
    print("Overall DiceLoss: {}".format(np.round_(overall_dice, decimals=4)))
    print("-"*50)
