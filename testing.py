import torch
import os
from dataloader import CarDataset
from torch.utils.data import DataLoader

# Get path of model savefile
model_savefile = 'optimal_model.pt'
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', model_savefile)

# Load model
model = torch.load(path)

# Load testdata
data_path = '.../test_data'
test_data = CarDataset(data_path)
dataloader = DataLoader(dataset=test_data, batch_size=1)
dataiter = iter(dataloader)

# Define dicecoefficient
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# Compute dice coefficients
dicecoeffs = []
for _ in range(len(test_data)):
    image, target = next(dataiter)
    pred = model(image)
    dicecoeffs.append(dice_coeff(pred, target))
