import torch
import os
from dataloader import CarDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#%%
# Get path of model savefile
model_savefile = 'vae_v2_0.pt'
path = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\\"

# Load model
model = torch.load(path+model_savefile)
model.cpu()
#%%
#model = vae.cpu()
# Load testdata
data_path = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\test_data"


imagewidth = 128
augmentations_test = transforms.Compose([transforms.Resize(size = imagewidth),
                                    ])
test_data = CarDataset(data_path, transform=augmentations_test)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# Define dicecoefficient
def dice_coeff(mask1, mask2):
    #hack to ensure only 1's and 0's
    mask1[mask1>0] = 1
    mask2[mask2>0] = 1
    
    intersect = torch.sum(mask1*mask2)
    fsum = torch.sum(mask1)
    ssum = torch.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = torch.mean(dice)
    #dice = round(dice, 3) # for easy reading
    return dice    

# Compute dice coefficients
dicecoeffs = []
for inputs,targets in test_loader:
    pred = model(inputs)
    dicecoeffs.append(dice_coeff(pred, targets).item())
