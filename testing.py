import torch
import os
from dataloader import CarDataset, plot_things
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_toolbelt.losses.functional import soft_dice_score
import numpy as np


#%%
# Get path of model savefile
model_savefile = 'unet.pt'
path = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\\"

# Load model
model = torch.load(path+model_savefile)
model.cpu()
#%%
#model = vae.cpu()
# Load testdata
data_path = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\test_data"



test_data = CarDataset(data_path,changelabel=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)



# Compute dice coefficients
dicecoeffs = []
for inputs,targets in test_loader:
    pred = model(inputs)
    plot_things(inputs,targets,pred,idx=0)
    pred = torch.argmax(pred, dim = 1)
    targets = torch.argmax(targets,dim=1)
    # divide pred by 7 so all values are between 0 and 1
    dicecoeffs.append(soft_dice_score(pred/7,targets).item())
    
    
print(np.mean(dicecoeffs))
